"""
Production-grade downloader for personal match replays from HSReplay.net.
Uses TLS-fingerprint impersonation (curl_cffi) to bypass Cloudflare WAF,
fetches full game history with pagination, and downloads .hsreplay.xml files from AWS S3 in parallel.
"""

import gzip
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from curl_cffi import requests

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_HSREPLAY_DIR = Path("data/replays_hsreplay")
GAMES_INDEX_FILE = DEFAULT_CACHE_DIR / "hsreplay_games_index.json"
SESSION_FILE = DEFAULT_CACHE_DIR / "hsreplay_session.json"


@dataclass
class HSReplayGameMeta:
    shortid: str
    won: bool
    num_turns: int
    format: int
    match_start: str
    match_end: str
    player_name: str
    player_hero: str
    opponent_name: str
    opponent_hero: str
    build: int
    rank: Optional[int] = None
    legend_rank: Optional[int] = None


class HSReplayDownloader:
    """
    Client to scrape user's match history and download replay XMLs from HSReplay.net.
    """

    def __init__(
        self,
        sessionid: Optional[str] = None,
        cf_clearance: Optional[str] = None,
        default_account: str = "2-28379340",
        output_dir: Path = DEFAULT_HSREPLAY_DIR,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if sessionid and cf_clearance:
            self.sessionid = sessionid
            self.cf_clearance = cf_clearance
            self.default_account = default_account
            self._save_session()
        else:
            loaded = self._load_session()
            self.sessionid = loaded.get("sessionid", "")
            self.cf_clearance = loaded.get("cf_clearance", "")
            self.default_account = loaded.get("default_account", "2-28379340")

    def _save_session(self) -> None:
        data = {
            "sessionid": self.sessionid,
            "cf_clearance": self.cf_clearance,
            "default_account": self.default_account,
        }
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_session(self) -> Dict[str, str]:
        if SESSION_FILE.exists():
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _create_session(self) -> requests.Session:
        s = requests.Session(impersonate="chrome120")
        cookie_header = f"sessionid={self.sessionid}; cf_clearance={self.cf_clearance}; default-account={self.default_account};"
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Cookie": cookie_header,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://hsreplay.net/games/mine/",
        })
        return s

    def scan_all_games_metadata(self, max_pages: Optional[int] = None) -> List[HSReplayGameMeta]:
        """
        Scans all paginated game records from https://hsreplay.net/api/v1/games/
        and saves index to data/cache/hsreplay_games_index.json.
        """
        session = self._create_session()
        url = "https://hsreplay.net/api/v1/games/"
        games: List[HSReplayGameMeta] = []
        page = 1

        print("🔍 Начало сканирования истории матчей с HSReplay.net...")

        while url:
            for attempt in range(3):
                try:
                    resp = session.get(url, timeout=15)
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:100]}")

                    data = json.loads(resp.text)
                    results = data.get("results", [])

                    for r in results:
                        g_meta = r.get("global_game", {})
                        f_player = r.get("friendly_player", {})
                        o_player = r.get("opposing_player", {})

                        meta = HSReplayGameMeta(
                            shortid=r.get("shortid", ""),
                            won=bool(r.get("won")),
                            num_turns=g_meta.get("num_turns", 0),
                            format=g_meta.get("format", 0),
                            match_start=g_meta.get("match_start", ""),
                            match_end=g_meta.get("match_end", ""),
                            player_name=f_player.get("name", "Player"),
                            player_hero=f_player.get("hero_class_name", "UNKNOWN"),
                            opponent_name=o_player.get("name", "Opponent"),
                            opponent_hero=o_player.get("hero_class_name", "UNKNOWN"),
                            build=r.get("build", 0),
                            rank=f_player.get("rank"),
                            legend_rank=f_player.get("legend_rank"),
                        )
                        games.append(meta)

                    wins = sum(1 for g in games if g.won)
                    print(f"  Страница {page:02d}: получено {len(results)} игр (Всего в индексе: {len(games)}, Побед: {wins})")

                    url = data.get("next")
                    page += 1
                    time.sleep(0.15)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"❌ Ошибка на странице {page} после 3 попыток: {e}")
                        url = None
                        break
                    time.sleep(1.0)

            if max_pages and page > max_pages:
                break

        # Save to disk
        with open(GAMES_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(g) for g in games], f, indent=2, ensure_ascii=False)

        print(f"✅ Индекс сохранен: {len(games)} матчей записано в {GAMES_INDEX_FILE}")
        return games

    def download_single_replay(self, shortid: str, session: Optional[requests.Session] = None) -> Optional[Path]:
        """
        Fetches presigned S3 XML link and downloads decompressed .hsreplay.xml.
        """
        target_path = self.output_dir / f"{shortid}.hsreplay.xml"
        if target_path.exists() and target_path.stat().st_size > 500:
            return target_path

        s = session or self._create_session()
        api_url = f"https://hsreplay.net/api/v1/games/{shortid}/"

        for attempt in range(5):
            try:
                resp = s.get(api_url, headers={"Referer": f"https://hsreplay.net/replay/{shortid}"}, timeout=15)
                if resp.status_code == 429:
                    wait_time = 2.0 * (attempt + 1)
                    time.sleep(wait_time)
                    continue

                if resp.status_code != 200:
                    logger.warning("Game %s returned status %d", shortid, resp.status_code)
                    return None

                meta_json = json.loads(resp.text)
                xml_s3_url = meta_json.get("replay_xml")
                if not xml_s3_url:
                    return None

                # S3 direct download
                resp_s3 = s.get(xml_s3_url, timeout=20)
                if resp_s3.status_code != 200:
                    logger.warning("S3 download failed for %s: %d", shortid, resp_s3.status_code)
                    return None

                # Step 3: Decompress if gzipped, or use text
                if resp_s3.content.startswith(b"\x1f\x8b"):
                    xml_text = gzip.decompress(resp_s3.content).decode("utf-8")
                else:
                    xml_text = resp_s3.text

                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(xml_text)

                return target_path

            except Exception as e:
                if attempt == 4:
                    logger.warning("Failed to download replay %s: %s", shortid, e)
                    return None
                time.sleep(1.5)

        return None

    def download_all_replays(
        self,
        only_wins: bool = True,
        max_workers: int = 2,
        limit: Optional[int] = None,
    ) -> List[Path]:
        """
        Downloads all available replay XMLs in parallel.
        """
        if not GAMES_INDEX_FILE.exists():
            self.scan_all_games_metadata()

        with open(GAMES_INDEX_FILE, "r", encoding="utf-8") as f:
            raw_games = json.load(f)

        candidates = [g["shortid"] for g in raw_games if (not only_wins or g.get("won"))]
        if limit:
            candidates = candidates[:limit]

        already_downloaded = sum(1 for cid in candidates if (self.output_dir / f"{cid}.hsreplay.xml").exists())
        to_download = [cid for cid in candidates if not (self.output_dir / f"{cid}.hsreplay.xml").exists()]

        print(f"📥 Скачивание XML реплеев:")
        print(f"  Всего кандидатов (победы): {len(candidates)}")
        print(f"  Уже скачано локально:      {already_downloaded}")
        print(f"  Осталось скачать:          {len(to_download)}")

        if not to_download:
            print("✅ Все реплеи уже скачаны!")
            return [self.output_dir / f"{cid}.hsreplay.xml" for cid in candidates]

        downloaded_paths: List[Path] = []
        count = 0
        total = len(to_download)

        def worker_task(short_id: str) -> Optional[Path]:
            time.sleep(0.35)
            thread_session = self._create_session()
            return self.download_single_replay(short_id, session=thread_session)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(worker_task, cid): cid for cid in to_download}
            for future in as_completed(future_to_id):
                cid = future_to_id[future]
                count += 1
                try:
                    path = future.result()
                    if path:
                        downloaded_paths.append(path)
                    if count % 20 == 0 or count == total:
                        print(f"  Прогресс: {count}/{total} реплеев обработано ({(count/total)*100:.1f}%) | Сохранено: {len(downloaded_paths)}")
                except Exception as e:
                    logger.warning("Download error on %s: %s", cid, e)

        print(f"🎉 Завершено! Успешно сохранено {len(downloaded_paths)} новых .hsreplay.xml файлов в {self.output_dir}")
        return [self.output_dir / f"{cid}.hsreplay.xml" for cid in candidates if (self.output_dir / f"{cid}.hsreplay.xml").exists()]
