"""
Batch dataset generator for HSReplay.net match archives.
Parses all downloaded .hsreplay.xml files and writes (Prompt -> Completion)
turn pairs into JSONL for fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from src.card_db import CardDatabase
from .dataset_generator import format_turn_completion, format_turn_prompt
from .hsreplay_downloader import HSReplayDownloader
from .hsreplay_xml_parser import parse_hsreplay_xml_file

logger = logging.getLogger(__name__)

DEFAULT_HSREPLAY_DATASET = Path("data/processed/train_hsreplay_actions.jsonl")


def build_hsreplay_dataset(
    max_games: Optional[int] = None,
    output_path: Path = DEFAULT_HSREPLAY_DATASET,
    download_missing: bool = False,
    max_workers: int = 2,
) -> int:
    """
    Parses .hsreplay.xml files from data/replays_hsreplay and writes
    (Prompt -> Completion) turn pairs into JSONL.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card_db = CardDatabase(auto_load=True)
    downloader = HSReplayDownloader()

    if download_missing:
        print(f"📥 Проверка и загрузка недостающих реплеев с HSReplay.net (лимит: {max_games or 'все'})...", flush=True)
        xml_paths = downloader.download_all_replays(only_wins=True, max_workers=max_workers, limit=max_games)
    else:
        xml_paths = sorted(list(downloader.output_dir.glob("*.hsreplay.xml")))
        if max_games:
            xml_paths = xml_paths[:max_games]

    print(f"\n🔄 Парсинг {len(xml_paths)} скачанных XML файлов и генерация обучающих пар...", flush=True)
    records_count = 0
    games_processed = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, xf in enumerate(xml_paths, start=1):
            try:
                replay = parse_hsreplay_xml_file(xf, card_db=card_db)
                if not any(ts.is_friendly_turn and ts.actions for ts in replay.turn_snapshots):
                    continue

                games_processed += 1
                for snap in replay.friendly_turns:
                    if not snap.actions:
                        continue  # dataset keeps only turns with actions to imitate
                    prompt = format_turn_prompt(
                        snapshot=snap,
                        player_hero=replay.metadata.player_hero,
                        opponent_hero=replay.metadata.opponent_hero,
                    )
                    completion = format_turn_completion(snap.actions)

                    record = {
                        "game_id": replay.metadata.game_id,
                        "turn": snap.turn_number,
                        "player_hero": replay.metadata.player_hero,
                        "opponent_hero": replay.metadata.opponent_hero,
                        "prompt": prompt,
                        "completion": completion,
                        "actions_count": len(snap.actions),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_count += 1

                if idx % 50 == 0 or idx == len(xml_paths):
                    print(f"  Обработано {idx}/{len(xml_paths)} реплеев ({(idx/len(xml_paths))*100:.1f}%) -> {records_count} обучающих пар", flush=True)

            except Exception as e:
                logger.warning("Error processing %s: %s", xf.name, e)

    print(f"✅ Готово! Извлечено {records_count} обучающих ходов из {games_processed} матчей -> {output_path}", flush=True)
    return records_count


if __name__ == "__main__":
    build_hsreplay_dataset(download_missing=False)
