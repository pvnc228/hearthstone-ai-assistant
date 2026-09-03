"""
HDT Replay Reader and Game History extractor.
Streams and parses .hdtreplay zip files and links metadata from DeckStats.xml.
"""

import logging
import os
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

from src.card_db import CardDatabase
from .log_parser import parse_power_log_lines
from .state_tracker import DecisionPoint, GameStateTracker, OptionDecision, TurnSnapshot

logger = logging.getLogger(__name__)

DEFAULT_HDT_DIR = os.path.expandvars(r"%APPDATA%\HearthstoneDeckTracker")
DEFAULT_REPLAY_DIR = Path(DEFAULT_HDT_DIR) / "Replays"
DEFAULT_DECK_STATS_XML = Path(DEFAULT_HDT_DIR) / "DeckStats.xml"


@dataclass
class GameMetadata:
    game_id: str = ""
    replay_file: str = ""
    player_name: str = ""
    opponent_name: str = ""
    player_hero: str = ""
    opponent_hero: str = ""
    player_hero_card_id: str = ""
    opponent_hero_card_id: str = ""
    result: str = ""  # Win, Loss
    game_mode: str = ""  # Ranked, Casual, Arena
    format: str = ""  # Standard, Wild
    deck_name: str = ""
    turns_count: int = 0
    start_time: str = ""
    end_time: str = ""
    was_conceded: bool = False
    friendly_player_id: int = 1
    opponent_player_id: int = 2


@dataclass
class GameReplay:
    metadata: GameMetadata
    turn_snapshots: List[TurnSnapshot] = field(default_factory=list)
    decision_points: List[DecisionPoint] = field(default_factory=list)
    option_decisions: List[OptionDecision] = field(default_factory=list)

    @property
    def friendly_turns(self) -> List[TurnSnapshot]:
        """All snapshots where the friendly player was taking their turn
        (including pass turns with no actions — needed for tempo-loss analysis)."""
        return [ts for ts in self.turn_snapshots if ts.is_friendly_turn]


def load_deck_stats_index(deck_stats_path: Optional[Path | str] = None) -> Dict[str, GameMetadata]:
    """
    Parses DeckStats.xml into a dictionary keyed by replay_file basename.
    """
    path = Path(deck_stats_path) if deck_stats_path else DEFAULT_DECK_STATS_XML
    if not path.exists():
        logger.warning("DeckStats.xml not found at %s", path)
        return {}

    tree = ET.parse(path)
    root = tree.getroot()
    index: Dict[str, GameMetadata] = {}

    for game_node in root.findall(".//Game"):
        replay_file = game_node.findtext("ReplayFile") or ""
        if not replay_file:
            continue

        meta = GameMetadata(
            game_id=game_node.findtext("GameId") or "",
            replay_file=replay_file,
            player_name=game_node.findtext("PlayerName") or "HappyBread#21597",
            opponent_name=game_node.findtext("OpponentName") or "Opponent",
            player_hero=game_node.findtext("PlayerHero") or "",
            opponent_hero=game_node.findtext("OpponentHero") or "",
            player_hero_card_id=game_node.findtext("PlayerHeroCardId") or "",
            opponent_hero_card_id=game_node.findtext("OpponentHeroCardId") or "",
            result=game_node.findtext("Result") or "",
            game_mode=game_node.findtext("GameMode") or "",
            format=game_node.findtext("Format") or "Wild",
            deck_name=game_node.findtext("DeckName") or "",
            turns_count=int(game_node.findtext("Turns") or "0"),
            start_time=game_node.findtext("StartTime") or "",
            end_time=game_node.findtext("EndTime") or "",
            was_conceded=(game_node.findtext("WasConceded", "").lower() == "true"),
            friendly_player_id=int(game_node.findtext("FriendlyPlayerId") or "1"),
            opponent_player_id=int(game_node.findtext("OpponentPlayerId") or "2"),
        )
        index[replay_file] = meta

    logger.info("Loaded %d game records from DeckStats.xml", len(index))
    return index


def _iter_decoded_log_lines(raw_lines: Iterable[bytes]) -> Generator[str, None, None]:
    """Decodes normal HDT lines and the escaped-newline single-line variant."""
    for raw_line in raw_lines:
        text = raw_line.decode("utf-8", errors="replace")
        # Some HDT exports serialize the whole log as one physical line while
        # retaining literal ``\\n`` separators. Normalize only that shape;
        # normal log lines may legitimately contain backslash characters.
        if "\\n" in text and text.count("\n") <= 1:
            text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
            yield from text.splitlines(keepends=True)
        else:
            yield text


def parse_replay_file(
    replay_path: Path | str,
    card_db: Optional[CardDatabase] = None,
    metadata: Optional[GameMetadata] = None,
    deck_stats_index: Optional[Dict[str, GameMetadata]] = None,
) -> GameReplay:
    """
    Parses a single .hdtreplay zip archive in streaming mode.
    """
    path = Path(replay_path)
    if not path.exists():
        raise FileNotFoundError(f"Replay archive not found: {path}")

    # Resolve metadata
    if metadata is None:
        if deck_stats_index is not None and path.name in deck_stats_index:
            metadata = deck_stats_index[path.name]
        else:
            metadata = GameMetadata(replay_file=path.name, player_name="HappyBread#21597")

    tracker = GameStateTracker(card_db=card_db, friendly_player_name=metadata.player_name)
    tracker.friendly_player_id = metadata.friendly_player_id

    # Stream output_log.txt directly from zip
    with zipfile.ZipFile(path, "r") as zf:
        if "output_log.txt" not in zf.namelist():
            raise ValueError(f"Invalid replay archive {path.name}: output_log.txt not found.")

        with zf.open("output_log.txt", "r") as log_file:

            for event in parse_power_log_lines(_iter_decoded_log_lines(log_file)):
                tracker.process_event(event)

    tracker.finalize()

    return GameReplay(
        metadata=metadata,
        turn_snapshots=tracker.turn_snapshots,
        decision_points=tracker.decision_points,
        option_decisions=tracker.option_decisions,
    )


def iterate_replays(
    replay_dir: Optional[Path | str] = None,
    deck_stats_path: Optional[Path | str] = None,
    card_db: Optional[CardDatabase] = None,
    filter_ranked_wins: bool = False,
    max_count: Optional[int] = None,
) -> Generator[GameReplay, None, None]:
    """
    Yields parsed GameReplay instances across the HDT replays directory.
    """
    r_dir = Path(replay_dir) if replay_dir else DEFAULT_REPLAY_DIR
    deck_stats = load_deck_stats_index(deck_stats_path)
    db = card_db or CardDatabase(auto_load=True)

    count = 0
    for file_path in sorted(r_dir.glob("*.hdtreplay"), key=lambda path: path.name.casefold()):
        meta = deck_stats.get(file_path.name)
        if filter_ranked_wins:
            if not meta or meta.result != "Win" or meta.game_mode != "Ranked":
                continue

        try:
            replay = parse_replay_file(
                replay_path=file_path,
                card_db=db,
                metadata=meta,
            )
            yield replay
            count += 1
            if max_count is not None and count >= max_count:
                break
        except Exception as e:
            logger.warning("Failed to parse replay %s: %s", file_path.name, e)
