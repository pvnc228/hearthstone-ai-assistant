"""
Hearthstone Parser & Replay Pipeline package.
"""

from .dataset_generator import format_turn_completion, format_turn_prompt, generate_dataset
from .log_parser import PowerEvent, parse_entity_ref, parse_power_log_lines
from .replay_reader import (
    DEFAULT_DECK_STATS_XML,
    DEFAULT_HDT_DIR,
    DEFAULT_REPLAY_DIR,
    GameMetadata,
    GameReplay,
    iterate_replays,
    load_deck_stats_index,
    parse_replay_file,
)
from .state_tracker import (
    Entity,
    GameStateTracker,
    PlayerAction,
    PlayerState,
    TurnSnapshot,
)

__all__ = [
    "PowerEvent",
    "parse_entity_ref",
    "parse_power_log_lines",
    "Entity",
    "PlayerState",
    "PlayerAction",
    "TurnSnapshot",
    "GameStateTracker",
    "GameMetadata",
    "GameReplay",
    "DEFAULT_REPLAY_DIR",
    "load_deck_stats_index",
    "parse_replay_file",
    "iterate_replays",
    "format_turn_prompt",
    "format_turn_completion",
    "generate_dataset",
]
