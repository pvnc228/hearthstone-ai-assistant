"""
Main entrypoint for Hearthstone AI Assistant & Replay Simulator TUI.
Provides an interactive, full-screen tactical terminal player
for live games and offline replay playback.

Usage:
  # Replay simulation & tactical review in TUI:
  python -m src.live.main --replay "path/to/game.hdtreplay" --speed 1.0

  # Watch live game in TUI:
  python -m src.live.main --live

  # Enable Ollama LLM tactical reasoning:
  python -m src.live.main --replay "path/to/game.hdtreplay" --llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.card_db import CardDatabase
from .dispatcher import LiveAdvisorDispatcher
from .event_hub import LiveGameSession
from .tui import HearthstoneTUIApp
from .watcher import HearthstoneLogWatcher, MockLogWatcher


def run_tui(
    replay_file: Optional[str] = None,
    is_live: bool = False,
    log_dir: Optional[str] = None,
    player_name: str = "HappyBread#21597",
    enable_llm: bool = False,
    model_name: Optional[str] = None,
    speed: float = 1.0,
) -> None:
    """Launches the interactive Textual TUI simulator and assistant."""
    card_db = CardDatabase(auto_load=True)
    dispatcher = LiveAdvisorDispatcher(
        card_db=card_db,
        enable_llm=enable_llm,
        model_name=model_name,
    )
    session = LiveGameSession(
        friendly_player_name=player_name,
        card_db=card_db,
        dispatcher=dispatcher,
    )

    if replay_file:
        source_p = Path(replay_file)
        if not source_p.exists():
            print(f"❌ Файл реплея не найден: {replay_file}")
            sys.exit(1)
        source_label = f"Replay: {source_p.name}"

        def line_stream_factory():
            return MockLogWatcher(source=source_p, speed=speed).stream_lines()

    else:
        log_watcher = HearthstoneLogWatcher(log_dir=log_dir, catch_up=True)
        active_log = log_watcher.find_active_power_log()
        source_label = str(active_log.name if active_log else "D:\\Hearthstone\\Logs")

        def line_stream_factory():
            return log_watcher.stream_lines(poll_interval=0.1)

    app = HearthstoneTUIApp(
        session=session,
        line_generator_factory=line_stream_factory,
        source_label=source_label,
    )
    app.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hearthstone AI Assistant — Replay Simulator & Tactical TUI Player"
    )
    parser.add_argument(
        "--replay", "--mock",
        dest="replay",
        type=str,
        default=None,
        help="Path to .hdtreplay or Power.log file to simulate and review",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Watch live Hearthstone process instead of replay",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for replay mode (default: 1.0)",
    )
    parser.add_argument(
        "--player",
        type=str,
        default="HappyBread#21597",
        help="Friendly player BattleTag (default: HappyBread#21597)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable Ollama LLM for tactical commentary",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: auto-detect)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom Hearthstone log directory for live mode",
    )
    args = parser.parse_args()

    # Default to live if no replay specified
    if not args.replay and not args.live:
        args.live = True

    run_tui(
        replay_file=args.replay,
        is_live=args.live,
        log_dir=args.log_dir,
        player_name=args.player,
        enable_llm=args.llm,
        model_name=args.model,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
