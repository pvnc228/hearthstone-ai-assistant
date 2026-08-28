"""
Dataset Generator for Hearthstone AI Assistant.
Extracts [GameState -> Winning Actions] pairs from 1,041 .hdtreplay archives
and formats them into JSONL for LLM fine-tuning and evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.card_db import CardDatabase, format_board_minion, format_card_compact
from .replay_reader import GameReplay, iterate_replays, load_deck_stats_index
from .state_tracker import PlayerAction, TurnSnapshot

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "train_actions.jsonl"


def format_turn_prompt(snapshot: TurnSnapshot, player_hero: str, opponent_hero: str) -> str:
    """
    Constructs a concise, structured markdown prompt representing the board state at turn start.
    """
    lines = []
    lines.append(f"### Ход {snapshot.turn_number}")
    lines.append(
        f"**Ваш герой ({player_hero})**: {snapshot.friendly_hero['health']} HP (+{snapshot.friendly_hero['armor']} брони)"
    )
    lines.append(
        f"**Герой противника ({opponent_hero})**: {snapshot.opponent_hero['health']} HP (+{snapshot.opponent_hero['armor']} брони)"
    )
    lines.append(f"**Доступная мана**: {snapshot.friendly_mana}/{snapshot.friendly_max_mana}")

    # Hand
    if snapshot.friendly_hand:
        hand_items = []
        for c in snapshot.friendly_hand:
            cost = c.get("cost", 0)
            atk = c.get("attack")
            hp = c.get("health")
            name = c.get("name", "")
            stats_str = f" {atk}/{hp}" if atk is not None and hp is not None else ""
            hand_items.append(f"[{cost}м{stats_str}] {name}")
        lines.append("**Рука**: " + ", ".join(hand_items))
    else:
        lines.append("**Рука**: [Пусто]")

    # Friendly Board
    if snapshot.friendly_board:
        b_items = []
        for m in snapshot.friendly_board:
            b_items.append(
                format_board_minion(
                    name=m.get("name", "Существо"),
                    atk=m.get("attack", 0),
                    hp=m.get("health", 0),
                    max_hp=m.get("max_health", 0),
                    can_attack=m.get("can_attack", False),
                    is_taunt=m.get("is_taunt", False),
                    is_divine_shield=m.get("is_divine_shield", False),
                    is_stealthed=m.get("is_stealthed", False),
                    is_frozen=m.get("is_frozen", False),
                    is_reborn=m.get("is_reborn", False),
                    is_silenced=m.get("is_silenced", False),
                    is_dormant=m.get("is_dormant", False),
                )
            )
        lines.append("**Ваш стол**: " + "; ".join(b_items))
    else:
        lines.append("**Ваш стол**: [Пусто]")

    # Opponent Board
    if snapshot.opponent_board:
        ob_items = []
        for m in snapshot.opponent_board:
            ob_items.append(
                format_board_minion(
                    name=m.get("name", "Существо"),
                    atk=m.get("attack", 0),
                    hp=m.get("health", 0),
                    max_hp=m.get("max_health", 0),
                    can_attack=False,
                    is_taunt=m.get("is_taunt", False),
                    is_divine_shield=m.get("is_divine_shield", False),
                    is_stealthed=m.get("is_stealthed", False),
                    is_frozen=m.get("is_frozen", False),
                    is_reborn=m.get("is_reborn", False),
                    is_silenced=m.get("is_silenced", False),
                    is_dormant=m.get("is_dormant", False),
                )
            )
        lines.append("**Стол противника**: " + "; ".join(ob_items))
    else:
        lines.append("**Стол противника**: [Пусто]")

    # Secrets / Locations
    if snapshot.friendly_locations:
        locs = [f"{l['name']} ({l['durability']} пр.)" for l in snapshot.friendly_locations]
        lines.append("**Ваши области**: " + ", ".join(locs))
    if snapshot.opponent_secrets_count > 0:
        lines.append(f"**Секреты противника**: {snapshot.opponent_secrets_count} активных")

    lines.append("\nКаковы наилучшие действия на этом ходу?")
    return "\n".join(lines)


def format_turn_completion(actions: List[PlayerAction]) -> str:
    """
    Formats the sequence of actions played by the winning player.
    """
    if not actions:
        return "1. Конец хода"

    lines = []
    for idx, act in enumerate(actions, start=1):
        if act.action_type == "PLAY":
            target = f" на {act.target_name}" if act.target_name else ""
            lines.append(f"{idx}. Разыграть карту: {act.entity_name}{target}")
        elif act.action_type == "ATTACK":
            target = act.target_name or "Героя противника"
            lines.append(f"{idx}. Атаковать: {act.entity_name} -> {target}")
        elif act.action_type == "HERO_POWER":
            target = f" на {act.target_name}" if act.target_name else ""
            lines.append(f"{idx}. Использовать силу героя: {act.entity_name}{target}")
        elif act.action_type == "LOCATION":
            target = f" на {act.target_name}" if act.target_name else ""
            lines.append(f"{idx}. Активировать область: {act.entity_name}{target}")
        else:
            lines.append(f"{idx}. Действие: {act.entity_name}")

    lines.append(f"{len(actions) + 1}. Завершить ход")
    return "\n".join(lines)


def generate_dataset(
    output_path: Optional[Path | str] = None,
    filter_ranked_wins: bool = True,
    max_replays: Optional[int] = None,
) -> int:
    """
    Extracts all eligible turns and writes them into a JSONL dataset.
    """
    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)

    card_db = CardDatabase(auto_load=True)
    total_records = 0
    total_games = 0

    with open(out_file, "w", encoding="utf-8") as f_out:
        for replay in iterate_replays(filter_ranked_wins=filter_ranked_wins, max_count=max_replays, card_db=card_db):
            total_games += 1
            meta = replay.metadata

            for snapshot in replay.turn_snapshots:
                # Keep friendly turns where the player performed at least 1 action
                if not snapshot.is_friendly_turn or not snapshot.actions:
                    continue

                prompt = format_turn_prompt(
                    snapshot=snapshot,
                    player_hero=meta.player_hero or "Friendly",
                    opponent_hero=meta.opponent_hero or "Opponent",
                )
                completion = format_turn_completion(snapshot.actions)

                record = {
                    "game_id": meta.game_id,
                    "replay_file": meta.replay_file,
                    "result": meta.result,
                    "game_mode": meta.game_mode,
                    "deck_name": meta.deck_name,
                    "player_hero": meta.player_hero,
                    "opponent_hero": meta.opponent_hero,
                    "turn_number": snapshot.turn_number,
                    "prompt": prompt,
                    "completion": completion,
                    "friendly_mana": snapshot.friendly_mana,
                    "friendly_max_mana": snapshot.friendly_max_mana,
                    "hand_size": len(snapshot.friendly_hand),
                    "friendly_minions_count": len(snapshot.friendly_board),
                    "opponent_minions_count": len(snapshot.opponent_board),
                    "actions_count": len(snapshot.actions),
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1

    logger.info("Generated %d training records from %d replays at %s", total_records, total_games, out_file)
    print(f"Generated {total_records} turn action records from {total_games} games -> {out_file}")
    return total_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from Hearthstone replays")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of replays to process")
    parser.add_argument(
        "--all-games", action="store_true", help="Include all games, not just ranked winning games"
    )
    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        filter_ranked_wins=not args.all_games,
        max_replays=args.limit,
    )
