"""
Command-line Interface for Post-Game Match Coach.
Usage:
  python -m src.coach.cli --latest
  python -m src.coach.cli --replay "HappyBread#21597(Priest) vs Enemy#1234(Rogue) 1200-100826.hdtreplay"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.card_db import CardDatabase
from src.llm import OllamaClient
from src.parser import DEFAULT_REPLAY_DIR, load_deck_stats_index, parse_replay_file
from .analyzer import MatchCoach, MatchReport


def print_match_report(report: MatchReport) -> None:
    """
    Renders a clean, structured match analysis to console.
    """
    print("\n" + "=" * 60)
    print("🏆 РАЗБОР МАТЧА — HEARTHSTONE AI COACH")
    print("=" * 60)
    result_icon = "🟢 ПОБЕДА" if report.result == "Win" else "🔴 ПОРАЖЕНИЕ"
    print(f"Игрок:     {report.player_name} ({report.player_hero})")
    print(f"Оппонент:  {report.opponent_name} ({report.opponent_hero})")
    print(f"Колода:    {report.deck_name or 'Пользовательская колода'}")
    print(f"Результат: {result_icon} | Всего ходов: {report.total_turns}")
    print("-" * 60)

    for ta in report.turn_analyses:
        print(f"\n👉 [ХОД {ta.turn_number}] (Мана: {ta.mana_available})")

        print("  Сыграно вами:")
        if ta.actual_actions:
            for act in ta.actual_actions:
                print(f"    • {act}")
        else:
            print("    • (Ход пропущен / нет действий)")

        print("  Рекомендация Coach LLM:")
        if ta.ai_actions:
            for act in ta.ai_actions:
                print(f"    ✓ {act}")
            if ta.ai_reasoning:
                print(f"    💬 {ta.ai_reasoning}")
        else:
            print("    ✓ Конец хода")

        if ta.notes:
            for note in ta.notes:
                print(f"  {note}")

    print("\n" + "=" * 60)
    print("📋 ИТОГОВЫЙ ВЫВОД ТАКТИЧЕСКОГО ТРЕНЕРА:")
    print(f"  {report.overall_summary}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hearthstone Post-Game Match Coach CLI")
    parser.add_argument("--latest", action="store_true", help="Analyze the most recent replay")
    parser.add_argument("--replay", type=str, default=None, help="Path or name of the .hdtreplay file")
    parser.add_argument("--model", type=str, default=None, help="Ollama model to use (default: auto-select)")
    parser.add_argument("--no-llm", action="store_true", help="Fast mode without querying LLM")
    args = parser.parse_args()

    replay_dir = DEFAULT_REPLAY_DIR
    target_file: Optional[Path] = None

    if args.latest:
        files = sorted(replay_dir.glob("*.hdtreplay"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            print("❌ Реплеи не найдены в", replay_dir)
            sys.exit(1)
        target_file = files[0]
    elif args.replay:
        p = Path(args.replay)
        if p.exists():
            target_file = p
        else:
            cand = replay_dir / args.replay
            if cand.exists():
                target_file = cand
            else:
                print(f"❌ Файл реплея не найден: {args.replay}")
                sys.exit(1)
    else:
        # Default to latest
        files = sorted(replay_dir.glob("*.hdtreplay"), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            target_file = files[0]
        else:
            print("❌ Укажите --replay или --latest")
            sys.exit(1)

    print(f"🔍 Загрузка и анализ матча: {target_file.name}...")

    stats_index = load_deck_stats_index()
    card_db = CardDatabase(auto_load=True)
    replay = parse_replay_file(target_file, card_db=card_db, deck_stats_index=stats_index)

    # In --no-llm mode skip the client entirely: its constructor probes Ollama over HTTP.
    coach = MatchCoach(card_db=card_db, ollama_client=None if args.no_llm else OllamaClient(model=args.model))

    report = coach.analyze_replay(replay, query_llm=not args.no_llm)
    print_match_report(report)


if __name__ == "__main__":
    main()
