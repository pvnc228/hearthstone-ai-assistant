"""
Full Game Simulation runner from Turn 1 (Mulligan/Opening) to Lethal.
Evaluates every turn using qwen2.5:1.5b-instruct-q8_0 and records complete
untruncated model responses into data/full_game_simulation.md.
"""

import json
import time
from pathlib import Path
from src.card_db import CardDatabase
from src.llm import NEXT_ACTION_SYSTEM_PROMPT, OllamaClient, generate_legal_candidates, parse_model_response
from src.parser import DEFAULT_REPLAY_DIR, load_deck_stats_index, parse_replay_file
from src.coach.analyzer import MatchCoach

OUTPUT_FILE = Path("data/full_game_simulation.md")


def run_full_game_simulation():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    card_db = CardDatabase(auto_load=True)
    client = OllamaClient()  # auto-select installed model
    coach = MatchCoach(card_db=card_db, ollama_client=client)

    stats_index = load_deck_stats_index()
    replay_files = sorted(DEFAULT_REPLAY_DIR.glob("*.hdtreplay"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not replay_files:
        print("No replay files found.")
        return

    # Select a clear, action-rich winning ranked replay
    target_replay = None
    for rf in replay_files:
        meta = stats_index.get(rf.name)
        if meta and meta.result == "Win" and meta.game_mode == "Ranked" and meta.turns_count >= 5:
            target_replay = parse_replay_file(rf, card_db=card_db, deck_stats_index=stats_index)
            if len([s for s in target_replay.friendly_turns if s.actions]) >= 4:
                break

    if not target_replay:
        target_replay = parse_replay_file(replay_files[0], card_db=card_db, deck_stats_index=stats_index)

    meta = target_replay.metadata
    print(f"Running Full Game Simulation on match: {meta.player_name} ({meta.player_hero}) vs {meta.opponent_name} ({meta.opponent_hero}) [{meta.result}]")

    lines = []
    lines.append(f"# Полная симуляция матча Hearthstone (От открытия до финала)\n")
    lines.append(f"**Матч**: {meta.player_name} ({meta.player_hero}) vs {meta.opponent_name} ({meta.opponent_hero})")
    lines.append(f"**Колода игрока**: {meta.deck_name or 'Основная колода'}")
    lines.append(f"**Режим**: {meta.game_mode} ({meta.format}) | **Результат**: 🟢 {meta.result}")
    lines.append(f"**Модель ИИ**: `{client.model}`")
    lines.append(f"**Дата симуляции**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    for idx, snap in enumerate(target_replay.friendly_turns, start=1):
        try:
            candidates = generate_legal_candidates(snap, card_db)
            prompt = coach.build_llm_prompt(snap, candidates)

            t0 = time.time()
            raw_resp = client.generate(
                prompt=prompt,
                system=NEXT_ACTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=250,
            )
            latency = time.time() - t0

            parsed = parse_model_response(raw_resp, candidates, max_mana=snap.friendly_mana)
            burst = coach.calculate_max_burst_damage(snap)
            opp_hp = snap.opponent_hero.get("health", 30) + snap.opponent_hero.get("armor", 0)
            is_lethal = (burst >= opp_hp)
        except Exception as e:
            # One bad turn (e.g. Ollama hiccup) must not kill the whole run
            import logging
            logging.getLogger(__name__).warning("Turn %s failed: %s", getattr(snap, "turn_number", idx), e)
            lines.append(f"## 🎮 Ход {getattr(snap, 'turn_number', idx)} — ОШИБКА: {e}\n")
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            continue

        print(f"Processing Turn {snap.turn_number} (Mana: {snap.friendly_mana}/{snap.friendly_max_mana}, Latency: {latency:.2f}s)...")

        lines.append(f"## 🎮 Ход {snap.turn_number} (Мана: {snap.friendly_mana}/{snap.friendly_max_mana})\n")
        lines.append(f"- **Здоровье вашего героя**: {snap.friendly_hero.get('health', 30)} HP (+{snap.friendly_hero.get('armor', 0)} брони)")
        lines.append(f"- **Здоровье оппонента**: {opp_hp} HP")
        lines.append(f"- **Карты в руке ({len(snap.friendly_hand)})**: {', '.join([c.get('name', '') for c in snap.friendly_hand]) or '[Пусто]'}")
        lines.append(f"- **Ваш стол ({len(snap.friendly_board)})**: {', '.join([m.get('name', '') + ' (' + str(m.get('attack', 0)) + '/' + str(m.get('health', 0)) + ')' for m in snap.friendly_board]) or '[Пусто]'}")
        lines.append(f"- **Стол оппонента ({len(snap.opponent_board)})**: {', '.join([m.get('name', '') + ' (' + str(m.get('attack', 0)) + '/' + str(m.get('health', 0)) + ')' for m in snap.opponent_board]) or '[Пусто]'}")
        if is_lethal:
            lines.append(f"- 💥 **Детектор летала**: Возможен летальный урон ({burst} урона при {opp_hp} HP врага)!\n")
        else:
            lines.append(f"- **Потенциальный взрывной урон в лицо**: {burst}\n")

        lines.append("### 1. Промпт переданный в модель:")
        lines.append("```text")
        lines.append(prompt)
        lines.append("```\n")

        lines.append(f"### 2. Сырой, полный ответ модели (Время отклика: {latency:.2f} сек):")
        lines.append("```text")
        lines.append(raw_resp)
        lines.append("```\n")

        lines.append("### 3. Распарсенное решение модели:")
        lines.append("```text")
        if parsed.actions:
            for a in parsed.actions:
                lines.append(f"• {a.description}")
        else:
            lines.append("• (Конец хода)")
        lines.append(f"Обоснование модели: {parsed.reasoning}")
        lines.append(f"Потрачено маны: {parsed.total_mana_spent}/{snap.friendly_mana}")
        lines.append("```\n")

        lines.append("### 4. Что вы сыграли в реальной партии:")
        lines.append("```text")
        if snap.actions:
            for act in snap.actions:
                t = f" -> {act.target_name}" if act.target_name else ""
                lines.append(f"• {act.action_type}: {act.entity_name}{t}")
        else:
            lines.append("• (Нет действий / пропуск)")
        lines.append("```\n")
        lines.append("---\n")
        # Incremental write: results survive interruption
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"\nFull game simulation complete and saved to {OUTPUT_FILE}!")


if __name__ == "__main__":
    run_full_game_simulation()
