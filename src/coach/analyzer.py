"""
Post-Game Match Coach & Tactical Turn Analyzer.
Evaluates played matches, detects missed lethal damage, tempo inefficiencies,
and generates step-by-step coaching reports with local LLM recommendations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.card_db import CardDatabase
from src.llm import ActionCandidate, OllamaClient, ParsedPlan, generate_legal_candidates, parse_model_response
from src.parser import GameReplay, PlayerAction, TurnSnapshot


@dataclass
class TurnAnalysis:
    turn_number: int
    mana_available: int
    mana_spent: int
    actual_actions: List[str]
    ai_actions: List[str]
    ai_reasoning: str
    is_lethal_possible: bool = False
    lethal_damage_available: int = 0
    opponent_health: int = 0
    tempo_loss: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class MatchReport:
    player_name: str
    opponent_name: str
    player_hero: str
    opponent_hero: str
    result: str
    deck_name: str
    total_turns: int
    turn_analyses: List[TurnAnalysis] = field(default_factory=list)
    missed_lethals_count: int = 0
    tempo_loss_turns_count: int = 0
    overall_summary: str = ""


class MatchCoach:
    """
    Coach analyzer that inspects replay turns, tests for tactical mistakes,
    and queries local LLM for optimal decisions.
    """

    def __init__(
        self,
        card_db: Optional[CardDatabase] = None,
        ollama_client: Optional[OllamaClient] = None,
        model_name: Optional[str] = None,
    ):
        self.card_db = card_db or CardDatabase(auto_load=True)
        self.llm_client = ollama_client or OllamaClient(model=model_name)

    def calculate_max_burst_damage(self, snapshot: TurnSnapshot) -> int:
        """
        Calculates maximum possible burst damage directly to the opponent's face on this turn.
        """
        damage = 0

        # Check for enemy Taunt
        has_taunt = any(
            m.get("is_taunt") and not m.get("is_stealthed") and not m.get("is_dormant")
            for m in snapshot.opponent_board
        )

        # Minion attacks (only if no Taunt)
        if not has_taunt:
            for m in snapshot.friendly_board:
                if m.get("can_attack"):
                    damage += m.get("attack", 0)

        # Spells and Charge from hand within mana limit
        mana_left = snapshot.friendly_mana
        # Simple burst spell lookups
        for card_data in snapshot.friendly_hand:
            cid = card_data.get("card_id", "")
            cost = card_data.get("cost", 0)
            if cost <= mana_left:
                c_info = self.card_db.get_by_id(cid)
                if c_info:
                    # Known common burst spells
                    if cid in ("CS2_029", "CORE_CS2_029"):  # Fireball (6)
                        damage += 6
                        mana_left -= cost
                    elif cid in ("CS2_024", "CORE_CS2_024"):  # Frostbolt (3)
                        damage += 3
                        mana_left -= cost
                    elif cid in ("EX1_277", "CORE_EX1_277"):  # Arcane Missiles (3)
                        damage += 3
                        mana_left -= cost
                    elif cid in ("EX1_116", "CORE_EX1_116") and not has_taunt:  # Leeroy (6)
                        damage += 6
                        mana_left -= cost
                    elif cid in ("DS1_185", "CORE_DS1_185"):  # Arcane Shot (2)
                        damage += 2
                        mana_left -= cost
                    elif cid in ("CS2_087", "CORE_CS2_087"):  # Blessing of Might (+3)
                        damage += 3
                        mana_left -= cost

        return damage

    def build_llm_prompt(self, snapshot: TurnSnapshot, candidates: List[ActionCandidate]) -> str:
        """
        Builds a compact prompt for small LLMs with candidate numbers.
        """
        opp_hp = snapshot.opponent_hero.get("health", 30)
        opp_armor = snapshot.opponent_hero.get("armor", 0)
        armor_str = f"+{opp_armor}" if opp_armor else ""

        lines = [
            "Ты — тактический ассистент Hearthstone. Выбери лучшую комбинацию действий из списка.",
            f"Ход {snapshot.turn_number}. Доступно маны: {snapshot.friendly_mana}/{snapshot.friendly_max_mana}.",
            f"Враг: {opp_hp}{armor_str} HP.",
            "\nДоступные действия:",
        ]

        for cand in candidates:
            lines.append(f"[{cand.index}] {cand.description}")

        lines.append(
            "\nОтветь в формате:\nПЛАН: [номера выбранных действий через запятую, например: 1, 3, 5]\nОБОСНОВАНИЕ: [кратко 1 предложение]"
        )
        return "\n".join(lines)

    def analyze_turn(self, snapshot: TurnSnapshot, query_llm: bool = True) -> TurnAnalysis:
        """
        Analyzes a single turn snapshot and returns TurnAnalysis.
        """
        candidates = generate_legal_candidates(snapshot, self.card_db)
        opp_total_hp = snapshot.opponent_hero.get("health", 30) + snapshot.opponent_hero.get("armor", 0)

        # 1. Check Missed Lethal
        burst = self.calculate_max_burst_damage(snapshot)
        is_lethal = (burst >= opp_total_hp)

        # 2. Check Mana Tempo Loss
        # If player had 2+ unspent mana and unplayed on-curve minions in hand
        actual_actions_str = []
        actual_mana_spent = 0
        for act in snapshot.actions:
            target = f" -> {act.target_name}" if act.target_name else ""
            actual_actions_str.append(f"{act.action_type}: {act.entity_name}{target}")

        tempo_loss = False
        if snapshot.friendly_mana >= 2 and len(snapshot.friendly_hand) > 0 and len(snapshot.actions) == 0:
            tempo_loss = True

        # 3. Query LLM for recommendation
        ai_actions: List[str] = []
        ai_reasoning: str = ""

        if query_llm and candidates:
            prompt = self.build_llm_prompt(snapshot, candidates)
            try:
                raw_resp = self.llm_client.generate(prompt=prompt, temperature=0.1, max_tokens=100)
                parsed = parse_model_response(raw_resp, candidates, max_mana=snapshot.friendly_mana)
                ai_actions = parsed.action_descriptions
                ai_reasoning = parsed.reasoning
            except Exception as e:
                logger.warning("LLM generation failed: %s. Using heuristic.", e)
                parsed = parse_model_response("", candidates, max_mana=snapshot.friendly_mana)
                ai_actions = parsed.action_descriptions
                ai_reasoning = "Эвристический план (LLM оффлайн)"

        notes = []
        if is_lethal:
            notes.append(f"💥 ВНИМАНИЕ: На этом ходу был возможен ЛЕТАЛ ({burst} урона при {opp_total_hp} HP врага)!")
        if tempo_loss:
            notes.append("⚠️ Потеря темпа: пропущен ход с неиспользованной маной при наличии карт в руке.")

        return TurnAnalysis(
            turn_number=snapshot.turn_number,
            mana_available=snapshot.friendly_mana,
            mana_spent=actual_mana_spent,
            actual_actions=actual_actions_str,
            ai_actions=ai_actions,
            ai_reasoning=ai_reasoning,
            is_lethal_possible=is_lethal,
            lethal_damage_available=burst,
            opponent_health=opp_total_hp,
            tempo_loss=tempo_loss,
            notes=notes,
        )

    def analyze_replay(self, replay: GameReplay, query_llm: bool = True) -> MatchReport:
        """
        Analyzes all friendly turns in a match and generates complete MatchReport.
        """
        meta = replay.metadata
        turn_analyses: List[TurnAnalysis] = []
        missed_lethals = 0
        tempo_losses = 0

        for snapshot in replay.friendly_turns:
            analysis = self.analyze_turn(snapshot, query_llm=query_llm)
            turn_analyses.append(analysis)
            if analysis.is_lethal_possible:
                missed_lethals += 1
            if analysis.tempo_loss:
                tempo_losses += 1

        # Overall summary
        summary_parts = []
        if missed_lethals > 0:
            summary_parts.append(f"Обнаружено {missed_lethals} ситуаций с потенциальным летальным уроном.")
        if tempo_losses > 0:
            summary_parts.append(f"Зафиксировано {tempo_losses} ходов с неэффективным расходом маны/темпа.")
        if not summary_parts:
            summary_parts.append("Матч сыгран стабильно и уверенно без грубых тактических потерь темпа.")

        return MatchReport(
            player_name=meta.player_name,
            opponent_name=meta.opponent_name,
            player_hero=meta.player_hero,
            opponent_hero=meta.opponent_hero,
            result=meta.result,
            deck_name=meta.deck_name,
            total_turns=len(turn_analyses),
            turn_analyses=turn_analyses,
            missed_lethals_count=missed_lethals,
            tempo_loss_turns_count=tempo_losses,
            overall_summary=" ".join(summary_parts),
        )
