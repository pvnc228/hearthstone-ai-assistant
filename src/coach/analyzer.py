"""
Post-Game Match Coach & Tactical Turn Analyzer.
Evaluates played matches, detects missed lethal damage, tempo inefficiencies,
and generates step-by-step coaching reports with local LLM recommendations.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.card_db import CardDatabase
from src.llm import ActionCandidate, OllamaClient, ParsedPlan, generate_legal_candidates, parse_model_response
from src.parser import GameReplay, PlayerAction, TurnSnapshot

logger = logging.getLogger(__name__)

# Known direct-damage spells: card_id -> face damage
# (Arcane Missiles counted optimistically — its damage is randomly distributed)
BURST_SPELLS: Dict[str, int] = {
    "CS2_029": 6,       # Fireball
    "CORE_CS2_029": 6,
    "CS2_024": 3,       # Frostbolt
    "CORE_CS2_024": 3,
    "DS1_185": 2,       # Arcane Shot
    "CORE_DS1_185": 2,
    "EX1_277": 3,       # Arcane Missiles
    "CORE_EX1_277": 3,
}
LEEROY_IDS = {"EX1_116", "CORE_EX1_116"}  # 6 dmg face, blocked by Taunt


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
        self._model_name = model_name
        # Lazy: OllamaClient probes the HTTP server on construction; don't pay
        # that cost in --no-llm mode (query_llm=False creates the client only on first use).
        self.llm_client = ollama_client

    def _get_llm_client(self) -> OllamaClient:
        if self.llm_client is None:
            self.llm_client = OllamaClient(model=self._model_name)
        return self.llm_client

    def calculate_max_burst_damage(self, snapshot: TurnSnapshot) -> int:
        """
        Calculates maximum possible burst damage directly to the opponent's face this turn.
        Exact small-scale knapsack over damage spells (not greedy), plus ready minion
        attacks (RUSH minions excluded — they cannot hit face).
        """
        has_taunt = any(
            m.get("is_taunt") and not m.get("is_stealthed") and not m.get("is_dormant")
            for m in snapshot.opponent_board
        )

        # 1. Minion attacks to face (only if no Taunt; rush minions can't hit face)
        minion_damage = 0
        if not has_taunt:
            for m in snapshot.friendly_board:
                can_face = m.get("can_attack_hero", m.get("can_attack", False))
                if can_face:
                    minion_damage += m.get("attack", 0)

        # 2. Damage spells from hand: exact knapsack over (cost, damage) options
        spell_options = []
        for card_data in snapshot.friendly_hand:
            cid = card_data.get("card_id", "")
            cost = card_data.get("cost", 0)
            if cost > snapshot.friendly_mana:
                continue
            if cid in BURST_SPELLS:
                spell_options.append((cost, BURST_SPELLS[cid]))
            elif cid in LEEROY_IDS and not has_taunt:
                spell_options.append((cost, 6))

        best_spell_damage = self._max_damage_within_mana(spell_options, snapshot.friendly_mana)

        return minion_damage + best_spell_damage

    @staticmethod
    def _max_damage_within_mana(options: List[tuple], mana: int) -> int:
        """Exact max damage within mana budget — O(n*mana) knapsack DP, not 2^n subsets."""
        if not options or mana <= 0:
            return 0
        # best[c] = max damage achievable with exactly <= c mana
        best = [0] * (mana + 1)
        for cost, dmg in options:
            if cost > mana or dmg <= 0:
                continue
            for c in range(mana, cost - 1, -1):
                cand = best[c - cost] + dmg
                if cand > best[c]:
                    best[c] = cand
        return best[mana]

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

        # 2. Check Mana Tempo Loss (player passed the turn with unspent mana)
        actual_actions_str = []
        actual_mana_spent = 0
        for act in snapshot.actions:
            target = f" -> {act.target_name}" if act.target_name and act.target_name != "Target" else ""
            actual_actions_str.append(f"{act.action_type}: {act.entity_name}{target}")
            if act.action_type in ("PLAY", "HERO_POWER"):
                c_info = self.card_db.get_by_id(act.entity_card_id) if act.entity_card_id else None
                actual_mana_spent += c_info.cost if c_info else 0

        # Mana actually spent according to the log's own accounting (RESOURCES delta
        # is captured in the tracker); approximate with mana at snapshot vs after actions.
        tempo_loss = (
            snapshot.is_friendly_turn
            and snapshot.friendly_mana >= 2
            and len(snapshot.friendly_hand) > 0
            and len(snapshot.actions) == 0
            and not snapshot.game_ended
        )

        # 3. Query LLM for recommendation
        ai_actions: List[str] = []
        ai_reasoning: str = ""

        if query_llm and candidates:
            prompt = self.build_llm_prompt(snapshot, candidates)
            try:
                raw_resp = self._get_llm_client().generate(prompt=prompt, temperature=0.1, max_tokens=100)
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

        friendly_turns = replay.friendly_turns
        for i, snapshot in enumerate(friendly_turns):
            analysis = self.analyze_turn(snapshot, query_llm=query_llm)
            turn_analyses.append(analysis)

            # Suppress "missed lethal" when the game ended on this turn
            # (the player either executed the lethal or the outcome was already decided).
            if analysis.is_lethal_possible and not snapshot.game_ended:
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
