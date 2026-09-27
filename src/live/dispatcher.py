"""
Model inference and recommendation dispatcher for the live assistant harness.
Combines fast deterministic Lethal Detection, Candidate Ranking,
and optional Ollama LLM tactical reasoning into a low-latency recommendation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.card_db import CardDatabase
from src.coach.analyzer import BURST_SPELLS, LEEROY_IDS, MatchCoach
from src.llm import OllamaClient
from src.llm.next_action_contract import (
    NEXT_ACTION_SYSTEM_PROMPT,
    NextActionParse,
    _SINGLE_PLAN_RE,
    build_next_action_prompt,
    candidate_to_prompt_dict,
    snapshot_to_prompt_state,
)
from src.parser import TurnSnapshot
from .guidance import ActionGuidance, format_guidance

logger = logging.getLogger(__name__)


@dataclass
class LiveRecommendation:
    """Tactical recommendation for the current player turn."""

    top_guidances: List[ActionGuidance] = field(default_factory=list)
    is_lethal: bool = False
    burst_damage: int = 0
    opponent_total_hp: int = 0
    chosen_candidate_id: int = 0
    coach_note: str = ""
    model_source: str = "rule_ranker"  # lethal_detector, rule_ranker, ollama, modernce
    latency_ms: float = 0.0


class LiveAdvisorDispatcher:
    """
    Coordinates model inference for live decision points.
    Enforces a strict waterfall:
    1. Instant Lethal Check (< 1ms).
    2. Fast Candidate Ranking (< 10ms).
    3. Optional Ollama CoT reasoning (< 2.5s) if enabled.
    """

    def __init__(
        self,
        card_db: Optional[CardDatabase] = None,
        ollama_client: Optional[OllamaClient] = None,
        enable_llm: bool = False,
        model_name: Optional[str] = None,
    ):
        self.card_db = card_db or CardDatabase(auto_load=True)
        self.enable_llm = enable_llm
        self.model_name = model_name
        self.coach = MatchCoach(card_db=self.card_db)
        self.ollama_client = ollama_client if ollama_client else (OllamaClient(model=model_name) if enable_llm else None)

    def evaluate_decision(
        self,
        snapshot: TurnSnapshot,
        candidates: List[Any],
    ) -> LiveRecommendation:
        """
        Evaluates legal candidates for the current turn snapshot and produces
        top-3 recommended actions with guidance.
        """
        t0 = time.perf_counter()

        if not candidates:
            return LiveRecommendation(
                top_guidances=[],
                coach_note="Нет доступных легальных действий.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        opp_hp = snapshot.opponent_hero.get("health", 30)
        opp_armor = snapshot.opponent_hero.get("armor", 0)
        opp_total_hp = opp_hp + opp_armor

        # 1. Fast Lethal Check
        burst = self.coach.calculate_max_burst_damage(snapshot)
        is_lethal = burst >= opp_total_hp and opp_total_hp > 0

        if is_lethal:
            lethal_guidances = self._prioritize_lethal(candidates, snapshot)
            latency_ms = (time.perf_counter() - t0) * 1000
            return LiveRecommendation(
                top_guidances=lethal_guidances[:3],
                is_lethal=True,
                burst_damage=burst,
                opponent_total_hp=opp_total_hp,
                chosen_candidate_id=lethal_guidances[0].candidate_id if lethal_guidances else 0,
                coach_note=f"🔥 ОБНАРУЖЕН ЛЕТАЛЬНЫЙ УРОН! Нанесите {burst} урона в лицо оппонента ({opp_total_hp} HP).",
                model_source="lethal_detector",
                latency_ms=latency_ms,
            )

        # 2. Fast Rule-Based Candidate Scoring
        scored_candidates = self._score_candidates(candidates, snapshot)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        chosen_id = scored_candidates[0][1].candidate_id if hasattr(scored_candidates[0][1], "candidate_id") else getattr(scored_candidates[0][1], "index", 1)
        coach_note = self._generate_tactical_note(scored_candidates[0][1], snapshot)
        model_source = "rule_ranker"

        # 3. Optional LLM Query (if enabled and reachable)
        if self.enable_llm and self.ollama_client:
            llm_cand_id, llm_note = self._query_llm(snapshot, candidates)
            if llm_cand_id is not None:
                chosen_id = llm_cand_id
                model_source = "ollama"
                if llm_note:
                    coach_note = llm_note
                # Re-order so LLM chosen action is first
        # Build guidance for top-3 distinct actions
        top_guidances: List[ActionGuidance] = []
        seen_signatures = set()
        for _, cand in scored_candidates:
            act_type = getattr(cand, "action_type", "")
            ent_name = getattr(cand, "entity_name", "")
            tgt_name = getattr(cand, "target_name", "")
            sub_name = getattr(cand, "sub_entity_name", "")
            sig = (act_type, ent_name, tgt_name, sub_name)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            top_guidances.append(format_guidance(cand, snapshot))
            if len(top_guidances) >= 3:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return LiveRecommendation(
            top_guidances=top_guidances,
            is_lethal=False,
            burst_damage=burst,
            opponent_total_hp=opp_total_hp,
            chosen_candidate_id=chosen_id,
            coach_note=coach_note,
            model_source=model_source,
            latency_ms=latency_ms,
        )

    def _score_candidates(
        self,
        candidates: List[Any],
        snapshot: TurnSnapshot,
    ) -> List[tuple[float, Any]]:
        """
        Calculates a baseline heuristic score for each legal candidate:
        - Prioritizes spending mana efficiently.
        - Prioritizes minion development and trades on board.
        - Penalizes premature End Turn if unspent mana and plays exist.
        """
        mana = snapshot.friendly_mana
        has_playable = any(
            getattr(c, "action_type", "") == "PLAY" and getattr(c, "mana_cost", 99) <= mana
            for c in candidates
        )

        scored: List[tuple[float, Any]] = []
        for cand in candidates:
            score = 10.0
            act_type = getattr(cand, "action_type", "")
            cost = getattr(cand, "mana_cost", 0)
            target_name = getattr(cand, "target_name", "") or ""

            if act_type == "END_TURN":
                if mana > 0 and has_playable:
                    score = 0.5  # Do not pass turn with usable mana
                else:
                    score = 25.0  # Safe end turn when out of plays

            elif act_type == "PLAY":
                # Spending exact mana is good
                score += 30.0
                if cost == mana:
                    score += 15.0  # Curve tempo
                elif cost > 0:
                    score += (cost / max(1, mana)) * 10.0

            elif act_type == "ATTACK":
                score += 25.0
                # Trade bonus against enemy minions vs face
                if target_name and "Герой" not in target_name:
                    score += 5.0

            elif act_type == "HERO_POWER":
                score += 15.0
                if mana == cost:
                    score += 10.0  # Floating mana hero power

            elif act_type == "LOCATION":
                score += 20.0

            scored.append((score, cand))

        return scored

    def _prioritize_lethal(
        self,
        candidates: List[Any],
        snapshot: TurnSnapshot,
    ) -> List[ActionGuidance]:
        """Arranges face-hitting attacks and burst damage first for lethal turn."""
        lethal_list: List[Any] = []
        other_list: List[Any] = []

        opp_name = snapshot.opponent_hero.get("name", "")

        for cand in candidates:
            act_type = getattr(cand, "action_type", "")
            target_name = getattr(cand, "target_name", "") or ""
            card_id = getattr(cand, "entity_card_id", "")

            is_face_attack = (act_type == "ATTACK" and (target_name == opp_name or not target_name or "Герой" in target_name))
            is_burst_spell = (act_type == "PLAY" and card_id in BURST_SPELLS)

            if is_face_attack or is_burst_spell:
                lethal_list.append(cand)
            elif act_type != "END_TURN":
                other_list.append(cand)

        all_ordered = lethal_list + other_list
        unique_lethal: List[ActionGuidance] = []
        seen = set()
        for c in all_ordered:
            sig = (getattr(c, "action_type", ""), getattr(c, "entity_name", ""), getattr(c, "target_name", ""))
            if sig in seen:
                continue
            seen.add(sig)
            unique_lethal.append(format_guidance(c, snapshot))
            if len(unique_lethal) >= 3:
                break
        return unique_lethal

    def _generate_tactical_note(self, best_cand: Any, snapshot: TurnSnapshot) -> str:
        act_type = getattr(best_cand, "action_type", "")
        card_name = getattr(best_cand, "entity_name", "")
        mana = snapshot.friendly_mana

        if act_type == "PLAY":
            return f"Темповый розыгрыш '{card_name}' на {snapshot.turn_number}-м ходу."
        if act_type == "ATTACK":
            tgt = getattr(best_cand, "target_name", "цель")
            return f"Тактический размен: {card_name} атакует {tgt}."
        if act_type == "HERO_POWER":
            return f"Эффективная реализация оставшейся маны ({mana}м) через силу героя."
        if act_type == "END_TURN":
            return "Все ключевые действия хода выполнены, передаём ход оппоненту."
        return "Оптимальное действие по темпу и мане."

    def _query_llm(
        self,
        snapshot: TurnSnapshot,
        candidates: List[Any],
    ) -> tuple[Optional[int], str]:
        """Queries local Ollama using the schema-v2 shared prompt contract."""
        if not self.ollama_client:
            return None, ""
        try:
            state_dict = snapshot_to_prompt_state(snapshot)
            prompt = build_next_action_prompt(state_dict, candidates)
            response = self.ollama_client.generate(
                prompt=prompt,
                system=NEXT_ACTION_SYSTEM_PROMPT,
                temperature=0.1,
            )
            # Parse candidate id
            m = _SINGLE_PLAN_RE.search(response)
            if m:
                cid = int(m.group(1))
                return cid, response.strip()
        except Exception as e:
            logger.warning("Ollama live inference failed/timed out: %s", e)
        return None, ""
