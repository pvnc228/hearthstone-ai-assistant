"""
Resilient Multi-Strategy Response Parser for Small LLMs (1.5B - 7B).
Extracts action sequences from raw, noisy, or semi-structured model output
and guarantees strict compliance with game rules, mana budgets, and board state.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .candidate_generator import ActionCandidate

RE_PLAN_TAG = re.compile(r"(?:ПЛАН|PLAN|ДЕЙСТВИЯ|ACTIONS)\s*:\s*\[?([\d,\s\-]+)\]?", re.IGNORECASE)
RE_BRACKET_NUMS = re.compile(r"\[([\d,\s\-]+)\]")
RE_NUM_LIST = re.compile(r"\b(\d+)\b")
RE_REASONING = re.compile(r"(?:ОБОСНОВАНИЕ|REASONING|ПОЧЕМУ)\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedPlan:
    actions: List[ActionCandidate] = field(default_factory=list)
    reasoning: str = ""
    total_mana_spent: int = 0
    is_fallback: bool = False

    @property
    def action_descriptions(self) -> List[str]:
        return [a.description for a in self.actions]


def parse_model_response(
    raw_text: str,
    candidates: List[ActionCandidate],
    max_mana: int,
) -> ParsedPlan:
    """
    Parses LLM response into a validated list of ActionCandidates.
    Enforces mana constraints and drops illegal duplicate actions.
    """
    if not candidates:
        return ParsedPlan(actions=[], reasoning="Нет доступных действий", total_mana_spent=0)

    candidate_by_idx: Dict[int, ActionCandidate] = {c.index: c for c in candidates}
    selected_indices: List[int] = []

    # Extract reasoning if provided
    reasoning = ""
    m_reas = RE_REASONING.search(raw_text)
    if m_reas:
        reasoning = m_reas.group(1).strip().split("\n")[0].strip()

    # --- Strategy 1: Explicit PLAN: [1, 2, 3] Tag ---
    m_plan = RE_PLAN_TAG.search(raw_text)
    if m_plan:
        raw_nums = m_plan.group(1)
        for num_str in re.findall(r"\d+", raw_nums):
            val = int(num_str)
            if val in candidate_by_idx and val not in selected_indices:
                selected_indices.append(val)

    # --- Strategy 2: Bracketed numbers [1, 5] ---
    if not selected_indices:
        for m_bracket in RE_BRACKET_NUMS.finditer(raw_text):
            for num_str in re.findall(r"\d+", m_bracket.group(1)):
                val = int(num_str)
                if val in candidate_by_idx and val not in selected_indices:
                    selected_indices.append(val)

    # --- Strategy 3: Numbered list items in text (e.g. 1. ... 5. ...) ---
    if not selected_indices:
        for num_str in RE_NUM_LIST.findall(raw_text):
            val = int(num_str)
            if val in candidate_by_idx and val not in selected_indices:
                selected_indices.append(val)

    # --- Strategy 4: Fuzzy Text / Substring Match on Entity Names ---
    if not selected_indices:
        raw_lower = raw_text.lower()
        for cand in candidates:
            if cand.entity_name.lower() in raw_lower:
                if cand.index not in selected_indices:
                    selected_indices.append(cand.index)

    # --- Step 5: Compliance Filter & Mana Budgeting ---
    valid_actions: List[ActionCandidate] = []
    mana_spent = 0
    attacked_entities: Set[str] = set()

    for idx in selected_indices:
        cand = candidate_by_idx[idx]

        # Prevent minion from attacking multiple times (without windfury)
        if cand.action_type == "ATTACK":
            if cand.entity_name in attacked_entities:
                continue
            attacked_entities.add(cand.entity_name)

        # Check mana
        if mana_spent + cand.mana_cost > max_mana:
            continue

        mana_spent += cand.mana_cost
        valid_actions.append(cand)

    # --- Step 6: Fallback if model returned empty or totally invalid plan ---
    if not valid_actions:
        # Heuristic: Take all free attacks to face or minions + highest mana playable card
        fallback_actions: List[ActionCandidate] = []
        fb_mana = 0
        fb_attacked: Set[str] = set()

        for cand in candidates:
            if cand.action_type == "ATTACK" and cand.entity_name not in fb_attacked:
                fallback_actions.append(cand)
                fb_attacked.add(cand.entity_name)

        # Find best playable card
        play_cards = [c for c in candidates if c.action_type == "PLAY" and c.mana_cost <= max_mana]
        if play_cards:
            # Sort by highest mana cost for max tempo
            play_cards.sort(key=lambda c: c.mana_cost, reverse=True)
            best_card = play_cards[0]
            if fb_mana + best_card.mana_cost <= max_mana:
                fallback_actions.insert(0, best_card)
                fb_mana += best_card.mana_cost

        return ParsedPlan(
            actions=fallback_actions,
            reasoning="Автоматический эвристический план (модель не вернула корректных действий)",
            total_mana_spent=fb_mana,
            is_fallback=True,
        )

    return ParsedPlan(
        actions=valid_actions,
        reasoning=reasoning or "План успешно сформирован моделью",
        total_mana_spent=mana_spent,
        is_fallback=False,
    )
