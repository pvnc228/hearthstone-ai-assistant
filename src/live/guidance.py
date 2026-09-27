"""
Human-in-the-loop action guidance and step translation for Hearthstone.
Converts model decision candidates into clear, unambiguous player instructions
(hand card position, board target position, mana cost, and step description)
without any memory injection into the game client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ActionGuidance:
    """Action guidance item presented to the player."""

    candidate_id: int
    action_type: str  # PLAY, ATTACK, HERO_POWER, LOCATION, END_TURN
    card_name: str
    mana_cost: int
    hand_position: Optional[int] = None  # 1-indexed from left to right
    attacker_name: Optional[str] = None
    attacker_board_pos: Optional[int] = None
    target_name: Optional[str] = None
    target_board_pos: Optional[int] = None
    target_is_hero: bool = False
    sub_option_name: Optional[str] = None
    placement_position: Optional[int] = None
    formatted_instruction: str = ""
    rationale: str = ""

    def __str__(self) -> str:
        return self.formatted_instruction


def format_guidance(
    candidate: Any,
    snapshot: Optional[Any] = None,
    entities: Optional[Dict[int, Any]] = None,
) -> ActionGuidance:
    """
    Transforms a ReplayOptionCandidate or ActionCandidate into an ActionGuidance instruction.
    """
    cand_id = getattr(candidate, "candidate_id", getattr(candidate, "index", 0))
    action_type = getattr(candidate, "action_type", "")
    card_name = getattr(candidate, "entity_name", "") or ""
    mana_cost = getattr(candidate, "mana_cost", 0)
    target_name = getattr(candidate, "target_name", None)
    sub_entity_name = getattr(candidate, "sub_entity_name", None)
    entity_id = getattr(candidate, "entity_id", None)
    target_entity_id = getattr(candidate, "target_entity_id", None)
    pos = getattr(candidate, "position", 0)

    hand_pos: Optional[int] = None
    target_board_pos: Optional[int] = None
    attacker_board_pos: Optional[int] = None
    target_is_hero = False

    # 1. Determine hand position for PLAY actions
    if snapshot and action_type == "PLAY" and hasattr(snapshot, "friendly_hand"):
        for i, card in enumerate(snapshot.friendly_hand, start=1):
            if entity_id is not None and card.get("entity_id") == entity_id:
                hand_pos = card.get("zone_position", i)
                break
            if not hand_pos and card.get("name") == card_name:
                hand_pos = card.get("zone_position", i)

    # 2. Determine attacker board position for ATTACK actions
    if snapshot and action_type == "ATTACK" and hasattr(snapshot, "friendly_board"):
        for i, minion in enumerate(snapshot.friendly_board, start=1):
            if entity_id is not None and minion.get("entity_id") == entity_id:
                attacker_board_pos = minion.get("zone_position", i)
                break
            if not attacker_board_pos and minion.get("name") == card_name:
                attacker_board_pos = minion.get("zone_position", i)

    # 3. Determine target position
    if snapshot and target_name:
        o_hero = getattr(snapshot, "opponent_hero", {})
        if o_hero and (target_name == o_hero.get("name") or target_entity_id == o_hero.get("entity_id")):
            target_is_hero = True
            target_name = f"Герой оппонента [{o_hero.get('name', 'Лицо')}]"
        elif hasattr(snapshot, "opponent_board"):
            for i, enemy_minion in enumerate(snapshot.opponent_board, start=1):
                if target_entity_id is not None and enemy_minion.get("entity_id") == target_entity_id:
                    target_board_pos = enemy_minion.get("zone_position", i)
                    break
                if not target_board_pos and enemy_minion.get("name") == target_name:
                    target_board_pos = enemy_minion.get("zone_position", i)

    # 4. Format clean, readable instruction
    if action_type == "END_TURN":
        instruction = "⏹️  Завершить ход"
    elif action_type == "PLAY":
        pos_str = f"[Рука #{hand_pos}] " if hand_pos else "[Рука] "
        mana_str = f"({mana_cost}м) " if mana_cost else ""
        sub_str = f" [{sub_entity_name}]" if sub_entity_name else ""
        tgt_str = f" -> {target_name}" if target_name else ""
        if target_board_pos:
            tgt_str += f" (поз. {target_board_pos})"
        board_slot_str = f" (на стол: поз. {pos})" if pos > 0 else ""
        instruction = f"{pos_str}Разыграть '{card_name}' {mana_str}{sub_str}{board_slot_str}{tgt_str}".strip()
    elif action_type == "ATTACK":
        atk_pos = f"[Стол #{attacker_board_pos}] " if attacker_board_pos else "[Стол] "
        tgt_str = f" -> {target_name}" if target_name else ""
        if target_board_pos:
            tgt_str += f" (поз. {target_board_pos})"
        instruction = f"{atk_pos}Атака: '{card_name}'{tgt_str}".strip()
    elif action_type == "HERO_POWER":
        tgt_str = f" -> {target_name}" if target_name else ""
        mana_str = f"({mana_cost}м)" if mana_cost else ""
        instruction = f"[Сила героя] '{card_name}' {mana_str}{tgt_str}".strip()
    elif action_type == "LOCATION":
        tgt_str = f" -> {target_name}" if target_name else ""
        instruction = f"[Область] Активировать '{card_name}'{tgt_str}".strip()
    else:
        desc = getattr(candidate, "description", card_name)
        instruction = f"Действие: {desc}"

    return ActionGuidance(
        candidate_id=cand_id,
        action_type=action_type,
        card_name=card_name,
        mana_cost=mana_cost,
        hand_position=hand_pos,
        attacker_name=card_name if action_type == "ATTACK" else None,
        attacker_board_pos=attacker_board_pos,
        target_name=target_name,
        target_board_pos=target_board_pos,
        target_is_hero=target_is_hero,
        sub_option_name=sub_entity_name,
        placement_position=pos if pos > 0 else None,
        formatted_instruction=instruction,
    )
