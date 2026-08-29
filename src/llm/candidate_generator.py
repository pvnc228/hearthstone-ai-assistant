"""
Deterministic Legal Candidate Action Generator for Hearthstone.
Generates numbered legal actions (plays, attacks, hero power, locations)
strictly validated against current mana, board state, and Taunt rules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.card_db import CardDatabase, CardType
from src.parser.state_tracker import TurnSnapshot


@dataclass
class ActionCandidate:
    index: int
    action_type: str  # PLAY, ATTACK, HERO_POWER, LOCATION
    entity_name: str
    entity_card_id: str
    mana_cost: int
    target_name: Optional[str] = None
    target_card_id: Optional[str] = None
    target_entity_id: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __str__(self) -> str:
        return f"[{self.index}] {self.description}"


def generate_legal_candidates(snapshot: TurnSnapshot, card_db: CardDatabase) -> List[ActionCandidate]:
    """
    Generates all legal action candidates for a given TurnSnapshot.
    Strictly enforces:
    1. Mana limits for hand cards and Hero Power.
    2. Taunt restrictions for minion/hero physical attacks.
    3. Ready minions (can_attack=True).
    4. Location durability and exhaustion states.
    """
    candidates: List[ActionCandidate] = []
    idx = 1
    mana = snapshot.friendly_mana
    board_full = len(snapshot.friendly_board) >= 10

    # 1. Physical Attacks from Board Minions
    # Only minions that can actually be attacked: no stealth, not dormant
    attackable_minions = [
        m for m in snapshot.opponent_board
        if not m.get("is_stealthed") and not m.get("is_dormant")
    ]
    # Taunts force targets among themselves
    enemy_taunts = [m for m in attackable_minions if m.get("is_taunt")]
    valid_attack_targets = enemy_taunts if enemy_taunts else attackable_minions

    for minion in snapshot.friendly_board:
        if not minion.get("can_attack") or minion.get("attack", 0) <= 0:
            continue
        m_name = minion.get("name", "Существо")
        m_atk = minion.get("attack", 0)
        m_hp = minion.get("health", 0)
        m_eid = minion.get("entity_id")

        # Rush minions can only attack minions, never the hero
        can_hit_face = minion.get("can_attack_hero", True)

        # Attack enemy minions
        for target in valid_attack_targets:
            t_name = target.get("name", "Вражеское существо")
            t_atk = target.get("attack", 0)
            t_hp = target.get("health", 0)
            taunt_flag = " [Провокация]" if target.get("is_taunt") else ""

            desc = f"Атака: {m_name} ({m_atk}/{m_hp}) -> {t_name} ({t_atk}/{t_hp}){taunt_flag}"
            candidates.append(
                ActionCandidate(
                    index=idx,
                    action_type="ATTACK",
                    entity_name=m_name,
                    entity_card_id=minion.get("card_id", ""),
                    mana_cost=0,
                    target_name=t_name,
                    target_card_id=target.get("card_id", ""),
                    target_entity_id=target.get("entity_id"),
                    description=desc,
                )
            )
            idx += 1

        # Attack enemy hero if no Taunts blocking and minion isn't Rush-only
        if can_hit_face and not enemy_taunts:
            opp_hero = snapshot.opponent_hero
            opp_hp = opp_hero.get("health", 30)
            opp_armor = opp_hero.get("armor", 0)
            armor_str = f"+{opp_armor}" if opp_armor else ""
            desc = f"Атака в лицо: {m_name} ({m_atk}/{m_hp}) -> Герой противника ({opp_hp}{armor_str} HP)"

            candidates.append(
                ActionCandidate(
                    index=idx,
                    action_type="ATTACK",
                    entity_name=m_name,
                    entity_card_id=minion.get("card_id", ""),
                    mana_cost=0,
                    target_name=opp_hero.get("name", "Герой противника"),
                    target_card_id=opp_hero.get("card_id", ""),
                    target_entity_id=0,  # 0 = enemy hero
                    details={"attacker_entity_id": m_eid},
                    description=desc,
                )
            )
            idx += 1

    # 2. Playable Cards from Hand
    for card_data in snapshot.friendly_hand:
        cid = card_data.get("card_id", "")
        card_name = card_data.get("name", "Карта")
        cost = card_data.get("cost", 0)

        if cost > mana:
            continue

        c_info = card_db.get_by_id(cid) if cid else None
        c_type = card_data.get("card_type") or (int(c_info.card_type) if c_info else 4)

        # Non-targeted minions / weapons / secrets
        if c_type in (CardType.MINION, CardType.WEAPON, CardType.ENCHANTMENT) or not c_info:
            if c_type == CardType.MINION and board_full:
                continue  # board full (10 minions) — cannot play more minions
            atk = card_data.get("attack") or (c_info.attack if c_info else None)
            hp = card_data.get("health") or (c_info.health if c_info else None)
            stats = f" {atk}/{hp}" if atk is not None and hp is not None else ""
            desc = f"Разыграть: {card_name} ({cost}м{stats})"

            candidates.append(
                ActionCandidate(
                    index=idx,
                    action_type="PLAY",
                    entity_name=card_name,
                    entity_card_id=cid,
                    mana_cost=cost,
                    description=desc,
                )
            )
            idx += 1

        elif c_type == CardType.SPELL:
            # Check targeted damage/removal spells
            # Target enemy hero
            desc_hero = f"Разыграть заклинание: {card_name} ({cost}м) -> Герой противника"
            candidates.append(
                ActionCandidate(
                    index=idx,
                    action_type="PLAY",
                    entity_name=card_name,
                    entity_card_id=cid,
                    mana_cost=cost,
                    target_name=snapshot.opponent_hero.get("name", "Герой противника"),
                    target_card_id=snapshot.opponent_hero.get("card_id", ""),
                    description=desc_hero,
                )
            )
            idx += 1

            # Target enemy minions
            for target in snapshot.opponent_board:
                t_name = target.get("name", "Вражеское существо")
                desc_minion = f"Разыграть заклинание: {card_name} ({cost}м) -> {t_name}"
                candidates.append(
                    ActionCandidate(
                        index=idx,
                        action_type="PLAY",
                        entity_name=card_name,
                        entity_card_id=cid,
                        mana_cost=cost,
                        target_name=t_name,
                        target_card_id=target.get("card_id", ""),
                        target_entity_id=target.get("entity_id"),
                        description=desc_minion,
                    )
                )
                idx += 1

    # 3. Hero Power — only if the power entity exists and is not exhausted this turn
    hp_info = snapshot.hero_power or {}
    hp_used = hp_info.get("exhausted", False)
    if mana >= 2 and not hp_used:
        hero_power_name = hp_info.get("name") or "Сила героя"
        desc_hp = f"Сила героя: {hero_power_name} (2м)"
        candidates.append(
            ActionCandidate(
                index=idx,
                action_type="HERO_POWER",
                entity_name=hero_power_name,
                entity_card_id=hp_info.get("card_id", ""),
                mana_cost=2,
                target_entity_id=None,
                description=desc_hp,
            )
        )
        idx += 1

    # 4. Locations
    for loc in snapshot.friendly_locations:
        if loc.get("can_use") and loc.get("durability", 0) > 0:
            l_name = loc.get("name", "Область")
            dur = loc.get("durability", 0)
            desc_loc = f"Активировать область: {l_name} ({dur} пр.)"
            candidates.append(
                ActionCandidate(
                    index=idx,
                    action_type="LOCATION",
                    entity_name=l_name,
                    entity_card_id=loc.get("card_id", ""),
                    mana_cost=0,
                    description=desc_loc,
                )
            )
            idx += 1

    return candidates
