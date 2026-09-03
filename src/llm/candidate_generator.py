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
    entity_id: Optional[int] = None
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
    board_full = len(snapshot.friendly_board) + len(snapshot.friendly_locations) >= 7

    # 1. Physical Attacks from Board Minions
    # Only minions that can actually be attacked: no stealth, not dormant
    attackable_minions = [
        m for m in snapshot.opponent_board
        if not m.get("is_stealthed")
        and not m.get("is_dormant")
        and not m.get("is_immune")
        and not m.get("cant_be_attacked")
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
                    entity_id=m_eid,
                    target_name=t_name,
                    target_card_id=target.get("card_id", ""),
                    target_entity_id=target.get("entity_id"),
                    details={"attacker_entity_id": m_eid},
                    description=desc,
                )
            )
            idx += 1

        # Attack enemy hero if no Taunts blocking and minion isn't Rush-only
        if (
            can_hit_face
            and not enemy_taunts
            and not snapshot.opponent_hero.get("is_immune")
            and not snapshot.opponent_hero.get("cant_be_attacked")
        ):
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
                    entity_id=m_eid,
                    target_name=opp_hero.get("name", "Герой противника"),
                    target_card_id=opp_hero.get("card_id", ""),
                    target_entity_id=opp_hero.get("entity_id", 0),
                    details={"attacker_entity_id": m_eid},
                    description=desc,
                )
            )
            idx += 1

    # Hero attacks use the same Taunt and attackability rules as minions.
    friendly_hero = snapshot.friendly_hero
    if friendly_hero.get("can_attack") and friendly_hero.get("attack", 0) > 0:
        hero_name = friendly_hero.get("name", "Ваш герой")
        hero_attack = friendly_hero.get("attack", 0)
        hero_eid = friendly_hero.get("entity_id")
        for target in valid_attack_targets:
            t_name = target.get("name", "Вражеское существо")
            desc = f"Атака героем: {hero_name} ({hero_attack}) -> {t_name} ({target.get('attack', 0)}/{target.get('health', 0)})"
            candidates.append(
                ActionCandidate(
                    index=idx,
                    action_type="ATTACK",
                    entity_name=hero_name,
                    entity_card_id=friendly_hero.get("card_id", ""),
                    mana_cost=0,
                    entity_id=hero_eid,
                    target_name=t_name,
                    target_card_id=target.get("card_id", ""),
                    target_entity_id=target.get("entity_id"),
                    details={"attacker_entity_id": hero_eid},
                    description=desc,
                )
            )
            idx += 1

        if (
            not enemy_taunts
            and not snapshot.opponent_hero.get("is_immune")
            and not snapshot.opponent_hero.get("cant_be_attacked")
        ):
            opp_hero = snapshot.opponent_hero
            desc = f"Атака героем: {hero_name} ({hero_attack}) -> Герой противника"
            candidates.append(
                ActionCandidate(
                    index=idx,
                    action_type="ATTACK",
                    entity_name=hero_name,
                    entity_card_id=friendly_hero.get("card_id", ""),
                    mana_cost=0,
                    entity_id=hero_eid,
                    target_name=opp_hero.get("name", "Герой противника"),
                    target_card_id=opp_hero.get("card_id", ""),
                    target_entity_id=opp_hero.get("entity_id", 0),
                    details={"attacker_entity_id": hero_eid},
                    description=desc,
                )
            )
            idx += 1

    # 2. Playable Cards from Hand
    for card_data in snapshot.friendly_hand:
        cid = card_data.get("card_id", "")
        card_name = card_data.get("name", "Карта")
        cost = card_data.get("cost")
        if cost is None:
            continue  # unknown cost — cannot verify mana compliance
        if cost > mana:
            continue

        c_info = card_db.get_by_id(cid) if cid else None
        c_type = card_data.get("card_type") or (int(c_info.card_type) if c_info else 0)

        if c_type == CardType.ENCHANTMENT:
            continue  # enchantments never sit in hand / are never played directly

        # Non-targeted minions / weapons / secrets
        if c_type in (CardType.MINION, CardType.WEAPON) or not c_type:
            if c_type == CardType.MINION and board_full:
                continue  # board full (7 minions/locations) — cannot play more minions
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
                    entity_id=card_data.get("entity_id"),
                    description=desc,
                )
            )
            idx += 1

        elif c_type == CardType.SPELL:
            # CardDefs has no verified target contract; never guess spell legality from card text.
            continue

    # 3. Hero Power — only if the power entity exists and is not exhausted this turn.
    # Missing hero_power info means unknown state: treat as used (conservative, no illegal suggestion).
    hp_info = snapshot.hero_power or {}
    hp_used = hp_info.get("exhausted", True)
    hp_cost = hp_info.get("cost", 2)
    if mana >= hp_cost and not hp_used and hp_info:
        hero_power_name = hp_info.get("name") or "Сила героя"
        desc_hp = f"Сила героя: {hero_power_name} (2м)"
        candidates.append(
            ActionCandidate(
                index=idx,
                action_type="HERO_POWER",
                entity_name=hero_power_name,
                entity_card_id=hp_info.get("card_id", ""),
                mana_cost=hp_cost,
                entity_id=hp_info.get("entity_id"),
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
                    entity_id=loc.get("entity_id"),
                    description=desc_loc,
                )
            )
            idx += 1

    candidates.append(
        ActionCandidate(
            index=idx,
            action_type="END_TURN",
            entity_name="END_TURN",
            entity_card_id="",
            mana_cost=0,
            description="Завершить ход",
        )
    )

    return candidates
