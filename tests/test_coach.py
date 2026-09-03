"""
Unit and integration tests for Coach Analyzer, Legal Candidate Generator, and Resilient Response Parser.
"""

import pytest

from src.card_db import CardDatabase
from src.coach import MatchCoach, MatchReport
from src.llm import ActionCandidate, generate_legal_candidates, parse_model_response
from src.parser import TurnSnapshot


def test_candidate_generator_taunt_and_mana():
    db = CardDatabase()

    # Board with 1 friendly ready minion (3/2) and 1 enemy Taunt minion (4/3) and enemy hero (15 HP)
    snap = TurnSnapshot(
        turn_number=3,
        active_player_id=1,
        active_player_name="HappyBread#21597",
        is_friendly_turn=True,
        friendly_mana=3,
        friendly_max_mana=3,
        friendly_hero={"health": 30, "armor": 0, "name": "Mage"},
        opponent_hero={"health": 15, "armor": 0, "name": "Hunter"},
        friendly_hand=[
            {"card_id": "CS2_024", "name": "Ледяная стрела", "cost": 2, "card_type": 5},
            {"card_id": "CS2_029", "name": "Огненный шар", "cost": 4, "card_type": 5},  # Cannot afford with 3 mana
        ],
        friendly_board=[
            {"entity_id": 10, "card_id": "", "name": "Вожак волков", "attack": 3, "health": 2, "can_attack": True}
        ],
        opponent_board=[
            {"entity_id": 20, "card_id": "", "name": "Псарь", "attack": 4, "health": 3, "is_taunt": True}
        ],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=3,
    )

    candidates = generate_legal_candidates(snap, db)
    assert len(candidates) > 0

    # 1. Check that minion CANNOT attack enemy hero directly because of Taunt
    hero_attacks = [c for c in candidates if c.action_type == "ATTACK" and "Герой противника" in c.description]
    assert len(hero_attacks) == 0

    # 2. Check that minion CAN attack Taunt minion
    taunt_attacks = [c for c in candidates if c.action_type == "ATTACK" and "Псарь" in c.description]
    assert len(taunt_attacks) == 1

    # 3. Spell targets are not offered without a verified target contract.
    frostbolt_plays = [c for c in candidates if c.action_type == "PLAY" and "Ледяная стрела" in c.description]
    assert len(frostbolt_plays) == 0

    fireball_plays = [c for c in candidates if c.action_type == "PLAY" and "Огненный шар" in c.description]
    assert len(fireball_plays) == 0


def test_resilient_response_parser():
    candidates = [
        ActionCandidate(index=1, action_type="PLAY", entity_name="Ледяная стрела", entity_card_id="CS2_024", mana_cost=2, description="Разыграть: Ледяная стрела (2м) -> Псарь"),
        ActionCandidate(index=2, action_type="ATTACK", entity_name="Вожак волков", entity_card_id="", mana_cost=0, description="Атака: Вожак волков -> Псарь"),
        ActionCandidate(index=3, action_type="HERO_POWER", entity_name="Сила героя", entity_card_id="", mana_cost=2, description="Сила героя (2м)"),
    ]

    # Test Format 1: PLAN: [1, 2]
    raw1 = "ПЛАН: [1, 2]\nОБОСНОВАНИЕ: Уничтожаем провокатора без потери существа."
    p1 = parse_model_response(raw1, candidates, max_mana=3)
    assert len(p1.actions) == 2
    assert p1.actions[0].index == 1
    assert p1.actions[1].index == 2
    assert p1.total_mana_spent == 2
    assert p1.is_fallback is False

    # Test Format 2: Overspending mana (1: 2m + 3: 2m = 4m > 3m available)
    raw2 = "PLAN: [1, 3, 2]"
    p2 = parse_model_response(raw2, candidates, max_mana=3)
    # Action 3 should be dropped due to mana limit, action 2 (0 mana) kept
    assert len(p2.actions) == 2
    assert p2.total_mana_spent == 2

    # Test Format 3: Free text with card name
    raw3 = "Рекомендую использовать Ледяная стрела и добить существо."
    p3 = parse_model_response(raw3, candidates, max_mana=3)
    assert len(p3.actions) >= 1
    assert p3.actions[0].index == 1

    # Test Format 4: Empty garbage output -> triggers safe heuristic fallback
    raw4 = "бла бла бла ничего не понятно"
    p4 = parse_model_response(raw4, candidates, max_mana=3)
    assert len(p4.actions) > 0
    assert p4.is_fallback is True


def test_lethal_calculation():
    db = CardDatabase()
    coach = MatchCoach(card_db=db)

    # Opponent at 9 HP, no taunts, friendly board has 3 attack minion + Fireball in hand (6 dmg) = 9 damage lethal!
    snap = TurnSnapshot(
        turn_number=5,
        active_player_id=1,
        active_player_name="HappyBread#21597",
        is_friendly_turn=True,
        friendly_mana=4,
        friendly_max_mana=4,
        friendly_hero={"health": 30, "armor": 0, "name": "Mage"},
        opponent_hero={"health": 9, "armor": 0, "name": "Hunter"},
        friendly_hand=[
            {"card_id": "CS2_029", "name": "Огненный шар", "cost": 4, "card_type": 5}
        ],
        friendly_board=[
            {"entity_id": 10, "name": "Существо", "attack": 3, "health": 3, "can_attack": True}
        ],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=3,
    )

    burst = coach.calculate_max_burst_damage(snap)
    assert burst == 9

    analysis = coach.analyze_turn(snap, query_llm=False)
    assert analysis.is_lethal_possible is True
    assert any("ЛЕТАЛ" in n for n in analysis.notes)


def test_burst_knapsack_not_greedy():
    """Regression: greedy hand-order spent Arcane Shot (1м/2dmg) before Fireball
    (4м/6dmg), undercounting burst and missing lethals."""
    db = CardDatabase()
    coach = MatchCoach(card_db=db)
    snap = TurnSnapshot(
        turn_number=5,
        active_player_id=1,
        active_player_name="x",
        is_friendly_turn=True,
        friendly_mana=4,
        friendly_max_mana=4,
        friendly_hero={"health": 30, "armor": 0, "name": "Mage"},
        opponent_hero={"health": 8, "armor": 0, "name": "Hunter"},
        friendly_hand=[
            # Hand order: cheap shot FIRST — greedy consumes it and can't afford fireball
            {"card_id": "DS1_185", "name": "Волшебная стрела", "cost": 1, "card_type": 5},
            {"card_id": "CS2_029", "name": "Огненный шар", "cost": 4, "card_type": 5},
        ],
        friendly_board=[],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
    )
    # Exact: Fireball (4м, 6 dmg) — NOT greedy shot-first (2 dmg, fireball unaffordable)
    assert coach.calculate_max_burst_damage(snap) == 6


def test_rush_minion_cannot_hit_face():
    """Regression: RUSH minions were offered 'Атака в лицо' candidates."""
    db = CardDatabase()
    snap = TurnSnapshot(
        turn_number=4,
        active_player_id=1,
        active_player_name="x",
        is_friendly_turn=True,
        friendly_mana=4,
        friendly_max_mana=4,
        friendly_hero={"health": 30, "armor": 0, "name": "Mage"},
        opponent_hero={"health": 30, "armor": 0, "name": "Hunter"},
        friendly_hand=[],
        friendly_board=[
            # Rush minion: ready but face-forbidden
            {"entity_id": 10, "name": "Рывок", "attack": 3, "health": 3, "can_attack": True, "can_attack_hero": False},
        ],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
    )
    candidates = generate_legal_candidates(snap, db)
    face_attacks = [c for c in candidates if "Атака в лицо" in c.description]
    assert len(face_attacks) == 0


def test_stealthed_enemies_not_attackable():
    """Regression: stealthed/dormant enemy minions were offered as targets."""
    db = CardDatabase()
    snap = TurnSnapshot(
        turn_number=4,
        active_player_id=1,
        active_player_name="x",
        is_friendly_turn=True,
        friendly_mana=4,
        friendly_max_mana=4,
        friendly_hero={"health": 30, "armor": 0, "name": "Mage"},
        opponent_hero={"health": 30, "armor": 0, "name": "Hunter"},
        friendly_hand=[],
        friendly_board=[
            {"entity_id": 10, "name": "Атакующий", "attack": 2, "health": 2, "can_attack": True},
        ],
        opponent_board=[
            {"entity_id": 20, "name": "Маскировщик", "attack": 1, "health": 1, "is_stealthed": True},
            {"entity_id": 21, "name": "Спящий", "attack": 1, "health": 5, "is_dormant": True},
        ],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
    )
    candidates = generate_legal_candidates(snap, db)
    # No minion attack targets (all stealthed/dormant) — only face attack (+ hero power) remain
    assert not any(c.action_type == "ATTACK" and "Атака в лицо" not in c.description for c in candidates)
    assert not any("Маскировщик" in (c.description or "") for c in candidates)
    assert not any("Спящий" in (c.description or "") for c in candidates)


def test_candidate_ids_board_limit_hero_attack_and_end_turn():
    db = CardDatabase()
    snap = TurnSnapshot(
        turn_number=6,
        active_player_id=1,
        active_player_name="x",
        is_friendly_turn=True,
        friendly_mana=10,
        friendly_max_mana=10,
        friendly_hero={
            "entity_id": 2,
            "card_id": "HERO_03",
            "name": "Rogue",
            "health": 30,
            "armor": 0,
            "attack": 2,
            "can_attack": True,
        },
        opponent_hero={
            "entity_id": 3,
            "card_id": "HERO_08",
            "name": "Mage",
            "health": 20,
            "armor": 0,
        },
        friendly_hand=[
            {
                "entity_id": 100,
                "card_id": "EX1_116",
                "name": "Лирой Дженкинс",
                "cost": 5,
                "card_type": 4,
            }
        ],
        friendly_board=[
            {
                "entity_id": idx,
                "name": f"Существо {idx}",
                "attack": 1,
                "health": 1,
                "can_attack": False,
            }
            for idx in range(10, 17)
        ],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
    )

    candidates = generate_legal_candidates(snap, db)
    assert not any(c.action_type == "PLAY" and c.entity_id == 100 for c in candidates)
    hero_attack = next(c for c in candidates if c.action_type == "ATTACK")
    assert hero_attack.entity_id == 2
    assert hero_attack.target_entity_id == 3
    assert candidates[-1].action_type == "END_TURN"


def test_board_limit_counts_friendly_locations():
    db = CardDatabase()
    snap = TurnSnapshot(
        turn_number=6,
        active_player_id=1,
        active_player_name="x",
        is_friendly_turn=True,
        friendly_mana=10,
        friendly_max_mana=10,
        friendly_hero={"health": 30, "armor": 0, "name": "Rogue"},
        opponent_hero={"health": 30, "armor": 0, "name": "Mage"},
        friendly_hand=[
            {
                "entity_id": 100,
                "card_id": "EX1_116",
                "name": "Лирой Дженкинс",
                "cost": 5,
                "card_type": 4,
            }
        ],
        friendly_board=[
            {
                "entity_id": idx,
                "name": f"Существо {idx}",
                "attack": 1,
                "health": 1,
                "can_attack": False,
            }
            for idx in range(10, 16)
        ],
        opponent_board=[],
        friendly_locations=[
            {"entity_id": 200, "card_id": "CATA_301", "name": "Локация"}
        ],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
    )

    candidates = generate_legal_candidates(snap, db)

    assert not any(c.action_type == "PLAY" and c.entity_id == 100 for c in candidates)


def test_spells_without_verified_target_contract_get_no_play_candidates():
    db = CardDatabase()
    snap = TurnSnapshot(
        turn_number=6,
        active_player_id=1,
        active_player_name="x",
        is_friendly_turn=True,
        friendly_mana=10,
        friendly_max_mana=10,
        friendly_hero={"health": 30, "armor": 0, "name": "Mage"},
        opponent_hero={"entity_id": 3, "health": 30, "armor": 0, "name": "Hunter"},
        friendly_hand=[
            {
                "entity_id": 101,
                "card_id": "ETC_COIN1",
                "name": "Монетка",
                "cost": 0,
                "card_type": 5,
            },
            {
                "entity_id": 102,
                "card_id": "UNVERIFIED_SPELL",
                "name": "Неразрешённое заклинание",
                "cost": 0,
                "card_type": 5,
            },
        ],
        friendly_board=[],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
    )

    candidates = generate_legal_candidates(snap, db)

    assert not any(c.action_type == "PLAY" and c.entity_id in {101, 102} for c in candidates)
