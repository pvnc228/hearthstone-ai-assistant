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

    # 3. Check that 2-mana Frostbolt is playable, but 4-mana Fireball is NOT playable
    frostbolt_plays = [c for c in candidates if c.action_type == "PLAY" and "Ледяная стрела" in c.description]
    assert len(frostbolt_plays) > 0

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
