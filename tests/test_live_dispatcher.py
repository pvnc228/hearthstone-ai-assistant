import pytest
from src.live.dispatcher import LiveAdvisorDispatcher
from src.parser.state_tracker import ReplayOptionCandidate, TurnSnapshot


def make_snapshot(**kwargs):
    defaults = {
        "turn_number": 1,
        "active_player_id": 1,
        "active_player_name": "HappyBread#21597",
        "is_friendly_turn": True,
        "friendly_mana": 1,
        "friendly_max_mana": 1,
        "friendly_hero": {"name": "Джайна", "entity_id": 50, "health": 30, "armor": 0},
        "opponent_hero": {"name": "Андуин", "entity_id": 54, "health": 30, "armor": 0},
        "friendly_hand": [],
        "friendly_board": [],
        "opponent_board": [],
        "friendly_locations": [],
        "opponent_locations": [],
        "friendly_secrets": [],
        "opponent_secrets_count": 0,
        "opponent_hand_count": 4,
    }
    defaults.update(kwargs)
    return TurnSnapshot(**defaults)


def test_dispatcher_lethal_detection():
    dispatcher = LiveAdvisorDispatcher(enable_llm=False)

    # Opponent has 6 HP, Fireball costs 4 and deals 6
    snap = make_snapshot(
        turn_number=4,
        friendly_mana=4,
        friendly_max_mana=4,
        opponent_hero={"name": "Андуин", "entity_id": 54, "health": 6, "armor": 0},
        friendly_hand=[
            {"entity_id": 10, "card_id": "CS2_029", "name": "Огненный шар", "cost": 4, "zone_position": 1}
        ],
    )
    candidates = [
        ReplayOptionCandidate(
            candidate_id=1,
            option_id=0,
            option_type="END_TURN",
            action_type="END_TURN",
            entity_id=None,
            entity_name="",
            entity_card_id="",
            description="Завершить ход",
        ),
        ReplayOptionCandidate(
            candidate_id=2,
            option_id=1,
            option_type="POWER",
            action_type="PLAY",
            entity_id=10,
            entity_name="Огненный шар",
            entity_card_id="CS2_029",
            mana_cost=4,
            target_name="Андуин",
            target_entity_id=54,
            description="Огненный шар -> Андуин",
        ),
    ]

    rec = dispatcher.evaluate_decision(snap, candidates)
    assert rec.is_lethal is True
    assert rec.burst_damage >= 6
    assert "ОБНАРУЖЕН ЛЕТАЛЬНЫЙ УРОН" in rec.coach_note
    assert rec.top_guidances[0].card_name == "Огненный шар"


def test_dispatcher_normal_ranking():
    dispatcher = LiveAdvisorDispatcher(enable_llm=False)

    snap = make_snapshot(
        turn_number=2,
        friendly_mana=2,
        friendly_max_mana=2,
        friendly_hand=[
            {"entity_id": 15, "card_id": "CORE_CS2_024", "name": "Ледяная стрела", "cost": 2, "zone_position": 1}
        ],
    )
    candidates = [
        ReplayOptionCandidate(
            candidate_id=1,
            option_id=0,
            option_type="END_TURN",
            action_type="END_TURN",
            entity_id=None,
            entity_name="",
            entity_card_id="",
            description="Завершить ход",
        ),
        ReplayOptionCandidate(
            candidate_id=2,
            option_id=1,
            option_type="POWER",
            action_type="PLAY",
            entity_id=15,
            entity_name="Ледяная стрела",
            entity_card_id="CORE_CS2_024",
            mana_cost=2,
            target_name="Андуин",
            description="Ледяная стрела -> Андуин",
        ),
    ]

    rec = dispatcher.evaluate_decision(snap, candidates)
    assert rec.is_lethal is False
    assert len(rec.top_guidances) >= 1
    # Play card should rank higher than End Turn when mana is available
    assert rec.top_guidances[0].action_type == "PLAY"
    assert rec.latency_ms < 50.0  # Must be fast (<50ms)
