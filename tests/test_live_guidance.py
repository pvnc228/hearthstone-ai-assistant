import pytest
from src.live.guidance import ActionGuidance, format_guidance
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


def test_format_guidance_end_turn():
    cand = ReplayOptionCandidate(
        candidate_id=1,
        option_id=0,
        option_type="END_TURN",
        action_type="END_TURN",
        entity_id=None,
        entity_name="",
        entity_card_id="",
        description="Завершить ход",
    )
    guidance = format_guidance(cand)
    assert guidance.action_type == "END_TURN"
    assert "Завершить ход" in guidance.formatted_instruction


def test_format_guidance_play_card():
    cand = ReplayOptionCandidate(
        candidate_id=2,
        option_id=1,
        option_type="POWER",
        action_type="PLAY",
        entity_id=10,
        entity_name="Огненный шар",
        entity_card_id="CS2_029",
        mana_cost=4,
        target_name="Герой оппонента",
        target_entity_id=54,
    )
    snap = make_snapshot(
        turn_number=4,
        friendly_mana=4,
        friendly_max_mana=4,
        friendly_hero={"name": "Джайна", "entity_id": 50},
        opponent_hero={"name": "Герой оппонента", "entity_id": 54},
        friendly_hand=[
            {"entity_id": 9, "name": "Монетка", "zone_position": 1},
            {"entity_id": 10, "name": "Огненный шар", "zone_position": 2},
        ],
    )
    guidance = format_guidance(cand, snap)
    assert guidance.action_type == "PLAY"
    assert guidance.hand_position == 2
    assert "Рука #2" in guidance.formatted_instruction
    assert "Огненный шар" in guidance.formatted_instruction
    assert "(4м)" in guidance.formatted_instruction
    assert guidance.target_is_hero is True


def test_format_guidance_attack():
    cand = ReplayOptionCandidate(
        candidate_id=3,
        option_id=2,
        option_type="POWER",
        action_type="ATTACK",
        entity_id=20,
        entity_name="Маназмей",
        entity_card_id="NEW1_012",
        mana_cost=0,
        target_name="Вражеское существо",
        target_entity_id=30,
    )
    snap = make_snapshot(
        turn_number=3,
        friendly_mana=3,
        friendly_max_mana=3,
        friendly_board=[
            {"entity_id": 20, "name": "Маназмей", "zone_position": 1, "attack": 2, "health": 3}
        ],
        opponent_board=[
            {"entity_id": 30, "name": "Вражеское существо", "zone_position": 1, "attack": 1, "health": 2}
        ],
    )
    guidance = format_guidance(cand, snap)
    assert guidance.action_type == "ATTACK"
    assert guidance.attacker_board_pos == 1
    assert "Стол #1" in guidance.formatted_instruction
    assert "Маназмей" in guidance.formatted_instruction
    assert "Вражеское существо" in guidance.formatted_instruction
