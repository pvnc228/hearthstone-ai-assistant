from src.llm.next_action_contract import (
    NEXT_ACTION_SYSTEM_PROMPT,
    build_next_action_prompt,
    format_next_action_completion,
    parse_next_action_response,
    snapshot_to_prompt_state,
)
from src.llm.candidate_generator import ActionCandidate
from src.coach.analyzer import MatchCoach
from src.parser import TurnSnapshot


def test_prompt_and_completion_use_one_shared_candidate_contract():
    state = {
        "turn": 4,
        "mana": 3,
        "max_mana": 5,
        "friendly_hero": {"name": "Mage", "health": 28, "armor": 2},
        "opponent_hero": {"name": "Hunter", "health": 17, "armor": 0},
        "hand": [{"name": "Карта", "cost": 2}],
    }
    candidates = [{"id": 1, "description": "Разыграть карту"}, {"id": 2, "description": "Завершить ход"}]

    prompt = build_next_action_prompt(state, candidates)

    assert "Ход 4" in prompt
    assert "[1] Разыграть карту" in prompt
    assert "[2] Завершить ход" in prompt
    assert "один идентификатор кандидата" in prompt
    assert format_next_action_completion(2) == "PLAN: [2]"
    assert NEXT_ACTION_SYSTEM_PROMPT


def test_next_action_response_requires_single_existing_candidate():
    parsed = parse_next_action_response("PLAN: [2]", [1, 2])
    assert parsed.candidate_id == 2
    assert parsed.format_valid is True
    assert parsed.candidate_exists is True

    assert parse_next_action_response("PLAN: [1, 2]", [1, 2]).format_valid is False
    assert parse_next_action_response("PLAN: [9]", [1, 2]).candidate_exists is False
    assert parse_next_action_response("Рекомендую карту 1", [1, 2]).format_valid is False
    assert parse_next_action_response("PLAN: [1]\nПояснение", [1, 2]).format_valid is False


def test_match_coach_uses_the_same_prompt_builder():
    snapshot = TurnSnapshot(
        turn_number=1,
        active_player_id=1,
        active_player_name="player",
        is_friendly_turn=True,
        friendly_mana=1,
        friendly_max_mana=1,
        friendly_hero={"name": "Mage", "health": 30, "armor": 0},
        opponent_hero={"name": "Hunter", "health": 30, "armor": 0},
        friendly_hand=[],
        friendly_board=[],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
    )
    candidates = [ActionCandidate(1, "END_TURN", "END_TURN", "", 0, description="Завершить ход")]

    runtime_prompt = MatchCoach.build_llm_prompt(None, snapshot, candidates)
    contract_prompt = build_next_action_prompt(snapshot_to_prompt_state(snapshot), candidates)

    assert runtime_prompt == contract_prompt
