import pytest
from src.live.event_hub import LiveGameSession
from src.live.dispatcher import LiveAdvisorDispatcher


def test_live_session_events():
    session = LiveGameSession(friendly_player_name="HappyBread#21597")

    sample_log = [
        "D 12:00:00.0000000 GameState.DebugPrintPower() - CREATE_GAME\n",
        "D 12:00:00.0000001 GameState.DebugPrintGame() - Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=1]\n",
        "D 12:00:00.0000002 GameState.DebugPrintGame() - Player EntityID=3 PlayerID=2 GameAccountId=[hi=1 lo=2]\n",
        "D 12:00:00.0000003 GameState.DebugPrintPower() - TAG_CHANGE Entity=GameEntity tag=TURN value=1\n",
        "D 12:00:00.0000004 GameState.DebugPrintPower() - TAG_CHANGE Entity=HappyBread#21597 tag=CURRENT_PLAYER value=1\n",
        "D 12:00:00.0000005 GameState.DebugPrintPower() - TAG_CHANGE Entity=HappyBread#21597 tag=RESOURCES value=1\n",
        "D 12:00:00.0000006 GameState.DebugPrintOptions() - id=1\n",
        "D 12:00:00.0000007 GameState.DebugPrintOptions() -   option 0 type=END_TURN mainEntity= error=NONE errorParam=\n",
        "D 12:00:00.0000008 GameState.DebugPrintPower() - TAG_CHANGE Entity=GameEntity tag=STEP value=MAIN_ACTION\n",
    ]

    updates = list(session.ingest_lines(sample_log))
    types = [u.event_type for u in updates]

    assert "TURN_START" in types
    assert "DECISION" in types

    decision_upd = next(u for u in updates if u.event_type == "DECISION")
    assert decision_upd.is_friendly_turn is True
    assert decision_upd.recommendation is not None
    assert len(decision_upd.recommendation.top_guidances) >= 1
