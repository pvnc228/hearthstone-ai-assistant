import pytest
from src.card_db import CardDatabase
from src.live.dispatcher import LiveAdvisorDispatcher, LiveRecommendation
from src.live.event_hub import GameStateUpdate, LiveGameSession
from src.live.guidance import ActionGuidance
from src.live.tui import AdvisorWidget, BoardWidget, HandWidget, HearthstoneTUIApp
from src.parser.state_tracker import TurnSnapshot


def make_snapshot():
    return TurnSnapshot(
        turn_number=3,
        active_player_id=1,
        active_player_name="HappyBread#21597",
        is_friendly_turn=True,
        friendly_mana=3,
        friendly_max_mana=3,
        friendly_hero={"name": "Джайна", "entity_id": 50, "health": 30, "armor": 0},
        opponent_hero={"name": "Андуин", "entity_id": 54, "health": 28, "armor": 0},
        friendly_hand=[
            {"entity_id": 10, "name": "Монетка", "cost": 0, "zone_position": 1},
            {"entity_id": 11, "name": "Огненный шар", "cost": 4, "zone_position": 2},
        ],
        friendly_board=[
            {"entity_id": 20, "name": "Маназмей", "zone_position": 1, "attack": 2, "health": 3}
        ],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=4,
    )


def test_tui_widgets_render():
    snap = make_snapshot()

    # Board widget
    bw = BoardWidget()
    bw.update_board(snap, 3, 3, 3)
    assert bw is not None

    # Hand widget
    hw = HandWidget()
    hw.update_hand(snap, 3)
    assert hw is not None

    # Advisor widget
    aw = AdvisorWidget()
    rec = LiveRecommendation(
        top_guidances=[
            ActionGuidance(
                candidate_id=1,
                action_type="PLAY",
                card_name="Монетка",
                mana_cost=0,
                formatted_instruction="[Рука #1] Разыграть 'Монетка' (0м)",
            )
        ],
        is_lethal=False,
        coach_note="Тестовый совет тренера",
        latency_ms=12.5,
    )
    aw.update_recommendation(rec)
    assert aw is not None


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_tui_app_lifecycle():
    card_db = CardDatabase(auto_load=True)
    session = LiveGameSession(friendly_player_name="HappyBread#21597", card_db=card_db)

    def dummy_generator():
        return iter([])

    app = HearthstoneTUIApp(
        session=session,
        line_generator_factory=dummy_generator,
        source_label="Test",
    )

    async with app.run_test() as pilot:
        # Verify app mounts cleanly
        assert app.is_mounted
        await pilot.pause()
        # Verify widgets are present
        assert app.query_one(BoardWidget) is not None
        assert app.query_one(HandWidget) is not None
        assert app.query_one(AdvisorWidget) is not None
