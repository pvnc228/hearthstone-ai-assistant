"""
Live event hub and game session manager for the Hearthstone assistant harness.
Maintains state across PowerEvents, detects turns and option decision boundaries,
and produces structured GameStateUpdates for presentation and model consumers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

from src.card_db import CardDatabase
from src.llm import generate_legal_candidates
from src.parser import GameStateTracker, PowerEvent, TurnSnapshot, parse_power_log_lines
from .dispatcher import LiveAdvisorDispatcher, LiveRecommendation

logger = logging.getLogger(__name__)


@dataclass
class GameStateUpdate:
    """Emitted by LiveGameSession whenever a meaningful game event occurs."""

    event_type: str  # GAME_START, TURN_START, DECISION, ACTION, GAME_OVER, TICK
    turn_number: int
    is_friendly_turn: bool
    friendly_hero_name: str
    opponent_hero_name: str
    friendly_mana: int
    friendly_max_mana: int
    snapshot: Optional[TurnSnapshot] = None
    recommendation: Optional[LiveRecommendation] = None
    raw_action_desc: str = ""
    game_result: Optional[str] = None  # WON, LOST, CONCEDED


class LiveGameSession:
    """
    Manages a live Hearthstone match session.
    Consumes raw Power.log lines, updates GameStateTracker, triggers
    model evaluations on decision points, and notifies observers.
    """

    def __init__(
        self,
        friendly_player_name: str = "HappyBread#21597",
        card_db: Optional[CardDatabase] = None,
        dispatcher: Optional[LiveAdvisorDispatcher] = None,
    ):
        self.friendly_player_name = friendly_player_name
        self.card_db = card_db or CardDatabase(auto_load=True)
        self.dispatcher = dispatcher or LiveAdvisorDispatcher(card_db=self.card_db)
        self.tracker = GameStateTracker(
            card_db=self.card_db,
            friendly_player_name=self.friendly_player_name,
        )

        self.match_active: bool = False
        self.last_decision_options_id: int = -1
        self.last_turn_processed: int = 0
        self._in_options_block: bool = False

    def reset(self) -> None:
        """Resets session for a new game."""
        self.tracker = GameStateTracker(
            card_db=self.card_db,
            friendly_player_name=self.friendly_player_name,
        )
        self.match_active = False
        self.last_decision_options_id = -1
        self.last_turn_processed = 0
        self._in_options_block = False

    def ingest_lines(self, lines: Iterable[str]) -> Generator[GameStateUpdate, None, None]:
        """Processes raw lines from the log stream and yields GameStateUpdates."""
        events = parse_power_log_lines(lines)
        for event in events:
            updates = self.process_event(event)
            for upd in updates:
                yield upd

    def process_event(self, event: PowerEvent) -> List[GameStateUpdate]:
        """Processes a single PowerEvent and returns any generated GameStateUpdates."""
        updates: List[GameStateUpdate] = []
        etype = event.event_type
        data = event.data

        # 1. Match Start Detection
        if etype == "CREATE_GAME":
            self.reset()
            self.match_active = True
            self.tracker.process_event(event)
            return updates

        # 2. Options tracking
        if etype == "OPTIONS_START":
            self._in_options_block = True
            self.tracker.process_event(event)
            return updates

        if etype in ("OPTION", "OPTION_TARGET", "OPTION_SUB_OPTION"):
            self.tracker.process_event(event)
            return updates

        # If we were in an options block and a non-option event arrives, the options block is complete!
        if self._in_options_block and etype not in ("OPTIONS_START", "OPTION", "OPTION_TARGET", "OPTION_SUB_OPTION"):
            self._in_options_block = False
            dec_update = self._check_and_trigger_decision()
            if dec_update:
                updates.append(dec_update)

        # 3. Game Over Detection
        if etype == "TAG_CHANGE":
            tag = data.get("tag")
            val = str(data.get("value", "")).strip()
            ent = data.get("entity", {})

            # Turn change detection
            if tag == "TURN":
                pass
            elif tag == "CURRENT_PLAYER" and val == "1":
                # Current player changed
                pass
            elif tag == "PLAYSTATE" and val in ("WON", "LOST", "CONCEDED"):
                ent_name = ent.get("name") or ent.get("raw") or ""
                is_friendly = (
                    ent_name == self.friendly_player_name
                    or ent.get("id") == self.tracker.friendly_player_id
                )
                result_str = val if is_friendly else ("LOST" if val == "WON" else "WON")
                self.match_active = False
                self.tracker.process_event(event)
                snap = self.tracker.take_turn_snapshot()
                updates.append(
                    GameStateUpdate(
                        event_type="GAME_OVER",
                        turn_number=snap.turn_number,
                        is_friendly_turn=snap.is_friendly_turn,
                        friendly_hero_name=snap.friendly_hero.get("name", "Герой"),
                        opponent_hero_name=snap.opponent_hero.get("name", "Оппонент"),
                        friendly_mana=snap.friendly_mana,
                        friendly_max_mana=snap.friendly_max_mana,
                        snapshot=snap,
                        game_result=result_str,
                    )
                )
                return updates

        # Let tracker process the event
        self.tracker.process_event(event)

        # 4. Turn start detection
        if self.tracker.current_turn > self.last_turn_processed and self.tracker.current_turn > 0:
            self.last_turn_processed = self.tracker.current_turn
            snap = self.tracker.take_turn_snapshot()
            updates.append(
                GameStateUpdate(
                    event_type="TURN_START",
                    turn_number=snap.turn_number,
                    is_friendly_turn=snap.is_friendly_turn,
                    friendly_hero_name=snap.friendly_hero.get("name", "Герой"),
                    opponent_hero_name=snap.opponent_hero.get("name", "Оппонент"),
                    friendly_mana=snap.friendly_mana,
                    friendly_max_mana=snap.friendly_max_mana,
                    snapshot=snap,
                )
            )

        # 5. Friendly action executed (BLOCK_START)
        if etype == "BLOCK_START":
            btype = data.get("block_type")
            ent_ref = data.get("entity", {})
            if btype in ("PLAY", "ATTACK", "POWER"):
                ent_name = ent_ref.get("entityName") or ent_ref.get("raw") or ""
                tgt_ref = data.get("target", {})
                tgt_name = tgt_ref.get("entityName") or tgt_ref.get("raw") or ""
                desc = f"{btype}: {ent_name}"
                if tgt_name:
                    desc += f" -> {tgt_name}"
                snap = self.tracker.take_turn_snapshot()
                updates.append(
                    GameStateUpdate(
                        event_type="ACTION",
                        turn_number=snap.turn_number,
                        is_friendly_turn=snap.is_friendly_turn,
                        friendly_hero_name=snap.friendly_hero.get("name", "Герой"),
                        opponent_hero_name=snap.opponent_hero.get("name", "Оппонент"),
                        friendly_mana=snap.friendly_mana,
                        friendly_max_mana=snap.friendly_max_mana,
                        snapshot=snap,
                        raw_action_desc=desc,
                    )
                )

        return updates

    def _check_and_trigger_decision(self) -> Optional[GameStateUpdate]:
        """
        Triggers a decision evaluation when a complete set of legal options
        is available for the friendly player.
        """
        opt_id = getattr(self.tracker, "_current_options_id", 0)
        if opt_id == self.last_decision_options_id or opt_id <= 0:
            return None

        # Build Blizzard oracle candidates
        oracle_candidates = self.tracker._build_option_candidates()
        snap = self.tracker.take_turn_snapshot()

        # Only evaluate decisions when it is our turn
        if not snap.is_friendly_turn and self.tracker.active_player_id != self.tracker.friendly_player_id:
            return None

        # If oracle candidates are empty, try deterministic legal generator fallback
        candidates = oracle_candidates
        if not candidates and snap.friendly_mana >= 0:
            candidates = generate_legal_candidates(snap, self.card_db)

        if not candidates:
            return None

        self.last_decision_options_id = opt_id

        # Invoke model dispatcher
        recommendation = self.dispatcher.evaluate_decision(snap, candidates)

        return GameStateUpdate(
            event_type="DECISION",
            turn_number=snap.turn_number,
            is_friendly_turn=snap.is_friendly_turn,
            friendly_hero_name=snap.friendly_hero.get("name", "Герой"),
            opponent_hero_name=snap.opponent_hero.get("name", "Оппонент"),
            friendly_mana=snap.friendly_mana,
            friendly_max_mana=snap.friendly_max_mana,
            snapshot=snap,
            recommendation=recommendation,
        )
