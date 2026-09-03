"""
Deterministic GameState Tracker for Hearthstone.
Processes PowerEvents and maintains accurate board states, hand cards, player health/mana,
and captures per-turn snapshots with player actions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.card_db import CardDatabase, CardType

# String name to CardType int mapping from Power.log
STR_TO_CARD_TYPE = {
    "GAME": 1,
    "PLAYER": 2,
    "HERO": 3,
    "MINION": 4,
    "SPELL": 5,
    "ENCHANTMENT": 6,
    "WEAPON": 7,
    "ITEM": 8,
    "TOKEN": 9,
    "HERO_POWER": 10,
    "LOCATION": 39,
}


@dataclass
class Entity:
    entity_id: int
    card_id: str = ""
    name: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    def get_tag(self, tag_name: str, default: int = 0) -> int:
        val = self.tags.get(tag_name, default)
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                return int(val)
        return default

    def get_str_tag(self, tag_name: str, default: str = "") -> str:
        return str(self.tags.get(tag_name, default))

    @property
    def zone(self) -> str:
        return self.get_str_tag("ZONE", "INVALID")

    @property
    def card_type(self) -> int:
        # 3=HERO, 4=MINION, 5=SPELL, 7=WEAPON, 10=HERO_POWER, 39=LOCATION
        val = self.tags.get("CARDTYPE", 0)
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            if val in STR_TO_CARD_TYPE:
                return STR_TO_CARD_TYPE[val]
            if val.isdigit():
                return int(val)
        return 0

    @property
    def controller(self) -> int:
        return self.get_tag("CONTROLLER", 0)

    @property
    def cost(self) -> int:
        # ponytail: EXHAUSTED-as-cost hack removed; live COST tag is authoritative, DB lookup only when absent
        if "COST" in self.tags:
            return self.get_tag("COST", 0)
        return -1  # sentinel: caller falls back to CardDefs cost

    @property
    def attack(self) -> int:
        return self.get_tag("ATK", 0)

    @property
    def health(self) -> int:
        base_hp = self.get_tag("HEALTH", 0)
        damage = self.get_tag("DAMAGE", 0)
        return max(0, base_hp - damage)

    @property
    def max_health(self) -> int:
        return self.get_tag("HEALTH", 0)

    @property
    def armor(self) -> int:
        return self.get_tag("ARMOR", 0)

    @property
    def durability(self) -> int:
        base_dur = self.get_tag("DURABILITY", 0) or self.get_tag("HEALTH", 0)
        damage = self.get_tag("DAMAGE", 0)
        return max(0, base_dur - damage)

    @property
    def is_exhausted(self) -> bool:
        return self.get_tag("EXHAUSTED", 0) == 1

    @property
    def is_frozen(self) -> bool:
        return self.get_tag("FROZEN", 0) == 1

    @property
    def is_taunt(self) -> bool:
        return self.get_tag("TAUNT", 0) == 1

    @property
    def is_divine_shield(self) -> bool:
        return self.get_tag("DIVINE_SHIELD", 0) == 1

    @property
    def is_stealthed(self) -> bool:
        return self.get_tag("STEALTH", 0) == 1

    @property
    def is_reborn(self) -> bool:
        return self.get_tag("REBORN", 0) == 1

    @property
    def is_silenced(self) -> bool:
        return self.get_tag("SILENCED", 0) == 1

    @property
    def is_dormant(self) -> bool:
        return self.get_tag("DORMANT", 0) == 1

    @property
    def is_titan(self) -> bool:
        return self.get_tag("TITAN", 0) == 1

    @property
    def is_starship(self) -> bool:
        return self.get_tag("STARSHIP", 0) == 1 or self.get_tag("STARSHIP_PIECE", 0) == 1

    @property
    def can_attack(self) -> bool:
        if self.zone != "PLAY" or self.is_frozen or self.is_dormant or self.get_tag("CANT_ATTACK", 0) == 1:
            return False
        if self.attack <= 0:
            return False
        if not (not self.is_exhausted or self.get_tag("CHARGE", 0) == 1 or self.get_tag("RUSH", 0) == 1):
            return False
        # Attack budget: 1 normally, 2 with Windfury, 4 with Mega-Windfury.
        if self.get_tag("MEGA_WINDFURY", 0) == 1:
            max_attacks = 4
        elif self.get_tag("WINDFURY", 0) == 1:
            max_attacks = 2
        else:
            max_attacks = 1
        return self.num_attacks_this_turn < max_attacks

    @property
    def can_attack_hero(self) -> bool:
        """Fresh Rush minions cannot hit heroes; the restriction expires next turn."""
        if not self.can_attack or self.get_tag("CANT_ATTACK_HEROES", 0) == 1:
            return False
        is_fresh_rush = (
            self.get_tag("RUSH", 0) == 1
            and self.get_tag("CHARGE", 0) != 1
            and self.get_tag("NUM_TURNS_IN_PLAY", 0) == 0
        )
        return not is_fresh_rush

    @property
    def num_attacks_this_turn(self) -> int:
        return self.get_tag("NUM_ATTACKS_THIS_TURN", 0)


@dataclass
class PlayerState:
    player_id: int
    entity_id: int
    name: str = ""
    hero_entity_id: Optional[int] = None
    hero_power_entity_id: Optional[int] = None
    resources: int = 0
    resources_used: int = 0
    temp_resources: int = 0
    overload_locked: int = 0
    overload_owes: int = 0
    corpses: int = 0
    playstate: str = "PLAYING"

    @property
    def mana_available(self) -> int:
        return max(0, (self.resources + self.temp_resources) - self.resources_used)

@dataclass
class PlayerAction:
    turn: int
    action_type: str  # PLAY, ATTACK, HERO_POWER, LOCATION
    entity_name: str
    entity_card_id: str
    target_name: Optional[str] = None
    target_card_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnSnapshot:
    turn_number: int
    active_player_id: int
    active_player_name: str
    is_friendly_turn: bool
    friendly_mana: int
    friendly_max_mana: int
    friendly_hero: Dict[str, Any]
    opponent_hero: Dict[str, Any]
    friendly_hand: List[Dict[str, Any]]
    friendly_board: List[Dict[str, Any]]
    opponent_board: List[Dict[str, Any]]
    friendly_locations: List[Dict[str, Any]]
    opponent_locations: List[Dict[str, Any]]
    friendly_secrets: List[str]
    opponent_secrets_count: int
    opponent_hand_count: int
    hero_power: Dict[str, Any] = field(default_factory=dict)
    game_ended: bool = False
    actions: List[PlayerAction] = field(default_factory=list)


@dataclass
class DecisionPoint:
    """Immutable state immediately before one player-controlled action."""

    sequence: int
    snapshot: TurnSnapshot
    action: PlayerAction


@dataclass
class ReplayOptionCandidate:
    """One legal option/target combination reported by DebugPrintOptions."""

    candidate_id: int
    option_id: int
    option_type: str
    action_type: str
    entity_id: Optional[int]
    entity_name: str
    entity_card_id: str
    entity_card_type: int = 0
    controller_id: int = 0
    target_entity_id: Optional[int] = None
    target_name: Optional[str] = None
    target_card_id: Optional[str] = None
    sub_option_id: int = -1
    sub_entity_id: Optional[int] = None
    sub_entity_name: Optional[str] = None
    sub_entity_card_id: Optional[str] = None
    position: int = 0
    mana_cost: int = 0
    is_tradeable: bool = False
    description: str = ""


@dataclass
class OptionDecision:
    """State and complete legal options immediately before SendOption."""

    sequence: int
    options_id: int
    snapshot: TurnSnapshot
    candidates: List[ReplayOptionCandidate]
    selected_option: int
    selected_sub_option: int
    selected_target: Optional[int]
    selected_position: int


class GameStateTracker:
    """
    State machine that processes PowerEvents, updates game entities,
    and captures turn snapshots with player actions.
    """

    def __init__(self, card_db: Optional[CardDatabase] = None, friendly_player_name: Optional[str] = None):
        self.card_db = card_db or CardDatabase(auto_load=True)
        self.friendly_player_name = friendly_player_name

        self.entities: Dict[int, Entity] = {}
        self.players: Dict[int, PlayerState] = {}
        self.player_id_by_name: Dict[str, int] = {"GameEntity": 1}

        self.game_entity_id: int = 1
        self.current_turn: int = 0
        self.active_player_id: int = 1
        self.friendly_player_id: int = 1
        self.game_over: bool = False

        self.turn_snapshots: List[TurnSnapshot] = []
        self.decision_points: List[DecisionPoint] = []
        self.option_decisions: List[OptionDecision] = []
        self._current_snapshot: Optional[TurnSnapshot] = None
        self._actions_this_turn: List[PlayerAction] = []
        self._pending_turn_transition: bool = False
        # entity_id -> [(kind, PlayerAction)]: actions awaiting SHOW_ENTITY name backfill
        self._unresolved_actions: Dict[int, List[tuple]] = {}
        self._last_end_turn_key: Optional[Tuple[int, int]] = None
        self._current_options_id: int = 0
        self._current_options: Dict[int, Dict[str, Any]] = {}
        self._current_option_id: Optional[int] = None

    def get_or_create_entity(self, entity_id: int) -> Entity:
        if entity_id not in self.entities:
            self.entities[entity_id] = Entity(entity_id=entity_id)
        return self.entities[entity_id]

    def _set_entity_card(self, entity: Entity, card_id: str) -> None:
        """Applies a reveal/transform without leaking type or cost from the old card."""
        if not card_id:
            return
        if entity.card_id != card_id:
            entity.card_id = card_id
            entity.name = ""
            entity.tags.pop("CARDTYPE", None)
            entity.tags.pop("COST", None)
        card_info = self.card_db.get_by_id(card_id)
        if card_info:
            entity.name = card_info.name_ru or card_info.name_en
            entity.tags["CARDTYPE"] = int(card_info.card_type)

    def _resolve_entity_id(self, ent_ref: Dict[str, Any]) -> Optional[int]:
        if not ent_ref:
            return None
        if "id" in ent_ref and ent_ref["id"] is not None:
            return int(ent_ref["id"])
        name = ent_ref.get("name") or ent_ref.get("raw")
        if name:
            if name == "GameEntity":
                return self.game_entity_id
            if name in self.player_id_by_name:
                pid = self.player_id_by_name[name]
                if pid in self.players:
                    return self.players[pid].entity_id
        return None

    def _register_player_name_alias(self, name: str) -> None:
        """Associates a replay player name with the only still-unmapped player."""
        if not name or "#" not in name or name in self.player_id_by_name:
            return

        if self.friendly_player_name and (
            name == self.friendly_player_name
            or name in self.friendly_player_name
            or self.friendly_player_name in name
        ):
            pid = self.friendly_player_id
        else:
            mapped_ids = set(self.player_id_by_name.values()) & set(self.players)
            candidates = [
                pid
                for pid, player in self.players.items()
                if pid not in mapped_ids
                or (player.name and player.name.upper().startswith("UNKNOWN"))
            ]
            if len(candidates) != 1:
                return
            pid = candidates[0]

            for alias, alias_pid in list(self.player_id_by_name.items()):
                if alias_pid == pid and alias.upper().startswith("UNKNOWN"):
                    del self.player_id_by_name[alias]

        self.player_id_by_name[name] = pid
        if pid in self.players:
            self.players[pid].name = name

    def process_event(self, event: Any) -> None:
        """Processes a single PowerEvent."""
        etype = event.event_type
        data = event.data

        if etype == "CREATE_GAME":
            self.entities.clear()
            self.players.clear()
            self.turn_snapshots.clear()
            self.decision_points.clear()
            self.option_decisions.clear()
            self.player_id_by_name = {"GameEntity": 1}
            self.current_turn = 0
            self.game_over = False
            self._pending_turn_transition = False
            self._current_snapshot = None
            self._actions_this_turn.clear()
            self._unresolved_actions.clear()
            self._last_end_turn_key = None
            self._current_options_id = 0
            self._current_options.clear()
            self._current_option_id = None

        elif etype == "GAME_ENTITY":
            self.game_entity_id = data.get("entity_id", 1)
            self.get_or_create_entity(self.game_entity_id)

        elif etype == "PLAYER_ENTITY":
            eid = data["entity_id"]
            pid = data["player_id"]
            player = PlayerState(player_id=pid, entity_id=eid)
            self.players[pid] = player
            ent = self.get_or_create_entity(eid)
            ent.tags["CARDTYPE"] = 2  # PLAYER
            ent.tags["PLAYER_ID"] = pid

        elif etype == "PLAYER_NAME":
            pid = data["player_id"]
            pname = data["player_name"]
            self.player_id_by_name[pname] = pid
            if pid in self.players:
                self.players[pid].name = pname
            if self.friendly_player_name and (self.friendly_player_name in pname or pname in self.friendly_player_name):
                self.friendly_player_id = pid

        elif etype == "FULL_ENTITY":
            eid = data["entity_id"]
            cid = data.get("card_id", "")
            ent = self.get_or_create_entity(eid)
            self._set_entity_card(ent, cid)

        elif etype == "SHOW_ENTITY":
            ent_ref = data.get("entity", {})
            eid = self._resolve_entity_id(ent_ref)
            cid = data.get("card_id", "")
            if eid:
                ent = self.get_or_create_entity(eid)
                self._set_entity_card(ent, cid)
                self._refresh_unresolved_actions(eid, ent)

        elif etype == "CHANGE_ENTITY":
            ent_ref = data.get("entity", {})
            eid = self._resolve_entity_id(ent_ref)
            if eid:
                ent = self.get_or_create_entity(eid)
                self._set_entity_card(ent, data.get("card_id", ""))
                self._refresh_unresolved_actions(eid, ent)

        elif etype == "TAG":
            eid = data["entity_id"]
            tag = data["tag"]
            val = data["value"]
            ent = self.get_or_create_entity(eid)
            ent.tags[tag] = int(val) if val.isdigit() or (val.startswith("-") and val[1:].isdigit()) else val
            self._handle_tag_update(ent, tag, val)

        elif etype == "TAG_CHANGE":
            ent_ref = data.get("entity", {})
            tag = data.get("tag", "")
            val = data.get("value", "")

            # Auto-register player names
            ent_name = ent_ref.get("name")
            if ent_name and "#" in ent_name:
                self._register_player_name_alias(ent_name)
                if tag == "PLAYER_ID" and val.isdigit():
                    pid = int(val)
                    self.player_id_by_name[ent_name] = pid
                    if pid in self.players:
                        self.players[pid].name = ent_name

            eid = self._resolve_entity_id(ent_ref)

            if eid:
                ent = self.get_or_create_entity(eid)
                if ent_ref.get("cardId") and ent_ref["cardId"] != ent.card_id:
                    self._set_entity_card(ent, ent_ref["cardId"])
                if "entityName" in ent_ref and ent_ref["entityName"] and not ent.name:
                    ent.name = ent_ref["entityName"]
                if not ent.name and ent.card_id:
                    card_info = self.card_db.get_by_id(ent.card_id)
                    if card_info:
                        ent.name = card_info.name_ru or card_info.name_en
                        if "CARDTYPE" not in ent.tags:
                            ent.tags["CARDTYPE"] = int(card_info.card_type)

                ent.tags[tag] = int(val) if val.isdigit() or (val.startswith("-") and val[1:].isdigit()) else val
                self._handle_tag_update(ent, tag, val)

        elif etype == "OPTIONS_START":
            self._current_options_id = data.get("options_id", 0)
            self._current_options = {}
            self._current_option_id = None

        elif etype == "OPTION":
            option_id = data["option_id"]
            self._current_options[option_id] = {
                **data,
                "targets": [],
                "sub_options": [],
            }
            self._current_option_id = option_id

        elif etype == "OPTION_TARGET" and self._current_option_id in self._current_options:
            self._current_options[self._current_option_id]["targets"].append(data)

        elif etype == "OPTION_SUB_OPTION" and self._current_option_id in self._current_options:
            self._current_options[self._current_option_id]["sub_options"].append(data)

        elif etype == "SEND_OPTION":
            self._record_option_decision(data)

        elif etype == "BLOCK_START":
            self._handle_block_start(data)

    def _handle_tag_update(self, ent: Entity, tag: str, val: str) -> None:
        if tag == "PLAYER_ID":
            pid = int(val) if val.isdigit() else 0
            if pid in self.players:
                self.players[pid].entity_id = ent.entity_id

        elif tag == "HERO_ENTITY":
            hp_val = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else None
            for p in self.players.values():
                if p.entity_id == ent.entity_id or ent.tags.get("PLAYER_ID") == p.player_id:
                    p.hero_entity_id = hp_val

        elif tag == "HERO_POWER_ENTITY":
            hp_val = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else None
            for p in self.players.values():
                if p.entity_id == ent.entity_id or ent.tags.get("PLAYER_ID") == p.player_id:
                    p.hero_power_entity_id = hp_val

        elif tag == "PLAYSTATE":
            ps_val = str(val)
            if ps_val in ("WON", "LOST", "CONCEDED", "TIED"):
                for p in self.players.values():
                    if p.entity_id == ent.entity_id:
                        p.playstate = ps_val
                self.game_over = True

        elif tag == "RESOURCES":
            for p in self.players.values():
                if p.entity_id == ent.entity_id:
                    p.resources = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else 0

        elif tag == "RESOURCES_USED":
            for p in self.players.values():
                if p.entity_id == ent.entity_id:
                    p.resources_used = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else 0

        elif tag == "TEMP_RESOURCES":
            for p in self.players.values():
                if p.entity_id == ent.entity_id:
                    p.temp_resources = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else 0

        elif tag == "CURRENT_PLAYER" and str(val) == "1":
            # Resolve the player via their PLAYER_ID tag; fall back to entity_id only
            # when it matches a registered player (entity ids 2/3 collide with pids 2/3 otherwise).
            pid = ent.tags.get("PLAYER_ID")
            if pid is None and ent.entity_id in self.players:
                pid = ent.entity_id
            if pid in self.players and self.active_player_id != pid:
                self.active_player_id = pid
                self._on_turn_or_player_transition()

        elif tag == "TURN" and ent.entity_id == self.game_entity_id:
            new_turn = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else 0
            if new_turn != self.current_turn:
                self.current_turn = new_turn
                self._pending_turn_transition = True

        elif tag == "STEP" and ent.entity_id == self.game_entity_id:
            step_name = str(val)
            if step_name == "MAIN_END":
                self._record_end_turn_decision()
            elif step_name == "MAIN_ACTION":
                # A new turn officially starts at MAIN_ACTION: turn draw / mana
                # refresh have resolved, so the snapshot reflects reality.
                if self._pending_turn_transition:
                    self._pending_turn_transition = False
                    self._on_turn_or_player_transition()
                elif self._current_snapshot and not self._actions_this_turn:
                    # Same turn re-sync (e.g. after a stolen extra turn): refresh mutable fields.
                    refreshed = self.take_turn_snapshot()
                    self._current_snapshot.friendly_mana = refreshed.friendly_mana
                    self._current_snapshot.friendly_max_mana = refreshed.friendly_max_mana
                    self._current_snapshot.friendly_hand = refreshed.friendly_hand
                    self._current_snapshot.friendly_board = refreshed.friendly_board
                    self._current_snapshot.opponent_board = refreshed.opponent_board
                    self._current_snapshot.friendly_locations = refreshed.friendly_locations
                    self._current_snapshot.opponent_locations = refreshed.opponent_locations
                    self._current_snapshot.opponent_hand_count = refreshed.opponent_hand_count
                    self._current_snapshot.opponent_secrets_count = refreshed.opponent_secrets_count
                    self._current_snapshot.friendly_hero = refreshed.friendly_hero
                    self._current_snapshot.opponent_hero = refreshed.opponent_hero

    def _on_turn_or_player_transition(self) -> None:
        """Captures turn snapshot when turn or active player starts."""
        if self.current_turn < 1:
            return

        # Dedupe: don't re-snapshot if a snapshot for this (turn, active player) already exists.
        if self._current_snapshot is not None:
            if (
                self._current_snapshot.turn_number == self._player_turn_number()
                and self._current_snapshot.active_player_id == self.active_player_id
            ):
                return

        if self._current_snapshot and self._actions_this_turn:
            self._current_snapshot.actions = list(self._actions_this_turn)

        # Reset resources used for new turn
        for p in self.players.values():
            p.resources_used = 0

        self._actions_this_turn.clear()
        self._current_snapshot = self.take_turn_snapshot()
        self.turn_snapshots.append(self._current_snapshot)

    def _handle_block_start(self, data: Dict[str, Any]) -> None:
        btype = data.get("block_type")
        ent_ref = data.get("entity", {})
        target_ref = data.get("target", {})

        eid = self._resolve_entity_id(ent_ref)
        target_eid = self._resolve_entity_id(target_ref)

        ent = self.entities.get(eid) if eid else None
        target_ent = self.entities.get(target_eid) if target_eid else None

        # Never attribute an action to the active player without controller proof.
        if btype in ("PLAY", "ATTACK") and (
            ent is None or ent.controller == 0 or ent.controller != self.active_player_id
        ):
            return

        ent_name = ent.name if ent else ent_ref.get("entityName", "")
        ent_card_id = ent.card_id if ent else ent_ref.get("cardId", "")
        target_name = target_ent.name if target_ent else target_ref.get("entityName")
        target_card_id = target_ent.card_id if target_ent else target_ref.get("cardId")

        if not ent_name and ent_card_id:
            card_info = self.card_db.get_by_id(ent_card_id)
            if card_info:
                ent_name = card_info.name_ru or card_info.name_en
        if not target_name and target_card_id:
            card_info = self.card_db.get_by_id(target_card_id)
            if card_info:
                target_name = card_info.name_ru or card_info.name_en

        if btype == "PLAY":
            action_type = "PLAY"
            c_type = ent.card_type if ent else 0
            if c_type == 10:  # HERO_POWER
                action_type = "HERO_POWER"
            elif c_type == 39:  # LOCATION
                action_type = "LOCATION"

            action = PlayerAction(
                turn=self.current_turn,
                action_type=action_type,
                entity_name=ent_name or "Unknown Card",
                entity_card_id=ent_card_id,
                target_name=target_name,
                target_card_id=target_card_id,
                details={
                    "sub_option": data.get("sub_option", -1),
                    "_entity_id": eid,
                    "_target_entity_id": target_eid,
                    "controller_id": ent.controller if ent else 0,
                },
            )
            self._actions_this_turn.append(action)
            self._index_action_refs(action)
            self._record_decision(action)

        elif btype == "ATTACK":
            action = PlayerAction(
                turn=self.current_turn,
                action_type="ATTACK",
                entity_name=ent_name or "Attacker",
                entity_card_id=ent_card_id,
                target_name=target_name or "Target",
                target_card_id=target_card_id,
                details={
                    "_entity_id": eid,
                    "_target_entity_id": target_eid,
                    "controller_id": ent.controller if ent else 0,
                },
            )
            self._actions_this_turn.append(action)
            self._index_action_refs(action)
            self._record_decision(action)

    def _record_decision(self, action: PlayerAction) -> None:
        """Stores a fresh snapshot before replay events mutate action state."""
        if self.current_turn < 1 or self._current_snapshot is None:
            return
        self.decision_points.append(
            DecisionPoint(
                sequence=len(self.decision_points) + 1,
                snapshot=self.take_turn_snapshot(),
                action=action,
            )
        )

    def _record_end_turn_decision(self) -> None:
        """Records the explicit player pass at STEP=MAIN_END once per turn."""
        key = (self.current_turn, self.active_player_id)
        if self.current_turn < 1 or self._current_snapshot is None or key == self._last_end_turn_key:
            return
        action = PlayerAction(
            turn=self.current_turn,
            action_type="END_TURN",
            entity_name="END_TURN",
            entity_card_id="",
            details={"_entity_id": None, "_target_entity_id": None, "controller_id": self.active_player_id},
        )
        self._record_decision(action)
        self._last_end_turn_key = key

    def _option_entity(self, ref: Dict[str, Any]) -> Tuple[Optional[int], str, str, Optional[Entity]]:
        entity_id = self._resolve_entity_id(ref)
        entity = self.entities.get(entity_id) if entity_id else None
        ref_name = str(ref.get("entityName") or "").strip()
        tracker_name = str(entity.name or "").strip() if entity else ""
        ref_card_id = str(ref.get("cardId") or "").strip()
        card_id = ref_card_id or (str(entity.card_id or "").strip() if entity else "")

        card = self.card_db.get_by_id(card_id) if card_id else None
        name = str((card.name_ru or card.name_en) if card else "").strip()
        if not name and ref_name and not ref_name.casefold().startswith("unknown entity"):
            name = ref_name
        if not name and tracker_name and not tracker_name.casefold().startswith("unknown entity"):
            name = tracker_name
        if not name:
            name = ref_name or tracker_name
        return entity_id, name, card_id, entity

    def _option_action_type(
        self,
        option_type: str,
        main_ref: Dict[str, Any],
        entity: Optional[Entity],
        card_type: int,
        target_entity_id: Optional[int],
        sub_option_id: int,
    ) -> str:
        if option_type == "END_TURN":
            return "END_TURN"
        zone = (entity.zone if entity else "") or main_ref.get("zone", "")
        if zone == "HAND":
            return "PLAY"
        if card_type == int(CardType.HERO_POWER):
            return "HERO_POWER"
        if card_type == int(CardType.LOCATION):
            return "LOCATION"
        if card_type in (int(CardType.HERO), int(CardType.MINION)) and target_entity_id and sub_option_id < 0:
            return "ATTACK"
        return option_type

    def _build_option_candidates(self) -> List[ReplayOptionCandidate]:
        candidates: List[ReplayOptionCandidate] = []
        board_slots = 0
        for board_entity in self.entities.values():
            board_card_type = board_entity.card_type
            if not board_card_type and board_entity.card_id:
                board_card = self.card_db.get_by_id(board_entity.card_id)
                board_card_type = int(board_card.card_type) if board_card else 0
            if (
                board_entity.controller == self.active_player_id
                and board_entity.zone == "PLAY"
                and board_card_type in (int(CardType.MINION), int(CardType.LOCATION))
            ):
                board_slots += 1

        for option in self._current_options.values():
            option_type = option.get("option_type", "")
            if option_type != "END_TURN" and option.get("error") != "NONE":
                continue

            main_ref = option.get("main_entity", {})
            entity_id, entity_name, entity_card_id, entity = self._option_entity(main_ref)
            entity_card_type = 0
            card = None
            if entity_card_id:
                card = self.card_db.get_by_id(entity_card_id)
                entity_card_type = int(card.card_type) if card else 0
            if not entity_card_type and entity:
                entity_card_type = entity.card_type
            legal_targets = [target for target in option.get("targets", []) if target.get("error") == "NONE"]
            if option.get("targets") and not legal_targets:
                continue
            target_variants = legal_targets or [None]

            legal_sub_options = [sub for sub in option.get("sub_options", []) if sub.get("error") == "NONE"]
            if option.get("sub_options") and not legal_sub_options:
                continue
            sub_variants = legal_sub_options or [None]

            for sub_option in sub_variants:
                sub_option_id = sub_option.get("sub_option_id", -1) if sub_option else -1
                sub_entity_id: Optional[int] = None
                sub_entity_name: Optional[str] = None
                sub_entity_card_id: Optional[str] = None
                if sub_option:
                    sub_entity_id, sub_entity_name, sub_entity_card_id, _ = self._option_entity(
                        sub_option.get("entity", {})
                    )

                for target in target_variants:
                    target_entity_id: Optional[int] = None
                    target_name: Optional[str] = None
                    target_card_id: Optional[str] = None
                    if target:
                        target_entity_id, target_name, target_card_id, _ = self._option_entity(
                            target.get("entity", {})
                        )

                    action_type = self._option_action_type(
                        option_type,
                        main_ref,
                        entity,
                        entity_card_type,
                        target_entity_id,
                        sub_option_id,
                    )
                    if action_type == "END_TURN":
                        description = "Завершить ход"
                    else:
                        description = entity_name or option_type
                        if sub_entity_name:
                            description += f" [{sub_entity_name}]"
                        if target_name:
                            description += f" -> {target_name}"

                    mana_cost = 0
                    if entity and action_type in ("PLAY", "HERO_POWER"):
                        if entity.card_id == entity_card_id and entity.cost >= 0:
                            mana_cost = entity.cost
                        elif entity_card_id:
                            card = self.card_db.get_by_id(entity_card_id)
                            mana_cost = card.cost if card else 0

                    position_variants = [0]
                    if action_type == "PLAY" and entity_card_type in (
                        int(CardType.MINION),
                        int(CardType.LOCATION),
                    ):
                        position_variants = list(range(1, min(7, board_slots + 1) + 1))

                    for position in position_variants:
                        candidates.append(
                            ReplayOptionCandidate(
                                candidate_id=len(candidates) + 1,
                                option_id=option["option_id"],
                                option_type=option_type,
                                action_type=action_type,
                                entity_id=entity_id,
                                entity_name=entity_name,
                                entity_card_id=entity_card_id,
                                entity_card_type=entity_card_type,
                                controller_id=(entity.controller if entity else main_ref.get("player", 0)),
                                target_entity_id=target_entity_id,
                                target_name=target_name,
                                target_card_id=target_card_id,
                                sub_option_id=sub_option_id,
                                sub_entity_id=sub_entity_id,
                                sub_entity_name=sub_entity_name,
                                sub_entity_card_id=sub_entity_card_id,
                                position=position,
                                mana_cost=mana_cost,
                                is_tradeable=bool(card and "Tradeable" in card.mechanics),
                                description=description,
                            )
                        )
        return candidates

    def _record_option_decision(self, data: Dict[str, Any]) -> None:
        if self.current_turn < 1 or not self._current_options:
            return
        # A few HDT exports contain a complete option oracle but omit the
        # turn-transition marker that normally creates _current_snapshot.
        # The option set itself is a pre-action boundary, so capture a lazy
        # snapshot instead of dropping an otherwise recoverable decision.
        if self._current_snapshot is None:
            self._current_snapshot = self.take_turn_snapshot()
        selected_target = data.get("selected_target", 0) or None
        self.option_decisions.append(
            OptionDecision(
                sequence=len(self.option_decisions) + 1,
                options_id=self._current_options_id,
                snapshot=self.take_turn_snapshot(),
                candidates=self._build_option_candidates(),
                selected_option=data.get("selected_option", -1),
                selected_sub_option=data.get("selected_sub_option", -1),
                selected_target=selected_target,
                selected_position=data.get("selected_position", 0),
            )
        )
        self._current_options = {}
        self._current_option_id = None

    def finalize(self) -> None:
        """Finalizes the last turn's actions."""
        if self._current_snapshot and self._actions_this_turn:
            self._current_snapshot.actions = list(self._actions_this_turn)
            self._actions_this_turn.clear()
        # game_ended on the final snapshot: PLAYSTATE WON/LOST arrives mid-turn,
        # after the snapshot was taken — propagate the final state now.
        if self._current_snapshot and self.game_over:
            self._current_snapshot.game_ended = True

    def _index_action_refs(self, act: PlayerAction) -> None:
        """Registers an action's unresolved entity refs in the O(1) backfill index."""
        ent_id = act.details.get("_entity_id")
        if ent_id and act.entity_card_id == "":
            self._unresolved_actions.setdefault(ent_id, []).append(("entity", act))
        tgt_id = act.details.get("_target_entity_id")
        if tgt_id and act.target_card_id == "" and act.target_name:
            self._unresolved_actions.setdefault(tgt_id, []).append(("target", act))

    def _refresh_unresolved_actions(self, entity_id: int, ent: Entity) -> None:
        """Backfills names of actions recorded before SHOW_ENTITY revealed the card
        (opponent plays are logged as 'UNKNOWN ENTITY' until mid-block). O(1) per reveal."""
        if not ent.name and not ent.card_id:
            return
        pending = self._unresolved_actions.pop(entity_id, None)
        if not pending:
            return
        for kind, act in pending:
            if kind == "entity":
                if act.entity_card_id == "":
                    act.entity_name = ent.name or act.entity_name
                    act.entity_card_id = ent.card_id or act.entity_card_id
            else:  # target
                if act.target_card_id == "" and act.target_name:
                    act.target_name = ent.name or act.target_name
                    act.target_card_id = ent.card_id or act.target_card_id

    def _player_turn_number(self) -> int:
        """Player-perceived turn number (each player counts their own turns)."""
        if self.active_player_id == 1:
            n = (self.current_turn + 1) // 2
        else:
            n = self.current_turn // 2
        return max(1, n)

    def take_turn_snapshot(self) -> TurnSnapshot:
        """Constructs an immutable TurnSnapshot of the current board and hand."""
        f_pid = self.friendly_player_id
        o_pid = 2 if f_pid == 1 else 1

        f_player = self.players.get(f_pid, PlayerState(player_id=f_pid, entity_id=0))
        o_player = self.players.get(o_pid, PlayerState(player_id=o_pid, entity_id=0))

        # Hero states
        f_hero_ent = self.entities.get(f_player.hero_entity_id or 0)
        o_hero_ent = self.entities.get(o_player.hero_entity_id or 0)

        f_hero_info = {
            "entity_id": f_hero_ent.entity_id if f_hero_ent else 0,
            "name": f_hero_ent.name if f_hero_ent else "Friendly Hero",
            "card_id": f_hero_ent.card_id if f_hero_ent else "",
            "health": f_hero_ent.health if f_hero_ent else 30,
            "armor": f_hero_ent.armor if f_hero_ent else 0,
            "attack": f_hero_ent.attack if f_hero_ent else 0,
            "can_attack": f_hero_ent.can_attack if f_hero_ent else False,
            "is_immune": bool(f_hero_ent.get_tag("IMMUNE", 0)) if f_hero_ent else False,
            "cant_be_attacked": bool(f_hero_ent.get_tag("CANT_BE_ATTACKED", 0)) if f_hero_ent else False,
        }
        o_hero_info = {
            "entity_id": o_hero_ent.entity_id if o_hero_ent else 0,
            "name": o_hero_ent.name if o_hero_ent else "Opponent Hero",
            "card_id": o_hero_ent.card_id if o_hero_ent else "",
            "health": o_hero_ent.health if o_hero_ent else 30,
            "armor": o_hero_ent.armor if o_hero_ent else 0,
            "attack": o_hero_ent.attack if o_hero_ent else 0,
            "can_attack": o_hero_ent.can_attack if o_hero_ent else False,
            "is_immune": bool(o_hero_ent.get_tag("IMMUNE", 0)) if o_hero_ent else False,
            "cant_be_attacked": bool(o_hero_ent.get_tag("CANT_BE_ATTACKED", 0)) if o_hero_ent else False,
        }

        # Friendly Hand
        friendly_hand = []
        # Friendly Board (Minions)
        friendly_board = []
        # Opponent Board (Minions)
        opponent_board = []
        # Locations
        friendly_locations = []
        opponent_locations = []
        # Secrets
        friendly_secrets = []
        opponent_secrets_count = 0
        opponent_hand_count = 0

        for ent in self.entities.values():
            c_type = ent.card_type
            c_info = self.card_db.get_by_id(ent.card_id) if ent.card_id else None
            if not c_type and c_info:
                c_type = int(c_info.card_type)

            if ent.zone == "HAND":
                if ent.controller == f_pid:
                    name = ent.name or (c_info.name_ru if c_info else "Неизвестная карта")
                    live_cost = ent.cost
                    db_cost = c_info.cost if c_info else 0
                    cost = db_cost if live_cost < 0 else live_cost
                    friendly_hand.append(
                        {
                            "entity_id": ent.entity_id,
                            "card_id": ent.card_id,
                            "name": name,
                            "cost": cost,
                            "attack": ent.attack if c_type == 4 else None,
                            "health": ent.health if c_type == 4 else None,
                            "card_type": c_type,
                            "text": c_info.text_ru if c_info else "",
                        }
                    )
                else:
                    opponent_hand_count += 1

            elif ent.zone == "PLAY":
                if c_type == 4:  # MINION
                    name = ent.name or (c_info.name_ru if c_info else "Существо")
                    minion_data = {
                        "entity_id": ent.entity_id,
                        "card_id": ent.card_id,
                        "name": name,
                        "attack": ent.attack,
                        "health": ent.health,
                        "max_health": ent.max_health,
                        "can_attack": ent.can_attack,
                        "can_attack_hero": ent.can_attack_hero,
                        "attacks_used": ent.num_attacks_this_turn,
                        "is_taunt": ent.is_taunt,
                        "is_divine_shield": ent.is_divine_shield,
                        "is_stealthed": ent.is_stealthed,
                        "is_frozen": ent.is_frozen,
                        "is_reborn": ent.is_reborn,
                        "is_silenced": ent.is_silenced,
                        "is_dormant": ent.is_dormant,
                        "is_titan": ent.is_titan,
                        "is_starship": ent.is_starship,
                        "is_immune": bool(ent.get_tag("IMMUNE", 0)),
                        "cant_be_attacked": bool(ent.get_tag("CANT_BE_ATTACKED", 0)),
                    }
                    if ent.controller == f_pid:
                        friendly_board.append(minion_data)
                    else:
                        opponent_board.append(minion_data)

                elif c_type == 39:  # LOCATION
                    name = ent.name or (c_info.name_ru if c_info else "Область")
                    loc_data = {
                        "entity_id": ent.entity_id,
                        "card_id": ent.card_id,
                        "name": name,
                        "durability": ent.durability,
                        "can_use": not ent.is_exhausted,
                    }
                    if ent.controller == f_pid:
                        friendly_locations.append(loc_data)
                    else:
                        opponent_locations.append(loc_data)

            elif ent.zone == "SECRET":
                if ent.controller == f_pid:
                    friendly_secrets.append(ent.name or (c_info.name_ru if c_info else "Секрет"))
                else:
                    opponent_secrets_count += 1

        active_player = self.players.get(self.active_player_id)
        active_name = active_player.name if active_player else f"Player {self.active_player_id}"

        # Player turn number in standard game convention (Turn 1, Turn 2, etc.)
        player_turn_num = self._player_turn_number()

        f_max = f_player.resources + f_player.temp_resources
        if f_max <= 0:
            f_max = min(10, player_turn_num)
        f_avail = max(0, f_max - f_player.resources_used)

        # Hero power availability: entity exists, not exhausted, and 2 mana free is checked by caller
        f_hp_ent = self.entities.get(f_player.hero_power_entity_id or 0)
        hero_power_info = {
            "entity_id": f_player.hero_power_entity_id or 0,
            "card_id": f_hp_ent.card_id if f_hp_ent else "",
            "name": f_hp_ent.name if f_hp_ent else "",
            "cost": max(0, f_hp_ent.cost) if f_hp_ent and f_hp_ent.cost >= 0 else 2,
            "exhausted": bool(f_hp_ent.is_exhausted) if f_hp_ent else True,
        }

        return TurnSnapshot(
            turn_number=player_turn_num,
            active_player_id=self.active_player_id,
            active_player_name=active_name,
            is_friendly_turn=(self.active_player_id == f_pid),
            friendly_mana=f_avail,
            friendly_max_mana=f_max,
            friendly_hero=f_hero_info,
            opponent_hero=o_hero_info,
            hero_power=hero_power_info,
            friendly_hand=friendly_hand,
            friendly_board=friendly_board,
            opponent_board=opponent_board,
            friendly_locations=friendly_locations,
            opponent_locations=opponent_locations,
            friendly_secrets=friendly_secrets,
            opponent_secrets_count=opponent_secrets_count,
            opponent_hand_count=opponent_hand_count,
            game_ended=self.game_over,
        )
