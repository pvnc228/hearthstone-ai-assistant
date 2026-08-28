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
        return self.get_tag("COST", 0)

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
        if self.zone != "PLAY" or self.is_frozen or self.is_dormant:
            return False
        if self.attack <= 0:
            return False
        return not self.is_exhausted or self.get_tag("CHARGE", 0) == 1 or self.get_tag("RUSH", 0) == 1


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
    actions: List[PlayerAction] = field(default_factory=list)


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
        self._current_snapshot: Optional[TurnSnapshot] = None
        self._actions_this_turn: List[PlayerAction] = []

    def get_or_create_entity(self, entity_id: int) -> Entity:
        if entity_id not in self.entities:
            self.entities[entity_id] = Entity(entity_id=entity_id)
        return self.entities[entity_id]

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

    def process_event(self, event: Any) -> None:
        """Processes a single PowerEvent."""
        etype = event.event_type
        data = event.data

        if etype == "CREATE_GAME":
            self.entities.clear()
            self.players.clear()
            self.turn_snapshots.clear()
            self.player_id_by_name = {"GameEntity": 1}
            self.current_turn = 0
            self.game_over = False

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
            if cid:
                ent.card_id = cid
                card_info = self.card_db.get_by_id(cid)
                if card_info:
                    ent.name = card_info.name_ru or card_info.name_en
                    if "CARDTYPE" not in ent.tags:
                        ent.tags["CARDTYPE"] = int(card_info.card_type)

        elif etype == "SHOW_ENTITY":
            ent_ref = data.get("entity", {})
            eid = self._resolve_entity_id(ent_ref)
            cid = data.get("card_id", "")
            if eid:
                ent = self.get_or_create_entity(eid)
                if cid:
                    ent.card_id = cid
                    card_info = self.card_db.get_by_id(cid)
                    if card_info:
                        ent.name = card_info.name_ru or card_info.name_en
                        if "CARDTYPE" not in ent.tags:
                            ent.tags["CARDTYPE"] = int(card_info.card_type)

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
                if tag == "PLAYER_ID" and val.isdigit():
                    pid = int(val)
                    self.player_id_by_name[ent_name] = pid
                    if pid in self.players:
                        self.players[pid].name = ent_name

            eid = self._resolve_entity_id(ent_ref)

            if eid:
                ent = self.get_or_create_entity(eid)
                if "cardId" in ent_ref and ent_ref["cardId"] and not ent.card_id:
                    ent.card_id = ent_ref["cardId"]
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

        elif etype == "BLOCK_START":
            self._handle_block_start(data)

    def _handle_tag_update(self, ent: Entity, tag: str, val: str) -> None:
        if tag == "PLAYER_ID":
            pid = int(val) if val.isdigit() else 0
            if pid in self.players:
                self.players[pid].entity_id = ent.entity_id

        elif tag == "HERO_ENTITY":
            for p in self.players.values():
                if p.entity_id == ent.entity_id or ent.tags.get("PLAYER_ID") == p.player_id:
                    p.hero_entity_id = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else None

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
            pid = ent.tags.get("PLAYER_ID") or ent.entity_id
            for p in self.players.values():
                if p.entity_id == ent.entity_id or p.player_id == pid:
                    if self.active_player_id != p.player_id:
                        self.active_player_id = p.player_id
                        self._on_turn_or_player_transition()

        elif tag == "TURN" and ent.entity_id == self.game_entity_id:
            new_turn = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isdigit()) else 0
            if new_turn != self.current_turn:
                self.current_turn = new_turn
                self._on_turn_or_player_transition()

        elif tag == "STEP" and str(val) == "MAIN_ACTION" and ent.entity_id == self.game_entity_id:
            # Refresh snapshot state now that turn draw and mana refresh have resolved
            if self._current_snapshot and not self._actions_this_turn:
                refreshed = self.take_turn_snapshot()
                self._current_snapshot.friendly_mana = refreshed.friendly_mana
                self._current_snapshot.friendly_max_mana = refreshed.friendly_max_mana
                self._current_snapshot.friendly_hand = refreshed.friendly_hand
                self._current_snapshot.friendly_board = refreshed.friendly_board
                self._current_snapshot.opponent_board = refreshed.opponent_board
                self._current_snapshot.friendly_hero = refreshed.friendly_hero
                self._current_snapshot.opponent_hero = refreshed.opponent_hero

        elif tag == "PLAYSTATE" and val in ("WON", "LOST", "CONCEDED"):
            self.game_over = True

    def _on_turn_or_player_transition(self) -> None:
        """Captures turn snapshot when turn or active player starts."""
        if self.current_turn < 1:
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
                details={"sub_option": data.get("sub_option", -1)},
            )
            self._actions_this_turn.append(action)

        elif btype == "ATTACK":
            action = PlayerAction(
                turn=self.current_turn,
                action_type="ATTACK",
                entity_name=ent_name or "Attacker",
                entity_card_id=ent_card_id,
                target_name=target_name or "Target",
                target_card_id=target_card_id,
            )
            self._actions_this_turn.append(action)

    def finalize(self) -> None:
        """Finalizes the last turn's actions."""
        if self._current_snapshot and self._actions_this_turn:
            self._current_snapshot.actions = list(self._actions_this_turn)
            self._actions_this_turn.clear()

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
            "name": f_hero_ent.name if f_hero_ent else "Friendly Hero",
            "card_id": f_hero_ent.card_id if f_hero_ent else "",
            "health": f_hero_ent.health if f_hero_ent else 30,
            "armor": f_hero_ent.armor if f_hero_ent else 0,
        }
        o_hero_info = {
            "name": o_hero_ent.name if o_hero_ent else "Opponent Hero",
            "card_id": o_hero_ent.card_id if o_hero_ent else "",
            "health": o_hero_ent.health if o_hero_ent else 30,
            "armor": o_hero_ent.armor if o_hero_ent else 0,
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
                    friendly_hand.append(
                        {
                            "entity_id": ent.entity_id,
                            "card_id": ent.card_id,
                            "name": name,
                            "cost": ent.cost or (c_info.cost if c_info else 0),
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
                        "is_taunt": ent.is_taunt,
                        "is_divine_shield": ent.is_divine_shield,
                        "is_stealthed": ent.is_stealthed,
                        "is_frozen": ent.is_frozen,
                        "is_reborn": ent.is_reborn,
                        "is_silenced": ent.is_silenced,
                        "is_dormant": ent.is_dormant,
                        "is_titan": ent.is_titan,
                        "is_starship": ent.is_starship,
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
        player_turn_num = (self.current_turn + 1) // 2 if self.active_player_id == 1 else self.current_turn // 2
        if player_turn_num < 1:
            player_turn_num = 1

        f_max = f_player.resources + f_player.temp_resources
        if f_max <= 0:
            f_max = min(10, player_turn_num)
        f_avail = max(0, f_max - f_player.resources_used)

        return TurnSnapshot(
            turn_number=player_turn_num,
            active_player_id=self.active_player_id,
            active_player_name=active_name,
            is_friendly_turn=(self.active_player_id == f_pid),
            friendly_mana=f_avail,
            friendly_max_mana=f_max,
            friendly_hero=f_hero_info,
            opponent_hero=o_hero_info,
            friendly_hand=friendly_hand,
            friendly_board=friendly_board,
            opponent_board=opponent_board,
            friendly_locations=friendly_locations,
            opponent_locations=opponent_locations,
            friendly_secrets=friendly_secrets,
            opponent_secrets_count=opponent_secrets_count,
            opponent_hand_count=opponent_hand_count,
        )
