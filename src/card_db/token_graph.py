"""
Token Graph & Card Semantics Knowledge Graph.
Integrates structured token relationships, parent-child derivations, and normalized action mechanics
from the Hearthstone semantics dataset into in-memory fast lookups.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .models import Card

logger = logging.getLogger(__name__)

DEFAULT_SEMANTICS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "semantics"


@dataclass
class CardSemantics:
    card_id: int  # DBF ID
    name: str
    collectible: bool = True
    is_derived: bool = False
    parent_card_ids: List[int] = field(default_factory=list)
    child_card_ids: List[int] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    mechanic_tags: List[str] = field(default_factory=list)
    visual_tags: List[str] = field(default_factory=list)
    string_card_id: Optional[str] = None


class TokenGraph:
    """
    In-memory semantic graph connecting Hearthstone cards to their generated tokens and child cards.
    """

    def __init__(self, semantics_dir: Optional[Union[Path, str]] = None, auto_load: bool = True):
        self.semantics_dir = Path(semantics_dir) if semantics_dir else DEFAULT_SEMANTICS_DIR
        self._semantics_by_dbf: Dict[int, CardSemantics] = {}
        self._semantics_by_str_id: Dict[str, CardSemantics] = {}
        self._children_by_parent: Dict[int, Set[int]] = {}
        self._parents_by_child: Dict[int, Set[int]] = {}
        self._loaded: bool = False

        if auto_load:
            self.load()

    def load(self) -> None:
        """Loads semantic knowledge and token edges from JSONL cache."""
        cards_file = self.semantics_dir / "cards_semantics_base.jsonl"
        edges_file = self.semantics_dir / "derived_edges.jsonl"

        if not cards_file.exists() or not edges_file.exists():
            logger.warning(
                "Semantics files missing at %s. TokenGraph will be empty until files are downloaded.",
                self.semantics_dir,
            )
            return

        # Load card semantics
        count_cards = 0
        with open(cards_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    dbf_id = data.get("card_id")
                    if not dbf_id:
                        continue

                    str_id = data.get("source", {}).get("art_card_id") or None
                    sem = CardSemantics(
                        card_id=dbf_id,
                        name=data.get("name", ""),
                        collectible=data.get("collectible", True),
                        is_derived=data.get("is_derived", False),
                        parent_card_ids=data.get("parent_card_ids", []),
                        child_card_ids=data.get("child_card_ids", []),
                        keywords=data.get("keywords", []),
                        actions=data.get("actions", []),
                        mechanic_tags=data.get("mechanic_tags", []),
                        visual_tags=data.get("visual_tags", []),
                        string_card_id=str_id,
                    )
                    self._semantics_by_dbf[dbf_id] = sem
                    if str_id:
                        self._semantics_by_str_id[str_id] = sem
                    count_cards += 1
                except Exception as e:
                    logger.debug("Failed parsing card semantics row: %s", e)

        # Load edge connections
        count_edges = 0
        with open(edges_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    src = data.get("source")
                    tgt = data.get("target")
                    if src is not None and tgt is not None:
                        self._children_by_parent.setdefault(src, set()).add(tgt)
                        self._parents_by_child.setdefault(tgt, set()).add(src)
                        count_edges += 1
                except Exception as e:
                    logger.debug("Failed parsing edge row: %s", e)

        # Sync children from semantics records
        for dbf_id, sem in self._semantics_by_dbf.items():
            for child_id in sem.child_card_ids:
                self._children_by_parent.setdefault(dbf_id, set()).add(child_id)
                self._parents_by_child.setdefault(child_id, set()).add(dbf_id)

        self._loaded = True
        logger.info(
            "TokenGraph loaded %d card semantics and %d token derivation edges.",
            count_cards,
            count_edges,
        )

    def _resolve_dbf_id(self, card_or_id: Union[Card, str, int]) -> Optional[int]:
        """Resolves Card object, string CardID, or int DBF ID to numeric DBF ID."""
        if isinstance(card_or_id, Card):
            return card_or_id.dbf_id
        elif isinstance(card_or_id, int):
            return card_or_id
        elif isinstance(card_or_id, str):
            if card_or_id.isdigit():
                return int(card_or_id)
            sem = self._semantics_by_str_id.get(card_or_id)
            if sem:
                return sem.card_id
        return None

    def get_semantics(self, card_or_id: Union[Card, str, int]) -> Optional[CardSemantics]:
        """Retrieves structured semantics (actions, tags, tokens) for a given card."""
        dbf_id = self._resolve_dbf_id(card_or_id)
        if dbf_id:
            return self._semantics_by_dbf.get(dbf_id)
        if isinstance(card_or_id, str):
            return self._semantics_by_str_id.get(card_or_id)
        return None

    def get_child_dbf_ids(self, card_or_id: Union[Card, str, int]) -> List[int]:
        """Returns DBF IDs of child cards / tokens spawned or discovered by this card."""
        dbf_id = self._resolve_dbf_id(card_or_id)
        if not dbf_id:
            return []
        return sorted(self._children_by_parent.get(dbf_id, set()))

    def get_parent_dbf_ids(self, card_or_id: Union[Card, str, int]) -> List[int]:
        """Returns DBF IDs of cards that create or spawn this token."""
        dbf_id = self._resolve_dbf_id(card_or_id)
        if not dbf_id:
            return []
        return sorted(self._parents_by_child.get(dbf_id, set()))

    def get_child_cards(self, card_or_id: Union[Card, str, int], card_db) -> List[Card]:
        """Returns resolved Card model instances for all child tokens."""
        child_ids = self.get_child_dbf_ids(card_or_id)
        cards = []
        for cid in child_ids:
            c = card_db.get_by_dbf_id(cid)
            if c:
                cards.append(c)
        return cards

    def get_mechanic_tags(self, card_or_id: Union[Card, str, int]) -> List[str]:
        """Returns normalized semantic mechanic tags (e.g. ['deal_damage', 'lifesteal_damage'])."""
        sem = self.get_semantics(card_or_id)
        return sem.mechanic_tags if sem else []

    def format_token_summary(self, card_or_id: Union[Card, str, int], card_db, lang: str = "ru") -> str:
        """
        Formats a compact string describing child tokens generated by this card.
        Example: '[Порождает: Хаффер (4/2 Рывок), Леокк (2/4), Миша (3/4 Провокация)]'
        """
        children = self.get_child_cards(card_or_id, card_db)
        if not children:
            return ""

        parts = []
        seen = set()
        for ch in children[:5]:
            if ch.card_id in seen:
                continue
            seen.add(ch.card_id)
            cname = ch.name_ru if lang == "ru" and ch.name_ru else ch.name_en
            if ch.is_minion:
                mechs = ", ".join(ch.mechanics_ru if lang == "ru" else ch.mechanics)
                m_str = f" {mechs}" if mechs else ""
                parts.append(f"{cname} ({ch.attack or 0}/{ch.health or 0}{m_str})")
            elif ch.is_spell:
                parts.append(f"{cname} ({ch.cost}м заклинание)")
            else:
                parts.append(f"{cname}")

        if not parts:
            return ""

        prefix = "Порождает: " if lang == "ru" else "Generates: "
        return f"[{prefix}{', '.join(parts)}]"
