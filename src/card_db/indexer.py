"""
Card database indexer and query engine.
Parses HDT CardDefs XML files, indexes cards into SQLite cache, and provides fast in-memory O(1) lookups.
"""

import json
import logging
import os
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .cleaner import clean_card_text
from .enums import (
    CardClass,
    CardType,
    Race,
    Rarity,
    SpellSchool,
    MECHANIC_TAGS,
    TOURIST_TAGS,
)
from .models import Card

logger = logging.getLogger(__name__)

# Default locations for HDT CardDefs
DEFAULT_HDT_APP_DATA = os.path.expandvars(r"%APPDATA%\HearthstoneDeckTracker\CardDefs")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
DEFAULT_DB_PATH = DEFAULT_CACHE_DIR / "cards.db"


def _safe_enum(enum_cls, value):
    """Converts raw int to enum member, returning None for unknown values instead of raising."""
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


class CardDatabase:
    """
    High-performance card database with SQLite backing and in-memory indexing.
    """

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        hdt_card_defs_dir: Optional[Path | str] = None,
        auto_load: bool = True,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.hdt_card_defs_dir = Path(hdt_card_defs_dir) if hdt_card_defs_dir else Path(DEFAULT_HDT_APP_DATA)

        self._cards_by_id: Dict[str, Card] = {}
        self._cards_by_dbf_id: Dict[int, Card] = {}
        self._loaded: bool = False

        if auto_load:
            self.load()

    def load(self, force_rebuild: bool = False) -> None:
        """Loads cards into in-memory dictionaries. Rebuilds SQLite cache if needed."""
        if not self.db_path.exists() or force_rebuild:
            self.build_db(force=True)
        elif self._cache_is_stale():
            logger.info("CardDefs XML is newer than cache; rebuilding SQLite cache.")
            self.build_db(force=True)

        self._load_from_sqlite()

    def _cache_is_stale(self) -> bool:
        """True when any CardDefs XML is newer than the SQLite cache."""
        if not self.db_path.exists():
            return True
        cache_mtime = self.db_path.stat().st_mtime
        for name in ("CardDefs.base.xml", "CardDefs.ruRU.xml"):
            xml_path = self.hdt_card_defs_dir / name
            if xml_path.exists() and xml_path.stat().st_mtime > cache_mtime:
                return True
        return False

    def _load_from_sqlite(self) -> None:
        """Reads all indexed cards from SQLite database into memory for microsecond lookups."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Card database not found at {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM cards")
        rows = cursor.fetchall()

        self._cards_by_id.clear()
        self._cards_by_dbf_id.clear()

        for row in rows:
            card = Card(
                card_id=row["card_id"],
                dbf_id=row["dbf_id"],
                name_ru=row["name_ru"] or "",
                name_en=row["name_en"] or "",
                cost=row["cost"] or 0,
                attack=row["attack"],
                health=row["health"],
                durability=row["durability"],
                card_type=_safe_enum(CardType, row["card_type"]) or CardType.MINION,
                card_class=_safe_enum(CardClass, row["card_class"]) or CardClass.NEUTRAL,
                race=_safe_enum(Race, row["race"]),
                spell_school=_safe_enum(SpellSchool, row["spell_school"]),
                rarity=_safe_enum(Rarity, row["rarity"]),
                text_ru=row["text_ru"] or "",
                text_en=row["text_en"] or "",
                mechanics=json.loads(row["mechanics_json"]) if row["mechanics_json"] else [],
                mechanics_ru=json.loads(row["mechanics_ru_json"]) if row["mechanics_ru_json"] else [],
                runes=json.loads(row["runes_json"]) if row["runes_json"] else {},
                tourist_class=row["tourist_class"],
                collectible=bool(row["collectible"]),
                elite=bool(row["elite"]),
                card_set=row["card_set"] or 0,
                artist=row["artist"] or "",
            )
            self._cards_by_id[card.card_id] = card
            if card.dbf_id:
                self._cards_by_dbf_id[card.dbf_id] = card

        conn.close()
        self._loaded = True
        logger.info("Loaded %d cards into CardDatabase memory cache.", len(self._cards_by_id))

    def build_db(self, force: bool = False) -> int:
        """Parses CardDefs XML files and saves structured records into SQLite database."""
        base_xml = self.hdt_card_defs_dir / "CardDefs.base.xml"
        ru_xml = self.hdt_card_defs_dir / "CardDefs.ruRU.xml"

        if not base_xml.exists() or not ru_xml.exists():
            raise FileNotFoundError(
                f"CardDefs XML files missing in {self.hdt_card_defs_dir}. Expected CardDefs.base.xml and CardDefs.ruRU.xml"
            )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists() and force:
            self.db_path.unlink()

        # Step 1: Parse ruRU localized text
        ru_texts: Dict[str, Tuple[str, str]] = {}
        tree_ru = ET.parse(ru_xml)
        for ent in tree_ru.getroot():
            cid = ent.attrib.get("CardID")
            if not cid:
                continue
            name_ru = ""
            text_ru = ""
            for tag in ent:
                tname = tag.attrib.get("name")
                if tname == "CARDNAME":
                    for c in tag:
                        if c.tag == "ruRU":
                            name_ru = c.text or ""
                elif tname == "CARDTEXT":
                    for c in tag:
                        if c.tag == "ruRU":
                            text_ru = c.text or ""
            ru_texts[cid] = (name_ru, text_ru)

        # Step 2: Parse base XML and combine
        tree_base = ET.parse(base_xml)
        cards_to_insert = []

        for ent in tree_base.getroot():
            cid = ent.attrib.get("CardID")
            if not cid:
                continue

            dbf_id_str = ent.attrib.get("ID", "0")
            dbf_id = int(dbf_id_str) if dbf_id_str.isdigit() else 0

            name_en = ""
            text_en = ""
            cost = 0
            attack = None
            health = None
            durability = None
            card_type = CardType.MINION
            card_class = CardClass.NEUTRAL
            race = None
            spell_school = None
            rarity = None
            collectible = False
            elite = False
            card_set = 0
            artist = ""

            mechanics = []
            mechanics_ru = []
            runes = {}
            tourist_class = None

            for tag in ent:
                tname = tag.attrib.get("name")
                tval = tag.attrib.get("value")

                if tname == "CARDNAME":
                    for c in tag:
                        if c.tag == "enUS":
                            name_en = c.text or ""
                elif tname == "CARDTEXT":
                    for c in tag:
                        if c.tag == "enUS":
                            text_en = c.text or ""
                elif tname == "COST" and tval is not None:
                    cost = int(tval)
                elif tname == "ATK" and tval is not None:
                    attack = int(tval)
                elif tname == "HEALTH" and tval is not None:
                    health = int(tval)
                elif tname == "DURABILITY" and tval is not None:
                    durability = int(tval)
                elif tname == "CARDTYPE" and tval is not None:
                    try:
                        card_type = CardType(int(tval))
                    except ValueError:
                        pass
                elif tname == "CLASS" and tval is not None:
                    try:
                        card_class = CardClass(int(tval))
                    except ValueError:
                        pass
                elif tname == "CARDRACE" and tval is not None:
                    try:
                        race = Race(int(tval))
                    except ValueError:
                        pass
                elif tname == "SPELL_SCHOOL" and tval is not None:
                    try:
                        spell_school = SpellSchool(int(tval))
                    except ValueError:
                        pass
                elif tname == "RARITY" and tval is not None:
                    try:
                        rarity = Rarity(int(tval))
                    except ValueError:
                        pass
                elif tname == "COLLECTIBLE" and tval == "1":
                    collectible = True
                elif tname == "ELITE" and tval == "1":
                    elite = True
                elif tname == "CARD_SET" and tval is not None:
                    card_set = int(tval)
                elif tname == "ARTISTNAME":
                    artist = tag.text or ""

                # Runes
                elif tname == "COST_BLOOD" and tval is not None:
                    runes["blood"] = int(tval)
                elif tname == "COST_FROST" and tval is not None:
                    runes["frost"] = int(tval)
                elif tname == "COST_UNHOLY" and tval is not None:
                    runes["unholy"] = int(tval)

                # Tourists
                elif tname in TOURIST_TAGS and tval == "1":
                    tourist_class = TOURIST_TAGS[tname]
                    if "Tourist" not in mechanics:
                        mechanics.append("Tourist")
                        mechanics_ru.append(f"Турист ({tourist_class})")

                # Boolean game mechanics
                elif tname in MECHANIC_TAGS and tval == "1":
                    en_mech, ru_mech = MECHANIC_TAGS[tname]
                    if en_mech not in mechanics:
                        mechanics.append(en_mech)
                        mechanics_ru.append(ru_mech)

            raw_name_ru, raw_text_ru = ru_texts.get(cid, ("", ""))
            clean_name_ru = clean_card_text(raw_name_ru) or clean_card_text(name_en)
            clean_name_en = clean_card_text(name_en)
            cleaned_text_ru = clean_card_text(raw_text_ru)
            cleaned_text_en = clean_card_text(text_en)

            cards_to_insert.append(
                (
                    cid,
                    dbf_id,
                    clean_name_ru,
                    clean_name_en,
                    cost,
                    attack,
                    health,
                    durability,
                    int(card_type),
                    int(card_class),
                    int(race) if race is not None else None,
                    int(spell_school) if spell_school is not None else None,
                    int(rarity) if rarity is not None else None,
                    cleaned_text_ru,
                    cleaned_text_en,
                    json.dumps(mechanics, ensure_ascii=False),
                    json.dumps(mechanics_ru, ensure_ascii=False),
                    json.dumps(runes, ensure_ascii=False),
                    tourist_class,
                    1 if collectible else 0,
                    1 if elite else 0,
                    card_set,
                    artist,
                )
            )

        # Step 3: Insert into SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cards (
                card_id TEXT PRIMARY KEY,
                dbf_id INTEGER,
                name_ru TEXT,
                name_en TEXT,
                cost INTEGER,
                attack INTEGER,
                health INTEGER,
                durability INTEGER,
                card_type INTEGER,
                card_class INTEGER,
                race INTEGER,
                spell_school INTEGER,
                rarity INTEGER,
                text_ru TEXT,
                text_en TEXT,
                mechanics_json TEXT,
                mechanics_ru_json TEXT,
                runes_json TEXT,
                tourist_class TEXT,
                collectible INTEGER,
                elite INTEGER,
                card_set INTEGER,
                artist TEXT
            )
        """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_dbf_id ON cards(dbf_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_name_ru ON cards(name_ru)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_name_en ON cards(name_en)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_collectible ON cards(collectible)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_type ON cards(card_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_class ON cards(card_class)")

        cursor.executemany(
            """
            INSERT OR REPLACE INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            cards_to_insert,
        )

        conn.commit()
        conn.close()

        logger.info("Successfully built SQLite cards cache with %d cards at %s", len(cards_to_insert), self.db_path)
        return len(cards_to_insert)

    def get_by_id(self, card_id: str) -> Optional[Card]:
        """Returns Card by its alphanumeric CardID (e.g. 'CS2_029') in O(1)."""
        return self._cards_by_id.get(card_id)

    def get_by_dbf_id(self, dbf_id: int) -> Optional[Card]:
        """Returns Card by its integer DbfID (e.g. 315) in O(1)."""
        return self._cards_by_dbf_id.get(dbf_id)

    def search_by_name(self, query: str, lang: str = "ru", limit: int = 10) -> List[Card]:
        """Searches cards by substring in Russian or English name."""
        q = query.lower().strip()
        results = []
        for card in self._cards_by_id.values():
            target_name = card.name_ru.lower() if lang == "ru" else card.name_en.lower()
            if q in target_name:
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    def filter(
        self,
        card_class: Optional[CardClass] = None,
        card_type: Optional[CardType] = None,
        cost: Optional[int] = None,
        collectible_only: bool = True,
    ) -> List[Card]:
        """Filters cards by class, type, mana cost, and collectibility."""
        results = []
        for card in self._cards_by_id.values():
            if collectible_only and not card.collectible:
                continue
            if card_class is not None and card.card_class != card_class:
                continue
            if card_type is not None and card.card_type != card_type:
                continue
            if cost is not None and card.cost != cost:
                continue
            results.append(card)
        return results

    def __len__(self) -> int:
        return len(self._cards_by_id)

    def __contains__(self, card_id: str) -> bool:
        return card_id in self._cards_by_id
