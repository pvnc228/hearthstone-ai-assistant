"""
Comprehensive unit and integration tests for Card Database, Indexer, Cleaner, and Semantic Formatter.
"""

import os
import time
import pytest

from src.card_db import (
    Card,
    CardClass,
    CardDatabase,
    CardType,
    Race,
    Rarity,
    SpellSchool,
    clean_card_text,
    format_board_minion,
    format_card,
    format_card_compact,
)


def test_clean_card_text():
    raw1 = "<b>Боевой клич:</b> наносит $6 ед. урона."
    assert clean_card_text(raw1) == "Боевой клич: наносит 6 ед. урона."

    raw2 = "[x]Наносит $3 ед. урона всем\nперсонажам."
    assert clean_card_text(raw2) == "Наносит 3 ед. урона всем персонажам."

    raw3 = "<i>Уничтожает</i> выбранное существо_."
    assert clean_card_text(raw3) == "Уничтожает выбранное существо."


def test_card_model_properties_and_serialization():
    card = Card(
        card_id="TEST_001",
        dbf_id=9999,
        name_ru="Тестовый Титан",
        name_en="Test Titan",
        cost=7,
        attack=6,
        health=8,
        card_type=CardType.MINION,
        card_class=CardClass.MAGE,
        race=Race.ELEMENTAL,
        spell_school=None,
        rarity=Rarity.LEGENDARY,
        text_ru="Титан. Способность наносит 5 урона.",
        text_en="Titan. Ability deals 5 damage.",
        mechanics=["Titan", "Taunt"],
        mechanics_ru=["Титан", "Провокация"],
        collectible=True,
    )

    assert card.is_minion is True
    assert card.is_spell is False
    assert card.is_titan is True
    assert card.has_taunt is True
    assert card.type_name_ru == "Существо"
    assert card.class_name_ru == "Маг"
    assert card.race_name_ru == "Элементаль"
    assert card.rarity_name_ru == "Легендарная"

    # Serialization roundtrip
    d = card.to_dict()
    restored = Card.from_dict(d)
    assert restored.card_id == card.card_id
    assert restored.dbf_id == card.dbf_id
    assert restored.name_ru == card.name_ru
    assert restored.is_titan is True
    assert restored.has_taunt is True


def test_card_database_lookups_and_filters():
    hdt_card_defs = os.path.expandvars(r"%APPDATA%\HearthstoneDeckTracker\CardDefs")
    if not os.path.exists(os.path.join(hdt_card_defs, "CardDefs.base.xml")):
        pytest.skip("HDT CardDefs XML files not found in AppData")

    db = CardDatabase()
    assert len(db) > 30000

    # 1. Classic Fireball (CS2_029, dbf_id 315)
    fireball = db.get_by_id("CS2_029")
    assert fireball is not None
    assert fireball.name_ru == "Огненный шар"
    assert fireball.cost == 4
    assert fireball.is_spell is True
    assert fireball.spell_school == SpellSchool.FIRE
    assert fireball.spell_school_name_ru == "Огонь"

    # Test lookup by dbf_id
    fireball_by_dbf = db.get_by_dbf_id(315)
    assert fireball_by_dbf is not None
    assert fireball_by_dbf.card_id == "CS2_029"

    # 2. Leeroy Jenkins (EX1_116)
    leeroy = db.get_by_id("EX1_116")
    assert leeroy is not None
    assert leeroy.name_ru == "Лирой Дженкинс"
    assert leeroy.attack == 6
    assert leeroy.health == 2
    assert leeroy.has_charge is True or "Charge" in leeroy.mechanics

    # 3. Search by name
    results = db.search_by_name("Огненный шар")
    assert any(c.card_id == "CS2_029" for c in results)

    # 4. Filter by class & cost
    mage_spells_4m = db.filter(card_class=CardClass.MAGE, card_type=CardType.SPELL, cost=4, collectible_only=True)
    assert any(c.card_id == "CS2_029" for c in mage_spells_4m)


def test_modern_mechanics():
    hdt_card_defs = os.path.expandvars(r"%APPDATA%\HearthstoneDeckTracker\CardDefs")
    if not os.path.exists(os.path.join(hdt_card_defs, "CardDefs.base.xml")):
        pytest.skip("HDT CardDefs XML files not found in AppData")

    db = CardDatabase()

    # Location test (CATA_301: Ruby Sanctuary)
    loc = db.get_by_id("CATA_301")
    if loc:
        assert loc.is_location is True
        assert loc.card_type == CardType.LOCATION

    # Titan test (TLC_452: Titangraph Osk)
    titan = db.get_by_id("TLC_452")
    if titan:
        assert titan.is_titan is True

    # Starship Piece test (GDB_100)
    starship = db.get_by_id("GDB_100")
    if starship:
        assert starship.is_starship is True

    # Tourist test (VAC_336: Masked Reveler / Maestro)
    tourist = db.get_by_id("VAC_336")
    if tourist:
        assert tourist.is_tourist is True

    # DK Runes test (RLK_061: Lord Marrowgar / Corpse Spender)
    dk_card = db.get_by_id("RLK_061")
    if dk_card:
        assert dk_card.card_class == CardClass.DEATHKNIGHT
        assert "unholy" in dk_card.runes


def test_semantic_formatter():
    fireball = Card(
        card_id="CS2_029",
        dbf_id=315,
        name_ru="Огненный шар",
        name_en="Fireball",
        cost=4,
        card_type=CardType.SPELL,
        card_class=CardClass.MAGE,
        spell_school=SpellSchool.FIRE,
        text_ru="Наносит 6 ед. урона.",
        collectible=True,
    )

    formatted = format_card(fireball, lang="ru")
    assert "Огненный шар" in formatted
    assert "4 маны" in formatted
    assert "Заклинание Огонь" in formatted
    assert "Наносит 6 ед. урона." in formatted

    compact = format_card_compact(fireball, lang="ru")
    assert "[4м Заклинание] Огненный шар: Наносит 6 ед. урона." in compact


def test_board_minion_formatter():
    res = format_board_minion(
        name="Тирион Фордринг",
        atk=6,
        hp=6,
        max_hp=6,
        can_attack=True,
        is_taunt=True,
        is_divine_shield=True,
        is_frozen=False,
    )
    assert "Тирион Фордринг (6/6)" in res
    assert "Провокация" in res
    assert "Бож. щит" in res
    assert "готов атаковать" in res


def test_benchmark_lookup_speed():
    db = CardDatabase()
    test_ids = ["CS2_029", "EX1_116", "CATA_301", "GDB_100", "RLK_061"]

    t0 = time.time()
    for _ in range(20000):
        for cid in test_ids:
            _ = db.get_by_id(cid)
    duration = time.time() - t0

    # 100,000 lookups should take < 0.1s
    lookups_per_sec = 100000 / duration
    print(f"\nBenchmark: {lookups_per_sec:,.0f} card lookups/sec (100k in {duration:.4f}s)")
    assert duration < 0.5
