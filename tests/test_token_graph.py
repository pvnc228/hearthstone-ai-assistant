"""
Unit tests for TokenGraph and card semantic relationships.
"""

import pytest
from src.card_db.indexer import CardDatabase
from src.card_db.token_graph import TokenGraph


@pytest.fixture(scope="module")
def card_db():
    return CardDatabase()


@pytest.fixture(scope="module")
def token_graph():
    return TokenGraph()


def test_token_graph_loading(token_graph):
    assert token_graph._loaded is True
    assert len(token_graph._semantics_by_dbf) > 1000
    assert len(token_graph._children_by_parent) > 500


def test_animal_companion_tokens(token_graph, card_db):
    # Animal Companion is NEW1_031 / DBF 437
    companion = card_db.get_by_id("NEW1_031")
    assert companion is not None

    children = token_graph.get_child_cards(companion, card_db)
    child_names = [c.name_ru for c in children]

    # Should contain Huffer, Leokk, Misha
    assert "Хаффер" in child_names or "Huffer" in [c.name_en for c in children]
    assert "Леокк" in child_names or "Leokk" in [c.name_en for c in children]
    assert "Миша" in child_names or "Misha" in [c.name_en for c in children]


def test_format_token_summary(token_graph, card_db):
    companion = card_db.get_by_id("NEW1_031")
    summary_ru = token_graph.format_token_summary(companion, card_db, lang="ru")
    assert "Порождает:" in summary_ru
    assert "Хаффер" in summary_ru or "Леокк" in summary_ru or "Миша" in summary_ru


def test_card_semantics_actions_and_tags(token_graph, card_db):
    # VAC_951 ("Health" Drink / DBF 107923)
    drink = card_db.get_by_id("VAC_951")
    if drink:
        sem = token_graph.get_semantics(drink)
        assert sem is not None
        assert "Lifesteal" in sem.keywords
        assert any(tag in sem.mechanic_tags for tag in ["deal_damage", "lifesteal_damage"])
