"""
Unit tests for dataset formatting and ChatML / Alpaca conversion.
"""

from pathlib import Path
from src.llm.dataset_formatter import format_dataset, to_alpaca, to_chatml


def test_to_chatml_structure():
    prompt = "Hero HP: 30, Mana: 2"
    completion = "1. Play card: Minion\n2. End turn"
    res = to_chatml(prompt, completion)

    assert "messages" in res
    assert len(res["messages"]) == 3
    assert res["messages"][0]["role"] == "system"
    assert res["messages"][1]["role"] == "user"
    assert res["messages"][1]["content"] == prompt
    assert res["messages"][2]["role"] == "assistant"
    assert res["messages"][2]["content"] == completion


def test_to_alpaca_structure():
    prompt = "Hero HP: 30, Mana: 2"
    completion = "1. Play card: Minion\n2. End turn"
    res = to_alpaca(prompt, completion)

    assert "instruction" in res
    assert "input" in res
    assert "output" in res
    assert res["input"] == prompt
    assert res["output"] == completion


def test_format_dataset_generation(tmp_path):
    train_count, eval_count = format_dataset(output_dir=tmp_path, fmt="chatml", train_ratio=0.8)
    assert train_count > 0
    assert eval_count > 0
    assert (tmp_path / "sft_train_chatml.jsonl").exists()
    assert (tmp_path / "sft_eval_chatml.jsonl").exists()
