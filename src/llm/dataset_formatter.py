"""
Dataset Formatter for LLM Fine-Tuning (SFT / QLoRA / DPO).
Converts extracted Hearthstone game-action pairs into standard ChatML, ShareGPT, and Alpaca formats.
Splits by game_id to guarantee zero data leakage between train and evaluation sets.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Ты — элитный гроссмейстер и тактический ИИ-ассистент Hearthstone. "
    "Твоя задача — проанализировать текущее состояние матча (здоровье героев, доступная мана, карты в руке, состояние стола) "
    "и выбрать оптимальную последовательность легальных действий на ход игрока, максимизируя темп, контроль стола или реализацию летального урона."
)

DEFAULT_INPUT_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "train_actions.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def to_chatml(prompt: str, completion: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Dict[str, Any]:
    """Converts a prompt-completion pair into OpenAI / HuggingFace ChatML format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }


def to_alpaca(prompt: str, completion: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Dict[str, Any]:
    """Converts a prompt-completion pair into Alpaca format."""
    return {
        "instruction": system_prompt,
        "input": prompt,
        "output": completion,
    }


def format_dataset(
    input_file: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    train_ratio: float = 0.9,
    seed: int = 42,
    fmt: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[int, int]:
    """
    Reads train_actions.jsonl, splits by game_id into train/eval subsets,
    and writes formatted JSONL files ready for HuggingFace / TRL / SFTTrainer.
    """
    input_path = Path(input_file) if input_file else DEFAULT_INPUT_FILE
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Source actions dataset not found at {input_path}")

    # Read records and group by game_id
    games: Dict[str, List[Dict[str, Any]]] = {}
    total_records = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gid = rec.get("game_id", "default_game")
            games.setdefault(gid, []).append(rec)
            total_records += 1

    # Shuffle games deterministically
    game_ids = list(games.keys())
    rng = random.Random(seed)
    rng.shuffle(game_ids)

    split_idx = int(len(game_ids) * train_ratio)
    train_game_ids = set(game_ids[:split_idx])
    eval_game_ids = set(game_ids[split_idx:])

    train_records = []
    eval_records = []

    for gid, rows in games.items():
        target_list = train_records if gid in train_game_ids else eval_records
        for r in rows:
            p = r["prompt"]
            c = r["completion"]
            if fmt == "chatml":
                item = to_chatml(p, c, system_prompt=system_prompt)
            elif fmt == "alpaca":
                item = to_alpaca(p, c, system_prompt=system_prompt)
            else:
                item = {"prompt": p, "completion": c}
            
            # Attach metadata
            item["game_id"] = gid
            item["turn_number"] = r.get("turn_number")
            item["player_hero"] = r.get("player_hero")
            item["opponent_hero"] = r.get("opponent_hero")
            target_list.append(item)

    # Write output files
    train_file = out_dir / f"sft_train_{fmt}.jsonl"
    eval_file = out_dir / f"sft_eval_{fmt}.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_file, "w", encoding="utf-8") as f:
        for item in eval_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Formatted dataset (%s): %d train pairs (from %d games), %d eval pairs (from %d games).",
        fmt,
        len(train_records),
        len(train_game_ids),
        len(eval_records),
        len(eval_game_ids),
    )

    return len(train_records), len(eval_records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    n_train, n_eval = format_dataset(fmt="chatml")
    print(f"ChatML SFT dataset generated: {n_train} train samples, {n_eval} eval samples.")
