"""
Master Dataset Builder:
Combines local .hdtreplay files and online HSReplay.net XML archives
into a unified, high-volume master dataset for QLoRA fine-tuning.
"""

import json
from pathlib import Path
from typing import Optional

from .dataset_generator import DEFAULT_OUTPUT_FILE as LOCAL_DATASET_FILE
from .hsreplay_dataset_builder import DEFAULT_HSREPLAY_DATASET, build_hsreplay_dataset

MASTER_DATASET_FILE = Path("data/processed/train_master_actions.jsonl")


def build_master_dataset(download_missing_online: bool = False, max_online_games: Optional[int] = None) -> None:
    """
    Parses HSReplay games, extracts turn action pairs, and merges
    with local replay data into data/processed/train_master_actions.jsonl.
    """
    MASTER_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65, flush=True)
    print("🚀 СБОРКА ЕДИНОГО ОБУЧАЮЩЕГО ДАТАСЕТА (LOCAL + HSREPLAY.NET)", flush=True)
    print("=" * 65, flush=True)

    print("\n[ШАГ 1/2] Парсинг скачанных XML матчей с HSReplay.net...", flush=True)
    build_hsreplay_dataset(max_games=max_online_games, download_missing=download_missing_online)

    print("\n[ШАГ 2/2] Объединение локальных и онлайн выборок...", flush=True)
    total_records = 0
    seen_prompts = set()

    with open(MASTER_DATASET_FILE, "w", encoding="utf-8") as master_f:
        # 1. Local dataset
        if LOCAL_DATASET_FILE.exists():
            local_count = 0
            with open(LOCAL_DATASET_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        p_hash = hash(record.get("prompt", ""))
                        seen_prompts.add(p_hash)
                        master_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        local_count += 1
                        total_records += 1
            print(f"  • Добавлено из локальных .hdtreplay: {local_count} записей", flush=True)

        # 2. HSReplay dataset
        if DEFAULT_HSREPLAY_DATASET.exists():
            online_count = 0
            duplicates = 0
            with open(DEFAULT_HSREPLAY_DATASET, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        p_hash = hash(record.get("prompt", ""))
                        if p_hash in seen_prompts:
                            duplicates += 1
                            continue
                        seen_prompts.add(p_hash)
                        master_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        online_count += 1
                        total_records += 1
            print(f"  • Добавлено из HSReplay.net:          {online_count} уникальных записей (дубликатов отсеяно: {duplicates})", flush=True)

    print("\n" + "=" * 65, flush=True)
    print(f"🎉 ИТОГОВЫЙ МАСТЕР-ДАТАСЕТ ГОТОВ:", flush=True)
    print(f"  Файл:        {MASTER_DATASET_FILE}", flush=True)
    print(f"  Всего пар:   {total_records} обучающих ходов", flush=True)
    print(f"  Размер:      {MASTER_DATASET_FILE.stat().st_size / (1024 * 1024):.2f} MB", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    build_master_dataset(download_missing_online=False)
