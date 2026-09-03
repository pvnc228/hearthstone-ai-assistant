# QLoRA readiness — 2026-09-03

## Current verdict

Schema-v2 is connected to the training contract and frozen game-level splits exist. QLoRA is still blocked by the environment, nine source replays without option events, unverified dynamic state transitions, and the missing live baseline run.

## Completed in this stage

- Shared prompt builder in `src/llm/next_action_contract.py` is used by both `MatchCoach` and formatted schema-v2 records.
- `data/processed/next_action_split_manifest_v1.json` freezes all 540 accepted games with no game-level overlap:
  - train: 390 games / 9,320 records;
  - validation: 48 games / 1,198 records;
  - test: 48 games / 1,117 records;
  - temporal holdout: 54 games / 1,205 records.
- Manifest source hash is `49d5a8163d77d7aef8a158ba69d9c4850dce84911d5b8a44d28c4e8235e6e3df`.
- Quarantine policy is explicit: only accepted schema-v2 records enter the first pilot; quarantine remains excluded.
- QLoRA config points to `next_action_train_chatml.jsonl` and `next_action_validation_chatml.jsonl` and declares `dataset_contract=next_action_v2`.
- `--validate-only` verifies the source hash, ChatML prompt/completion consistency, and split membership.
- `requirements-qlora.txt` records the bounded optional dependency family required by the trainer; it was not installed in this run.

## Verification evidence

```text
py -m pytest -q --tb=short -p no:cacheprovider --basetemp C:\Users\mist8\AppData\Local\Temp\hearthstone-ai-final-3
57 passed in 13.29s

py -m compileall -q src
exit code 0

py -m src.llm.train_qlora --validate-only
train 9320 / 390 games; eval 1198 / 48 games; test 1117 / 48 games; temporal 1205 / 54 games
```

## Remaining blockers

`py -m src.llm.train_qlora --check-environment` reports:

- missing `datasets`, `trl`, and `bitsandbytes`;
- `torch 2.9.0+cpu`, CUDA unavailable, no GPU smoke possible;
- installed: `transformers 4.57.6`, `peft 0.17.1`, `accelerate 1.10.1`.

The bounded Ollama smoke against the frozen test file timed out at `http://127.0.0.1:11434`; its saved report has status `blocked`, so no baseline metrics are claimed. Nine ranked-win replays still contain zero `DebugPrintOptions` and `SendOption` events; one replay with option events was recovered by the lazy snapshot fix.

No full training, dependency installation, Git commit, or push was performed.
