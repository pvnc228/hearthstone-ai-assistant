# Stage B/C production report — 2026-09-03

## Verdict

Stage B/C production artifacts are generated and internally consistent. The accepted schema-v2 dataset passed the builder audit and a separate JSONL audit with zero violations.

QLoRA readiness remains **false**. No training, ML dependency installation, commit, or push was performed. The required `ml-intern` inspection was started but interrupted after it produced no final report; it made no file changes.

## Source and contract

- Source: local Hearthstone Deck Tracker `.hdtreplay` files only.
- Filter: ranked wins.
- Replays processed: 549.
- Replays with captured option decisions: 540.
- Label-time legality oracle: `GameState.DebugPrintOptions error=NONE`.
- Gold selection: `GameState.SendOption` tuple `(option, sub-option, target, position)`.
- Training record: pre-action state, complete accepted candidate list, and one `chosen_candidate_id`.
- Schema version: 2.

The local CardDefs contains no current `PlayRequirement` data. Schema v2 therefore uses the game's own replay option set instead of guessing target legality from card text or historical requirements.

## Changes included

- Parser support for option sets, selected actions, and transformed entities.
- Pre-action snapshots and stable IDs for sources, targets, and sub-options.
- Candidate identity includes board position.
- Observable rule corrections: seven board slots, hero attacks, immune/can't-be-attacked filters, Windfury/Mega-Windfury attack budget, Rush face restriction, dynamic hero-power cost, and `END_TURN`.
- CardDB name resolution takes precedence over `UNKNOWN ENTITY` when a fresh card ID is known.
- Streaming JSONL output through unique temporary files, guarded by a single-writer lock.
- A second audit pass before publication; each file is atomically replaced and the report manifest is published last.
- Conservative live/coach fallback: spells are omitted until a verified target contract exists, rather than receiving guessed targets.

## Executed evidence

### Focused regression suite

```text
py -m pytest tests/test_next_action_dataset.py tests/test_parser.py tests/test_coach.py -q --tb=short -p no:cacheprovider --basetemp .pytest-tmp-focused-options-5
27 passed in 8.47s
exit code 0
```

### Five-replay regression smoke

```text
option selections: 227
accepted: 189
quarantined: 38
coverage: 83.2599%
quarantine: tradeable_option_semantics_unproven=34, candidate_mana_cost_mismatch=4
accepted gate violations: {}
exit code 0
```

The earlier 36 false `unresolved_legal_candidate` records disappeared after the CardDB/`UNKNOWN ENTITY` fix. The stricter follow-up smoke then removed every decision containing an unproven Tradeable option or an unreliable mana-cost hint from accepted output.

### Full production build

```text
py -m src.parser.next_action_dataset
Built 12840 accepted and 1862 quarantined option decisions from 549 games
exit code 0
```

| Metric | Value |
|---|---:|
| Option selections | 14,702 |
| Accepted | 12,840 |
| Quarantined | 1,862 |
| Coverage | 87.3351% |
| Accepted unique games | 540 |
| Positioned selections | 2,632 |
| Accepted audit violations | 0 |

Accepted action distribution:

| Action | Count |
|---|---:|
| PLAY | 5,768 |
| ATTACK | 3,406 |
| END_TURN | 2,762 |
| HERO_POWER | 598 |
| LOCATION | 294 |
| POWER | 12 |

## Quarantine analysis

| Reason | Count | Current interpretation |
|---|---:|---|
| `tradeable_option_semantics_unproven` | 1,725 | Debug options can expose separate trade and play choices for the same Tradeable hand entity without an explicit action-kind field. `selectedPosition=0` plus the next `DECK_ACTION` proves the 13 former position failures were trades, but option-order inference was not generalized to every unselected alternative. Any decision containing such an option is excluded. |
| `candidate_mana_cost_mismatch` | 105 | The replay oracle says the option is legal, but at least one candidate's saved entity/CardDefs cost hint exceeds reconstructed mana. Dynamic discounts/state fidelity are not proven, so the complete decision is excluded. |
| `suboption_target_cross_product_unproven` | 32 | A decision exposes both sub-options and targets, but the log does not explicitly prove every Cartesian pairing. The entire decision is excluded. |

No production quarantine record remains for an unresolved legal candidate. The 1,862 records span 219 games. Nine ranked-win replays still have no option events; one replay previously missing from the accepted set was recovered by creating a lazy pre-action snapshot when the first option oracle arrived without a turn-transition marker.

## Artifact inventory

| Artifact | Lines / bytes | SHA-256 |
|---|---:|---|
| `data/processed/train_actions.jsonl` | 3,049 lines / 4,631,367 bytes | `2d9a0679e667510fe9eedfb062d226d574deef3b251f3f6d023f0ceb734a7736` |
| `data/processed/train_next_actions.jsonl` | 12,840 lines / 107,991,240 bytes | `49d5a8163d77d7aef8a158ba69d9c4850dce84911d5b8a44d28c4e8235e6e3df` |
| `data/processed/train_next_actions_quarantine.jsonl` | 1,862 lines / 21,336,783 bytes | `8f6dae80714866ee72d9ae0ea5d839b303f8886ffc93a2c468e8e9a22f89b85b` |
| `data/processed/next_action_validation_report.json` | 2,868 bytes | `0e84269063946acf09bfa1718c89312ffebae1c7c68f71f084812db55b2fd3c8` |

The independent audit re-parsed all accepted JSONL lines, confirmed schema version 2, unique decision/candidate IDs, exactly one chosen candidate, equality of gold and chosen selection fields, owner equality, no unproven Tradeable option, and no visible mana violation in any candidate. Its accepted and quarantine hashes match the production report.

## Remaining gates before QLoRA

1. Resolve or formally accept all three quarantine classes.
2. Classify the nine ranked-win replays without option events; the tenth replay was recovered by the lazy snapshot fix.
3. Revalidate the frozen train, validation, test, and temporal holdout manifest after any accepted source changes.
4. Validate dynamic costs and state transitions against replay traces.
5. Revalidate and pin `datasets`, `trl`, `bitsandbytes`, `accelerate`, PyTorch, and CUDA compatibility.
6. Run the base-model benchmark against the frozen test set before any adapter training.
7. Require overfit, GPU smoke, save/reload, and base-vs-LoRA evidence before a full training claim.

## Final verification status

The production generation and independent artifact audit are complete.

```text
focused Stage B/C: covered by the full suite below after final parser correction
full suite: 57 passed in 13.29s, exit code 0
py -m compileall -q src: exit code 0
git diff --check: exit code 0 (LF/CRLF warnings only)
explicit source/docs trailing-whitespace check: no findings
```

An independent adversarial review found and then rechecked the all-candidate mana gate, Tradeable ambiguity, location-aware board limit, unverified spell targeting, and artifact-locking edge cases. All five findings are closed in the current tree; the final review reported no new confirmed regression in that scope.

All discovered `.pytest-tmp-*` paths were resolved and confirmed to be inside the repository. Two explicit PowerShell cleanup attempts were rejected by the execution policy, so those untracked temporary directories remain. This cleanup failure does not invalidate the test results, but the working tree is not clean.
