# Spec: Stage B/C next-action dataset

## Goal

Produce a reproducible local next-action dataset from ranked winning HDT replays in the contract `pre-action state + complete replay-reported legal candidates -> chosen candidate ID`.

## Scope

- Preserve the regenerated turn-sequence baseline in `data/processed/train_actions.jsonl`.
- Parse `DebugPrintOptions` and `SendOption` and capture state immediately before every selection.
- Build candidates only from replay options reported with `error=NONE`, plus the explicit end-turn option.
- Include stable entity IDs, target IDs, sub-option IDs, and board position in candidate identity.
- Write accepted JSONL, quarantine JSONL, and a machine-readable validation report through unique temporary files; prevent concurrent writers, audit before publication, and publish the report manifest last.
- Use local HDT ranked-win replays only; keep HSReplay disabled.

## Acceptance criteria

- Board capacity is seven minions/locations.
- Remaining Windfury/Mega-Windfury attacks, hero attacks, attackability flags, Rush face restrictions, dynamic hero-power cost, and `END_TURN` are represented.
- A known option `cardId` resolves through CardDB before a placeholder `UNKNOWN ENTITY` name can be used.
- Every accepted decision has exactly one candidate matching `option/sub-option/target/position`.
- Every accepted chosen candidate has the active owner, no unresolved action entity, and no mana violation visible in the saved snapshot.
- Ambiguous Tradeable play/trade options, sub-option/target cross-products, and reconstructed mana-cost mismatches in any candidate are quarantined with reason codes rather than guessed.
- The report records source counts and hashes, accepted/quarantined counts, action distributions, gate violations, and an explicit QLoRA readiness verdict.
- Focused and full project tests, production generation, independent JSONL audit, compilation, and `git diff --check` pass.

## Non-goals

- No QLoRA run, dependency installation, HSReplay inclusion, inferred target legality, or model benchmark.
- No claim that the dataset is training-ready before frozen game-level splits and the remaining readiness gates exist.
