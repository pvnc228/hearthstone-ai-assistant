# Plan: Stage B/C next-action dataset

Status: production artifacts generated and verification gates passed; pytest temporary-directory cleanup was blocked by the execution policy.

1. [x] Add regression tests for pre-action state, replay option parsing, `END_TURN`, position identity, attack budgets, and placeholder entity resolution.
2. [x] Extend replay state with `DecisionPoint`, `ReplayOptionCandidate`, `OptionDecision`, stable entity IDs, and `CHANGE_ENTITY` handling.
3. [x] Correct observable candidate rules: seven-slot board, hero attacks, attackability filters, Windfury/Mega-Windfury, Rush, dynamic hero-power cost, and `END_TURN`.
4. [x] Replace CardDefs target inference with the replay oracle: `DebugPrintOptions error=NONE` candidates plus the `SendOption` selected tuple.
5. [x] Build schema-v2 accepted/quarantine/report artifacts with unique temporary names, a single-writer lock, pre-publication audit, atomic replacement per file, and report publication last.
6. [x] Run focused tests, a five-replay smoke, the full 549-replay production build, and an independent JSONL audit.
7. [x] Run the final full test suite (`49 passed`), compileall, `git diff --check`, and an explicit whitespace check.
8. [ ] Resolve or formally accept the three quarantine classes before training-data splits are frozen.
9. [ ] Remove verified in-repository `.pytest-tmp-*` directories; two PowerShell removal attempts were rejected by the execution policy.

## Challenge log

- Rebuilding all current card requirements was rejected because the replay already exposes a stronger label-time legality oracle.
- Treating Tradeable option ordering, dynamic discounts, or sub-option/target products as legal by convention was rejected because their semantics are not yet proven across the corpus.
- The implementation reuses the existing parser, state tracker, CardDB, and stdlib streaming JSONL; no simulator or new dependency was added.
- Production result: 12,829 accepted and 1,862 quarantined decisions out of 14,691; accepted gate violations are zero. This completes a conservative artifact generation, not QLoRA readiness.
