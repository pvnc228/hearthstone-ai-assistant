# Research: replay-option-grounded next-action dataset

## Evidence

- Production `train_actions.jsonl` was regenerated from 549 ranked-win HDT replays: 3,049 turn records, 538 non-empty unique `game_id` values, and 11,704 recorded actions.
- HDT replays contain `GameState.DebugPrintOptions` and `GameState.SendOption`. Together they provide the actual option set (`error=NONE`) and the selected `option/sub-option/target/position` tuple immediately before an action.
- The local `CardDefs.base.xml` contains 35,807 cards but no `<PlayRequirement>` elements. That prevents CardDefs-only reconstruction of current target rules, but it does not block replay labels when the game's own option oracle is present.
- Ten of 549 ranked-win replays contain no captured option decisions. They remain outside the schema-v2 dataset instead of receiving inferred labels.
- The final conservative schema-v2 build processed 14,691 option selections: 12,829 accepted and 1,862 quarantined (87.3256% coverage). The independent serialized-file audit found no accepted-record violations.

## Alternatives considered

1. Generate targets from card text. This can create illegal candidates and is not a deterministic legality proof.
2. Import a historical PlayRequirements snapshot. It is incomplete for current cards and may encode obsolete behavior.
3. Use the replay's `DebugPrintOptions error=NONE` set and `SendOption` selection. This is the strongest available label-time oracle and avoids rebuilding Hearthstone rules.
4. Admit ambiguous position, dynamic-cost, or sub-option/target combinations heuristically. This raises coverage but weakens the training contract.

## Recommendation

Use option 3 and quarantine the remaining ambiguous records. Candidate IDs must include option, sub-option, target, and board position. Resolve a known card ID through CardDB before accepting an `UNKNOWN ENTITY` label. Keep all decisions containing an unresolved Tradeable option, any candidate with a stale/dynamic mana hint, or an unproven sub-option/target binding out of training until each class has a reproducible rule.

The resulting dataset is suitable as the Stage C production artifact, but it is not yet a training-ready split. QLoRA remains blocked by unresolved quarantine classes, ten source games without option data, missing frozen validation/test splits, an unvalidated training environment, and the missing base-model benchmark.
