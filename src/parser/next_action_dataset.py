"""Build a replay-option-grounded state + candidates -> next action dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.card_db import CardDatabase

from .dataset_generator import DEFAULT_OUTPUT_FILE as TURN_DATASET_FILE
from .replay_reader import DEFAULT_REPLAY_DIR, iterate_replays
from .state_tracker import OptionDecision, ReplayOptionCandidate


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
DEFAULT_DATASET_FILE = DEFAULT_OUTPUT_DIR / "train_next_actions.jsonl"
DEFAULT_QUARANTINE_FILE = DEFAULT_OUTPUT_DIR / "train_next_actions_quarantine.jsonl"
DEFAULT_REPORT_FILE = DEFAULT_OUTPUT_DIR / "next_action_validation_report.json"

UNKNOWN_NAMES = {
    "attacker",
    "target",
    "unknown card",
    "unknown entity",
    "неизвестная карта",
}


@dataclass(frozen=True)
class OptionValidation:
    accepted: bool
    reason: str = ""
    chosen_candidate_id: Optional[int] = None
    match_count: int = 0


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_unknown(name: Optional[str]) -> bool:
    normalized = (name or "").casefold()
    return not normalized or normalized in UNKNOWN_NAMES or normalized.startswith("unknown entity")


def _candidate_matches_selection(decision: OptionDecision, candidate: ReplayOptionCandidate) -> bool:
    return (
        candidate.option_id == decision.selected_option
        and candidate.sub_option_id == decision.selected_sub_option
        and candidate.target_entity_id == decision.selected_target
        and candidate.position == decision.selected_position
    )


def _candidate_is_resolved(candidate: ReplayOptionCandidate) -> bool:
    if candidate.action_type == "END_TURN":
        return True
    if not candidate.entity_id or _is_unknown(candidate.entity_name) or not candidate.entity_card_id:
        return False
    if candidate.target_entity_id and (_is_unknown(candidate.target_name) or not candidate.target_card_id):
        return False
    if candidate.sub_option_id >= 0:
        return bool(
            candidate.sub_entity_id
            and not _is_unknown(candidate.sub_entity_name)
            and candidate.sub_entity_card_id
        )
    return True


def validate_option_decision(decision: OptionDecision) -> OptionValidation:
    """Admits only selections proven legal by the replay's latest option set."""
    if not decision.snapshot.is_friendly_turn:
        return OptionValidation(False, "not_friendly_turn")
    if not decision.candidates:
        return OptionValidation(False, "no_legal_candidates")
    if any(
        candidate.sub_option_id >= 0 and candidate.target_entity_id
        for candidate in decision.candidates
    ):
        return OptionValidation(False, "suboption_target_cross_product_unproven")
    if any(not _candidate_is_resolved(candidate) for candidate in decision.candidates):
        return OptionValidation(False, "unresolved_legal_candidate")
    if any(candidate.is_tradeable for candidate in decision.candidates):
        return OptionValidation(False, "tradeable_option_semantics_unproven")

    signatures = {
        (candidate.option_id, candidate.sub_option_id, candidate.target_entity_id, candidate.position)
        for candidate in decision.candidates
    }
    if len(signatures) != len(decision.candidates):
        return OptionValidation(False, "duplicate_legal_candidate")
    if any(candidate.mana_cost > decision.snapshot.friendly_mana for candidate in decision.candidates):
        return OptionValidation(False, "candidate_mana_cost_mismatch")

    matches = [candidate for candidate in decision.candidates if _candidate_matches_selection(decision, candidate)]
    if not matches:
        without_position = [
            candidate
            for candidate in decision.candidates
            if candidate.option_id == decision.selected_option
            and candidate.sub_option_id == decision.selected_sub_option
            and candidate.target_entity_id == decision.selected_target
        ]
        if without_position:
            return OptionValidation(False, "selected_position_outside_derived_range")
        return OptionValidation(False, "selected_option_not_legal")
    if len(matches) != 1:
        return OptionValidation(False, "ambiguous_selected_candidate", match_count=len(matches))

    chosen = matches[0]
    if chosen.action_type != "END_TURN" and chosen.controller_id != decision.snapshot.active_player_id:
        return OptionValidation(False, "owner_mismatch", match_count=1)
    return OptionValidation(True, chosen_candidate_id=chosen.candidate_id, match_count=1)


def _state_dict(decision: OptionDecision) -> Dict[str, Any]:
    snapshot = decision.snapshot
    return {
        "turn": snapshot.turn_number,
        "active_player_id": snapshot.active_player_id,
        "mana": snapshot.friendly_mana,
        "max_mana": snapshot.friendly_max_mana,
        "friendly_hero": snapshot.friendly_hero,
        "opponent_hero": snapshot.opponent_hero,
        "hand": snapshot.friendly_hand,
        "friendly_board": snapshot.friendly_board,
        "opponent_board": snapshot.opponent_board,
        "hero_power": snapshot.hero_power,
        "friendly_locations": snapshot.friendly_locations,
        "opponent_locations": snapshot.opponent_locations,
        "friendly_secrets": snapshot.friendly_secrets,
        "opponent_secrets_count": snapshot.opponent_secrets_count,
        "opponent_hand_count": snapshot.opponent_hand_count,
    }


def _candidate_dict(candidate: ReplayOptionCandidate) -> Dict[str, Any]:
    legality = "power_log_end_turn" if candidate.action_type == "END_TURN" else "power_log_error_none"
    return {
        "id": candidate.candidate_id,
        "option_id": candidate.option_id,
        "option_type": candidate.option_type,
        "type": candidate.action_type,
        "entity_id": candidate.entity_id,
        "card_id": candidate.entity_card_id or None,
        "card_type": candidate.entity_card_type or None,
        "controller_id": candidate.controller_id or None,
        "target_id": candidate.target_entity_id,
        "target_card_id": candidate.target_card_id or None,
        "sub_option_id": candidate.sub_option_id,
        "sub_entity_id": candidate.sub_entity_id,
        "sub_card_id": candidate.sub_entity_card_id or None,
        "position": candidate.position,
        "mana_cost": candidate.mana_cost,
        "mana_cost_source": "tracked_entity_or_carddefs_hint",
        "is_tradeable": candidate.is_tradeable,
        "description": candidate.description,
        "legality": legality,
    }


def _gold_dict(decision: OptionDecision, chosen: Optional[ReplayOptionCandidate]) -> Dict[str, Any]:
    if chosen is None:
        return {
            "option_id": decision.selected_option,
            "sub_option_id": decision.selected_sub_option,
            "target_id": decision.selected_target,
            "position": decision.selected_position,
        }
    gold = _candidate_dict(chosen)
    gold.pop("id")
    gold.pop("legality")
    gold["position"] = decision.selected_position
    return gold


def _chosen_candidate(decision: OptionDecision) -> Optional[ReplayOptionCandidate]:
    matches = [candidate for candidate in decision.candidates if _candidate_matches_selection(decision, candidate)]
    return matches[0] if len(matches) == 1 else None


def _audit_written_dataset(path: Path) -> Dict[str, int]:
    """Independently verifies the serialized admission invariants."""
    violations: Counter[str] = Counter()
    seen_decisions = set()
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                violations["invalid_json"] += 1
                continue

            decision_id = record.get("decision_id")
            if decision_id in seen_decisions:
                violations["duplicate_decision_id"] += 1
            seen_decisions.add(decision_id)

            candidates = record.get("candidates", [])
            chosen_id = record.get("chosen_candidate_id")
            chosen_matches = [candidate for candidate in candidates if candidate.get("id") == chosen_id]
            if len(chosen_matches) != 1:
                violations["chosen_candidate_count"] += 1
                continue

            chosen = chosen_matches[0]
            gold = record.get("gold_action", {})
            state = record.get("state", {})
            for field in ("option_id", "sub_option_id", "target_id", "position"):
                if chosen.get(field) != gold.get(field):
                    violations[f"{field}_mismatch"] += 1
            if chosen.get("type") != gold.get("type"):
                violations["action_type_mismatch"] += 1
            if chosen.get("type") != "END_TURN" and chosen.get("controller_id") != state.get("active_player_id"):
                violations["owner_mismatch"] += 1
            if any(candidate.get("mana_cost", 0) > state.get("mana", 0) for candidate in candidates):
                violations["candidate_mana_violation"] += 1
            if any(candidate.get("is_tradeable") for candidate in candidates):
                violations["unproven_tradeable_option"] += 1
            if chosen.get("type") != "END_TURN" and (
                not chosen.get("entity_id") or not chosen.get("card_id") or _is_unknown(chosen.get("description"))
            ):
                violations["unresolved_action_entity"] += 1
            if any(
                candidate.get("legality") not in {"power_log_error_none", "power_log_end_turn"}
                for candidate in candidates
            ):
                violations["unproven_candidate"] += 1

            candidate_ids = [candidate.get("id") for candidate in candidates]
            signatures = [
                (
                    candidate.get("option_id"),
                    candidate.get("sub_option_id"),
                    candidate.get("target_id"),
                    candidate.get("position"),
                )
                for candidate in candidates
            ]
            if len(candidate_ids) != len(set(candidate_ids)):
                violations["duplicate_candidate_id"] += 1
            if len(signatures) != len(set(signatures)):
                violations["duplicate_candidate_signature"] += 1
    return dict(sorted(violations.items()))


def _carddefs_has_play_requirements(card_db: CardDatabase) -> bool:
    path = card_db.hdt_card_defs_dir / "CardDefs.base.xml"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        return any("<PlayRequirement" in line for line in source)


def _build_next_action_dataset_unlocked(
    output_path: Path | str = DEFAULT_DATASET_FILE,
    quarantine_path: Path | str = DEFAULT_QUARANTINE_FILE,
    report_path: Path | str = DEFAULT_REPORT_FILE,
    max_replays: Optional[int] = None,
) -> Dict[str, Any]:
    """Builds local-only Stage C artifacts and returns the validation report."""
    output = Path(output_path)
    quarantine = Path(quarantine_path)
    report_file = Path(report_path)
    card_db = CardDatabase(auto_load=True)

    reasons: Counter[str] = Counter()
    accepted_actions: Counter[str] = Counter()
    rejected_actions: Counter[str] = Counter()
    game_ids: Counter[str] = Counter()
    games = 0
    games_with_options = 0
    legacy_friendly_decisions = 0
    decisions = 0
    positioned_selections = 0
    accepted_count = 0
    rejected_count = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    temporary_token = uuid4().hex
    output_temporary = output.with_name(f"{output.name}.{temporary_token}.tmp")
    quarantine_temporary = quarantine.with_name(f"{quarantine.name}.{temporary_token}.tmp")
    accepted_gate_violations: Dict[str, int] = {}
    try:
        with output_temporary.open("w", encoding="utf-8", newline="\n") as accepted_file, quarantine_temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as quarantine_file:
            for replay in iterate_replays(
                filter_ranked_wins=True,
                max_count=max_replays,
                card_db=card_db,
            ):
                games += 1
                if replay.option_decisions:
                    games_with_options += 1
                legacy_friendly_decisions += sum(
                    1 for decision in replay.decision_points if decision.snapshot.is_friendly_turn
                )
                metadata = replay.metadata
                source_game_id = metadata.game_id or Path(metadata.replay_file).stem
                game_ids[source_game_id] += 1

                for decision in replay.option_decisions:
                    decisions += 1
                    if decision.selected_position > 0:
                        positioned_selections += 1
                    validation = validate_option_decision(decision)
                    chosen = _chosen_candidate(decision)
                    base = {
                        "schema_version": 2,
                        "game_id": source_game_id,
                        "decision_id": f"{source_game_id}:option:{decision.sequence:04d}",
                        "replay_file": metadata.replay_file,
                        "turn_number": decision.snapshot.turn_number,
                        "options_id": decision.options_id,
                        "state": _state_dict(decision),
                        "candidates": [_candidate_dict(candidate) for candidate in decision.candidates],
                        "gold_action": _gold_dict(decision, chosen),
                    }

                    action_type = chosen.action_type if chosen else "UNRESOLVED"
                    if validation.accepted:
                        base["chosen_candidate_id"] = validation.chosen_candidate_id
                        accepted_file.write(json.dumps(base, ensure_ascii=False, separators=(",", ":")) + "\n")
                        accepted_count += 1
                        accepted_actions[action_type] += 1
                    else:
                        base["reason"] = validation.reason
                        base["match_count"] = validation.match_count
                        quarantine_file.write(json.dumps(base, ensure_ascii=False, separators=(",", ":")) + "\n")
                        rejected_count += 1
                        reasons[validation.reason] += 1
                        rejected_actions[action_type] += 1

        accepted_gate_violations = _audit_written_dataset(output_temporary)
        if accepted_gate_violations:
            raise RuntimeError(f"Refusing to publish dataset with gate violations: {accepted_gate_violations}")
        output_temporary.replace(output)
        quarantine_temporary.replace(quarantine)
    finally:
        if output_temporary.exists():
            output_temporary.unlink()
        if quarantine_temporary.exists():
            quarantine_temporary.unlink()

    requirements_available = _carddefs_has_play_requirements(card_db)
    duplicate_game_ids = sorted(game_id for game_id, count in game_ids.items() if count > 1)
    blockers: List[str] = []
    if reasons:
        blockers.append("quarantined_option_decisions_present")
    if duplicate_game_ids:
        blockers.append("duplicate_game_ids_in_source")
    if accepted_gate_violations:
        blockers.append("accepted_dataset_gate_violation")
    if games_with_options != games:
        blockers.append("ranked_games_without_option_decisions")
    blockers.extend(
        [
            "validation_and_test_splits_not_frozen",
            "training_environment_not_revalidated",
            "base_model_benchmark_not_ready",
        ]
    )

    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "replay_directory": str(DEFAULT_REPLAY_DIR),
            "filter": "Ranked wins; local HDT only",
            "games": games,
            "games_with_option_decisions": games_with_options,
            "legality_source": "GameState.DebugPrintOptions error=NONE + GameState.SendOption selection",
            "turn_dataset_path": str(TURN_DATASET_FILE),
            "turn_dataset_sha256": _sha256(TURN_DATASET_FILE),
            "carddefs_base_sha256": _sha256(card_db.hdt_card_defs_dir / "CardDefs.base.xml"),
            "carddefs_play_requirements_available": requirements_available,
        },
        "decisions": {
            "legacy_friendly_action_and_end_turn_count": legacy_friendly_decisions,
            "option_selections_total": decisions,
            "accepted": accepted_count,
            "quarantined": rejected_count,
            "coverage": (accepted_count / decisions) if decisions else 0.0,
            "positioned_selections": positioned_selections,
            "accepted_action_counts": dict(sorted(accepted_actions.items())),
            "quarantined_action_counts": dict(sorted(rejected_actions.items())),
            "quarantine_reasons": dict(sorted(reasons.items())),
            "duplicate_game_ids": duplicate_game_ids,
            "accepted_gate_violations": accepted_gate_violations,
        },
        "artifacts": {
            "dataset": str(output),
            "dataset_sha256": _sha256(output),
            "quarantine": str(quarantine),
            "quarantine_sha256": _sha256(quarantine),
        },
        "stage_b": {
            "board_limit": 7,
            "stable_entity_ids": True,
            "board_position_in_candidate_id": True,
            "end_turn_candidate": True,
            "windfury_attack_budget": True,
            "target_legality_from_replay_options": True,
            "all_candidate_mana_gate": True,
            "tradeable_options_quarantined": True,
            "single_writer_lock": True,
            "report_manifest_published_last": True,
            "carddefs_target_requirements_complete": requirements_available,
        },
        "qlora_ready": False,
        "blockers": blockers,
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_file.with_name(f"{report_file.name}.{temporary_token}.tmp")
    try:
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_file)
    finally:
        if temporary_report.exists():
            temporary_report.unlink()

    print(
        f"Built {accepted_count} accepted and {rejected_count} quarantined option decisions "
        f"from {games} games -> {output}"
    )
    return report


def build_next_action_dataset(
    output_path: Path | str = DEFAULT_DATASET_FILE,
    quarantine_path: Path | str = DEFAULT_QUARANTINE_FILE,
    report_path: Path | str = DEFAULT_REPORT_FILE,
    max_replays: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs one isolated build; the report is published last as the artifact manifest."""
    artifact_paths = tuple(
        Path(path).resolve() for path in (output_path, quarantine_path, report_path)
    )
    normalized_paths = {os.path.normcase(str(path)) for path in artifact_paths}
    if len(normalized_paths) != len(artifact_paths):
        raise ValueError("Dataset, quarantine, and report paths must be distinct")

    for artifact_path in artifact_paths:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_paths = sorted(
        (path.with_suffix(path.suffix + ".lock") for path in artifact_paths),
        key=lambda path: os.path.normcase(str(path)),
    )
    normalized_lock_paths = {os.path.normcase(str(path)) for path in lock_paths}
    if normalized_paths & normalized_lock_paths:
        raise ValueError("An artifact path collides with another artifact's lock path")
    acquired_locks = []
    try:
        for lock_path in lock_paths:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            acquired_locks.append((lock_path, lock_fd))
            os.write(lock_fd, str(os.getpid()).encode("ascii"))
    except FileExistsError as exc:
        for acquired_path, acquired_fd in reversed(acquired_locks):
            os.close(acquired_fd)
            acquired_path.unlink(missing_ok=True)
        raise RuntimeError(f"Another build is active or left a stale lock: {lock_path}") from exc
    except Exception:
        for acquired_path, acquired_fd in reversed(acquired_locks):
            os.close(acquired_fd)
            acquired_path.unlink(missing_ok=True)
        raise

    try:
        return _build_next_action_dataset_unlocked(
            output_path=artifact_paths[0],
            quarantine_path=artifact_paths[1],
            report_path=artifact_paths[2],
            max_replays=max_replays,
        )
    finally:
        for lock_path, lock_fd in reversed(acquired_locks):
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the replay-option-grounded next-action dataset")
    parser.add_argument("--output", default=str(DEFAULT_DATASET_FILE))
    parser.add_argument("--quarantine", default=str(DEFAULT_QUARANTINE_FILE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_FILE))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_next_action_dataset(
        output_path=args.output,
        quarantine_path=args.quarantine,
        report_path=args.report,
        max_replays=args.limit,
    )
