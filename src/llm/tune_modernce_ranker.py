"""Tune CPU-only listwise heads on cached frozen ModernCE features."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .train_modernce_ranker import PROJECT_ROOT, atomic_write_json, cache_paths, iter_records


DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "processed" / "modernce_ranker"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "tuning"
ACTION_TYPES = ("ATTACK", "END_TURN", "HERO_POWER", "LOCATION", "PLAY", "POWER", "UNKNOWN")


def split_groups_by_game(
    groups: list[dict[str, Any]], *, dev_ratio: float = 0.15, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")
    game_ids = sorted({str(group["game_id"]) for group in groups})
    if len(game_ids) < 2:
        raise ValueError("At least two games are required for a game-level split")
    random.Random(seed).shuffle(game_ids)
    dev_count = max(1, min(len(game_ids) - 1, round(len(game_ids) * dev_ratio)))
    dev_games = set(game_ids[:dev_count])
    fit = [group for group in groups if str(group["game_id"]) not in dev_games]
    dev = [group for group in groups if str(group["game_id"]) in dev_games]
    return fit, dev


def sqrt_inverse_action_weights(
    groups: Iterable[dict[str, Any]], *, max_weight: float = 5.0
) -> dict[str, float]:
    groups = list(groups)
    counts = Counter(str(group["action_type"]) for group in groups)
    if not counts:
        raise ValueError("Cannot calculate weights for an empty group list")
    total = sum(counts.values())
    raw = {
        action_type: math.sqrt(total / (len(counts) * count))
        for action_type, count in counts.items()
    }
    weighted_mean = sum(counts[action_type] * weight for action_type, weight in raw.items()) / total
    normalized = {action_type: weight / weighted_mean for action_type, weight in raw.items()}
    clipped = {action_type: min(weight, max_weight) for action_type, weight in normalized.items()}
    clipped_mean = sum(counts[action_type] * weight for action_type, weight in clipped.items()) / total
    return {action_type: weight / clipped_mean for action_type, weight in clipped.items()}


def tempered_action_weights(
    groups: Iterable[dict[str, Any]], *, exponent: float, max_weight: float = 5.0
) -> dict[str, float]:
    if not 0.0 < exponent <= 1.0:
        raise ValueError("exponent must be in (0, 1]")
    groups = list(groups)
    counts = Counter(str(group["action_type"]) for group in groups)
    base = sqrt_inverse_action_weights(groups, max_weight=max_weight)
    tempered = {action_type: weight**exponent for action_type, weight in base.items()}
    total = sum(counts.values())
    weighted_mean = (
        sum(counts[action_type] * weight for action_type, weight in tempered.items()) / total
    )
    return {action_type: weight / weighted_mean for action_type, weight in tempered.items()}


def assess_candidate_gates(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    ndcg_ci_lower: float,
    play_retention: float = 0.8,
    rare_action_floor: float = 0.1,
    ndcg_noninferiority_margin: float = 0.01,
) -> dict[str, Any]:
    def action_accuracy(metrics: dict[str, Any], action_type: str) -> float:
        return float(
            metrics.get("by_action_type", {})
            .get(action_type, {})
            .get("top1_accuracy_n_gt_1", 0.0)
        )

    baseline_play = action_accuracy(baseline_metrics, "PLAY")
    candidate_play = action_accuracy(candidate_metrics, "PLAY")
    end_turn = action_accuracy(candidate_metrics, "END_TURN")
    hero_power = action_accuracy(candidate_metrics, "HERO_POWER")
    thresholds = {
        "play_top1_n_gt_1": baseline_play * play_retention,
        "end_turn_top1_n_gt_1": rare_action_floor,
        "hero_power_top1_n_gt_1": rare_action_floor,
        "ndcg_at_5_ci_lower": -ndcg_noninferiority_margin,
    }
    observed = {
        "play_top1_n_gt_1": candidate_play,
        "end_turn_top1_n_gt_1": end_turn,
        "hero_power_top1_n_gt_1": hero_power,
        "ndcg_at_5_ci_lower": ndcg_ci_lower,
    }
    checks = {
        "play_retention": candidate_play >= thresholds["play_top1_n_gt_1"],
        "end_turn_floor": end_turn >= rare_action_floor,
        "hero_power_floor": hero_power >= rare_action_floor,
        "ndcg_noninferiority": ndcg_ci_lower >= -ndcg_noninferiority_margin,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "observed": observed,
    }


def candidate_type_ids(
    dataset_path: Path | str, groups: list[dict[str, Any]]
) -> list[int]:
    type_to_id = {action_type: index for index, action_type in enumerate(ACTION_TYPES)}
    result: list[int] = []
    records = iter_records(dataset_path)
    for group_index, (group, record) in enumerate(zip(groups, records, strict=True)):
        if str(record.get("decision_id", "")) != str(group["decision_id"]):
            raise ValueError(f"Decision order mismatch at group {group_index}")
        candidates = list(record.get("candidates", []))
        if len(candidates) != int(group["count"]):
            raise ValueError(f"Candidate count mismatch at {group['decision_id']}")
        for candidate in candidates:
            action_type = str(candidate.get("type") or candidate.get("option_type") or "UNKNOWN")
            result.append(type_to_id.get(action_type, type_to_id["UNKNOWN"]))
    expected_pairs = sum(int(group["count"]) for group in groups)
    if len(result) != expected_pairs:
        raise ValueError(f"Candidate type count {len(result)} != expected {expected_pairs}")
    return result


def action_counts(groups: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(group["action_type"]) for group in groups).items()))


def feature_batch(
    features,
    type_ids: list[int],
    start: int,
    stop: int,
    *,
    use_type_features: bool,
    torch,
):
    batch = torch.from_numpy(features[start:stop].astype("float32"))
    if not use_type_features:
        return batch
    ids = torch.tensor(type_ids[start:stop], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(ids, num_classes=len(ACTION_TYPES)).to(dtype=batch.dtype)
    return torch.cat((batch, one_hot), dim=1)


def build_head(kind: str, input_dim: int, torch):
    if kind == "linear":
        return torch.nn.Linear(input_dim, 1)
    if kind == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1),
        )
    raise ValueError(f"Unknown head kind: {kind}")


def summarize_rows(rows: list[dict[str, Any]], min_action_support: int = 10) -> dict[str, Any]:
    if not rows:
        raise ValueError("No prediction rows")

    def accuracy(items: list[dict[str, Any]], field: str) -> float:
        return sum(bool(item[field]) for item in items) / len(items) if items else 0.0

    multi = [row for row in rows if row["count"] > 1]
    expected_random_top1 = statistics.mean(1.0 / row["count"] for row in rows)
    expected_random_top1_multi = statistics.mean(1.0 / row["count"] for row in multi)
    expected_random_ndcg_at_5 = statistics.mean(
        sum(1.0 / math.log2(rank + 1) for rank in range(1, min(row["count"], 5) + 1))
        / row["count"]
        for row in rows
    )
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_game_ndcg: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_action[row["gold_action_type"]].append(row)
        confusion[row["gold_action_type"]][row["predicted_action_type"]] += 1
        by_game_ndcg[row["game_id"]].append(row["ndcg_at_5"])

    action_metrics = {}
    macro_values = []
    for action_type, action_rows in sorted(by_action.items()):
        action_multi = [row for row in action_rows if row["count"] > 1]
        multi_accuracy = accuracy(action_multi, "top1")
        action_metrics[action_type] = {
            "total": len(action_rows),
            "n_gt_1": len(action_multi),
            "top1_accuracy": accuracy(action_rows, "top1"),
            "top1_accuracy_n_gt_1": multi_accuracy,
        }
        if len(action_multi) >= min_action_support:
            macro_values.append(multi_accuracy)

    macro = statistics.mean(macro_values) if macro_values else 0.0
    top1_multi = accuracy(multi, "top1")
    return {
        "decisions": len(rows),
        "n_gt_1": len(multi),
        "top1_accuracy": accuracy(rows, "top1"),
        "top1_accuracy_n_gt_1": top1_multi,
        "top3_accuracy": accuracy(rows, "top3"),
        "ndcg_at_5": statistics.mean(row["ndcg_at_5"] for row in rows),
        "mrr": statistics.mean(row["reciprocal_rank"] for row in rows),
        "expected_random_top1": expected_random_top1,
        "expected_random_top1_n_gt_1": expected_random_top1_multi,
        "expected_random_ndcg_at_5": expected_random_ndcg_at_5,
        "macro_action_top1_n_gt_1": macro,
        "selection_score": statistics.mean(row["ndcg_at_5"] for row in rows),
        "selection_formula": "mean NDCG@5 over decisions",
        "per_game_ndcg_at_5": {
            game_id: statistics.mean(values) for game_id, values in sorted(by_game_ndcg.items())
        },
        "by_action_type": action_metrics,
        "predicted_action_counts": dict(sorted(Counter(row["predicted_action_type"] for row in rows).items())),
        "confusion": {
            gold: dict(sorted(predicted.items())) for gold, predicted in sorted(confusion.items())
        },
    }


def evaluate_model(
    model,
    features,
    groups: list[dict[str, Any]],
    type_ids: list[int],
    *,
    use_type_features: bool,
    torch,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for group in groups:
            start = int(group["offset"])
            stop = start + int(group["count"])
            batch = feature_batch(
                features,
                type_ids,
                start,
                stop,
                use_type_features=use_type_features,
                torch=torch,
            )
            scores = model(batch).squeeze(-1)
            target = torch.tensor([int(group["gold_position"])])
            losses.append(float(torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)))
            ranking = torch.argsort(scores, descending=True).tolist()
            predicted_position = int(ranking[0])
            gold_rank = ranking.index(int(group["gold_position"])) + 1
            rows.append(
                {
                    "decision_id": group["decision_id"],
                    "game_id": group["game_id"],
                    "count": int(group["count"]),
                    "top1": predicted_position == int(group["gold_position"]),
                    "top3": int(group["gold_position"]) in ranking[:3],
                    "gold_rank": gold_rank,
                    "ndcg_at_5": 1.0 / math.log2(gold_rank + 1) if gold_rank <= 5 else 0.0,
                    "reciprocal_rank": 1.0 / gold_rank,
                    "gold_action_type": str(group["action_type"]),
                    "predicted_action_type": ACTION_TYPES[type_ids[start + predicted_position]],
                }
            )
    return statistics.mean(losses), summarize_rows(rows)


def paired_game_bootstrap_ci(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    seed: int = 42,
    samples: int = 2000,
) -> tuple[float, float, float]:
    game_ids = sorted(set(baseline) & set(candidate))
    if not game_ids:
        raise ValueError("No shared games for paired bootstrap")
    differences = [candidate[game_id] - baseline[game_id] for game_id in game_ids]
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(rng.choice(differences) for _ in differences) for _ in range(samples)
    )
    lower = estimates[max(0, math.floor(0.025 * samples))]
    upper = estimates[min(samples - 1, math.ceil(0.975 * samples) - 1)]
    return statistics.mean(differences), lower, upper


def train_with_dev(
    *,
    name: str,
    kind: str,
    balanced: bool,
    weight_exponent: float,
    use_type_features: bool,
    features,
    fit_groups: list[dict[str, Any]],
    dev_groups: list[dict[str, Any]],
    type_ids: list[int],
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    group_batch_size: int,
    seed: int,
    torch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    input_dim = hidden_size + (len(ACTION_TYPES) if use_type_features else 0)
    model = build_head(kind, input_dim, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    train_groups = [group for group in fit_groups if int(group["count"]) > 1]
    weights = (
        tempered_action_weights(train_groups, exponent=weight_exponent) if balanced else {}
    )
    best_score = -1.0
    best_state = None
    best_epoch = 0
    best_metrics = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_groups)))
        random.Random(seed + epoch).shuffle(order)
        epoch_losses = []
        for offset in range(0, len(order), group_batch_size):
            optimizer.zero_grad(set_to_none=True)
            weighted_losses = []
            batch_weights = []
            for group_index in order[offset : offset + group_batch_size]:
                group = train_groups[group_index]
                start = int(group["offset"])
                stop = start + int(group["count"])
                batch = feature_batch(
                    features,
                    type_ids,
                    start,
                    stop,
                    use_type_features=use_type_features,
                    torch=torch,
                )
                scores = model(batch).squeeze(-1)
                target = torch.tensor([int(group["gold_position"])])
                item_loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
                item_weight = weights.get(str(group["action_type"]), 1.0)
                weighted_losses.append(item_loss * item_weight)
                batch_weights.append(item_weight)
            loss = torch.stack(weighted_losses).sum() / sum(batch_weights)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        dev_loss, metrics = evaluate_model(
            model,
            features,
            dev_groups,
            type_ids,
            use_type_features=use_type_features,
            torch=torch,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": statistics.mean(epoch_losses),
                "dev_loss": dev_loss,
                "dev_metrics": metrics,
            }
        )
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = metrics
        print(
            f"[{name}/seed={seed}] epoch={epoch}/{epochs} "
            f"dev_n_gt_1={metrics['top1_accuracy_n_gt_1']:.4f} "
            f"macro={metrics['macro_action_top1_n_gt_1']:.4f}",
            flush=True,
        )

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"No checkpoint produced for {name}")
    report = {
        "name": name,
        "kind": kind,
        "balanced": balanced,
        "weight_exponent": weight_exponent,
        "use_type_features": use_type_features,
        "input_dim": input_dim,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev_metrics": best_metrics,
        "action_weights": weights,
        "history": history,
    }
    return best_state, report


def train_fixed_epochs(
    *,
    kind: str,
    balanced: bool,
    weight_exponent: float,
    use_type_features: bool,
    features,
    groups: list[dict[str, Any]],
    type_ids: list[int],
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    group_batch_size: int,
    seed: int,
    torch,
):
    random.seed(seed)
    torch.manual_seed(seed)
    input_dim = hidden_size + (len(ACTION_TYPES) if use_type_features else 0)
    model = build_head(kind, input_dim, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    train_groups = [group for group in groups if int(group["count"]) > 1]
    weights = (
        tempered_action_weights(train_groups, exponent=weight_exponent) if balanced else {}
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_groups)))
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for offset in range(0, len(order), group_batch_size):
            optimizer.zero_grad(set_to_none=True)
            weighted_losses = []
            batch_weights = []
            for group_index in order[offset : offset + group_batch_size]:
                group = train_groups[group_index]
                start = int(group["offset"])
                stop = start + int(group["count"])
                batch = feature_batch(
                    features,
                    type_ids,
                    start,
                    stop,
                    use_type_features=use_type_features,
                    torch=torch,
                )
                scores = model(batch).squeeze(-1)
                target = torch.tensor([int(group["gold_position"])])
                item_loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
                item_weight = weights.get(str(group["action_type"]), 1.0)
                weighted_losses.append(item_loss * item_weight)
                batch_weights.append(item_weight)
            loss = torch.stack(weighted_losses).sum() / sum(batch_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = statistics.mean(losses)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"[final-fit] epoch={epoch}/{epochs} train_loss={epoch_loss:.4f}", flush=True)
    return model, history, weights


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--group-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")

    os.environ.update(
        {
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
            "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
        }
    )
    import numpy as np
    import torch

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(min(2, args.cpu_threads))
    except RuntimeError:
        pass

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = run_dir / "cache"
    train_metadata = load_json(cache_dir / "train.metadata.json")
    validation_metadata = load_json(cache_dir / "validation.metadata.json")
    for split in ("train", "validation"):
        checkpoint = load_json(cache_dir / f"{split}.checkpoint.json")
        if not checkpoint.get("complete"):
            raise ValueError(f"{split} feature cache is incomplete")

    hidden_size = int(train_metadata["hidden_size"])
    train_features = np.memmap(
        cache_paths(cache_dir, "train")["features"],
        dtype=np.float16,
        mode="r",
        shape=(train_metadata["pairs"], hidden_size),
    )
    validation_features = np.memmap(
        cache_paths(cache_dir, "validation")["features"],
        dtype=np.float16,
        mode="r",
        shape=(validation_metadata["pairs"], hidden_size),
    )
    train_type_ids = candidate_type_ids(train_metadata["dataset"], train_metadata["groups"])
    validation_type_ids = candidate_type_ids(
        validation_metadata["dataset"], validation_metadata["groups"]
    )
    fit_groups, dev_groups = split_groups_by_game(
        train_metadata["groups"], dev_ratio=args.dev_ratio, seed=args.seed
    )
    print(
        json.dumps(
            {
                "fit_decisions": len(fit_groups),
                "dev_decisions": len(dev_groups),
                "fit_games": len({group["game_id"] for group in fit_groups}),
                "dev_games": len({group["game_id"] for group in dev_groups}),
                "fit_action_counts": action_counts(fit_groups),
                "dev_action_counts": action_counts(dev_groups),
            },
            indent=2,
        ),
        flush=True,
    )

    variants = [
        {
            "name": "linear_unweighted",
            "kind": "linear",
            "balanced": False,
            "weight_exponent": 0.0,
            "use_type_features": False,
        }
    ]
    for exponent in (0.25, 0.5, 0.75):
        suffix = str(exponent).replace(".", "")
        variants.extend(
            [
                {
                    "name": f"linear_tempered_{suffix}",
                    "kind": "linear",
                    "balanced": True,
                    "weight_exponent": exponent,
                    "use_type_features": False,
                },
                {
                    "name": f"linear_tempered_type_{suffix}",
                    "kind": "linear",
                    "balanced": True,
                    "weight_exponent": exponent,
                    "use_type_features": True,
                },
            ]
        )
    candidate_runs = []
    started = time.monotonic()
    for variant in variants:
        state, report = train_with_dev(
            **variant,
            features=train_features,
            fit_groups=fit_groups,
            dev_groups=dev_groups,
            type_ids=train_type_ids,
            hidden_size=hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            group_batch_size=args.group_batch_size,
            seed=args.seed,
            torch=torch,
        )
        candidate_runs.append(report)
        del state

    baseline_run = next(run for run in candidate_runs if run["name"] == "linear_unweighted")
    baseline_games = baseline_run["best_dev_metrics"]["per_game_ndcg_at_5"]
    for run in candidate_runs:
        delta, lower, upper = paired_game_bootstrap_ci(
            baseline_games,
            run["best_dev_metrics"]["per_game_ndcg_at_5"],
            seed=args.seed,
        )
        run["ndcg_at_5_delta_vs_linear"] = delta
        run["paired_game_bootstrap_95_ci"] = [lower, upper]
        run["clears_linear_baseline"] = run["name"] == "linear_unweighted" or lower > 0.0
        run["selection_gates"] = (
            {"eligible": True, "fallback_baseline": True}
            if run["name"] == "linear_unweighted"
            else assess_candidate_gates(
                run["best_dev_metrics"],
                baseline_run["best_dev_metrics"],
                ndcg_ci_lower=lower,
            )
        )
    eligible_runs = [run for run in candidate_runs if run["selection_gates"]["eligible"]]
    selected = max(
        eligible_runs,
        key=lambda run: (
            run["best_dev_metrics"]["top1_accuracy_n_gt_1"],
            run["best_dev_metrics"]["selection_score"],
        ),
    )
    selected_config = next(variant for variant in variants if variant["name"] == selected["name"])
    robustness_runs = [selected]
    for seed in (41, 43):
        _, report = train_with_dev(
            **selected_config,
            features=train_features,
            fit_groups=fit_groups,
            dev_groups=dev_groups,
            type_ids=train_type_ids,
            hidden_size=hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            group_batch_size=args.group_batch_size,
            seed=seed,
            torch=torch,
        )
        robustness_runs.append(report)

    final_epochs = int(statistics.median(run["best_epoch"] for run in robustness_runs))
    final_model, final_history, final_weights = train_fixed_epochs(
        kind=selected_config["kind"],
        balanced=selected_config["balanced"],
        weight_exponent=selected_config["weight_exponent"],
        use_type_features=selected_config["use_type_features"],
        features=train_features,
        groups=train_metadata["groups"],
        type_ids=train_type_ids,
        hidden_size=hidden_size,
        epochs=final_epochs,
        learning_rate=args.learning_rate,
        group_batch_size=args.group_batch_size,
        seed=args.seed,
        torch=torch,
    )
    validation_loss, validation_metrics = evaluate_model(
        final_model,
        validation_features,
        validation_metadata["groups"],
        validation_type_ids,
        use_type_features=selected_config["use_type_features"],
        torch=torch,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "head.pt"
    temporary_artifact = artifact_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "variant": selected_config,
            "hidden_size": hidden_size,
            "action_types": ACTION_TYPES,
            "epochs": final_epochs,
            "seed": args.seed,
            "validation_metrics": validation_metrics,
        },
        temporary_artifact,
    )
    os.replace(temporary_artifact, artifact_path)

    robustness_scores = [run["best_dev_metrics"]["selection_score"] for run in robustness_runs]
    report = {
        "method": "cpu_only_cached_feature_head_tuning",
        "selection_scope": "game-level internal dev split from train; validation evaluated once after refit",
        "selection_metric": "highest multi-candidate top-1 among gated candidates; PLAY >= 80% of baseline, END_TURN/HERO_POWER >= 10%, and paired game-bootstrap NDCG@5 lower CI >= -0.01",
        "cpu_threads": args.cpu_threads,
        "train_dataset": train_metadata["dataset"],
        "validation_dataset": validation_metadata["dataset"],
        "hidden_size": hidden_size,
        "action_types": ACTION_TYPES,
        "split": {
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "fit_decisions": len(fit_groups),
            "dev_decisions": len(dev_groups),
            "fit_games": len({group["game_id"] for group in fit_groups}),
            "dev_games": len({group["game_id"] for group in dev_groups}),
            "fit_action_counts": action_counts(fit_groups),
            "dev_action_counts": action_counts(dev_groups),
        },
        "candidate_runs": candidate_runs,
        "selected_variant": selected_config,
        "robustness_runs": robustness_runs,
        "robustness_selection_score_mean": statistics.mean(robustness_scores),
        "robustness_selection_score_stdev": statistics.pstdev(robustness_scores),
        "final_epochs": final_epochs,
        "final_action_weights": final_weights,
        "final_train_history": final_history,
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "artifact": str(artifact_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "selected_variant": selected_config,
                "final_epochs": final_epochs,
                "robustness_selection_score_mean": report["robustness_selection_score_mean"],
                "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
                "artifact": str(artifact_path),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
