"""Compare reproducible checkpoint ensembles on the labelled validation split only."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import numpy as np

from data import load_selected
from development_data import remove_labels
from evaluate import evaluate
from make_submission import (
    score_unlabelled_rows,
    within_user_rank_average,
    within_user_rank_weighted_average,
)


GROUP_WEIGHT_CANDIDATES = (
    (5.0, 3.0, 1.0),  # current equal-checkpoint ensemble
    (4.0, 4.0, 1.0),
    (4.0, 3.0, 2.0),
    (5.0, 4.0, 1.0),
    (5.0, 3.0, 2.0),
    (6.0, 3.0, 1.0),
)


def checkpoint_weights_from_group_totals(feature_sets, group_totals):
    """Distribute fixed base/hour/session mass equally inside each model family."""
    group_names = ("base", "hour", "session")
    if len(group_totals) != len(group_names):
        raise ValueError("Expected base, hour, and session group totals")
    counts = {name: feature_sets.count(name) for name in group_names}
    if any(counts[name] == 0 for name in group_names):
        raise ValueError("Weighted temporal ensemble requires every model family")
    totals = dict(zip(group_names, group_totals))
    return np.asarray(
        [totals[feature_set] / counts[feature_set] for feature_set in feature_sets],
        dtype=np.float64,
    )


def within_user_zscore_average(score_arrays, users):
    """Average zero-mean, unit-variance scores independently within each user."""
    if not score_arrays:
        raise ValueError("At least one score array is required")
    expected = len(users)
    if any(len(scores) != expected for scores in score_arrays):
        raise ValueError("Every score array must align with the target rows")
    groups = collections.defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    combined = np.zeros(expected, dtype=np.float32)
    for scores in score_arrays:
        values = np.asarray(scores)
        normalized = np.empty(expected, dtype=np.float32)
        for indices in groups.values():
            group = np.asarray(indices, dtype=np.int64)
            group_scores = values[group]
            normalized[group] = (group_scores - np.mean(group_scores)) / max(
                float(np.std(group_scores)), 1e-6
            )
        combined += normalized / len(score_arrays)
    return combined


def train_history_scores(train_rows, target_rows, mode):
    """Build smoothed item or personalized history logits from train labels only."""
    global_positive = sum(row[6] for row in train_rows)
    global_rate = global_positive / max(1, len(train_rows))
    stats = {
        "video": collections.defaultdict(lambda: [0, 0]),
        "author": collections.defaultdict(lambda: [0, 0]),
        "user_video": collections.defaultdict(lambda: [0, 0]),
        "user_author": collections.defaultdict(lambda: [0, 0]),
    }
    for row in train_rows:
        label = int(row[6])
        for key, value in (
            ("video", row[2]),
            ("author", row[3]),
            ("user_video", (row[1], row[2])),
            ("user_author", (row[1], row[3])),
        ):
            stats[key][value][0] += label
            stats[key][value][1] += 1

    scores = np.empty(len(target_rows), dtype=np.float32)
    for index, row in enumerate(target_rows):
        author_positive, author_count = stats["author"][row[3]]
        author_rate = (author_positive + 30.0 * global_rate) / (author_count + 30.0)
        video_positive, video_count = stats["video"][row[2]]
        video_rate = (video_positive + 20.0 * author_rate) / (video_count + 20.0)
        if mode == "video":
            rate = video_rate
        else:
            affinity_positive, affinity_count = stats["user_author"][(row[1], row[3])]
            affinity_rate = (affinity_positive + 5.0 * author_rate) / (
                affinity_count + 5.0
            )
            history_positive, history_count = stats["user_video"][(row[1], row[2])]
            rate = (history_positive + 2.0 * (video_rate + affinity_rate) / 2.0) / (
                history_count + 2.0
            )
        rate = float(np.clip(rate, 1e-5, 1.0 - 1e-5))
        scores[index] = np.log(rate / (1.0 - rate))
    return scores


def within_user_blend(base_scores, extra_scores, users, alpha):
    groups = collections.defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    combined = np.empty(len(users), dtype=np.float32)
    for indices in groups.values():
        group = np.asarray(indices, dtype=np.int64)
        base = np.asarray(base_scores)[group]
        extra = np.asarray(extra_scores)[group]
        base = (base - np.mean(base)) / max(float(np.std(base)), 1e-6)
        extra = (extra - np.mean(extra)) / max(float(np.std(extra)), 1e-6)
        combined[group] = base + alpha * extra
    return combined


def validation_ensemble_results(checkpoints, data_dir, full_only=False):
    splits = load_selected(data_dir, ["train", "valid"])
    train_rows = splits["train"]
    valid_rows = splits["valid"]
    unlabelled_valid_rows = remove_labels(valid_rows)
    users = [row[1] for row in valid_rows]
    labels = np.asarray([row[6] for row in valid_rows], dtype=np.float32)
    scored = [
        score_unlabelled_rows(checkpoint, train_rows, unlabelled_valid_rows)
        for checkpoint in checkpoints
    ]
    score_arrays = [item[0] for item in scored]
    feature_sets = [item[1]["feature_set"] for item in scored]
    results = []
    sizes = (len(checkpoints),) if full_only else range(1, len(checkpoints) + 1)
    for size in sizes:
        for members in itertools.combinations(range(len(checkpoints)), size):
            selected = [score_arrays[index] for index in members]
            aggregations = {
                "rank_average": within_user_rank_average(selected, users),
                "zscore_average": within_user_zscore_average(selected, users),
            }
            for aggregation, scores in aggregations.items():
                metrics = evaluate(users, labels, scores)
                results.append(
                    {
                        "aggregation": aggregation,
                        "members": list(members),
                        "checkpoints": [str(checkpoints[index]) for index in members],
                        "metrics": {
                            "GAUC": float(metrics["GAUC"]),
                            "nDCG@5": float(metrics["nDCG@5"]),
                            "primary": float(metrics["primary"]),
                        },
                    }
                )
    full_rank_scores = within_user_rank_average(score_arrays, users)
    if set(feature_sets) == {"base", "hour", "session"}:
        for group_totals in GROUP_WEIGHT_CANDIDATES:
            weights = checkpoint_weights_from_group_totals(feature_sets, group_totals)
            scores = within_user_rank_weighted_average(score_arrays, users, weights)
            metrics = evaluate(users, labels, scores)
            results.append(
                {
                    "aggregation": "fixed_group_weighted_rank_average",
                    "members": list(range(len(checkpoints))),
                    "checkpoints": [str(checkpoint) for checkpoint in checkpoints],
                    "feature_sets": feature_sets,
                    "group_totals": dict(
                        zip(("base", "hour", "session"), group_totals)
                    ),
                    "weights": weights.tolist(),
                    "metrics": {
                        "GAUC": float(metrics["GAUC"]),
                        "nDCG@5": float(metrics["nDCG@5"]),
                        "primary": float(metrics["primary"]),
                    },
                }
            )
    for mode in ("video", "personalized"):
        history_scores = train_history_scores(train_rows, valid_rows, mode)
        for alpha in (0.025, 0.05, 0.1, 0.2, 0.4):
            scores = within_user_blend(full_rank_scores, history_scores, users, alpha)
            metrics = evaluate(users, labels, scores)
            results.append(
                {
                    "aggregation": f"rank_average_plus_{mode}_history",
                    "members": list(range(len(checkpoints))),
                    "checkpoints": [str(checkpoint) for checkpoint in checkpoints],
                    "alpha": alpha,
                    "metrics": {
                        "GAUC": float(metrics["GAUC"]),
                        "nDCG@5": float(metrics["nDCG@5"]),
                        "primary": float(metrics["primary"]),
                    },
                }
            )
    return sorted(results, key=lambda result: result["metrics"]["primary"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Evaluate only the complete checkpoint ensemble, avoiding subset selection.",
    )
    args = parser.parse_args()
    checkpoints = [Path(value) for value in args.checkpoint]
    results = validation_ensemble_results(
        checkpoints, args.data_dir, full_only=args.full_only
    )
    print(
        json.dumps(
            {
                "status": "success",
                "split": "valid",
                "hidden_test_accessed": False,
                "candidates_evaluated": len(results),
                "top_results": results[: args.top],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
