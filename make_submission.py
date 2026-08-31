"""Generate checkpoint predictions without reading target labels."""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from baseline import FM
from candidate.model import FieldWeightedFM
from data import (
    FIELDS,
    HOUR_FIELDS,
    USER_AUTHOR_FIELDS,
    WEEKDAY_FIELDS,
    fit_feature_encoder,
    load_selected,
    load_unlabelled,
    transform_rows,
)

HEADER = ["row_id", "user_id", "video_id", "score"]
FEATURE_SETS = {
    "base": FIELDS,
    "weekday": WEEKDAY_FIELDS,
    "hour": HOUR_FIELDS,
    "user_author": USER_AUTHOR_FIELDS,
}


def score_unlabelled_rows(checkpoint_path, train_rows, target_rows):
    """Reconstruct the train-fitted encoder and score rows that contain no label."""
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"]))
        feature_set = metadata["feature_set"]
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unsupported final-inference feature set: {feature_set}")
        encoder = fit_feature_encoder(train_rows, fields=FEATURE_SETS[feature_set])
        X, labels, _ = transform_rows(target_rows, encoder, include_labels=False)
        if labels is not None:
            raise AssertionError("Final inference must not materialise target labels")
        V = checkpoint["V"]
        W = checkpoint["W"]
        b = np.float32(checkpoint["b"])
        if V.shape[0] != encoder["dimension"] or W.shape != (encoder["dimension"],):
            raise ValueError("Checkpoint dimensions do not match the train-fitted encoder")
        architecture = metadata.get("architecture", "fm")
        if architecture == "fm":
            model = FM(encoder["dimension"], k=V.shape[1], lr=metadata["lr"], seed=0)
        elif architecture == "field_weighted":
            model = FieldWeightedFM(
                encoder["dimension"],
                field_count=X.shape[1],
                k=V.shape[1],
                lr=metadata["lr"],
                seed=0,
            )
            if "field_pair_weights" not in checkpoint.files:
                raise ValueError("Field-weighted checkpoint is missing field_pair_weights")
            model.field_pair_weights = checkpoint["field_pair_weights"].copy()
        else:
            raise ValueError(f"Unsupported final-inference architecture: {architecture}")
        model.V, model.W, model.b = V.copy(), W.copy(), b
        scores = model.predict(X)
    if len(scores) != len(target_rows) or not np.isfinite(scores).all():
        raise ValueError("Final predictions must be finite and match the target row count")
    return scores, metadata


def within_user_rank_average(score_arrays, users):
    """Average normalized within-user ranks from independently trained models."""
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
        ranked = np.empty(expected, dtype=np.float32)
        for indices in groups.values():
            group = np.asarray(indices, dtype=np.int64)
            order = np.argsort(
                np.argsort(np.asarray(scores)[group], kind="stable"), kind="stable"
            )
            ranked[group] = order / max(1, len(group) - 1)
        combined += ranked / len(score_arrays)
    return combined


def write_submission(path, rows, scores):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row_id, (row, score) in enumerate(zip(rows, scores)):
            writer.writerow([row_id, row[1], row[2], f"{float(score):.9g}"])
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat for rank-averaged checkpoint ensembling.",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="Defaults to validation; choose test explicitly only for the final submission.",
    )
    args = parser.parse_args()
    train_rows = load_selected(args.data_dir, ["train"])["train"]
    target_rows = load_unlabelled(args.data_dir, args.split)
    scored = [
        score_unlabelled_rows(checkpoint, train_rows, target_rows)
        for checkpoint in args.checkpoint
    ]
    score_arrays = [item[0] for item in scored]
    metadata = scored[0][1]
    feature_sets = [item[1]["feature_set"] for item in scored]
    scores = (
        score_arrays[0]
        if len(score_arrays) == 1
        else within_user_rank_average(
            score_arrays, [row[1] for row in target_rows]
        )
    )
    output = write_submission(args.output, target_rows, scores)
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(output),
                "rows": len(target_rows),
                "split": args.split,
                "checkpoints": args.checkpoint,
                "ensemble": "within_user_rank_average" if len(scored) > 1 else None,
                "feature_sets": feature_sets,
                "target_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
