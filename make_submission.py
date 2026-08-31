"""Generate checkpoint predictions without reading target labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from baseline import FM
from data import (
    FIELDS,
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
        model = FM(encoder["dimension"], k=V.shape[1], lr=metadata["lr"], seed=0)
        model.V, model.W, model.b = V.copy(), W.copy(), b
        scores = model.predict(X)
    if len(scores) != len(target_rows) or not np.isfinite(scores).all():
        raise ValueError("Final predictions must be finite and match the target row count")
    return scores, metadata


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
    parser.add_argument("--checkpoint", required=True)
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
    scores, metadata = score_unlabelled_rows(args.checkpoint, train_rows, target_rows)
    output = write_submission(args.output, target_rows, scores)
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(output),
                "rows": len(target_rows),
                "split": args.split,
                "checkpoint": args.checkpoint,
                "feature_set": metadata["feature_set"],
                "target_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
