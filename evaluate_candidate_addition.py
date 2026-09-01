"""Evaluate one candidate checkpoint only as an addition to the accepted ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import load_selected
from development_data import remove_labels
from evaluate import evaluate
from make_submission import score_unlabelled_rows, within_user_rank_average


def candidate_addition_metrics(incumbents, candidate, data_dir):
    splits = load_selected(data_dir, ["train", "valid"])
    train_rows, valid_rows = splits["train"], splits["valid"]
    unlabelled_valid = remove_labels(valid_rows)
    users = [row[1] for row in valid_rows]
    labels = np.asarray([row[6] for row in valid_rows], dtype=np.float32)
    checkpoints = [*incumbents, candidate]
    scores = [
        score_unlabelled_rows(path, train_rows, unlabelled_valid)[0]
        for path in checkpoints
    ]
    metrics = evaluate(users, labels, within_user_rank_average(scores, users))
    return {
        "GAUC": float(metrics["GAUC"]),
        "nDCG@5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", action="append", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    metrics = candidate_addition_metrics(
        [Path(path) for path in args.incumbent],
        Path(args.candidate),
        args.data_dir,
    )
    print(json.dumps({"status": "success", "valid": metrics}, indent=2))


if __name__ == "__main__":
    main()
