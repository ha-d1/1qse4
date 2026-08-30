"""Verify dataset invariants and reproduce the official validation baseline.

This script never evaluates the test split. It is the gate that must pass before
the autonomous research loop is allowed to start.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from baseline import FM
from data import encode
from development_data import load_development_splits
from evaluate import evaluate

EXPECTED_ROWS = {"train": 1_141_112, "valid": 124_909}
EXPECTED_RANDOM_PRIMARY = 0.4834
EXPECTED_FM_PRIMARY = 0.6016


def serialisable_metrics(metrics: dict) -> dict:
    """Convert NumPy scalar values returned by evaluation into JSON types."""
    output = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        output[key] = value
    return output


def verify_rows(splits: dict) -> None:
    actual = {name: len(rows) for name, rows in splits.items()}
    if actual != EXPECTED_ROWS:
        raise AssertionError(f"Unexpected split sizes: {actual}; expected {EXPECTED_ROWS}")


def random_sanity(splits: dict, seed: int = 0) -> dict:
    rows = splits["valid"]
    rng = np.random.default_rng(seed)
    result = evaluate(
        [row[1] for row in rows],
        [row[6] for row in rows],
        rng.random(len(rows)),
    )
    if abs(result["primary"] - EXPECTED_RANDOM_PRIMARY) > 0.003:
        raise AssertionError(f"Random sanity check failed: {result}")
    return serialisable_metrics(result)


def train_fm_validation(splits: dict, seed: int = 0, epochs: int = 40) -> dict:
    encoded, dimension = encode(splits)
    X_train, y_train, _ = encoded["train"]
    X_valid, y_valid, valid_users = encoded["valid"]
    model = FM(dimension, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best_score = float("-inf")
    best_state = None
    stale = 0
    for _ in range(epochs):
        indices = rng.permutation(len(y_train))
        for start in range(0, len(indices), 8192):
            batch = indices[start : start + 8192]
            model.step(X_train[batch], y_train[batch])
        metrics = evaluate(valid_users, y_valid, model.predict(X_valid))
        if metrics["primary"] > best_score + 1e-5:
            best_score = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is None:
        raise RuntimeError("FM did not produce a checkpoint")
    model.V, model.W, model.b = best_state
    result = evaluate(valid_users, y_valid, model.predict(X_valid))
    if abs(result["primary"] - EXPECTED_FM_PRIMARY) > 0.004:
        raise AssertionError(f"FM validation baseline is outside tolerance: {result}")
    return serialisable_metrics(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--skip-fm", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    splits = load_development_splits(args.data_dir)
    verify_rows(splits)
    output = {
        "status": "verified",
        "rows": {name: len(rows) for name, rows in splits.items()},
        "random_valid": random_sanity(splits, args.seed),
    }
    if not args.skip_fm:
        output["fm_valid"] = train_fm_validation(splits, args.seed, args.epochs)
    output["runtime_seconds"] = time.monotonic() - started
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
