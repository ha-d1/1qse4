"""Cheap end-to-end smoke test for a generated candidate command."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def synthetic_splits() -> dict:
    train = []
    valid = []
    for user_index in range(4):
        user = f"u{user_index}"
        for item_index in range(6):
            label = 1 if item_index % 3 == 0 else 0
            row = (
                20220408,
                user,
                f"v{item_index}",
                f"a{item_index % 2}",
                "1",
                float(1000 + item_index * 100),
                label,
            )
            train.append(row)
            valid.append((20220422,) + row[1:])
    edge_cases = {
        "all_positive": [1, 1, 1],
        "all_negative": [0, 0, 0],
        "singleton": [1],
        "long_mixed": [1 if index % 4 == 0 else 0 for index in range(20)],
    }
    for user, labels in edge_cases.items():
        for item_index, label in enumerate(labels):
            row = (
                20220408,
                user,
                f"{user}_v{item_index}",
                f"edge_a{item_index % 2}",
                "1",
                float(1200 + item_index * 50),
                label,
            )
            train.append(row)
            valid.append((20220422,) + row[1:])
    return {"train": train, "valid": valid}


def synthetic_training_auxiliary() -> dict:
    """Train-only auxiliary targets aligned with the synthetic training rows."""
    train = synthetic_splits()["train"]
    labels = np.asarray([row[6] for row in train], dtype=np.float32)
    durations = np.asarray([row[5] for row in train], dtype=np.float32)
    return {
        "is_click": labels.copy(),
        "is_like": (np.arange(len(train)) % 7 == 0).astype(np.float32),
        "is_follow": (np.arange(len(train)) % 11 == 0).astype(np.float32),
        "is_comment": (np.arange(len(train)) % 9 == 0).astype(np.float32),
        "is_forward": (np.arange(len(train)) % 13 == 0).astype(np.float32),
        "play_time_ms": durations * (0.25 + 0.75 * labels),
    }


def smoke_test(command: list[str]) -> None:
    if len(command) < 2 or command[1] != "candidate/train.py":
        raise ValueError("Preflight requires candidate/train.py")

    import development_data

    development_data.load_training_auxiliary = lambda _data_dir, fields=None: {
        key: value
        for key, value in synthetic_training_auxiliary().items()
        if fields is None or key in fields
    }
    import candidate.train as candidate_train

    candidate_train.load_development_splits = lambda _data_dir: synthetic_splits()
    if hasattr(candidate_train, "load_training_auxiliary"):
        candidate_train.load_training_auxiliary = development_data.load_training_auxiliary
    cli_args = list(command[1:])
    if "--epochs" in cli_args:
        index = cli_args.index("--epochs")
        cli_args[index + 1] = "1"
    else:
        cli_args.extend(["--epochs", "1"])
    original_argv = sys.argv
    try:
        sys.argv = cli_args
        candidate_train.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", required=True)
    args = parser.parse_args()
    command = json.loads(args.command_json)
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("command-json must contain a string argv list")
    smoke_test(command)


if __name__ == "__main__":
    main()
