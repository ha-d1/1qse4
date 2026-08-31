"""Cheap end-to-end smoke test for a generated candidate command."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

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


_AUXILIARY_OPTION_PREFIXES = ("--aux-", "--auxiliary-", "--multitask-")
_INFERENCE_PARAMETERS = ("V", "W", "b")


def _one_epoch(command: list[str]) -> list[str]:
    cli_args = list(command[1:])
    if "--epochs" in cli_args:
        index = cli_args.index("--epochs")
        cli_args[index + 1] = "1"
    else:
        cli_args.extend(["--epochs", "1"])
    return cli_args


def auxiliary_control_command(command: list[str], checkpoint_path: str | Path) -> list[str]:
    """Disable explicit auxiliary options while preserving every other experiment setting."""
    control = []
    removed_auxiliary_option = False
    index = 0
    while index < len(command):
        argument = command[index]
        if argument.startswith(_AUXILIARY_OPTION_PREFIXES):
            removed_auxiliary_option = True
            index += 1
            if index < len(command) and not command[index].startswith("--"):
                index += 1
            continue
        if argument in {"--checkpoint-out", "--checkpoint_out"}:
            if index + 1 >= len(command):
                raise ValueError(f"Missing value for {argument}")
            control.extend([argument, str(checkpoint_path)])
            index += 2
            continue
        control.append(argument)
        index += 1
    if not removed_auxiliary_option:
        raise ValueError(
            "Auxiliary experiment must expose a removable --aux-*, --auxiliary-*, "
            "or --multitask-* command option for its control comparison"
        )
    return control


def assert_shared_parameter_effect(
    proposed_checkpoint: str | Path, control_checkpoint: str | Path
) -> None:
    """Reject auxiliary work that has no effect on the final ranking parameters."""
    with np.load(proposed_checkpoint) as proposed, np.load(control_checkpoint) as control:
        missing = [
            name
            for name in _INFERENCE_PARAMETERS
            if name not in proposed.files or name not in control.files
        ]
        if missing:
            raise ValueError(
                f"Semantic preflight checkpoints are missing inference parameters: {missing}"
            )
        changed = any(
            not np.allclose(proposed[name], control[name], rtol=0.0, atol=1e-8)
            for name in _INFERENCE_PARAMETERS
        )
    if not changed:
        raise ValueError(
            "Auxiliary experiment does not change checkpointed inference parameters V, W, "
            "or b relative to the same-code control; it cannot affect final ranking"
        )


def _run_candidate(candidate_train, command: list[str]) -> None:
    original_argv = sys.argv
    try:
        sys.argv = _one_epoch(command)
        candidate_train.main()
    finally:
        sys.argv = original_argv


def smoke_test(command: list[str], require_shared_parameter_effect: bool = False) -> None:
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
    _run_candidate(candidate_train, command)
    if not require_shared_parameter_effect:
        return

    checkpoint_option = next(
        (option for option in ("--checkpoint-out", "--checkpoint_out") if option in command),
        None,
    )
    if checkpoint_option is None:
        raise ValueError("Semantic preflight requires --checkpoint-out")
    proposed_checkpoint = command[command.index(checkpoint_option) + 1]
    if not Path(proposed_checkpoint).is_file():
        raise ValueError("Proposed synthetic run did not write its checkpoint")
    with tempfile.TemporaryDirectory(prefix="techjam_semantic_preflight_") as tmp:
        control_checkpoint = Path(tmp) / "control.npz"
        control_command = auxiliary_control_command(command, control_checkpoint)
        _run_candidate(candidate_train, control_command)
        if not control_checkpoint.is_file():
            raise ValueError("Control synthetic run did not write its checkpoint")
        assert_shared_parameter_effect(proposed_checkpoint, control_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", required=True)
    parser.add_argument("--require-shared-parameter-effect", action="store_true")
    args = parser.parse_args()
    command = json.loads(args.command_json)
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("command-json must contain a string argv list")
    smoke_test(command, require_shared_parameter_effect=args.require_shared_parameter_effect)


if __name__ == "__main__":
    main()
