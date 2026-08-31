"""Data-access boundary for autonomous development.

The research loop must only see train and validation labels. Final evaluation
rows are exposed without labels and must never be used to select experiments.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from data import AUXILIARY_FIELDS, load_selected, load_train_auxiliary, load_unlabelled

Row = Tuple[int, str, str, str, str, float, int]
UnlabelledRow = Tuple[int, str, str, str, str, float]
DEVELOPMENT_SPLITS = frozenset({"train", "valid"})


def ensure_development_splits(splits: Dict[str, Sequence[Row]]) -> None:
    """Reject mappings that expose labels outside train and validation."""
    unexpected = set(splits) - DEVELOPMENT_SPLITS
    missing = DEVELOPMENT_SPLITS - set(splits)
    if unexpected:
        raise ValueError(
            "Development code may only access train/valid labels; "
            f"unexpected splits: {sorted(unexpected)}"
        )
    if missing:
        raise ValueError(f"Missing development splits: {sorted(missing)}")


def load_development_splits(data_dir: str) -> Dict[str, List[Row]]:
    """Read and return labels for train/valid only; test rows are never loaded."""
    development = load_selected(data_dir, sorted(DEVELOPMENT_SPLITS))
    ensure_development_splits(development)
    return development


def load_training_auxiliary(data_dir: str, fields=AUXILIARY_FIELDS):
    """Expose auxiliary supervision for train only; validation/test targets are unavailable."""
    auxiliary = load_train_auxiliary(data_dir, fields=fields)
    train_rows = load_selected(data_dir, ["train"])["train"]
    lengths = {field: len(values) for field, values in auxiliary.items()}
    if any(length != len(train_rows) for length in lengths.values()):
        raise ValueError(
            "Auxiliary targets are not aligned with the training split: "
            f"train={len(train_rows)}, auxiliary={lengths}"
        )
    return auxiliary


def remove_labels(rows: Iterable[Row]) -> List[UnlabelledRow]:
    """Strip the label field from rows before final prediction."""
    return [row[:-1] for row in rows]


def load_final_prediction_rows(data_dir: str) -> List[UnlabelledRow]:
    """Return test features without labels for one final prediction pass."""
    return load_unlabelled(data_dir, "test")
