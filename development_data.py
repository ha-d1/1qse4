"""Data-access boundary for autonomous development.

The research loop must only see train and validation labels. Final evaluation
rows are exposed without labels and must never be used to select experiments.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from data import load

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
    """Return labelled train/valid data and deliberately discard test labels."""
    all_splits = load(data_dir)
    development = {name: all_splits[name] for name in sorted(DEVELOPMENT_SPLITS)}
    ensure_development_splits(development)
    return development


def remove_labels(rows: Iterable[Row]) -> List[UnlabelledRow]:
    """Strip the label field from rows before final prediction."""
    return [row[:-1] for row in rows]


def load_final_prediction_rows(data_dir: str) -> List[UnlabelledRow]:
    """Return test features without labels for one final prediction pass."""
    return remove_labels(load(data_dir)["test"])
