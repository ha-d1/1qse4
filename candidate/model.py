"""Validation-only FM/BPR candidate built on the starter-kit primitives."""
from __future__ import annotations

import collections
import time

import numpy as np

from baseline import FM, make_bpr_pairs
from data import encode
from evaluate import evaluate


def train_candidate(
    splits: dict,
    objective: str = "bpr",
    k: int = 16,
    lr: float = 0.001,
    epochs: int = 40,
    batch_size: int = 8192,
    patience: int = 4,
    seed: int = 0,
) -> tuple[dict, FM]:
    if set(splits) != {"train", "valid"}:
        raise ValueError("Candidate training accepts train/valid only")
    if objective not in {"pointwise", "bpr"}:
        raise ValueError(f"Unknown objective: {objective}")
    encoded, dimension = encode(splits)
    X_train, y_train, train_users = encoded["train"]
    X_valid, y_valid, valid_users = encoded["valid"]
    model = FM(dimension, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_score = float("-inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        losses = []
        if objective == "pointwise":
            indices = rng.permutation(len(y_train))
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                losses.append(model.step(X_train[batch], y_train[batch]))
        else:
            X_positive, X_negative = make_bpr_pairs(
                X_train, y_train, train_users, seed=seed + epoch
            )
            indices = rng.permutation(len(X_positive))
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                losses.append(model.step_bpr(X_positive[batch], X_negative[batch]))

        metrics = evaluate(valid_users, y_valid, model.predict(X_valid))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "primary": float(metrics["primary"]),
                "runtime_seconds": time.monotonic() - started,
            }
        )
        if metrics["primary"] > best_score + 1e-5:
            best_score = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Candidate training did not produce a checkpoint")
    model.V, model.W, model.b = best_state
    metrics = evaluate(valid_users, y_valid, model.predict(X_valid))
    return {
        "GAUC": float(metrics["GAUC"]),
        "nDCG@5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
        "epochs_completed": len(history),
        "history": history,
    }, model
