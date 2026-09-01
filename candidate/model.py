"""Validation-only FM/BPR candidate built on the starter-kit primitives."""
from __future__ import annotations

import collections
import time

import numpy as np

from baseline import FM, make_bpr_pairs
from candidate.data import (
    FIELDS,
    HOUR_FIELDS,
    SESSION_FIELDS,
    TEMPORAL_CROSS_FIELDS,
    RECENT_LONG_FIELDS,
    RECENT_SHORT_FIELDS,
    RECENT_TIME_FIELDS,
    USER_AUTHOR_FIELDS,
    WEEKDAY_FIELDS,
    encode,
)
from evaluate import evaluate


def bpr_user_balance_weights(
    positive_rows: np.ndarray, strength: float
) -> np.ndarray:
    """Interpolate between impression-weighted and equal-user BPR contributions."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("user balance strength must be between 0 and 1")
    if not len(positive_rows):
        return np.empty(0, dtype=np.float32)
    _, inverse, counts = np.unique(
        positive_rows[:, 0], return_inverse=True, return_counts=True
    )
    equal_user = 1.0 / counts[inverse].astype(np.float32)
    equal_user /= float(np.mean(equal_user))
    weights = (1.0 - strength) + strength * equal_user
    return (weights / float(np.mean(weights))).astype(np.float32)


def weighted_bpr_step(
    model: FM,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    """Apply a weighted BPR update to the standard categorical FM parameters."""
    if len(positive_rows) != len(negative_rows) or len(positive_rows) != len(sample_weights):
        raise ValueError("weighted BPR inputs must have equal lengths")
    weights = np.asarray(sample_weights, dtype=np.float32)
    if not len(weights) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weighted BPR requires finite positive sample weights")
    weights = weights / float(np.mean(weights))
    positive_scores, positive_embeddings, positive_sums = model.logits(positive_rows)
    negative_scores, negative_embeddings, negative_sums = model.logits(negative_rows)
    difference = positive_scores - negative_scores
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(difference, -30, 30)))
    score_gradient = (
        (sigmoid - 1.0) * weights / len(positive_rows)
    ).astype(np.float32)
    gradient_v = np.zeros_like(model.V)
    gradient_w = np.zeros_like(model.W)
    np.add.at(gradient_w, positive_rows, score_gradient[:, None])
    np.add.at(
        gradient_v,
        positive_rows,
        score_gradient[:, None, None]
        * (positive_sums[:, None, :] - positive_embeddings),
    )
    np.add.at(gradient_w, negative_rows, -score_gradient[:, None])
    np.add.at(
        gradient_v,
        negative_rows,
        -score_gradient[:, None, None]
        * (negative_sums[:, None, :] - negative_embeddings),
    )
    gradient_v += model.l2 * model.V
    gradient_w += model.l2 * model.W
    model.t += 1
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for parameter, gradient, momentum, variance in (
        (model.V, gradient_v, model.mV, model.vV),
        (model.W, gradient_w, model.mW, model.vW),
    ):
        momentum *= beta1
        momentum += (1.0 - beta1) * gradient
        variance *= beta2
        variance += (1.0 - beta2) * gradient * gradient
        parameter -= model.lr * (momentum / (1.0 - beta1**model.t)) / (
            np.sqrt(variance / (1.0 - beta2**model.t)) + epsilon
        )
    return float(-np.sum(weights * np.log(sigmoid + 1e-9)) / np.sum(weights))


class FieldWeightedFM(FM):
    """FM with one learned gate for every pair of categorical feature fields."""

    def __init__(self, dim, field_count, k=16, lr=0.001, l2=1e-6, seed=0):
        super().__init__(dim, k=k, lr=lr, l2=l2, seed=seed)
        self.field_count = int(field_count)
        self.field_pairs = [
            (left, right)
            for left in range(self.field_count)
            for right in range(left + 1, self.field_count)
        ]
        self.field_pair_weights = np.ones(len(self.field_pairs), dtype=np.float32)
        self.mP = np.zeros_like(self.field_pair_weights)
        self.vP = np.zeros_like(self.field_pair_weights)

    def _pair_features(self, embeddings):
        return np.stack(
            [
                np.sum(embeddings[:, left] * embeddings[:, right], axis=1)
                for left, right in self.field_pairs
            ],
            axis=1,
        )

    def _embedding_derivative(self, embeddings):
        derivative = np.zeros_like(embeddings)
        for pair_index, (left, right) in enumerate(self.field_pairs):
            weight = self.field_pair_weights[pair_index]
            derivative[:, left] += weight * embeddings[:, right]
            derivative[:, right] += weight * embeddings[:, left]
        return derivative

    def logits(self, X):
        embeddings = self.V[X]
        pair_features = self._pair_features(embeddings)
        interaction = pair_features @ self.field_pair_weights
        scores = self.b + self.W[X].sum(1) + interaction
        return scores, embeddings, pair_features

    def _adam_update(self, gradients):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        parameters = (
            (self.V, gradients[0], self.mV, self.vV),
            (self.W, gradients[1], self.mW, self.vW),
            (self.field_pair_weights, gradients[2], self.mP, self.vP),
        )
        for parameter, gradient, momentum, variance in parameters:
            momentum *= b1
            momentum += (1.0 - b1) * gradient
            variance *= b2
            variance += (1.0 - b2) * gradient * gradient
            parameter -= self.lr * (momentum / (1.0 - b1**self.t)) / (
                np.sqrt(variance / (1.0 - b2**self.t)) + eps
            )

    def step(self, X, y):
        batch_size = len(y)
        scores, embeddings, pair_features = self.logits(X)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
        score_gradient = ((probabilities - y) / batch_size).astype(np.float32)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, X, score_gradient[:, None])
        np.add.at(
            gradient_v,
            X,
            score_gradient[:, None, None] * self._embedding_derivative(embeddings),
        )
        gradient_p = np.sum(score_gradient[:, None] * pair_features, axis=0)
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        gradient_p += self.l2 * (self.field_pair_weights - 1.0)
        self._adam_update((gradient_v, gradient_w, gradient_p))
        self.b -= self.lr * score_gradient.sum()
        loss = -np.mean(
            y * np.log(probabilities + 1e-9)
            + (1.0 - y) * np.log(1.0 - probabilities + 1e-9)
        )
        return float(loss)

    def step_bpr(self, X_pos, X_neg):
        batch_size = len(X_pos)
        pos_scores, pos_embeddings, pos_pairs = self.logits(X_pos)
        neg_scores, neg_embeddings, neg_pairs = self.logits(X_neg)
        difference = pos_scores - neg_scores
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(difference, -30, 30)))
        score_gradient = ((sigmoid - 1.0) / batch_size).astype(np.float32)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, X_pos, score_gradient[:, None])
        np.add.at(gradient_w, X_neg, -score_gradient[:, None])
        np.add.at(
            gradient_v,
            X_pos,
            score_gradient[:, None, None]
            * self._embedding_derivative(pos_embeddings),
        )
        np.add.at(
            gradient_v,
            X_neg,
            -score_gradient[:, None, None]
            * self._embedding_derivative(neg_embeddings),
        )
        gradient_p = np.sum(
            score_gradient[:, None] * (pos_pairs - neg_pairs), axis=0
        )
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        gradient_p += self.l2 * (self.field_pair_weights - 1.0)
        self._adam_update((gradient_v, gradient_w, gradient_p))
        return float(-np.mean(np.log(sigmoid + 1e-9)))


def make_hard_bpr_pairs(
    X: np.ndarray,
    y: np.ndarray,
    users: list[str],
    model: FM,
    negatives_per_positive: int,
    candidate_multiplier: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the highest-scoring negatives from a bounded random candidate pool."""
    candidate_count = negatives_per_positive * candidate_multiplier
    sampled = [
        make_bpr_pairs(X, y, users, seed=seed + sample)
        for sample in range(candidate_count)
    ]
    if not sampled or not len(sampled[0][0]):
        return X[:0], X[:0]
    positive = sampled[0][0]
    negative_candidates = np.stack([pair[1] for pair in sampled], axis=1)
    candidate_scores = model.predict(
        negative_candidates.reshape(-1, negative_candidates.shape[-1])
    ).reshape(len(positive), candidate_count)
    first_selected = candidate_count - negatives_per_positive
    selected_columns = np.argpartition(
        candidate_scores, kth=first_selected, axis=1
    )[:, first_selected:]
    selected_negatives = negative_candidates[
        np.arange(len(positive))[:, None], selected_columns
    ]
    return (
        np.repeat(positive, negatives_per_positive, axis=0),
        selected_negatives.reshape(-1, negative_candidates.shape[-1]),
    )


def user_author_history_scores(train_rows: list, target_rows: list) -> np.ndarray:
    """Return train-only smoothed user-author preference logits for target rows."""
    global_positive = sum(row[6] for row in train_rows)
    global_rate = global_positive / max(1, len(train_rows))
    author_stats: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    affinity_stats: dict[tuple[str, str], list[int]] = collections.defaultdict(
        lambda: [0, 0]
    )
    for row in train_rows:
        label = int(row[6])
        author_stats[row[3]][0] += label
        author_stats[row[3]][1] += 1
        affinity_stats[(row[1], row[3])][0] += label
        affinity_stats[(row[1], row[3])][1] += 1

    scores = np.empty(len(target_rows), dtype=np.float32)
    for index, row in enumerate(target_rows):
        author_positive, author_count = author_stats[row[3]]
        author_rate = (author_positive + 20.0 * global_rate) / (author_count + 20.0)
        affinity_positive, affinity_count = affinity_stats[(row[1], row[3])]
        rate = (affinity_positive + 5.0 * author_rate) / (affinity_count + 5.0)
        rate = float(np.clip(rate, 1e-5, 1.0 - 1e-5))
        scores[index] = np.log(rate / (1.0 - rate))
    return scores


def blend_within_user(
    model_scores: np.ndarray,
    history_scores: np.ndarray,
    users: list[str],
    alpha: float,
) -> np.ndarray:
    """Blend components on comparable scales separately within each ranking group."""
    blended = np.empty(len(users), dtype=np.float32)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    for indices in groups.values():
        group = np.asarray(indices, dtype=np.int64)
        model_group = model_scores[group]
        history_group = history_scores[group]
        model_std = float(np.std(model_group))
        history_std = float(np.std(history_group))
        model_scaled = (model_group - np.mean(model_group)) / max(model_std, 1e-6)
        history_scaled = (history_group - np.mean(history_group)) / max(history_std, 1e-6)
        blended[group] = model_scaled + alpha * history_scaled
    return blended


def train_candidate(
    splits: dict,
    objective: str = "bpr",
    k: int = 16,
    lr: float = 0.001,
    epochs: int = 40,
    batch_size: int = 8192,
    patience: int = 4,
    seed: int = 0,
    feature_set: str = "base",
    negatives_per_positive: int = 1,
    auxiliary_targets: dict[str, np.ndarray] | None = None,
    auxiliary_task: str | None = None,
    auxiliary_ratio: float = 0.0,
    negative_strategy: str = "random",
    hard_candidate_multiplier: int = 2,
    architecture: str = "fm",
    user_balance: float = 0.0,
) -> tuple[dict, FM]:
    if set(splits) != {"train", "valid"}:
        raise ValueError("Candidate training accepts train/valid only")
    if objective not in {"pointwise", "bpr"}:
        raise ValueError(f"Unknown objective: {objective}")
    feature_sets = {
        "base": FIELDS,
        "weekday": WEEKDAY_FIELDS,
        "hour": HOUR_FIELDS,
        "session": SESSION_FIELDS,
        "temporal_cross": TEMPORAL_CROSS_FIELDS,
        "recent_short": RECENT_SHORT_FIELDS,
        "recent_long": RECENT_LONG_FIELDS,
        "recent_time": RECENT_TIME_FIELDS,
        "user_author": USER_AUTHOR_FIELDS,
        "history_blend": FIELDS,
    }
    if feature_set not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    if negative_strategy not in {"random", "hard"}:
        raise ValueError(f"Unknown negative strategy: {negative_strategy}")
    if hard_candidate_multiplier < 1:
        raise ValueError("hard_candidate_multiplier must be at least 1")
    if architecture not in {"fm", "field_weighted"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    if not 0.0 <= user_balance <= 1.0:
        raise ValueError("user_balance must be between 0 and 1")
    if user_balance and (objective != "bpr" or architecture != "fm"):
        raise ValueError("user-balanced training currently requires BPR with architecture=fm")
    if auxiliary_ratio < 0.0 or auxiliary_ratio > 1.0:
        raise ValueError("auxiliary_ratio must be between 0 and 1")
    if auxiliary_task is None and auxiliary_ratio != 0.0:
        raise ValueError("auxiliary_ratio requires an auxiliary_task")
    if auxiliary_task is not None and objective != "bpr":
        raise ValueError("auxiliary pairwise training requires the bpr objective")
    encoded, dimension = encode(splits, fields=feature_sets[feature_set])
    X_train, y_train, train_users = encoded["train"]
    X_valid, y_valid, valid_users = encoded["valid"]
    auxiliary_y = None
    if auxiliary_task is not None:
        if not auxiliary_targets or auxiliary_task not in auxiliary_targets:
            raise ValueError(f"Missing train-only auxiliary target: {auxiliary_task}")
        auxiliary_y = np.asarray(auxiliary_targets[auxiliary_task], dtype=np.float32)
        if len(auxiliary_y) != len(y_train):
            raise ValueError(
                "Auxiliary target is not aligned with training rows: "
                f"train={len(y_train)}, auxiliary={len(auxiliary_y)}"
            )
        if not np.isfinite(auxiliary_y).all() or not np.isin(auxiliary_y, (0.0, 1.0)).all():
            raise ValueError("Auxiliary pairwise target must be finite and binary")
    history_scores = (
        user_author_history_scores(splits["train"], splits["valid"])
        if feature_set == "history_blend"
        else None
    )
    model = (
        FM(dimension, k=k, lr=lr, seed=seed)
        if architecture == "fm"
        else FieldWeightedFM(
            dimension,
            field_count=X_train.shape[1],
            k=k,
            lr=lr,
            seed=seed,
        )
    )
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
            if negative_strategy == "hard":
                X_positive, X_negative = make_hard_bpr_pairs(
                    X_train,
                    y_train,
                    train_users,
                    model,
                    negatives_per_positive,
                    hard_candidate_multiplier,
                    seed=seed + epoch * negatives_per_positive,
                )
            else:
                sampled_pairs = [
                    make_bpr_pairs(
                        X_train,
                        y_train,
                        train_users,
                        seed=seed + epoch * negatives_per_positive + sample,
                    )
                    for sample in range(negatives_per_positive)
                ]
                X_positive = np.concatenate([pair[0] for pair in sampled_pairs])
                X_negative = np.concatenate([pair[1] for pair in sampled_pairs])
            auxiliary_pair_count = 0
            if auxiliary_y is not None and auxiliary_ratio > 0.0:
                auxiliary_positive, auxiliary_negative = make_bpr_pairs(
                    X_train,
                    auxiliary_y,
                    train_users,
                    seed=seed + 100_000 + epoch,
                )
                desired = min(
                    len(auxiliary_positive),
                    int(round(auxiliary_ratio * len(X_positive))),
                )
                if desired:
                    selected = rng.choice(
                        len(auxiliary_positive), size=desired, replace=False
                    )
                    X_positive = np.concatenate(
                        (X_positive, auxiliary_positive[selected]), axis=0
                    )
                    X_negative = np.concatenate(
                        (X_negative, auxiliary_negative[selected]), axis=0
                    )
                    auxiliary_pair_count = desired
            indices = rng.permutation(len(X_positive))
            pair_weights = bpr_user_balance_weights(X_positive, user_balance)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if user_balance:
                    losses.append(
                        weighted_bpr_step(
                            model,
                            X_positive[batch],
                            X_negative[batch],
                            pair_weights[batch],
                        )
                    )
                else:
                    losses.append(model.step_bpr(X_positive[batch], X_negative[batch]))

        model_scores = model.predict(X_valid)
        current_alpha = 0.0
        if history_scores is None:
            metrics = evaluate(valid_users, y_valid, model_scores)
        else:
            candidates = []
            for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
                scores = blend_within_user(
                    model_scores, history_scores, valid_users, alpha
                )
                candidates.append((evaluate(valid_users, y_valid, scores), alpha))
            metrics, current_alpha = max(
                candidates, key=lambda candidate: candidate[0]["primary"]
            )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "primary": float(metrics["primary"]),
                "blend_alpha": current_alpha,
                "runtime_seconds": time.monotonic() - started,
                "auxiliary_pairs": auxiliary_pair_count if objective == "bpr" else 0,
            }
        )
        if metrics["primary"] > best_score + 1e-5:
            best_score = metrics["primary"]
            best_state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b),
                current_alpha,
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Candidate training did not produce a checkpoint")
    model.V, model.W, model.b, best_alpha = best_state
    final_scores = model.predict(X_valid)
    if history_scores is not None:
        final_scores = blend_within_user(
            final_scores, history_scores, valid_users, best_alpha
        )
    metrics = evaluate(valid_users, y_valid, final_scores)
    return {
        "GAUC": float(metrics["GAUC"]),
        "nDCG@5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
        "epochs_completed": len(history),
        "feature_set": feature_set,
        "blend_alpha": best_alpha,
        "negatives_per_positive": negatives_per_positive,
        "auxiliary_task": auxiliary_task,
        "auxiliary_ratio": auxiliary_ratio,
        "negative_strategy": negative_strategy,
        "hard_candidate_multiplier": hard_candidate_multiplier,
        "architecture": architecture,
        "user_balance": user_balance,
        "history": history,
    }, model
