"""CLI used by the bounded experiment runner; emits JSON validation metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from candidate.model import train_candidate
from development_data import load_development_splits, load_training_auxiliary


def save_checkpoint(path: str | Path, model, metadata: dict) -> Path:
    """Persist the validation-selected FM state and reproducibility metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        V=model.V,
        W=model.W,
        b=np.asarray(model.b),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default=os.environ.get("TECHJAM_DATA_DIR", "./KuaiRand-Pure/data")
    )
    parser.add_argument("--objective", choices=["pointwise", "bpr"], default="bpr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument(
        "--negative-strategy", choices=["random", "hard"], default="random"
    )
    parser.add_argument("--hard-candidate-multiplier", type=int, default=2)
    parser.add_argument(
        "--auxiliary-task",
        choices=["is_click", "is_like", "is_follow", "is_comment", "is_forward"],
    )
    parser.add_argument("--auxiliary-ratio", type=float, default=0.0)
    parser.add_argument("--checkpoint-out")
    parser.add_argument(
        "--feature-set",
        choices=["base", "weekday", "user_author", "history_blend"],
        default="base",
    )
    args = parser.parse_args()
    splits = load_development_splits(args.data_dir)
    auxiliary_targets = (
        load_training_auxiliary(args.data_dir, fields=(args.auxiliary_task,))
        if args.auxiliary_task is not None
        else None
    )
    metrics, model = train_candidate(
        splits,
        objective=args.objective,
        seed=args.seed,
        epochs=args.epochs,
        k=args.k,
        lr=args.lr,
        feature_set=args.feature_set,
        negatives_per_positive=args.negatives_per_positive,
        auxiliary_targets=auxiliary_targets,
        auxiliary_task=args.auxiliary_task,
        auxiliary_ratio=args.auxiliary_ratio,
        negative_strategy=args.negative_strategy,
        hard_candidate_multiplier=args.hard_candidate_multiplier,
    )
    if args.checkpoint_out:
        checkpoint = save_checkpoint(
            args.checkpoint_out,
            model,
            {
                "objective": args.objective,
                "seed": args.seed,
                "epochs_requested": args.epochs,
                "k": args.k,
                "lr": args.lr,
                "feature_set": args.feature_set,
                "negatives_per_positive": args.negatives_per_positive,
                "auxiliary_task": args.auxiliary_task,
                "auxiliary_ratio": args.auxiliary_ratio,
                "negative_strategy": args.negative_strategy,
                "hard_candidate_multiplier": args.hard_candidate_multiplier,
                "validation_metrics": metrics,
            },
        )
        metrics["checkpoint"] = str(checkpoint)
    print(json.dumps({"status": "success", "valid": metrics}, indent=2))


if __name__ == "__main__":
    main()
