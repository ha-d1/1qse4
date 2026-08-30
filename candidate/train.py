"""CLI used by the bounded experiment runner; emits JSON validation metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from candidate.model import train_candidate
from development_data import load_development_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--objective", choices=["pointwise", "bpr"], default="bpr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    splits = load_development_splits(args.data_dir)
    metrics, _ = train_candidate(
        splits,
        objective=args.objective,
        seed=args.seed,
        epochs=args.epochs,
        k=args.k,
        lr=args.lr,
    )
    print(json.dumps({"status": "success", "valid": metrics}, indent=2))


if __name__ == "__main__":
    main()
