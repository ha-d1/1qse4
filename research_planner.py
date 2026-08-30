"""Build a compact, test-label-free context for Gemini."""
from __future__ import annotations

from typing import Any, Dict, List


def build_research_context(
    iteration: int,
    best_valid_metrics: Dict[str, float],
    experiment_history: List[Dict[str, Any]],
    remaining_seconds: float,
    remaining_iterations: int,
    candidate_sources: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    return {
        "benchmark": {
            "dataset": "KuaiRand-Pure",
            "task": "within-user ranking of logged impressions",
            "label": "long_view",
            "metrics": ["GAUC", "nDCG@5"],
            "official_valid_primary": 0.6016,
        },
        "iteration": iteration,
        "best_validation_metrics": best_valid_metrics,
        "recent_experiments": experiment_history[-8:],
        "candidate_sources": candidate_sources or {},
        "budget": {
            "remaining_iterations": remaining_iterations,
            "remaining_seconds": max(0.0, remaining_seconds),
        },
        "available_directions": [
            "pairwise negative sampling",
            "listwise ranking loss",
            "leakage-safe time features",
            "user history features using past events only",
            "multi-task auxiliary engagement labels",
        ],
        "constraints": [
            "train and validation only",
            "one main hypothesis per iteration",
            "candidate/ files only",
            "Python argv command without shell syntax",
        ],
    }
