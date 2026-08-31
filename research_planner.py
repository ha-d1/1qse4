"""Build a compact, test-label-free context for the research planner."""
from __future__ import annotations

from typing import Any, Dict, List


def build_research_context(
    iteration: int,
    best_valid_metrics: Dict[str, float],
    experiment_history: List[Dict[str, Any]],
    remaining_seconds: float,
    remaining_iterations: int,
    candidate_sources: Dict[str, str] | None = None,
    reference_sources: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    return {
        "benchmark": {
            "dataset": "KuaiRand-Pure",
            "task": "within-user ranking of logged impressions",
            "label": "long_view",
            "metrics": ["GAUC", "nDCG@5"],
            "official_valid_primary": 0.6016,
            "official_baseline": "Factorization Machine trained with pointwise log loss",
        },
        "existing_user_work": {
            "status": "completed before autonomous research",
            "candidate": (
                "FM trained with four independently sampled negatives per positive using BPR "
                "and learning rate 0.00025"
            ),
            "matched_seed_primary": {
                "seed_0": 0.6038926243782043,
                "seed_1": 0.603919267654419,
                "seed_2": 0.6036715507507324,
                "mean": 0.6038278142611185,
            },
            "accepted_submission_ensemble": {
                "method": "equal average of normalized within-user ranks from BPR FM seeds 0 through 4",
                "validation_primary": 0.604539155960083,
                "best_single_seed_primary": 0.60400390625,
            },
            "instruction": (
                "The five-seed rank ensemble is the current accepted incumbent. Do not rediscover plain BPR or "
                "multi-negative BPR. Propose a materially different representation, auxiliary "
                "task, history mechanism, or architecture that builds beyond it."
            ),
        },
        "completed_autonomous_directions": [
            {
                "direction": "ListNet/listwise softmax on user impression sets",
                "best_primary": 0.5967028737068176,
                "incumbent_primary": 0.6037,
                "decision": "rejected",
                "instruction": (
                    "Do not propose another ListNet or listwise-softmax variant. This direction "
                    "completed full validation and underperformed the incumbent."
                ),
            },
            {
                "direction": "Soft-NDCG/top-k-weighted pairwise loss",
                "decision": "suspended after repeated preflight failures",
                "instruction": (
                    "Do not propose Soft-NDCG, rank-discounted BPR, or another top-k-weighted "
                    "pairwise loss. Two autonomous implementations failed synthetic preflight "
                    "before full validation and further attempts are not token-efficient."
                ),
            },
            {
                "direction": "simple static and aggregate feature additions",
                "decision": "rejected on validation",
                "evidence": {
                    "weekday_primary": 0.6035299301147461,
                    "explicit_user_author_cross_primary": 0.6017320156097412,
                    "user_author_history_blend_primary": 0.6037293672561646,
                },
                "instruction": (
                    "Do not repeat weekday, explicit user-author crosses, or smoothed "
                    "user-author target-statistic blends. True past-event sequence modelling "
                    "remains available and is scientifically distinct."
                ),
            },
            {
                "direction": "generated past-event user-history implementations",
                "decision": "temporarily suspended after repeated implementation failures",
                "instruction": (
                    "Do not propose another user-history implementation in this run. The "
                    "scientific direction remains promising, but two generated versions failed "
                    "before validation and a dedicated scaffold is needed first."
                ),
            },
            {
                "direction": "naive click-pair mixing in the primary BPR score",
                "decision": "rejected on validation",
                "evidence": {
                    "is_click_pair_ratio": 0.1,
                    "seed_1_primary": 0.6030641794204712,
                    "incumbent_seed_1_primary": 0.603919267654419,
                },
                "instruction": (
                    "Do not repeat shared-score click-pair mixing. A separate auxiliary head "
                    "or a different train-only auxiliary objective remains scientifically distinct."
                ),
            },
            {
                "direction": "hard-negative mining from the current FM score",
                "decision": "rejected on validation",
                "evidence": {
                    "candidate_pool_per_positive": 8,
                    "selected_negatives_per_positive": 4,
                    "seed_1_primary": 0.591168999671936,
                    "incumbent_seed_1_primary": 0.603919267654419,
                },
                "instruction": (
                    "Do not repeat current-score hard-negative mining; it strongly degraded "
                    "both ranking metrics relative to random multi-negative sampling."
                ),
            },
        ],
        "data_contract": {
            "user_id_type": "opaque string",
            "video_id_type": "opaque string",
            "instruction": (
                "Never cast raw user_id or video_id values to integers. Group them as strings, "
                "or build an explicit deterministic dictionary mapping if numeric IDs are required. "
                "Validation labels are evaluation-only and must never be used to build history "
                "features; label-dependent history must be fitted on train and frozen for valid."
            ),
            "training_auxiliary_api": {
                "function": "development_data.load_training_auxiliary",
                "available_train_only_fields": [
                    "is_click",
                    "is_like",
                    "is_follow",
                    "is_comment",
                    "is_forward",
                    "play_time_ms",
                ],
                "instruction": (
                    "Auxiliary targets are aligned with train rows and are unavailable for "
                    "validation and test. Use long_view as the primary objective and metrics."
                ),
            },
        },
        "iteration": iteration,
        "best_validation_metrics": best_valid_metrics,
        "recent_experiments": experiment_history[-8:],
        "candidate_sources": candidate_sources or {},
        "read_only_reference_sources": reference_sources or {},
        "budget": {
            "remaining_iterations": remaining_iterations,
            "remaining_seconds": max(0.0, remaining_seconds),
        },
        "available_directions": [
            "multi-objective learning with train-only auxiliary feedback",
            "leakage-safe time features",
            "alternative interaction architectures evaluated against the multi-negative BPR incumbent",
        ],
        "constraints": [
            "train and validation only",
            "one main hypothesis per iteration",
            "candidate/ files only",
            "Python argv command without shell syntax",
            "plain and multi-negative BPR are existing work and must not be proposed as the new contribution",
            "the implemented gradient must mathematically match the claimed objective",
            "do not disguise pairwise BPR updates as a listwise method",
            "all command-line arguments must be supported by the returned candidate files",
            "inspect read_only_reference_sources and use only methods that actually exist",
            "read_only_reference_sources inform implementation but must never be modified",
            "new helpers in candidate/data.py must be imported from candidate.data, not root data",
            "validation labels are evaluation-only and cannot be used in features or history",
            "raw user_id values are strings such as u0 and must never be passed to int() or int64 arrays",
            "ListNet and listwise softmax have completed validation and must not be proposed again",
            "Soft-NDCG and top-k-weighted pairwise losses are suspended after repeated preflight failures",
            "weekday, explicit user-author crosses, and user-author statistic blends are rejected",
            "generated user-history implementations are temporarily suspended for this run",
            "naive shared-score click-pair mixing is rejected; use a separate auxiliary head if revisiting multitask learning",
            "current-score hard-negative mining is rejected",
        ],
    }
