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
                    "Do not repeat shared-score click-pair mixing. A different train-only "
                    "auxiliary objective remains scientifically distinct only if its gradients "
                    "change checkpointed ranking parameters."
                ),
            },
            {
                "direction": "detached click auxiliary head",
                "decision": "rejected by semantic preflight and validation",
                "evidence": {
                    "observed_primary": 0.6038926243782043,
                    "incumbent_primary": 0.604539155960083,
                    "failure": "auxiliary updates did not change checkpointed V, W, or b",
                },
                "instruction": (
                    "Do not propose a detached auxiliary scalar or head that is absent from "
                    "final ranking. Any multitask gradient must update shared inference parameters."
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
            {
                "direction": "more than four uniformly sampled BPR negatives",
                "decision": "rejected on validation",
                "evidence": {
                    "eight_negative_seed_1_primary": 0.6038511991500854,
                    "four_negative_seed_1_primary": 0.603919267654419,
                    "eight_negative_learning_rate": 0.000125,
                },
                "instruction": (
                    "Do not increase uniform negatives beyond four or retune that same BPR "
                    "density; the eight-negative proportional-learning-rate control did not improve."
                ),
            },
            {
                "direction": "simple train-label history post-processing",
                "decision": "rejected on validation",
                "evidence": {
                    "incumbent_recalculated_primary": 0.6045457124710083,
                    "best_video_history_primary": 0.6044692993164062,
                    "best_personalized_history_primary": 0.6044439673423767,
                },
                "instruction": (
                    "Do not repeat smoothed video, user-video, author, or user-author target-rate "
                    "blends. Sequence models using ordered events remain scientifically distinct."
                ),
            },
            {
                "direction": "auxiliary signals as appended input features",
                "decision": "suspended after repeated implementation failures",
                "instruction": (
                    "Do not retry appended auxiliary feature matrices in this run. The prior "
                    "attempt failed twice at patch/preflight before a valid validation result; "
                    "move to a different research direction."
                ),
            },
            {
                "direction": "generated continuous recency from unavailable event timestamps",
                "decision": "suspended after incompatible plan",
                "instruction": (
                    "Do not retry DataFrame-based time_diff or time-since-last-interaction. "
                    "Development rows are raw seven-field tuples containing date but no event "
                    "timestamp; the project has no pandas dependency."
                ),
            },
            {
                "direction": "shared-parameter watch-duration MSE auxiliary objective",
                "decision": "rejected on validation",
                "evidence": {
                    "seed_1_primary": 0.6038864850997925,
                    "seed_1_control_primary": 0.603919267654419,
                    "best_ensemble_with_watch_primary": 0.6044906973838806,
                    "incumbent_ensemble_primary": 0.6045457124710083,
                },
                "instruction": (
                    "Do not repeat normalized log(play_time_ms) MSE updates on the shared FM "
                    "parameters; the semantic preflight passed but full validation and ensemble "
                    "diversity did not improve."
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
            "simple train-label history-rate blends are rejected",
            "appended auxiliary feature matrices are suspended after repeated implementation failures",
            "DataFrame-based recency from unavailable event timestamps is suspended",
            "shared-parameter watch-duration MSE is rejected on validation",
            "detached auxiliary heads are rejected; multitask gradients must update shared ranking parameters",
            "uniform BPR sampling beyond four negatives is rejected",
            "current-score hard-negative mining is rejected",
        ],
    }
