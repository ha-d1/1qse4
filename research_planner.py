"""Build a compact, test-label-free context for the research planner."""
from __future__ import annotations

from typing import Any, Dict, List


INCUMBENT_CHECKPOINTS = [
    "runs/ensemble_multineg_seed0/best.npz",
    "runs/accepted_multineg_seed1/best.npz",
    "runs/ensemble_multineg_seed2/best.npz",
    "runs/ensemble_multineg_seed3/best.npz",
    "runs/ensemble_multineg_seed4/best.npz",
    "runs/hour_feature_seed0/best.npz",
    "runs/hour_feature_seed1/best.npz",
    "runs/hour_feature_seed2/best.npz",
    "runs/session_feature_seed2/best.npz",
]


def recent_interest_scaffolds(preflight_status: str = "pending") -> Dict[str, Any]:
    variants = {
        "short-window recent-interest exposure features": (
            "recent_short",
            "Add prior-five-exposure author/video repeat and author-distance fields.",
        ),
        "long-window recent-interest exposure features": (
            "recent_long",
            "Add prior-twenty-exposure author/video repeat and author-distance fields.",
        ),
        "time-decayed recent-interest exposure features": (
            "recent_time",
            "Add elapsed-time author/video recency and recent-author-frequency fields.",
        ),
    }
    scaffolds = {}
    for direction, (feature_set, scientific_change) in variants.items():
        scaffolds[direction] = {
            "implementation_mode": "scaffold",
            "scientific_change": scientific_change,
            "expected_effect": (
                "Improve the accepted nine-checkpoint ensemble through label-free, "
                "within-user recent-interest diversity."
            ),
            "target_files": [
                "candidate/data.py",
                "candidate/model.py",
                "candidate/train.py",
            ],
            "seed_schedule": [2],
            "command": [
                "python",
                "candidate/train.py",
                "--objective",
                "bpr",
                "--architecture",
                "fm",
                "--seed",
                "2",
                "--epochs",
                "40",
                "--k",
                "16",
                "--lr",
                "0.00025",
                "--negatives-per-positive",
                "4",
                "--batch-size",
                "8192",
                "--feature-set",
                feature_set,
            ],
            "ensemble_checkpoints": list(INCUMBENT_CHECKPOINTS),
            "instruction": (
                "Use the exact locally tested scaffold and judge the candidate by its addition "
                "to the accepted nine-checkpoint ensemble. Do not generate source code."
            ),
            "local_preflight": {"status": preflight_status},
        }
    return scaffolds


def build_research_context(
    iteration: int,
    best_valid_metrics: Dict[str, float],
    experiment_history: List[Dict[str, Any]],
    remaining_seconds: float,
    remaining_iterations: int,
    candidate_sources: Dict[str, str] | None = None,
    reference_sources: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    active_scaffolds: Dict[str, Any] = {}
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
                "method": (
                    "equal average of normalized within-user ranks from five base, three "
                    "hour-aware, and session-aware seed 2 BPR FMs"
                ),
                "validation_primary": 0.6051324605941772,
                "best_single_seed_primary": 0.6046919822692871,
            },
            "instruction": (
                "The nine-checkpoint rank ensemble is the current accepted incumbent. Do not rediscover plain BPR or "
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
            {
                "direction": "click-based auxiliary objectives",
                "decision": "closed after multiple non-improving or invalid implementations",
                "instruction": (
                    "Do not propose click BCE, click pair mixing, or another click auxiliary "
                    "head, whether detached or shared. Move to an explicitly available direction."
                ),
            },
            {
                "direction": "rank ensemble expansion beyond five BPR seeds",
                "decision": "rejected on validation",
                "evidence": {
                    "five_seed_primary": 0.6045457124710083,
                    "eight_seed_primary": 0.6043494343757629,
                    "additional_seeds": [5, 6, 7],
                },
                "instruction": (
                    "Do not train or add more plain four-negative BPR seeds. The complete "
                    "eight-seed ensemble underperformed the five-seed incumbent."
                ),
            },
            {
                "direction": "generated diagonal-bilinear FM rewrites",
                "decision": "closed after repeated implementation failures",
                "evidence": {
                    "sustained_runs": 3,
                    "failure": "categorical-ID matrices were repeatedly treated as dense inputs",
                },
                "instruction": (
                    "Do not generate another diagonal-bilinear FM rewrite. Use the prevalidated "
                    "field-pair-weighted FM scaffold to test the broader interaction-reweighting "
                    "hypothesis without repeating the implementation failure."
                ),
            },
            {
                "direction": "prevalidated field-pair-weighted FM interaction scaffold",
                "decision": "rejected after matched-seed validation and closed by reflection",
                "evidence": {
                    "seed_0_primary": 0.6035248637199402,
                    "seed_1_primary": 0.6030241250991821,
                    "seed_2_primary": 0.6031533479690552,
                    "mean_primary": 0.6032341122627258,
                    "incumbent_primary": 0.604539155960083,
                },
                "instruction": (
                    "Do not repeat field-pair gating. Three successful matched-seed validations "
                    "underperformed the incumbent and all three reflections closed the direction."
                ),
            },
            {
                "direction": "user-balanced BPR weighting",
                "decision": "rejected after seed and ensemble validation",
                "evidence": {
                    "seed_1_primary": 0.60227370262146,
                    "six_model_ensemble_primary": 0.6043605804443359,
                },
                "instruction": "Do not repeat inverse-positive-count BPR weighting.",
            },
            {
                "direction": "hour-of-day categorical interactions",
                "decision": "accepted after matched-seed validation",
                "evidence": {
                    "seed_0_primary": 0.6045704483985901,
                    "seed_1_primary": 0.6046919822692871,
                    "seed_2_primary": 0.6045923829078674,
                    "three_seed_mean": 0.6046182711919149,
                    "eight_checkpoint_ensemble_primary": 0.6049131155014038,
                },
                "instruction": (
                    "Retain the hour-aware checkpoints in the incumbent. Do not rerun the same "
                    "hour bucket; future temporal work must use session order or a new mechanism."
                ),
            },
            {
                "direction": "time_ms session gap and position interactions",
                "decision": "accepted as a diverse single-checkpoint ensemble component",
                "evidence": {
                    "seed_0_primary": 0.6045177578926086,
                    "seed_1_primary": 0.6042183637619019,
                    "seed_2_primary": 0.6046835780143738,
                    "three_seed_mean": 0.6044732332229614,
                    "selected_session_seed": 2,
                    "nine_checkpoint_ensemble_primary": 0.6051324605941772,
                    "all_three_session_ensemble_primary": 0.6049855947494507,
                },
                "instruction": (
                    "Retain session seed 2 only. Do not repeat the same fixed session gap or "
                    "position buckets; additional session seeds diluted the incumbent."
                ),
            },
            {
                "direction": "censored watch-ratio auxiliary objective on shared FM ranking parameters",
                "decision": "suspended after one bounded iteration and failed repair preflights",
                "evidence": {
                    "run_id": "run_20260831T141532Z",
                    "attempts": 2,
                    "full_validation_executed": False,
                    "attempt_1_failure": (
                        "auxiliary path did not change checkpointed V, W, or b relative to control"
                    ),
                    "attempt_2_failure": (
                        "repair did not expose a removable auxiliary CLI option for control comparison"
                    ),
                },
                "instruction": (
                    "Do not automatically retry this direction. Its scientific hypothesis remains "
                    "untested, but another API call is not justified until a human-reviewed local "
                    "scaffold passes the shared-parameter and removable-control preflights."
                ),
            },
            {
                "direction": "fixed family weights for the nine-checkpoint rank ensemble",
                "decision": "rejected on validation",
                "evidence": {
                    "candidates_evaluated": 6,
                    "best_group_totals": {"base": 4, "hour": 3, "session": 2},
                    "best_primary": 0.6051281690597534,
                    "incumbent_primary": 0.6051324605941772,
                },
                "instruction": (
                    "Do not tune more base/hour/session family weights on the same validation "
                    "split; the small fixed grid did not improve equal checkpoint weighting."
                ),
            },
            {
                "direction": "weekday-hour and hour-session-position temporal crosses",
                "decision": "rejected after seed-2 validation and ensemble check",
                "evidence": {
                    "seed_2_primary": 0.6046702861785889,
                    "session_seed_2_primary": 0.6046835780143738,
                    "best_ten_checkpoint_history_blend_primary": 0.6051959991455078,
                    "incumbent_primary": 0.6051324605941772,
                    "apparent_improvement": 0.0000635385513306,
                    "significance_epsilon": 0.002,
                },
                "instruction": (
                    "Do not run additional temporal-cross seeds. The tiny ensemble gain is far "
                    "below the declared significance threshold and does not justify promotion."
                ),
            },
            {
                "direction": "short-window recent-interest exposure features",
                "decision": "rejected by autonomous ensemble validation and closed by Qwen",
                "evidence": {
                    "run_id": "run_20260901T014142Z",
                    "standalone_primary": 0.6045159101486206,
                    "ensemble_primary": 0.6050341129302979,
                    "incumbent_primary": 0.6051324605941772,
                },
                "instruction": "Do not repeat prior-five-exposure author/video features.",
            },
            {
                "direction": "long-window recent-interest exposure features",
                "decision": "rejected by autonomous ensemble validation",
                "evidence": {
                    "run_id": "run_20260901T014142Z",
                    "standalone_primary": 0.6045420169830322,
                    "ensemble_primary": 0.6050402522087097,
                    "incumbent_primary": 0.6051324605941772,
                },
                "instruction": "Do not repeat prior-twenty-exposure author/video features.",
            },
            {
                "direction": "time-decayed recent-interest exposure features",
                "decision": "rejected by autonomous ensemble validation and closed by Qwen",
                "evidence": {
                    "run_id": "run_20260901T015212Z",
                    "standalone_primary": 0.6045085191726685,
                    "ensemble_primary": 0.6050400733947754,
                    "incumbent_primary": 0.6051324605941772,
                },
                "instruction": "Do not repeat elapsed-time recency/frequency FM features.",
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
        # Only locally preflighted, implementation-complete directions may reach Qwen.
        "available_directions": list(active_scaffolds),
        "direction_contracts": {},
        "prevalidated_experiment_scaffolds": active_scaffolds,
        "completed_prevalidated_experiment_scaffolds": {
            "prevalidated field-pair-weighted FM interaction scaffold": {
                "implementation_mode": "scaffold",
                "scientific_change": (
                    "Learn one gate for each pair of the five categorical fields while preserving "
                    "the four-negative BPR objective and the incumbent feature encoding."
                ),
                "expected_effect": (
                    "Test whether field-pair gates improve GAUC and nDCG@5 over standard FM."
                ),
                "target_files": ["candidate/model.py", "candidate/train.py"],
                "seed_schedule": [0, 1, 2],
                "command": [
                    "python",
                    "candidate/train.py",
                    "--objective",
                    "bpr",
                    "--architecture",
                    "field_weighted",
                    "--seed",
                    "0",
                    "--epochs",
                    "40",
                    "--k",
                    "16",
                    "--lr",
                    "0.00025",
                    "--negatives-per-positive",
                    "4",
                    "--batch-size",
                    "8192",
                ],
                "instruction": (
                    "Select this exact direction. The implementation is already locally tested; "
                    "do not request a generated rewrite. Evaluate the supplied command as the "
                    "controlled alternative-interaction experiment."
                ),
                "local_preflight": {"status": "passed"},
                "validation_decision": "completed and rejected",
            },
            "candidate-only temporal-cross FM scaffold": {
                "implementation_mode": "scaffold",
                "scientific_change": (
                    "Add label-free weekday-hour and hour-session-position categorical crosses "
                    "on top of the session feature set."
                ),
                "expected_effect": "Increase temporal representation diversity in FM ranking.",
                "target_files": [
                    "candidate/data.py",
                    "candidate/model.py",
                    "candidate/train.py",
                ],
                "seed_schedule": [2],
                "command": [
                    "python",
                    "candidate/train.py",
                    "--objective",
                    "bpr",
                    "--architecture",
                    "fm",
                    "--seed",
                    "2",
                    "--epochs",
                    "40",
                    "--k",
                    "16",
                    "--lr",
                    "0.00025",
                    "--negatives-per-positive",
                    "4",
                    "--batch-size",
                    "8192",
                    "--feature-set",
                    "temporal_cross",
                ],
                "local_preflight": {"status": "passed"},
                "validation_decision": "completed and rejected",
            },
            **{
                direction: {
                    **scaffold,
                    "validation_decision": "completed and rejected",
                }
                for direction, scaffold in recent_interest_scaffolds(
                    preflight_status="passed"
                ).items()
            },
        },
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
            "unscaffolded generated user-history implementations are suspended",
            "simple train-label history-rate blends are rejected",
            "appended auxiliary feature matrices are suspended after repeated implementation failures",
            "DataFrame-based recency from unavailable event timestamps is suspended",
            "shared-parameter watch-duration MSE is rejected on validation",
            "watch-duration MSE is closed and censored watch-ratio work is suspended pending a local scaffold",
            "all click-based auxiliary objectives are closed",
            "plain BPR ensemble expansion beyond five seeds is rejected",
            "generated diagonal-bilinear FM rewrites are closed after repeated preflight failures",
            "detached auxiliary heads are rejected; multitask gradients must update shared ranking parameters",
            "uniform BPR sampling beyond four negatives is rejected",
            "current-score hard-negative mining is rejected",
            "fixed ensemble family weights and temporal crosses are rejected",
            "all three recent-interest FM scaffolds are rejected by ensemble validation",
        ],
    }
