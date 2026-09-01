"""Prompt contract and request builders for the campaign coding agent."""

IMMUTABLE_RULES = (
    "Do not change evaluate.py, data.py, split dates, metrics, tuple order, or row identity.",
    "Return the complete replacement source for solution.py; never return a patch.",
    "Never use test labels or test metrics for model selection.",
    "Rank only each user's logged exposures; never retrieve from the full catalogue.",
    "The candidate score function must return one finite row-aligned score per target row.",
)

SYSTEM_PROMPT = (
    "You are the coding agent in an automated ML research loop. Return one complete, attributable experiment as the exact JSON action schema supplied by the controller.",
    "Test exactly one causal hypothesis per experiment. Keep behavior unrelated to that hypothesis identical to current_best_source so any score change is attributable; do not rewrite working components from scratch.",
    "The controller, not candidate code, loads labels, computes metrics with evaluate.py, persists scores, and runs validation/test. Candidate code only returns scores.",
    "Train only from splits['train']. The controller compares experiments on hidden validation metrics. Test labels and test metrics are unavailable and must never influence code, configuration, or stopping.",
    "The runtime imports solution.py and calls score(splits, data_access, target_split: str, seed: int, config: dict) exactly once. Return a one-dimensional finite numeric array with exactly one value for every row of splits[target_split], in unchanged row order.",
    "Evaluation groups and ranks rows independently within each user. Any term that is constant within a user cannot change GAUC or nDCG@5 and is not a useful experiment.",
    "All target-split labels presented to candidate code are zero-masked. Never train, early-stop, or select epochs on splits['valid'] or splits['test'].",
    "For model-side selection, construct a deterministic temporal holdout only from splits['train']; fit on 20220408-20220418 and select on 20220419-20220421.",
    "The baseline API is fit_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True). Never pass unsupported keywords. Calling it on the supplied masked valid split cannot perform meaningful early stopping.",
    "data_access.iter_rows(filename, columns_tuple, split='train' or target_split) exposes sanitized non-label CSV columns. The long_view label is deliberately rejected.",
    "Candidate code may use any Python package or system module already installed in the candidate environment. It may not install packages, access the network, download data, spawn subprocesses, or bypass the supplied data_access interface.",
    "Implement every behavioral change in the returned complete solution.py source. Config is passed unchanged to score and is metadata unless candidate code reads it.",
    "Do not reimplement or modify metrics, split dates, row identity, the executor, or protected files.",
    "A repair request contains the complete previous source, execution status, and bounded stdout, stderr, and traceback tails. Preserve working code, fix the diagnosed cause, and return the complete corrected source rather than rewriting unrelated logic.",
    "Do not conceal a broken hypothesis with catch-all exception handling, constant or all-zero scores, or a fallback that silently disables the new logic. Diagnose and repair the root cause; a no-op candidate wastes a full run even if it satisfies the shape check.",
    "For a timeout or memory-limit failure, use the supplied status and output tails to reduce the failed algorithm's complexity or allocation while preserving its hypothesis. Do not merely raise the limits or abandon the experiment.",
)

CANDIDATE_CONSTRAINTS = (
    "The candidate may use any Python package or system module already installed in the candidate environment.",
    "Use standard-library modules, numpy, baseline, data, and installed third-party dependencies declared by the project; never install packages, access the network, download data, spawn subprocesses, or bypass the supplied data_access interface.",
    "Return complete executable solution.py source in the experiment action's source field, with no Markdown fence, patch, diff, commentary, TODO, placeholder, or ellipsis.",
    "Keep the first experiment implementation small and self-contained.",
    "On source validation failure, correct the supplied source; do not repeat it or inspect again.",
    "Experiment config is metadata only; all behavior must be implemented in solution.py.",
)

PROTECTED_PATHS = (
    "evaluate.py", "data.py", "baseline.py", "experiment.py", "agent.py",
    "agent_codex.py", "agent_prompts.py", "agent_sandbox.py", "agent_state.py",
    "submit.py",
)

ACTION_NOTES = (
    "Inspection requests must be JSON objects with the exact kind/path/start/end or kind/path/pattern fields shown in action_schema.",
    "Do not encode inspection requests as strings such as 'file|solution.py'.",
    "Return exactly one action object and no Markdown wrapper.",
)


def build_generate_prompt(current_best_source, prior_experiments, best_metrics, target_primary):
    """Build the first request around the current accepted source and research log."""
    return {
        "instruction": (
            "Inspect if needed, then return one evidence-backed experiment. Implement one "
            "hypothesis against current_best_source and preserve unrelated working behavior. "
            "Do not finish before best_metrics.primary reaches target_primary."
        ),
        "current_best_source": current_best_source,
        "prior_experiments": prior_experiments,
        "best_metrics": best_metrics,
        "target_primary": target_primary,
    }


def build_repair_prompt(previous_action, previous_source, failure, repair_attempt):
    """Build an in-place retry request with the failed source and executor evidence."""
    return {
        "instruction": (
            "Repair the failed candidate in place. Return one experiment action with the same "
            "name and hypothesis, complete corrected solution.py source, and corrected config. "
            "Preserve working code and fix the supplied failure's root cause. Do not inspect, "
            "finish, abandon or replace the hypothesis, hide the failure with a no-op fallback, "
            "or return commentary."
        ),
        "previous_action": previous_action,
        "previous_source": previous_source,
        "failure": failure,
        "repair_attempt": repair_attempt,
    }
