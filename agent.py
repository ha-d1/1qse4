"""Bounded autonomous research loop for KuaiRand-Pure."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from convergence import ConvergenceState
from experiment_runner import ExperimentRunner
from gemini_client import GeminiClient
from patch_manager import PatchManager
from research_planner import build_research_context
from resource_tracker import ResourceTracker
from run_logger import RunLogger


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def candidate_sources(root: Path, limit: int = 80_000) -> dict[str, str]:
    sources = {}
    used = 0
    for path in sorted((root / "candidate").glob("*.py")):
        text = path.read_text()
        if used + len(text) > limit:
            break
        sources[str(path.relative_to(root))] = text
        used += len(text)
    return sources


def parse_metrics(stdout: str) -> dict:
    payload = json.loads(stdout)
    metrics = payload["valid"]
    required = {"GAUC", "nDCG@5", "primary"}
    if not required.issubset(metrics):
        raise ValueError(f"Missing validation metrics: {sorted(required - set(metrics))}")
    return {key: float(metrics[key]) for key in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="agent_config.json")
    parser.add_argument("--max-iterations", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    budget = config["budget"]
    if args.max_iterations is not None:
        budget = {**budget, "max_iterations": min(args.max_iterations, budget["max_iterations"])}

    logger = RunLogger(root / config["paths"]["runs_dir"])
    resources = ResourceTracker()
    convergence = ConvergenceState(
        epsilon=budget["epsilon"],
        patience=budget["consecutive_non_improving_iterations"],
        max_iterations=budget["max_iterations"],
        max_wall_clock_seconds=budget["max_wall_clock_seconds"],
        best_score=config["benchmark"]["official_valid_primary"],
    )
    runner = ExperimentRunner(root, budget["experiment_timeout_seconds"])
    patches = PatchManager(root, config["safety"]["mutable_roots"])
    gemini = GeminiClient(
        model=config["llm"]["model"],
        max_attempts=config["llm"]["max_attempts"],
    )
    history = []
    best_metrics = {"primary": convergence.best_score}

    try:
        while convergence.stop_reason(resources.snapshot()["wall_clock_seconds"]) is None:
            iteration = convergence.iterations + 1
            elapsed = resources.snapshot()["wall_clock_seconds"]
            context = build_research_context(
                iteration=iteration,
                best_valid_metrics=best_metrics,
                experiment_history=history,
                remaining_seconds=budget["max_wall_clock_seconds"] - elapsed,
                remaining_iterations=budget["max_iterations"] - convergence.iterations,
                candidate_sources=candidate_sources(root),
            )
            proposal_result = gemini.propose(context)
            resources.add_llm_usage(**proposal_result.usage)
            proposal = proposal_result.proposal
            iteration_record = {
                "iteration": iteration,
                "proposal": asdict(proposal),
                "llm_usage": proposal_result.usage,
                "interaction_id": proposal_result.interaction_id,
            }
            patch_applied = False
            try:
                changed_paths = patches.apply(proposal.patch)
                patch_applied = True
                if set(changed_paths) != set(proposal.target_files):
                    raise ValueError(
                        "Proposal target_files do not match patch headers: "
                        f"declared={proposal.target_files}, actual={changed_paths}"
                    )
                process = runner.run(proposal.command)
                iteration_record["process"] = asdict(process)
                if process.status != "success":
                    raise RuntimeError(process.stderr or f"experiment status={process.status}")
                metrics = parse_metrics(process.stdout)
                improvement = convergence.observe(metrics["primary"])
                accepted = metrics["primary"] > best_metrics["primary"]
                if accepted:
                    best_metrics = metrics
                else:
                    patches.rollback(proposal.patch)
                    patch_applied = False
                iteration_record.update(
                    {
                        "status": "success",
                        "changed_paths": changed_paths,
                        "metrics": metrics,
                        "improvement_over_prior_best": improvement,
                        "accepted": accepted,
                    }
                )
            except Exception as exc:
                if patch_applied:
                    patches.rollback(proposal.patch)
                convergence.observe(convergence.best_score)
                iteration_record.update({"status": "failed", "accepted": False, "error": repr(exc)})

            history.append(
                {
                    "iteration": iteration,
                    "hypothesis": proposal.hypothesis,
                    "status": iteration_record["status"],
                    "accepted": iteration_record["accepted"],
                    "metrics": iteration_record.get("metrics"),
                    "error": iteration_record.get("error"),
                }
            )
            resources.usage.iterations = convergence.iterations
            iteration_record["resources"] = resources.snapshot()
            logger.log_iteration(iteration, iteration_record)

        logger.write_json(
            "summary.json",
            {
                "stop_reason": convergence.stop_reason(resources.snapshot()["wall_clock_seconds"]),
                "best_validation_metrics": best_metrics,
                "resources": resources.snapshot(),
                "history": history,
            },
        )
        print(json.dumps({"run_dir": str(logger.run_dir), "best": best_metrics}, indent=2))
    finally:
        gemini.close()


if __name__ == "__main__":
    main()
