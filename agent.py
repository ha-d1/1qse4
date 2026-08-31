"""Bounded autonomous research loop for KuaiRand-Pure."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from convergence import ConvergenceState
from experiment_runner import ExperimentRunner
from llm_common import redact_secrets
from llm_factory import create_llm_client
from patch_manager import PatchManager
from proposal_materializer import materialize_patch
from research_planner import build_research_context
from resource_tracker import ResourceTracker
from run_logger import RunLogger


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def candidate_sources(root: Path, limit: int = 35_000) -> dict[str, str]:
    sources = {}
    used = 0
    for path in sorted((root / "candidate").glob("*.py")):
        text = path.read_text()
        if used + len(text) > limit:
            break
        sources[str(path.relative_to(root))] = text
        used += len(text)
    return sources


def reference_api_contracts() -> dict[str, str]:
    """Expose exact protected APIs without repeatedly sending full implementations."""
    return {
        "baseline.py": (
            "FM(dim, k=16, lr=0.001, l2=1e-6, seed=0); public state V, W, b, lr, l2; "
            "methods logits(X)->(scores,E,S), step(X,y)->loss, "
            "step_bpr(X_pos,X_neg)->loss, predict(X,bs=200000)->scores. "
            "make_bpr_pairs(X,y,users,seed=0)->(X_positive,X_negative)."
        ),
        "data.py": (
            "FIELDS, WEEKDAY_FIELDS, USER_AUTHOR_FIELDS; "
            "encode(splits,fields=None)->(encoded,dimension), where encoded[name] is "
            "(X:int32[n,fields], y:float32[n], users:list[str]); "
            "fit_feature_encoder(train_rows,fields=None)->encoder; "
            "transform_rows(rows,encoder,include_labels=True)->(X,y_or_None,users)."
        ),
        "development_data.py": (
            "load_development_splits(data_dir)->{'train': raw_rows,'valid': raw_rows}; "
            "load_training_auxiliary(data_dir,fields)->dict[str,float32 array] for TRAIN ONLY. "
            "Available auxiliary fields: is_click,is_like,is_follow,is_comment,is_forward,play_time_ms. "
            "No validation/test auxiliary targets exist."
        ),
        "evaluate.py": (
            "evaluate(users,labels,scores)->dict with GAUC,nDCG@5,primary. Protected; never modify."
        ),
    }


def parse_metrics(stdout: str) -> dict:
    payload = json.loads(stdout)
    metrics = payload["valid"]
    required = {"GAUC", "nDCG@5", "primary"}
    if not required.issubset(metrics):
        raise ValueError(f"Missing validation metrics: {sorted(required - set(metrics))}")
    return {key: float(metrics[key]) for key in required}


def record_failed_llm_usage(resources: ResourceTracker, error: Exception) -> None:
    """Count provider usage even when a structured response could not be parsed."""
    usage = getattr(error, "usage", None)
    if isinstance(usage, dict) and usage.get("total_tokens", 0):
        resources.add_llm_usage(**usage)


def load_prior_experiment_history(runs_dir: Path, limit: int = 8) -> list[dict]:
    """Load compact, redacted outcomes from earlier runs for cross-run memory."""
    records = []
    if not runs_dir.exists():
        return records
    result_paths = sorted(
        runs_dir.glob("run_*/iterations/iteration_*/result.json"), reverse=True
    )
    for path in result_paths[:limit]:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        attempts = payload.get("attempts", [])
        last_attempt = attempts[-1] if attempts else {}
        proposal = last_attempt.get("proposal", {})
        preflight = last_attempt.get("preflight") or {}
        error = payload.get("error")
        records.append(
            {
                "run_id": path.parents[2].name,
                "iteration": payload.get("iteration"),
                "hypothesis": proposal.get("hypothesis", "proposal unavailable"),
                "status": payload.get("status"),
                "accepted": payload.get("accepted", False),
                "metrics": payload.get("metrics"),
                "error": error[-1200:] if isinstance(error, str) else error,
                "last_preflight": {
                    "status": preflight.get("status"),
                    "stderr_tail": str(preflight.get("stderr", ""))[-1200:],
                }
                if preflight
                else None,
            }
        )
    records.reverse()
    return records


def command_with_verified_data_dir(command: list[str], data_dir: Path) -> list[str]:
    """Force generated commands to use the already verified development dataset path."""
    output = list(command)
    for option in ("--data_dir", "--data-dir", "--dataset", "--label"):
        while option in output:
            index = output.index(option)
            del output[index : min(index + 2, len(output))]
    if "--model" in output:
        index = output.index("--model")
        if "--objective" not in output and index + 1 < len(output):
            output[index] = "--objective"
        else:
            del output[index : min(index + 2, len(output))]
    output.extend(["--data_dir", str(data_dir)])
    return output


def command_with_checkpoint(command: list[str], checkpoint_path: Path) -> list[str]:
    """Force each experiment to materialise its validation-selected model state."""
    output = list(command)
    for option in ("--checkpoint-out", "--checkpoint_out"):
        while option in output:
            index = output.index(option)
            del output[index : min(index + 2, len(output))]
    output.extend(["--checkpoint-out", str(checkpoint_path)])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="agent_config.json")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--data-dir", help="Override the local KuaiRand-Pure data directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    configured_data_dir = Path(args.data_dir or config["paths"]["data_dir"])
    if not configured_data_dir.is_absolute():
        configured_data_dir = root / configured_data_dir
    data_dir = configured_data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")
    budget = config["budget"]
    if args.max_iterations is not None:
        budget = {**budget, "max_iterations": min(args.max_iterations, budget["max_iterations"])}

    runs_dir = root / config["paths"]["runs_dir"]
    history = load_prior_experiment_history(runs_dir)
    logger = RunLogger(runs_dir)
    resources = ResourceTracker()
    convergence = ConvergenceState(
        epsilon=budget["epsilon"],
        patience=budget["consecutive_non_improving_iterations"],
        max_iterations=budget["max_iterations"],
        max_wall_clock_seconds=budget["max_wall_clock_seconds"],
        best_score=config["benchmark"].get(
            "incumbent_valid_primary",
            config["benchmark"].get(
                "incumbent_valid_primary_seed_0",
                config["benchmark"]["official_valid_primary"],
            ),
        ),
    )
    runner = ExperimentRunner(root, budget["experiment_timeout_seconds"])
    preflight_runner = ExperimentRunner(root, min(90, budget["experiment_timeout_seconds"]))
    patches = PatchManager(root, config["safety"]["mutable_roots"])
    llm = create_llm_client(config["llm"])
    best_metrics = {"primary": convergence.best_score}
    initial_snapshot = logger.snapshot_candidate(root / config["paths"]["candidate_dir"], 0)
    initial_snapshot_relative = str(initial_snapshot.relative_to(logger.run_dir))
    logger.write_json(
        "run_config.json",
        {
            "config": config,
            "provider": config["llm"]["provider"],
            "model": config["llm"]["model"],
            "initial_candidate_snapshot": initial_snapshot_relative,
        },
    )
    logger.write_json(
        "best.json",
        {
            "iteration": 0,
            "metrics": best_metrics,
            "snapshot": initial_snapshot_relative,
            "command": None,
            "note": "Initial candidate retained until a validation improvement is accepted.",
        },
    )

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
                reference_sources=reference_api_contracts(),
            )
            iteration_record = {
                "iteration": iteration,
                "attempts": [],
                "recovery_events": [],
            }
            planning_failed = False
            if hasattr(llm, "plan"):
                planning_context = {
                    key: value
                    for key, value in context.items()
                    if key not in {"candidate_sources", "read_only_reference_sources"}
                }
                try:
                    planning_result = llm.plan(planning_context)
                    resources.add_llm_usage(**planning_result.usage)
                    context["approved_research_plan"] = planning_result.plan
                    iteration_record["planning"] = {
                        "plan": planning_result.plan,
                        "llm_usage": planning_result.usage,
                        "interaction_id": planning_result.interaction_id,
                    }
                    logger.write_json(
                        f"iterations/iteration_{iteration:03d}/research_plan.json",
                        planning_result.plan,
                    )
                except Exception as exc:
                    record_failed_llm_usage(resources, exc)
                    error = redact_secrets(repr(exc))
                    planning_failed = True
                    iteration_record.update(
                        {
                            "status": "failed",
                            "accepted": False,
                            "error": error,
                            "planning": {"status": "failed", "error": error},
                        }
                    )
            recovery = None
            proposal = None
            observed_score = None
            max_repairs = int(budget.get("max_repair_attempts", 0))
            for attempt_index in range(0 if planning_failed else max_repairs + 1):
                attempt_number = attempt_index + 1
                attempt_record = {"attempt": attempt_number, "kind": "proposal" if not recovery else "repair"}
                patch_applied = False
                try:
                    proposal_result = (
                        llm.propose(context) if recovery is None else llm.repair(context, recovery)
                    )
                    resources.add_llm_usage(**proposal_result.usage)
                    proposal = proposal_result.proposal
                    approved_plan = context.get("approved_research_plan")
                    if approved_plan and set(proposal.target_files) != set(
                        approved_plan["target_files"]
                    ):
                        raise ValueError(
                            "Coder target_files must exactly match the approved plan: "
                            f"approved={approved_plan['target_files']}, "
                            f"returned={proposal.target_files}"
                        )
                    artifact_root = (
                        f"iterations/iteration_{iteration:03d}/attempt_{attempt_number:02d}"
                    )
                    candidate_checkpoint = logger.run_dir / artifact_root / "candidate.npz"
                    experiment_command = command_with_checkpoint(
                        command_with_verified_data_dir(proposal.command, data_dir),
                        candidate_checkpoint,
                    )
                    experiment_patch = materialize_patch(root, proposal)
                    attempt_record.update(
                        {
                            "proposal": asdict(proposal),
                            "llm_usage": proposal_result.usage,
                            "interaction_id": proposal_result.interaction_id,
                            "executed_command": experiment_command,
                        }
                    )
                    logger.write_json(f"{artifact_root}/proposal.json", asdict(proposal))
                    logger.write_text(f"{artifact_root}/patch.diff", experiment_patch)
                    changed_paths = patches.apply(experiment_patch)
                    patch_applied = True
                    if not set(changed_paths).issubset(set(proposal.target_files)):
                        raise ValueError(
                            "Patch changed a path outside proposal target_files: "
                            f"declared={proposal.target_files}, actual={changed_paths}"
                        )
                    preflight = preflight_runner.run(
                        [
                            "python",
                            "candidate_preflight.py",
                            "--command-json",
                            json.dumps(experiment_command),
                        ]
                    )
                    preflight_record = asdict(preflight)
                    preflight_record["stdout"] = redact_secrets(preflight.stdout)
                    preflight_record["stderr"] = redact_secrets(preflight.stderr)
                    attempt_record["preflight"] = preflight_record
                    logger.write_text(
                        f"{artifact_root}/preflight_stdout.txt", preflight_record["stdout"]
                    )
                    logger.write_text(
                        f"{artifact_root}/preflight_stderr.txt", preflight_record["stderr"]
                    )
                    if preflight.status != "success":
                        raise RuntimeError(
                            "Candidate preflight failed:\n"
                            + (preflight.stderr or preflight.stdout or preflight.status)
                        )
                    process = runner.run(
                        experiment_command, env={"TECHJAM_DATA_DIR": str(data_dir)}
                    )
                    process_record = asdict(process)
                    process_record["stdout"] = redact_secrets(process.stdout)
                    process_record["stderr"] = redact_secrets(process.stderr)
                    attempt_record["process"] = process_record
                    logger.write_text(f"{artifact_root}/stdout.txt", process_record["stdout"])
                    logger.write_text(f"{artifact_root}/stderr.txt", process_record["stderr"])
                    if process.status != "success":
                        raise RuntimeError(process.stderr or f"experiment status={process.status}")
                    metrics = parse_metrics(process.stdout)
                    if not candidate_checkpoint.is_file():
                        raise RuntimeError(
                            "Candidate completed without writing the required checkpoint: "
                            f"{candidate_checkpoint}"
                        )
                    observed_score = metrics["primary"]
                    accepted = metrics["primary"] > best_metrics["primary"]
                    if accepted:
                        best_metrics = metrics
                        snapshot = logger.snapshot_candidate(
                            root / config["paths"]["candidate_dir"], iteration
                        )
                        logger.write_json(
                            "best.json",
                            {
                                "iteration": iteration,
                                "metrics": metrics,
                                "snapshot": str(snapshot.relative_to(logger.run_dir)),
                                "checkpoint": str(
                                    candidate_checkpoint.relative_to(logger.run_dir)
                                ),
                                "command": experiment_command,
                            },
                        )
                    else:
                        patches.rollback(experiment_patch)
                        patch_applied = False
                    attempt_record.update(
                        {
                            "status": "success",
                            "changed_paths": changed_paths,
                            "metrics": metrics,
                            "accepted": accepted,
                        }
                    )
                    iteration_record["attempts"].append(attempt_record)
                    iteration_record.update(
                        {"status": "success", "metrics": metrics, "accepted": accepted}
                    )
                    break
                except Exception as exc:
                    record_failed_llm_usage(resources, exc)
                    if patch_applied and proposal is not None:
                        patches.rollback(experiment_patch)
                    error = redact_secrets(repr(exc))
                    attempt_record.update({"status": "failed", "error": error})
                    iteration_record["attempts"].append(attempt_record)
                    if attempt_index < max_repairs:
                        recovery = {
                            "failed_proposal": {
                                "hypothesis": proposal.hypothesis,
                                "target_files": proposal.target_files,
                                "command": proposal.command,
                            }
                            if proposal is not None
                            else None,
                            "error": error,
                            "instruction": "Correct the failure without changing benchmark or safety constraints.",
                        }
                        iteration_record["recovery_events"].append(
                            {"after_attempt": attempt_number, "error": error}
                        )
                    else:
                        iteration_record.update(
                            {"status": "failed", "accepted": False, "error": error}
                        )

            score_for_convergence = (
                observed_score if observed_score is not None else convergence.best_score
            )
            improvement = convergence.observe(score_for_convergence)
            iteration_record["improvement_over_prior_best"] = improvement

            history.append(
                {
                    "iteration": iteration,
                    "hypothesis": proposal.hypothesis if proposal is not None else "proposal unavailable",
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
        llm.close()


if __name__ == "__main__":
    main()
