"""Bounded autonomous research loop for KuaiRand-Pure."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from convergence import ConvergenceState
from experiment_runner import ExperimentRunner
from experiment_schema import ExperimentProposal
from llm_common import ProposalResult, redact_secrets
from llm_common import (
    PLANNER_SYSTEM_INSTRUCTION,
    REFLECTION_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    planning_prompt,
    proposal_prompt,
    reflection_prompt,
)
from llm_factory import create_llm_client
from patch_manager import PatchManager
from proposal_materializer import materialize_patch
from research_planner import build_research_context
from resource_tracker import ResourceTracker
from run_logger import RunLogger
from token_budget import reserve_output_tokens


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
            "make_bpr_pairs(X,y,users,seed=0)->(X_positive,X_negative). "
            "X is an integer categorical-ID matrix shaped [rows, field_count], normally five "
            "fields; dim is the global encoded vocabulary size, not X.shape[1]. FM gathers "
            "V[X] and W[X]. Never compute X @ V or treat X as a dense one-hot matrix."
        ),
        "data.py": (
            "FIELDS, WEEKDAY_FIELDS, USER_AUTHOR_FIELDS, HOUR_FIELDS, SESSION_FIELDS; "
            "encode(splits,fields=None)->(encoded,dimension), where encoded[name] is "
            "(X:int32[n,fields], y:float32[n], users:list[str]); "
            "fit_feature_encoder(train_rows,fields=None)->encoder; "
            "transform_rows(rows,encoder,include_labels=True)->(X,y_or_None,users). "
            "Labelled raw rows are (date,user_id,video_id,author_id,tab,duration_ms,long_view,"
            "hourmin,time_ms); label remains index 6. SESSION_FIELDS are label-free hour, "
            "session-gap, and session-position categorical features derived from time_ms."
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


def load_prior_experiment_history(runs_dir: Path, limit: int = 6) -> list[dict]:
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
                "direction": payload.get("planning", {}).get("plan", {}).get(
                    "direction"
                ),
                "hypothesis": proposal.get("hypothesis", "proposal unavailable"),
                "status": payload.get("status"),
                "accepted": payload.get("accepted", False),
                "metrics": payload.get("metrics"),
                "error": error[-600:] if isinstance(error, str) else error,
                "last_preflight": {
                    "status": preflight.get("status"),
                    "stderr_tail": str(preflight.get("stderr", ""))[-600:],
                }
                if preflight
                else None,
                "reflection": payload.get("reflection", {}).get("content"),
            }
        )
    records.reverse()
    return records


def compact_planning_context(context: dict) -> dict:
    """Keep scientific memory while removing verbose evidence from planner prompts."""
    compact = {
        key: value
        for key, value in context.items()
        if key
        not in {
            "candidate_sources",
            "read_only_reference_sources",
            "completed_prevalidated_experiment_scaffolds",
        }
    }
    compact["completed_autonomous_directions"] = [
        {
            "direction": item.get("direction"),
            "decision": item.get("decision"),
            "instruction": item.get("instruction"),
        }
        for item in context.get("completed_autonomous_directions", [])
    ]
    compact["recent_experiments"] = [
        {
            "run_id": item.get("run_id"),
            "iteration": item.get("iteration"),
            "direction": item.get("direction"),
            "hypothesis": item.get("hypothesis"),
            "status": item.get("status"),
            "accepted": item.get("accepted", False),
            "metrics": item.get("metrics"),
            "error": item.get("error"),
            "reflection": {
                key: (item.get("reflection") or {}).get(key)
                for key in (
                    "outcome_summary",
                    "direction_decision",
                    "next_action",
                )
            }
            if item.get("reflection")
            else None,
        }
        for item in context.get("recent_experiments", [])
    ]
    return compact


def build_reflection_context(
    iteration_record: dict,
    approved_plan: dict | None,
    prior_best: float,
    current_best: dict,
    improvement: float,
) -> dict:
    """Build a compact, code-free evidence record for post-experiment reflection."""
    attempts = []
    for attempt in iteration_record.get("attempts", []):
        proposal = attempt.get("proposal") or {}
        preflight = attempt.get("preflight") or {}
        process = attempt.get("process") or {}
        attempts.append(
            {
                "attempt": attempt.get("attempt"),
                "kind": attempt.get("kind"),
                "hypothesis": proposal.get("hypothesis"),
                "status": attempt.get("status"),
                "error": attempt.get("error"),
                "preflight_status": preflight.get("status"),
                "experiment_status": process.get("status"),
                "metrics": attempt.get("metrics"),
                "accepted": attempt.get("accepted", False),
            }
        )
    return {
        "iteration": iteration_record.get("iteration"),
        "approved_plan": approved_plan,
        "outcome": {
            "status": iteration_record.get("status"),
            "accepted": iteration_record.get("accepted", False),
            "metrics": iteration_record.get("metrics"),
            "error": iteration_record.get("error"),
            "attempts": attempts,
        },
        "comparison": {
            "prior_best_primary": prior_best,
            "current_best_validation_metrics": current_best,
            "improvement_over_prior_best": improvement,
        },
        "evidence_boundary": "train and validation only; hidden test was not evaluated",
    }


def proposal_from_scaffold(plan: dict, scaffold: dict, iteration: int) -> ProposalResult:
    """Materialise a planner-selected, locally prevalidated experiment without generated code."""
    command = list(scaffold["command"])
    seed_schedule = list(scaffold.get("seed_schedule", [0]))
    scheduled_seed = seed_schedule[(iteration - 1) % len(seed_schedule)]
    if "--seed" in command:
        command[command.index("--seed") + 1] = str(scheduled_seed)
    preflight = scaffold.get("local_preflight", {})
    if preflight.get("status") != "passed":
        raise ValueError("Experiment scaffold must record a passed local preflight")
    proposal = ExperimentProposal(
        hypothesis=plan["hypothesis"],
        reasoning=plan["reasoning"],
        target_files=list(scaffold["target_files"]),
        expected_effect=scaffold["expected_effect"],
        risk=plan["risk"],
        patch="",
        command=command,
        implementation_mode="scaffold",
    )
    proposal.validate()
    return ProposalResult(
        proposal=proposal,
        usage={"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0},
        interaction_id=None,
        raw_text="prevalidated scaffold selected by planner",
    )


def llm_call_reservation(
    *,
    budget: dict,
    tokens_spent: int,
    system_instruction: str,
    user_prompt: str,
    requested_output_tokens: int,
    minimum_output_tokens: int,
) -> dict[str, int]:
    """Reserve a complete provider call against the current iteration token cap."""
    return reserve_output_tokens(
        token_cap=int(budget.get("max_llm_tokens_per_iteration", 0)),
        tokens_spent=tokens_spent,
        requested_output_tokens=requested_output_tokens,
        minimum_output_tokens=minimum_output_tokens,
        prompt_parts=(system_instruction, user_prompt),
        overhead_tokens=int(budget.get("llm_prompt_overhead_tokens", 512)),
    )


def command_with_verified_data_dir(command: list[str], data_dir: Path) -> list[str]:
    """Force generated commands to use the already verified development dataset path."""
    output = list(command)
    output = ["--batch-size" if argument == "--batch_size" else argument for argument in output]
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


def proposal_requires_shared_parameter_effect(proposal) -> bool:
    """Identify auxiliary/multitask proposals that need a same-code control check."""
    if getattr(proposal, "implementation_mode", "patch") == "scaffold":
        return False
    command_mentions_auxiliary = any(
        argument.startswith(("--aux-", "--auxiliary-", "--multitask-"))
        for argument in proposal.command
    )
    description = " ".join(
        [proposal.hypothesis, proposal.reasoning, proposal.expected_effect]
    ).lower()
    return command_mentions_auxiliary or any(
        term in description
        for term in ("auxiliary", "multi-task", "multitask", "multi-objective")
    )


def coding_token_budget(context: dict, compact_max_tokens: int, full_max_tokens: int) -> int:
    """Spend fewer output tokens on narrow diffs while retaining capacity for high-value work."""
    plan = context.get("approved_research_plan") or {}
    text = " ".join(
        str(plan.get(key, ""))
        for key in ("hypothesis", "reasoning", "direction", "implementation_requirements")
    ).lower()
    complex_direction = any(
        term in text
        for term in ("architecture", "duration", "censored", "sequence", "attention", "interaction")
    ) or len(plan.get("target_files", [])) >= 2
    return int(full_max_tokens if complex_direction else compact_max_tokens)


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

    terminal_stop_reason = None
    run_directions: set[str] = {
        record["direction"] for record in history if record.get("direction")
    }
    try:
        while convergence.stop_reason(resources.snapshot()["wall_clock_seconds"]) is None:
            iteration = convergence.iterations + 1
            prior_best_score = convergence.best_score
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
            context["available_directions"] = [
                direction
                for direction in context.get("available_directions", [])
                if direction not in run_directions
            ]
            context["prevalidated_experiment_scaffolds"] = {
                direction: scaffold
                for direction, scaffold in context.get(
                    "prevalidated_experiment_scaffolds", {}
                ).items()
                if direction in context["available_directions"]
            }
            if not context.get("available_directions"):
                terminal_stop_reason = "no_approved_research_direction"
                break
            iteration_record = {
                "iteration": iteration,
                "attempts": [],
                "recovery_events": [],
            }
            iteration_llm_start = resources.snapshot()["total_tokens"]
            planning_failed = False
            if hasattr(llm, "plan"):
                planning_context = compact_planning_context(context)
                try:
                    requested_planning_tokens = int(
                        config["llm"].get("planning_max_tokens", 1200)
                    )
                    planning_context["llm_budget"] = {
                        "planning_max_tokens": requested_planning_tokens
                    }
                    planning_reservation = llm_call_reservation(
                        budget=budget,
                        tokens_spent=(
                            resources.snapshot()["total_tokens"] - iteration_llm_start
                        ),
                        system_instruction=PLANNER_SYSTEM_INSTRUCTION,
                        user_prompt=planning_prompt(planning_context),
                        requested_output_tokens=requested_planning_tokens,
                        minimum_output_tokens=256,
                    )
                    planning_context["llm_budget"] = {
                        "planning_max_tokens": planning_reservation["output_reservation"]
                    }
                    planning_result = llm.plan(planning_context)
                    resources.add_llm_usage(**planning_result.usage)
                    if planning_result.plan["direction"] not in context["available_directions"]:
                        raise ValueError(
                            "Research plan must select an exact available direction: "
                            f"selected={planning_result.plan['direction']!r}, "
                            f"available={context['available_directions']!r}"
                        )
                    scaffold = context.get("prevalidated_experiment_scaffolds", {}).get(
                        planning_result.plan["direction"]
                    )
                    if scaffold:
                        planning_result.plan["target_files"] = list(scaffold["target_files"])
                        planning_result.plan["command"] = list(scaffold["command"])
                        planning_result.plan["implementation_requirements"] = [
                            scaffold["scientific_change"],
                            scaffold["instruction"],
                        ]
                    context["approved_research_plan"] = planning_result.plan
                    run_directions.add(planning_result.plan["direction"])
                    context["llm_budget"] = {
                        "coding_max_tokens": coding_token_budget(
                            context,
                            int(config["llm"].get("coding_compact_max_tokens", 4500)),
                            int(config["llm"].get("coding_max_tokens", 7000)),
                        )
                    }
                    iteration_record["planning"] = {
                        "plan": planning_result.plan,
                        "llm_usage": planning_result.usage,
                        "llm_reservation": planning_reservation,
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
            approved_plan = context.get("approved_research_plan") or {}
            scaffold = context.get("prevalidated_experiment_scaffolds", {}).get(
                approved_plan.get("direction")
            )
            max_repairs = (
                0 if scaffold else int(budget.get("max_repair_attempts", 0))
            )
            for attempt_index in range(0 if planning_failed else max_repairs + 1):
                attempt_number = attempt_index + 1
                attempt_record = {"attempt": attempt_number, "kind": "proposal" if not recovery else "repair"}
                patch_applied = False
                try:
                    if recovery is None and scaffold:
                        proposal_result = proposal_from_scaffold(
                            approved_plan, scaffold, iteration
                        )
                    else:
                        requested_coding_tokens = int(
                            context.get("llm_budget", {}).get(
                                "coding_max_tokens",
                                config["llm"].get("coding_max_tokens", 7000),
                            )
                        )
                        coding_reservation = llm_call_reservation(
                            budget=budget,
                            tokens_spent=(
                                resources.snapshot()["total_tokens"]
                                - iteration_llm_start
                            ),
                            system_instruction=SYSTEM_INSTRUCTION,
                            user_prompt=proposal_prompt(context, recovery),
                            requested_output_tokens=requested_coding_tokens,
                            minimum_output_tokens=1024,
                        )
                        context["llm_budget"]["coding_max_tokens"] = (
                            coding_reservation["output_reservation"]
                        )
                        attempt_record["llm_reservation"] = coding_reservation
                        proposal_result = (
                            llm.propose(context)
                            if recovery is None
                            else llm.repair(context, recovery)
                        )
                    if proposal_result.usage.get("total_tokens", 0):
                        resources.add_llm_usage(**proposal_result.usage)
                    proposal = proposal_result.proposal
                    if approved_plan:
                        approved_targets = set(approved_plan["target_files"])
                        proposed_targets = set(proposal.target_files)
                        if not proposed_targets.issubset(approved_targets):
                            raise ValueError(
                                "Coder target_files must remain within the approved plan: "
                                f"approved={approved_plan['target_files']}, "
                                f"returned={proposal.target_files}"
                            )
                        # An unchanged planned file need not be echoed by the coder. Retain the
                        # complete approved declaration while changed_paths is checked separately.
                        proposal.target_files = list(approved_plan["target_files"])
                    artifact_root = (
                        f"iterations/iteration_{iteration:03d}/attempt_{attempt_number:02d}"
                    )
                    candidate_checkpoint = logger.run_dir / artifact_root / "candidate.npz"
                    experiment_command = command_with_checkpoint(
                        command_with_verified_data_dir(proposal.command, data_dir),
                        candidate_checkpoint,
                    )
                    is_scaffold = proposal.implementation_mode == "scaffold"
                    experiment_patch = "" if is_scaffold else materialize_patch(root, proposal)
                    attempt_record.update(
                        {
                            "proposal": asdict(proposal),
                            "llm_usage": proposal_result.usage,
                            "interaction_id": proposal_result.interaction_id,
                            "executed_command": experiment_command,
                            "implementation_mode": proposal.implementation_mode,
                        }
                    )
                    logger.write_json(f"{artifact_root}/proposal.json", asdict(proposal))
                    logger.write_text(f"{artifact_root}/patch.diff", experiment_patch)
                    changed_paths = [] if is_scaffold else patches.apply(experiment_patch)
                    patch_applied = not is_scaffold
                    if not set(changed_paths).issubset(set(proposal.target_files)):
                        raise ValueError(
                            "Patch changed a path outside proposal target_files: "
                            f"declared={proposal.target_files}, actual={changed_paths}"
                        )
                    preflight_command = [
                        "python",
                        "candidate_preflight.py",
                        "--command-json",
                        json.dumps(experiment_command),
                    ]
                    if proposal_requires_shared_parameter_effect(proposal):
                        preflight_command.append("--require-shared-parameter-effect")
                    preflight = preflight_runner.run(preflight_command)
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
                    standalone_metrics = dict(metrics)
                    ensemble_checkpoints = (
                        list(scaffold.get("ensemble_checkpoints", []))
                        if scaffold
                        else []
                    )
                    if ensemble_checkpoints:
                        ensemble_command = ["python", "evaluate_candidate_addition.py"]
                        for checkpoint in ensemble_checkpoints:
                            ensemble_command.extend(["--incumbent", checkpoint])
                        ensemble_command.extend(
                            [
                                "--candidate",
                                str(candidate_checkpoint),
                                "--data_dir",
                                str(data_dir),
                            ]
                        )
                        ensemble_process = runner.run(ensemble_command)
                        ensemble_record = asdict(ensemble_process)
                        ensemble_record["stdout"] = redact_secrets(
                            ensemble_process.stdout
                        )
                        ensemble_record["stderr"] = redact_secrets(
                            ensemble_process.stderr
                        )
                        attempt_record["ensemble_evaluation"] = ensemble_record
                        attempt_record["standalone_metrics"] = standalone_metrics
                        if ensemble_process.status != "success":
                            raise RuntimeError(
                                ensemble_process.stderr
                                or f"ensemble status={ensemble_process.status}"
                            )
                        metrics = parse_metrics(ensemble_process.stdout)
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
                    elif patch_applied:
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

            if hasattr(llm, "reflect"):
                reflection_context = build_reflection_context(
                    iteration_record=iteration_record,
                    approved_plan=context.get("approved_research_plan"),
                    prior_best=prior_best_score,
                    current_best=best_metrics,
                    improvement=improvement,
                )
                try:
                    requested_reflection_tokens = int(
                        config["llm"].get("reflection_max_tokens", 900)
                    )
                    reflection_context["llm_budget"] = {
                        "reflection_max_tokens": requested_reflection_tokens
                    }
                    reflection_reservation = llm_call_reservation(
                        budget=budget,
                        tokens_spent=(
                            resources.snapshot()["total_tokens"] - iteration_llm_start
                        ),
                        system_instruction=REFLECTION_SYSTEM_INSTRUCTION,
                        user_prompt=reflection_prompt(reflection_context),
                        requested_output_tokens=requested_reflection_tokens,
                        minimum_output_tokens=256,
                    )
                    reflection_context["llm_budget"] = {
                        "reflection_max_tokens": reflection_reservation[
                            "output_reservation"
                        ]
                    }
                    reflection_result = llm.reflect(reflection_context)
                    resources.add_llm_usage(**reflection_result.usage)
                    iteration_record["reflection"] = {
                        "status": "success",
                        "content": reflection_result.reflection,
                        "llm_usage": reflection_result.usage,
                        "llm_reservation": reflection_reservation,
                        "interaction_id": reflection_result.interaction_id,
                    }
                    logger.write_json(
                        f"iterations/iteration_{iteration:03d}/reflection.json",
                        reflection_result.reflection,
                    )
                except Exception as exc:
                    record_failed_llm_usage(resources, exc)
                    reflection_error = redact_secrets(repr(exc))
                    iteration_record["reflection"] = {
                        "status": "failed",
                        "error": reflection_error,
                    }
                    logger.write_json(
                        f"iterations/iteration_{iteration:03d}/reflection.json",
                        {"status": "failed", "error": reflection_error},
                    )

            history.append(
                {
                    "iteration": iteration,
                    "direction": approved_plan.get("direction"),
                    "hypothesis": proposal.hypothesis if proposal is not None else "proposal unavailable",
                    "status": iteration_record["status"],
                    "accepted": iteration_record["accepted"],
                    "metrics": iteration_record.get("metrics"),
                    "error": iteration_record.get("error"),
                    "reflection": iteration_record.get("reflection", {}).get("content"),
                }
            )
            resources.usage.iterations = convergence.iterations
            iteration_record["resources"] = resources.snapshot()
            logger.log_iteration(iteration, iteration_record)

        logger.write_json(
            "summary.json",
            {
                "stop_reason": terminal_stop_reason
                or convergence.stop_reason(resources.snapshot()["wall_clock_seconds"]),
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
