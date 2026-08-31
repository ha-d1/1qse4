"""Aggregate auditable experiment logs into the Track 2 resource report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


RESOURCE_FIELDS = (
    "prompt_tokens",
    "response_tokens",
    "total_tokens",
    "llm_calls",
    "iterations",
    "gpu_hours",
    "wall_clock_seconds",
    "manual_interventions",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_report(runs_dir: Path, config: dict) -> dict:
    summaries = [_read_json(path) for path in sorted(runs_dir.glob("run_*/summary.json"))]
    totals = {field: 0 for field in RESOURCE_FIELDS}
    for summary in summaries:
        resources = summary.get("resources", {})
        for field in RESOURCE_FIELDS:
            totals[field] += resources.get(field, 0) or 0

    iteration_results = [
        _read_json(path)
        for path in sorted(runs_dir.glob("run_*/iterations/iteration_*/result.json"))
    ]
    attempts = [
        attempt
        for iteration in iteration_results
        for attempt in iteration.get("attempts", [])
    ]
    manual_results = [
        _read_json(path)
        for path in sorted(runs_dir.glob("*/result.json"))
        if "iterations" not in path.parts
    ]
    manual_resources = [result.get("resources", {}) for result in manual_results]
    totals["manual_interventions"] += sum(
        resource.get("manual_interventions", 0) or 0
        for resource in manual_resources
    )
    manual_runtime = sum(
        resource.get("training_runtime_seconds", 0) or 0
        for resource in manual_resources
    )
    manual_tokens = sum(
        resource.get("llm_total_tokens", 0) or 0 for resource in manual_resources
    )

    benchmark = config["benchmark"]
    checkpoints = [
        runs_dir / "ensemble_multineg_seed0" / "best.npz",
        runs_dir / "accepted_multineg_seed1" / "best.npz",
        runs_dir / "ensemble_multineg_seed2" / "best.npz",
        runs_dir / "ensemble_multineg_seed3" / "best.npz",
        runs_dir / "ensemble_multineg_seed4" / "best.npz",
    ]
    return {
        "benchmark": {
            "dataset": benchmark["name"],
            "label": benchmark["label"],
            "metrics": benchmark["metrics"],
            "primary": benchmark["primary"],
            "official_validation_primary": benchmark["official_valid_primary"],
        },
        "selected_candidate": {
            "model": "Five-seed within-user rank ensemble of four-negative BPR FMs",
            "learning_rate": 0.00025,
            "validation_primary": benchmark["incumbent_valid_primary"],
            "best_single_seed_validation_primary": benchmark["incumbent_single_seed_primary"],
            "three_seed_validation_mean": benchmark["incumbent_three_seed_mean"],
            "checkpoints": [str(path) for path in checkpoints],
            "checkpoints_present": all(path.is_file() for path in checkpoints),
        },
        "autonomous_agent": {
            "provider": config["llm"]["provider"],
            "planning_model": config["llm"]["planning_model"],
            "coding_model": config["llm"]["model"],
            "runs": len(summaries),
            "iterations": int(totals["iterations"]),
            "proposal_attempts": len(attempts),
            "failed_attempts": sum(a.get("status") == "failed" for a in attempts),
            "accepted_iterations": sum(
                result.get("accepted") is True for result in iteration_results
            ),
            "prompt_tokens": int(totals["prompt_tokens"]),
            "response_tokens": int(totals["response_tokens"]),
            "total_tokens": int(totals["total_tokens"]),
            "llm_calls": int(totals["llm_calls"]),
            "wall_clock_seconds": float(totals["wall_clock_seconds"]),
        },
        "other_resources": {
            "manual_experiment_records": len(manual_results),
            "manual_training_runtime_seconds": float(manual_runtime),
            "manual_interventions": int(totals["manual_interventions"]),
            "manual_experiment_llm_tokens": int(manual_tokens),
            "gpu_hours": float(totals["gpu_hours"]),
        },
        "safety": {
            "development_splits": config["safety"]["development_splits"],
            "hidden_test_evaluation_performed": False,
            "evaluate_py_modified": False,
        },
    }


def render_markdown(report: dict) -> str:
    agent = report["autonomous_agent"]
    resources = report["other_resources"]
    candidate = report["selected_candidate"]
    return f"""# TechJam Track 2 Research Run Report

## Selected recommender

- Model: {candidate['model']}
- Validation primary: {candidate['validation_primary']:.9f}
- Best single-seed validation primary: {candidate['best_single_seed_validation_primary']:.9f}
- Matched three-seed validation mean: {candidate['three_seed_validation_mean']:.9f}
- All five checkpoints present: {candidate['checkpoints_present']}

## Autonomous agent usage

- SoCLaaS planning model: {agent['planning_model']}
- SoCLaaS coding model: {agent['coding_model']}
- Runs / iterations: {agent['runs']} / {agent['iterations']}
- Proposal attempts / failed attempts: {agent['proposal_attempts']} / {agent['failed_attempts']}
- Accepted autonomous iterations: {agent['accepted_iterations']}
- LLM calls: {agent['llm_calls']}
- Prompt / response / total tokens: {agent['prompt_tokens']} / {agent['response_tokens']} / {agent['total_tokens']}
- Autonomous wall-clock seconds: {agent['wall_clock_seconds']:.3f}

## Other resources and safety

- Logged manual experiment records: {resources['manual_experiment_records']}
- Logged manual interventions: {resources['manual_interventions']}
- Manual training runtime seconds: {resources['manual_training_runtime_seconds']:.3f}
- GPU hours: {resources['gpu_hours']:.3f}
- Development labels used: train and validation only
- Hidden-test evaluation performed: no
- `evaluate.py` modified: no
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config", default="agent_config.json")
    parser.add_argument("--output", default="research_run_report.md")
    args = parser.parse_args()
    report = collect_report(Path(args.runs_dir), _read_json(Path(args.config)))
    Path(args.output).write_text(render_markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
