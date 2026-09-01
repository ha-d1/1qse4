"""Build a compact, tracked Track 2 evidence bundle from ignored local runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from generate_research_report import collect_report


ACCEPTED_CHECKPOINTS = (
    ("seed0.npz", "ensemble_multineg_seed0/best.npz"),
    ("seed1.npz", "accepted_multineg_seed1/best.npz"),
    ("seed2.npz", "ensemble_multineg_seed2/best.npz"),
    ("seed3.npz", "ensemble_multineg_seed3/best.npz"),
    ("seed4.npz", "ensemble_multineg_seed4/best.npz"),
    ("hour_seed0.npz", "hour_feature_seed0/best.npz"),
    ("hour_seed1.npz", "hour_feature_seed1/best.npz"),
    ("hour_seed2.npz", "hour_feature_seed2/best.npz"),
    ("session_seed2.npz", "session_feature_seed2/best.npz"),
)
RECENT_INTEREST_DIRECTIONS = (
    "short-window recent-interest exposure features",
    "long-window recent-interest exposure features",
    "time-decayed recent-interest exposure features",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_iteration(path: Path) -> dict:
    result = _read_json(path)
    attempts = result.get("attempts", [])
    hypotheses = [
        attempt.get("proposal", {}).get("hypothesis")
        for attempt in attempts
        if attempt.get("proposal", {}).get("hypothesis")
    ]
    return {
        "run_id": path.parents[2].name,
        "iteration": result.get("iteration"),
        "direction": result.get("planning", {}).get("plan", {}).get("direction"),
        "status": result.get("status"),
        "accepted": result.get("accepted", False),
        "primary": (result.get("metrics") or {}).get("primary"),
        "improvement_over_prior_best": result.get("improvement_over_prior_best"),
        "hypothesis": hypotheses[-1] if hypotheses else None,
        "attempts": len(attempts),
        "failed_attempts": sum(attempt.get("status") == "failed" for attempt in attempts),
        "errors": [
            str(attempt.get("error"))[-800:]
            for attempt in attempts
            if attempt.get("error")
        ],
        "reflection": (result.get("reflection") or {}).get("content"),
        "reflection_status": (result.get("reflection") or {}).get("status"),
        "resources": result.get("resources", {}),
    }


def recent_interest_campaign(runs_dir: Path) -> dict:
    """Link the bounded primary run and approved recovery into one auditable campaign."""
    completed = {}
    relevant_run_ids = set()
    failures = []
    for path in sorted(runs_dir.glob("run_*/iterations/iteration_*/result.json")):
        result = _read_json(path)
        direction = result.get("planning", {}).get("plan", {}).get("direction")
        run_id = path.parents[2].name
        if direction in RECENT_INTEREST_DIRECTIONS and result.get("status") == "success":
            attempt = (result.get("attempts") or [{}])[-1]
            completed[direction] = {
                "run_id": run_id,
                "iteration": result.get("iteration"),
                "direction": direction,
                "hypothesis": result.get("planning", {}).get("plan", {}).get("hypothesis"),
                "standalone_metrics": attempt.get("standalone_metrics"),
                "ensemble_metrics": result.get("metrics"),
                "accepted": result.get("accepted", False),
                "reflection_status": result.get("reflection", {}).get("status"),
                "reflection_decision": result.get("reflection", {})
                .get("content", {})
                .get("direction_decision"),
            }
            relevant_run_ids.add(run_id)
    for run_id in sorted(relevant_run_ids):
        for path in sorted((runs_dir / run_id / "iterations").glob("iteration_*/result.json")):
            result = _read_json(path)
            if result.get("status") != "success":
                failures.append(
                    {
                        "run_id": run_id,
                        "iteration": result.get("iteration"),
                        "error": result.get("error"),
                        "reflection_status": result.get("reflection", {}).get("status"),
                    }
                )
    resources = {
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "wall_clock_seconds": 0.0,
        "manual_interventions": 0,
        "gpu_hours": 0.0,
    }
    for run_id in sorted(relevant_run_ids):
        summary_path = runs_dir / run_id / "summary.json"
        if not summary_path.is_file():
            continue
        usage = _read_json(summary_path).get("resources", {})
        for field in resources:
            resources[field] += usage.get(field, 0) or 0
    ordered = [completed[direction] for direction in RECENT_INTEREST_DIRECTIONS if direction in completed]
    return {
        "status": "complete" if len(ordered) == len(RECENT_INTEREST_DIRECTIONS) else "incomplete",
        "campaign": "Qwen-controlled recent-interest FM template study",
        "run_ids": sorted(relevant_run_ids),
        "completed_scientific_experiments": ordered,
        "implementation_failures": failures,
        "best_validation_primary_after_campaign": 0.6051324605941772,
        "accepted_candidate": False,
        "resources": resources,
        "metric_audit_notes": [
            "Authoritative metrics are the standalone_metrics and ensemble_metrics fields above.",
            "The raw time-decayed Qwen reflection mislabeled primary as GAUC and compared nDCG@5 "
            "with a primary score; the raw reflection is preserved in experiment_index.json.",
        ],
        "hidden_test_labels_evaluated": False,
    }


def latest_sustained_run(runs_dir: Path, minimum_iterations: int = 3) -> dict | None:
    candidates = []
    for run_dir in sorted(runs_dir.glob("run_*")):
        paths = sorted(run_dir.glob("iterations/iteration_*/result.json"))
        if len(paths) >= minimum_iterations:
            iterations = [compact_iteration(path) for path in paths]
            completed = [
                item
                for item in iterations
                if item.get("status") == "success"
                and item.get("primary") is not None
                and item.get("reflection_status") == "success"
            ]
            if len(completed) < minimum_iterations:
                continue
            candidates.append(
                {
                    "run_id": run_dir.name,
                    "summary": _read_json(run_dir / "summary.json"),
                    "iterations": iterations,
                }
            )
    return candidates[-1] if candidates else None


def build_package(root: Path, output_dir: Path, include_checkpoints: bool = True) -> dict:
    runs_dir = root / "runs"
    config = _read_json(root / "agent_config.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = collect_report(runs_dir, config)
    report["selected_candidate"]["checkpoints"] = [
        f"evidence/checkpoints/{name}" for name, _ in ACCEPTED_CHECKPOINTS
    ]
    _write_json(output_dir / "resource_report.json", report)

    iteration_paths = sorted(runs_dir.glob("run_*/iterations/iteration_*/result.json"))
    index = [compact_iteration(path) for path in iteration_paths]
    _write_json(output_dir / "experiment_index.json", index)

    campaign = recent_interest_campaign(runs_dir)
    _write_json(output_dir / "recent_interest_campaign.json", campaign)

    sustained = latest_sustained_run(runs_dir)
    _write_json(
        output_dir / "sustained_run.json",
        sustained
        or {
            "status": "missing",
            "requirement": "Run at least three iterations in one agent process.",
        },
    )

    checkpoint_records = []
    for destination_name, relative_source in ACCEPTED_CHECKPOINTS:
        source = runs_dir / relative_source
        if not source.is_file():
            checkpoint_records.append(
                {"file": f"checkpoints/{destination_name}", "present": False}
            )
            continue
        destination = output_dir / "checkpoints" / destination_name
        if include_checkpoints:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        checkpoint_records.append(
            {
                "file": f"checkpoints/{destination_name}",
                "present": include_checkpoints and destination.is_file(),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "objective": "four-negative BPR FM",
                "learning_rate": 0.00025,
                "feature_set": (
                    "hour"
                    if destination_name.startswith("hour_")
                    else "session"
                    if destination_name.startswith("session_")
                    else "base"
                ),
            }
        )
    _write_json(output_dir / "checkpoint_manifest.json", checkpoint_records)

    evaluate_diff = subprocess.run(
        ["git", "diff", "54ec8c8..HEAD", "--", "evaluate.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "KuaiRand-Pure within-user ranking",
        "label": "long_view",
        "metrics": ["GAUC", "nDCG@5"],
        "primary": "mean(GAUC, nDCG@5)",
        "hidden_test_labels_evaluated": False,
        "evaluate_py_changed_from_starter": bool(evaluate_diff.stdout.strip()),
        "autonomous_iterations": len(index),
        "reflections_present": sum(item["reflection_status"] == "success" for item in index),
        "sustained_run_present": sustained is not None,
        "checkpoint_count": sum(record.get("present", False) for record in checkpoint_records),
        "recent_interest_campaign_present": campaign["status"] == "complete",
        "recent_interest_completed_experiments": len(
            campaign["completed_scientific_experiments"]
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)

    readme = """# Track 2 Evidence Package

This directory is the compact, Git-tracked audit trail for the autonomous recommender-system
research agent. It contains no API key, hidden-test labels, or hidden-test metric. Raw local runs
remain ignored because they contain large generated patches and transient outputs.

## Contents

- `manifest.json`: benchmark and evidence-integrity summary.
- `resource_report.json`: aggregate Qwen usage, runtime, iterations, failures, and interventions.
- `experiment_index.json`: compact outcome and reflection for every autonomous iteration.
- `recent_interest_campaign.json`: linked three-template Qwen study, recovery, metric audit, and usage.
- `sustained_run.json`: the latest run containing at least three iterations in one process.
- `checkpoint_manifest.json`: SHA-256 checksums and metadata for the nine accepted models.
- `checkpoints/`: five base, three hour-aware, and one session-aware BPR FM checkpoints.

## Reproduce the validation submission

From the repository root, after downloading KuaiRand-Pure:

```bash
.venv/bin/python make_submission.py \\
  --checkpoint evidence/checkpoints/seed0.npz \\
  --checkpoint evidence/checkpoints/seed1.npz \\
  --checkpoint evidence/checkpoints/seed2.npz \\
  --checkpoint evidence/checkpoints/seed3.npz \\
  --checkpoint evidence/checkpoints/seed4.npz \\
  --checkpoint evidence/checkpoints/hour_seed0.npz \\
  --checkpoint evidence/checkpoints/hour_seed1.npz \\
  --checkpoint evidence/checkpoints/hour_seed2.npz \\
  --checkpoint evidence/checkpoints/session_seed2.npz \\
  --data-dir /path/to/KuaiRand-Pure/data \\
  --split valid \\
  --output valid_submission.csv
.venv/bin/python submit.py --check --split valid \\
  --data_dir /path/to/KuaiRand-Pure/data valid_submission.csv
```

Run `python build_evidence_package.py` after new experiments to refresh this package. Run
`python verify_evidence_package.py` to verify checkpoint hashes, reflection coverage, and the
sustained-run requirement. Use `--split test` only once for the final unlabelled prediction.
"""
    (output_dir / "README.md").write_text(readme)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="evidence")
    parser.add_argument("--without-checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = build_package(
        root,
        root / args.output_dir,
        include_checkpoints=not args.without_checkpoints,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
