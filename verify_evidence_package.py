"""Verify the tracked Track 2 evidence bundle without accessing hidden-test labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence(evidence_dir: Path) -> dict:
    manifest = json.loads((evidence_dir / "manifest.json").read_text())
    sustained = json.loads((evidence_dir / "sustained_run.json").read_text())
    checkpoints = json.loads((evidence_dir / "checkpoint_manifest.json").read_text())
    campaign_path = evidence_dir / "recent_interest_campaign.json"
    if manifest["hidden_test_labels_evaluated"] is not False:
        raise ValueError("Evidence must not claim hidden-test label evaluation")
    if manifest["evaluate_py_changed_from_starter"] is not False:
        raise ValueError("evaluate.py differs from the protected starter version")
    iterations = sustained.get("iterations", [])
    if len(iterations) < 3:
        raise ValueError("Evidence does not contain a sustained run of at least three iterations")
    missing_reflections = [
        item["iteration"] for item in iterations if item.get("reflection_status") != "success"
    ]
    if missing_reflections:
        raise ValueError(f"Sustained-run reflections are missing: {missing_reflections}")
    completed_validations = [
        item for item in iterations if item.get("status") == "success" and item.get("primary") is not None
    ]
    if len(completed_validations) < 3:
        raise ValueError("Sustained run has fewer than three completed validation experiments")
    for record in checkpoints:
        path = evidence_dir / record["file"]
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"Checkpoint verification failed: {path}")
    campaign_experiments = 0
    if manifest.get("recent_interest_campaign_present"):
        campaign = json.loads(campaign_path.read_text())
        experiments = campaign.get("completed_scientific_experiments", [])
        if campaign.get("status") != "complete" or len(experiments) != 3:
            raise ValueError("Recent-interest autonomous campaign is incomplete")
        if any(item.get("reflection_status") != "success" for item in experiments):
            raise ValueError("Recent-interest campaign reflections are incomplete")
        if campaign.get("hidden_test_labels_evaluated") is not False:
            raise ValueError("Recent-interest campaign must not use hidden-test labels")
        campaign_experiments = len(experiments)
    return {
        "status": "verified",
        "sustained_run": sustained["run_id"],
        "iterations": len(iterations),
        "reflections": len(iterations) - len(missing_reflections),
        "completed_validations": len(completed_validations),
        "checkpoints": len(checkpoints),
        "recent_interest_experiments": campaign_experiments,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    print(json.dumps(verify_evidence(root / "evidence"), indent=2))


if __name__ == "__main__":
    main()
