"""Append-only, auditable run and iteration logging."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class RunLogger:
    def __init__(self, runs_dir: str | Path, run_id: str | None = None) -> None:
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        self.run_dir = Path(runs_dir).resolve() / run_id
        self.iterations_dir = self.run_dir / "iterations"
        self.iterations_dir.mkdir(parents=True, exist_ok=False)
        self.write_json("manifest.json", {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat()})

    def write_json(self, relative_path: str | Path, payload: Mapping[str, Any]) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
        return path

    def write_text(self, relative_path: str | Path, text: str) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def log_iteration(self, iteration: int, payload: Mapping[str, Any]) -> Path:
        return self.write_json(f"iterations/iteration_{iteration:03d}/result.json", payload)
