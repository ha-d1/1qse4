"""Run bounded experiments without invoking a shell."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass
class ProcessResult:
    status: str
    return_code: int | None
    runtime_seconds: float
    stdout: str
    stderr: str


class ExperimentRunner:
    def __init__(self, project_root: str | Path, timeout_seconds: float = 1200) -> None:
        self.project_root = Path(project_root).resolve()
        self.timeout_seconds = timeout_seconds

    def run(self, command: Sequence[str], env: Mapping[str, str] | None = None) -> ProcessResult:
        if not command:
            raise ValueError("Experiment command cannot be empty")
        executable = Path(command[0]).name
        if executable not in {"python", "python3"} and not executable.startswith("python3."):
            raise ValueError(f"Only Python experiment commands are allowed; got {command[0]!r}")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(command),
                cwd=self.project_root,
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            status = "success" if completed.returncode == 0 else "failed"
            return ProcessResult(
                status=status,
                return_code=completed.returncode,
                runtime_seconds=time.monotonic() - started,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return ProcessResult(
                status="timeout",
                return_code=None,
                runtime_seconds=time.monotonic() - started,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            )
