"""Validate, apply, and reverse Gemini-generated patches."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable


class PatchError(RuntimeError):
    pass


class PatchManager:
    def __init__(self, project_root: str | Path, mutable_roots: Iterable[str] = ("candidate",)) -> None:
        self.project_root = Path(project_root).resolve()
        self.mutable_roots = tuple(mutable_roots)

    def _paths(self, patch: str) -> list[str]:
        paths = []
        for match in re.finditer(r"^(?:---|\+\+\+)\s+(?:a/|b/)?([^\t\n]+)", patch, re.MULTILINE):
            path = match.group(1).strip()
            if path != "/dev/null":
                paths.append(path)
        return sorted(set(paths))

    def validate(self, patch: str) -> list[str]:
        if not patch.strip():
            raise PatchError("Patch is empty")
        paths = self._paths(patch)
        if not paths:
            raise PatchError("Patch has no file headers")
        for path in paths:
            parts = Path(path).parts
            if ".." in parts or not any(parts and parts[0] == root for root in self.mutable_roots):
                raise PatchError(f"Patch target is outside mutable roots: {path}")
            resolved = (self.project_root / path).resolve()
            if self.project_root not in resolved.parents:
                raise PatchError(f"Patch escapes project root: {path}")
        return paths

    def _git_apply(self, patch: str, *options: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "apply", *options, "-"],
            cwd=self.project_root,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )

    def apply(self, patch: str) -> list[str]:
        paths = self.validate(patch)
        checked = self._git_apply(patch, "--check", "--whitespace=error-all")
        if checked.returncode != 0:
            raise PatchError(checked.stderr.strip() or "git apply --check failed")
        applied = self._git_apply(patch, "--whitespace=error-all")
        if applied.returncode != 0:
            raise PatchError(applied.stderr.strip() or "git apply failed")
        return paths

    def rollback(self, patch: str) -> None:
        reversed_patch = self._git_apply(patch, "--reverse")
        if reversed_patch.returncode != 0:
            raise PatchError(reversed_patch.stderr.strip() or "git apply --reverse failed")
