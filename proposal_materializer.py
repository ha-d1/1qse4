"""Turn structured full-file updates into deterministic unified diffs."""
from __future__ import annotations

import difflib
from pathlib import Path

from experiment_schema import ExperimentProposal


def _normalise_generated_source(content: str) -> str:
    """Remove harmless LLM whitespace defects and ensure patch-safe EOF newlines."""
    lines = [line.rstrip() for line in content.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def materialize_patch(project_root: str | Path, proposal: ExperimentProposal) -> str:
    if not proposal.file_updates:
        return proposal.patch

    root = Path(project_root).resolve()
    updates = {
        update["path"]: _normalise_generated_source(update["content"])
        for update in proposal.file_updates
    }
    undeclared = set(updates) - set(proposal.target_files)
    if undeclared:
        raise ValueError(
            "file_updates contain undeclared target paths: "
            f"targets={proposal.target_files}, undeclared={sorted(undeclared)}"
        )

    chunks = []
    for relative_path in sorted(updates):
        path = root / relative_path
        old_text = path.read_text() if path.exists() else ""
        new_text = updates[relative_path]
        if new_text == old_text:
            continue
        from_name = f"a/{relative_path}" if path.exists() else "/dev/null"
        to_name = f"b/{relative_path}"
        chunks.extend(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=from_name,
                tofile=to_name,
                n=3,
            )
        )
    patch = "".join(chunks)
    if not patch.strip():
        raise ValueError("Proposal file updates do not change any candidate files")
    return patch
