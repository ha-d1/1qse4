"""Shared, serialisable records for proposals and experiment results."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ExperimentProposal:
    hypothesis: str
    reasoning: str
    target_files: List[str]
    expected_effect: str
    risk: str
    patch: str
    command: List[str]

    def validate(self) -> None:
        if not self.hypothesis.strip() or not self.reasoning.strip():
            raise ValueError("Proposal requires a hypothesis and reasoning")
        if not self.target_files:
            raise ValueError("Proposal must name at least one target file")
        if not self.patch.strip():
            raise ValueError("Proposal must include a unified diff patch")
        if not self.command:
            raise ValueError("Proposal must provide an argv-style command")
        for path in self.target_files:
            if not path.startswith("candidate/") or ".." in path.split("/"):
                raise ValueError(f"Target is outside candidate/: {path}")


@dataclass
class ValidationMetrics:
    GAUC: float
    nDCG_at_5: float
    primary: float


@dataclass
class ExperimentResult:
    status: Literal["success", "failed", "timeout"]
    iteration: int
    metrics: Optional[ValidationMetrics] = None
    runtime_seconds: float = 0.0
    return_code: Optional[int] = None
    checkpoint: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
