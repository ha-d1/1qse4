"""Track the resource fields required by the competition report."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass


@dataclass
class ResourceUsage:
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    iterations: int = 0
    manual_interventions: int = 0
    gpu_hours: float = 0.0
    wall_clock_seconds: float = 0.0


class ResourceTracker:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.usage = ResourceUsage()

    def add_llm_usage(self, prompt: int, response: int, total: int | None = None) -> None:
        self.usage.prompt_tokens += int(prompt)
        self.usage.response_tokens += int(response)
        self.usage.total_tokens += int(total if total is not None else prompt + response)
        self.usage.llm_calls += 1

    def snapshot(self) -> dict:
        self.usage.wall_clock_seconds = time.monotonic() - self.started_at
        return asdict(self.usage)
