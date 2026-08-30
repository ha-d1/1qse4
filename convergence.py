"""Competition budget and convergence tracking."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConvergenceState:
    epsilon: float = 0.002
    patience: int = 3
    max_iterations: int = 50
    max_wall_clock_seconds: float = 6 * 60 * 60
    best_score: float = float("-inf")
    iterations: int = 0
    consecutive_small_improvements: int = 0

    def observe(self, score: float) -> float:
        """Record one validation result and return improvement over prior best."""
        previous_best = self.best_score
        improvement = float("inf") if previous_best == float("-inf") else score - previous_best
        if score > self.best_score:
            self.best_score = score
        if improvement > self.epsilon:
            self.consecutive_small_improvements = 0
        else:
            self.consecutive_small_improvements += 1
        self.iterations += 1
        return improvement

    def stop_reason(self, elapsed_seconds: float) -> str | None:
        if self.iterations >= self.max_iterations:
            return "iteration_limit"
        if elapsed_seconds >= self.max_wall_clock_seconds:
            return "wall_clock_limit"
        if self.consecutive_small_improvements >= self.patience:
            return "converged"
        return None
