"""Construct the configured research-planner provider."""
from __future__ import annotations

from typing import Any, Mapping

from soclaas_client import SoCLaaSClient


def create_llm_client(config: Mapping[str, Any]) -> Any:
    provider = str(config.get("provider", "soclaas")).lower()
    if provider == "soclaas":
        return SoCLaaSClient(
            model=config["model"],
            planning_model=config.get("planning_model", "qwen3.6:35b"),
            max_attempts=int(config.get("max_attempts", 3)),
            temperature=float(config.get("temperature", 0.2)),
            planning_max_tokens=int(config.get("planning_max_tokens", 1200)),
            reflection_max_tokens=int(config.get("reflection_max_tokens", 900)),
            coding_max_tokens=int(config.get("coding_max_tokens", 7000)),
            coding_compact_max_tokens=int(config.get("coding_compact_max_tokens", 4500)),
            request_timeout_seconds=float(config.get("request_timeout_seconds", 180)),
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
