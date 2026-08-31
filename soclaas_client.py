"""OpenAI-compatible client for the NUS School of Computing LLM service."""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from llm_common import (
    PLAN_SCHEMA,
    PLANNER_SYSTEM_INSTRUCTION,
    REFLECTION_SYSTEM_INSTRUCTION,
    PlanningResult,
    ProposalResult,
    ReflectionResult,
    SYSTEM_INSTRUCTION,
    parse_plan,
    parse_proposal,
    parse_reflection,
    planning_prompt,
    proposal_prompt,
    reflection_prompt,
)

DEFAULT_BASE_URL = "https://soclaas-api.comp.nus.edu.sg/v1"


class SoCLaaSRequestError(RuntimeError):
    """Provider failure carrying any billable usage returned before parsing failed."""

    def __init__(self, message: str, usage: Dict[str, int]) -> None:
        super().__init__(message)
        self.usage = usage


def _add_usage(total: Dict[str, int], addition: Dict[str, int]) -> None:
    for field in ("prompt_tokens", "response_tokens", "total_tokens"):
        total[field] += int(addition.get(field, 0))


class SoCLaaSClient:
    def __init__(
        self,
        model: str = "qwen3-coder-next",
        max_attempts: int = 3,
        temperature: float = 0.2,
        planning_model: str = "qwen3.6:35b",
        planning_max_tokens: int = 1200,
        reflection_max_tokens: int = 900,
        coding_max_tokens: int = 7000,
        coding_compact_max_tokens: int = 4500,
        request_timeout_seconds: float = 180.0,
        client: Any | None = None,
    ) -> None:
        self.model = os.environ.get("SOCLAAS_MODEL", model)
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.planning_model = os.environ.get("SOCLAAS_PLANNING_MODEL", planning_model)
        self.planning_max_tokens = int(planning_max_tokens)
        self.reflection_max_tokens = int(reflection_max_tokens)
        self.coding_max_tokens = int(coding_max_tokens)
        self.coding_compact_max_tokens = int(coding_compact_max_tokens)
        self.request_timeout_seconds = float(request_timeout_seconds)
        if (
            self.planning_max_tokens < 256
            or self.reflection_max_tokens < 256
            or self.coding_max_tokens < 1024
        ):
            raise ValueError("SoCLaaS output token limits are too small for structured proposals")
        if self.coding_compact_max_tokens < 1024 or self.coding_compact_max_tokens > self.coding_max_tokens:
            raise ValueError("Invalid compact coding token limit")
        if client is None:
            api_key = os.environ.get("SOCLAAS_API_KEY")
            if not api_key:
                raise RuntimeError("Set SOCLAAS_API_KEY before starting the autonomous agent")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Install dependencies with: python3 -m pip install -r requirements.txt"
                ) from exc
            base_url = os.environ.get("SOCLAAS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.request_timeout_seconds,
                max_retries=0,
            )
        self.client = client

    @staticmethod
    def _usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        response_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", prompt + response_tokens) or 0)
        return {
            "prompt_tokens": prompt,
            "response_tokens": response_tokens,
            "total_tokens": total,
        }

    def _request(self, context: Dict[str, Any], recovery: Dict[str, Any] | None) -> ProposalResult:
        last_error: Exception | None = None
        cumulative_usage = {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": proposal_prompt(context, recovery)},
                    ],
                    temperature=self.temperature,
                    max_tokens=min(
                        self.coding_max_tokens,
                        int(
                            context.get("llm_budget", {}).get(
                                "coding_max_tokens", self.coding_max_tokens
                            )
                        ),
                    ),
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                _add_usage(cumulative_usage, self._usage(response))
                if not raw_text:
                    raise ValueError("SoCLaaS returned an empty proposal")
                return ProposalResult(
                    proposal=parse_proposal(
                        raw_text, approved_plan=context.get("approved_research_plan")
                    ),
                    usage=cumulative_usage,
                    interaction_id=getattr(response, "id", None),
                    raw_text=raw_text,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise SoCLaaSRequestError(
            f"SoCLaaS proposal failed after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}",
            cumulative_usage,
        ) from last_error

    def propose(self, context: Dict[str, Any]) -> ProposalResult:
        return self._request(context, recovery=None)

    def plan(self, context: Dict[str, Any]) -> PlanningResult:
        last_error: Exception | None = None
        cumulative_usage = {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.planning_model,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": planning_prompt(context)},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.planning_max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                _add_usage(cumulative_usage, self._usage(response))
                if not raw_text:
                    raise ValueError("SoCLaaS returned an empty research plan")
                return PlanningResult(
                    plan=parse_plan(raw_text, context=context),
                    usage=cumulative_usage,
                    interaction_id=getattr(response, "id", None),
                    raw_text=raw_text,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise SoCLaaSRequestError(
            f"SoCLaaS planning failed after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}",
            cumulative_usage,
        ) from last_error

    def repair(self, context: Dict[str, Any], recovery: Dict[str, Any]) -> ProposalResult:
        return self._request(context, recovery=recovery)

    def reflect(self, context: Dict[str, Any]) -> ReflectionResult:
        last_error: Exception | None = None
        cumulative_usage = {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.planning_model,
                    messages=[
                        {"role": "system", "content": REFLECTION_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": reflection_prompt(context)},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.reflection_max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                _add_usage(cumulative_usage, self._usage(response))
                if not raw_text:
                    raise ValueError("SoCLaaS returned an empty reflection")
                return ReflectionResult(
                    reflection=parse_reflection(raw_text),
                    usage=cumulative_usage,
                    interaction_id=getattr(response, "id", None),
                    raw_text=raw_text,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise SoCLaaSRequestError(
            f"SoCLaaS reflection failed after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}",
            cumulative_usage,
        ) from last_error

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
