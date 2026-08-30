"""Minimal Gemini client for structured research proposals.

The project remains Python 3.9 compatible, so it uses the fully supported
generate_content interface available in google-genai 1.x.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

from experiment_schema import ExperimentProposal

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "description": "One falsifiable ML hypothesis."},
        "reasoning": {"type": "string", "description": "Why this follows from prior results."},
        "target_files": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Files under candidate/ only.",
        },
        "expected_effect": {"type": "string"},
        "risk": {"type": "string"},
        "patch": {
            "type": "string",
            "description": "A complete git-compatible unified diff changing candidate/ files only.",
        },
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "description": "A Python command represented as argv, never shell syntax.",
        },
    },
    "required": [
        "hypothesis",
        "reasoning",
        "target_files",
        "expected_effect",
        "risk",
        "patch",
        "command",
    ],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTION = """You are the research planner for KuaiRand-Pure.
Optimise validation primary = mean(GAUC, nDCG@5), using long_view labels.
You may use train and validation only. Never request, infer, or evaluate test labels.
Propose exactly one controlled, falsifiable change per iteration.
Only files under candidate/ may be changed. Do not modify official evaluation code.
Return a complete git-compatible unified diff in the patch field.
Use prior evidence, avoid repeated failed ideas, and consider runtime cost.
Return only the schema-constrained proposal."""


@dataclass
class GeminiProposalResult:
    proposal: ExperimentProposal
    usage: Dict[str, int]
    interaction_id: str | None
    raw_text: str


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_attempts: int = 3,
        client: Any | None = None,
    ) -> None:
        self.model = os.environ.get("GEMINI_MODEL", model)
        self.max_attempts = max_attempts
        if client is None:
            if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
                raise RuntimeError("Set GEMINI_API_KEY before starting the autonomous agent")
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Install dependencies with: python3 -m pip install -r requirements.txt") from exc
            client = genai.Client()
        self.client = client

    @staticmethod
    def _usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "response_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        }

    def propose(self, context: Dict[str, Any]) -> GeminiProposalResult:
        prompt = (
            "Review the following bounded experiment context and propose the next single experiment.\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "system_instruction": SYSTEM_INSTRUCTION,
                        "temperature": 0.2,
                        "response_mime_type": "application/json",
                        "response_schema": PROPOSAL_SCHEMA,
                    },
                )
                raw_text = response.text
                payload = json.loads(raw_text)
                proposal = ExperimentProposal(**payload)
                proposal.validate()
                return GeminiProposalResult(
                    proposal=proposal,
                    usage=self._usage(response),
                    interaction_id=getattr(response, "response_id", None),
                    raw_text=raw_text,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"Gemini proposal failed after {self.max_attempts} attempts") from last_error

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
