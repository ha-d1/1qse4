"""Provider-neutral proposal records and prompt construction."""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Dict

from experiment_schema import ExperimentProposal


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "reasoning": {"type": "string"},
        "target_files": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "expected_effect": {"type": "string"},
        "risk": {"type": "string"},
        "patch": {"type": "string"},
        "file_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
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

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "reasoning": {"type": "string"},
        "direction": {"type": "string"},
        "target_files": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "implementation_requirements": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "risk": {"type": "string"},
    },
    "required": [
        "hypothesis",
        "reasoning",
        "direction",
        "target_files",
        "implementation_requirements",
        "command",
        "risk",
    ],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTION = """You are the research planner for KuaiRand-Pure.
Optimise validation primary = mean(GAUC, nDCG@5), using long_view labels.
Use train and validation only. Never request, infer, or evaluate test labels.
Propose exactly one controlled, falsifiable change per iteration.
Only files under candidate/ may be changed. Do not modify official evaluation code.
Prefer a concise unified diff in patch and omit file_updates. Use exact
--- a/candidate/... and +++ b/candidate/... headers with enough unchanged context for git apply.
Use full-file file_updates only as a fallback when a safe concise diff is impractical; if used,
set patch to an empty string. Never populate both patch and file_updates.
Use prior evidence, avoid repeated failed ideas, and consider runtime cost.
The implementation must genuinely optimise the objective named in the hypothesis. Do not
describe a listwise loss while secretly using BPR updates, and do not pass score gradients
to an API that expects binary labels. Ensure every command-line objective is implemented by
both candidate/model.py and candidate/train.py. Preserve a runnable control path.
Preserve candidate/train.py's `--checkpoint-out` interface and write the validation-selected
model state to that exact path. The harness requires this artifact after every successful run.
Treat read_only_reference_sources as exact API documentation: never invent methods that are
absent from those sources, and never include those protected files in target_files.
Root data.py is read-only. If the approved plan creates candidate/data.py, import its new names
with `from candidate.data import ...`; never import candidate-only helpers with `from data import`.
Validation labels are evaluation-only: never use them to construct history, features, targets,
or training examples. History statistics that depend on labels must be fitted on train labels and
then frozen for validation. Validation impression features may be transformed without labels.
For multi-objective learning, use development_data.load_training_auxiliary. It returns train-only
arrays aligned exactly with splits['train']; long_view remains the primary target and the only
validation target. Never attempt to load auxiliary validation or test targets.
Expose auxiliary behaviour through removable --aux-*, --auxiliary-*, or --multitask-* command
options. Auxiliary gradients must change checkpointed inference parameters V, W, or b; updating
only a detached auxiliary scalar or head that is absent from final ranking will fail preflight.
Before returning, perform a symbol-completeness check over the proposed files: every imported
name and every method called on the FM instance must either already exist in the supplied
sources or be implemented in the proposed patch or file_updates. In particular, adding an objective dispatcher is
incomplete unless its loss/update method is also present and callable with the supplied arguments.
Raw user_id and video_id values are opaque strings. Never cast them with int() or store them
in integer NumPy arrays; group by string or create an explicit dictionary mapping.
load_development_splits returns dictionaries of raw seven-field row tuples, not encoded
(X, y, users) triples. Use data.encode(splits) before unpacking encoded arrays, following the
existing candidate/model.py control path.
Return only one JSON object matching the requested schema."""

PLANNER_SYSTEM_INSTRUCTION = """You are the senior autonomous ML researcher for
KuaiRand-Pure. Select exactly one falsifiable experiment that can improve validation
mean(GAUC, nDCG@5) beyond the user's existing BPR candidate. Use long_view and train/valid
only. Do not rediscover plain BPR, access test labels, or modify files outside candidate/.
Choose a scientifically valid objective and provide concrete mathematical and implementation
requirements for a coding model. Keep the plan narrow enough for one bounded iteration.
Use recent_experiments as persistent memory. If the latest attempt reached preflight and has a
small, localized implementation error, prefer a focused repair of that promising experiment;
otherwise avoid repeating the failed implementation.
Treat completed_autonomous_directions as closed research branches. Never propose a direction
whose decision is rejected, even if you believe a minor implementation variant might improve it.
Return only one JSON object matching the requested schema."""


@dataclass
class ProposalResult:
    proposal: ExperimentProposal
    usage: Dict[str, int]
    interaction_id: str | None
    raw_text: str


@dataclass
class PlanningResult:
    plan: Dict[str, Any]
    usage: Dict[str, int]
    interaction_id: str | None
    raw_text: str


def proposal_prompt(context: Dict[str, Any], recovery: Dict[str, Any] | None = None) -> str:
    task = "Propose the next single bounded experiment."
    if context.get("approved_research_plan"):
        task = (
            "Implement the approved_research_plan exactly. Do not substitute another "
            "objective or omit any required target file."
        )
    request = {
        "task": task,
        "context": context,
        "proposal_schema": PROPOSAL_SCHEMA,
    }
    if recovery is not None:
        request["recovery"] = recovery
        request["task"] = (
            "Repair the failed proposal within this same iteration. Return a corrected, "
            "complete proposal and do not repeat the reported failure. Recheck every imported "
            "symbol, model method, function signature, and CLI option before returning."
        )
    return json.dumps(request, ensure_ascii=False, indent=2)


def planning_prompt(context: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Select the next single bounded research experiment.",
            "context": context,
            "plan_schema": PLAN_SCHEMA,
        },
        ensure_ascii=False,
        indent=2,
    )


def parse_plan(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    payload = json.loads(text)
    missing = set(PLAN_SCHEMA["required"]) - set(payload)
    if missing:
        raise ValueError(f"Invalid research plan fields: missing={sorted(missing)}")
    payload = {key: payload[key] for key in PLAN_SCHEMA["properties"] if key in payload}
    # The starter candidate calls its FM implementation model.py. Planning models sometimes
    # describe that role as fm.py; canonicalise the harmless alias before the coder contract
    # is enforced so a correct implementation is not rejected over naming alone.
    target_aliases = {"candidate/fm.py": "candidate/model.py"}
    payload["target_files"] = list(
        dict.fromkeys(target_aliases.get(path, path) for path in payload["target_files"])
    )
    payload["implementation_requirements"] = [
        requirement.replace("candidate/fm.py", "candidate/model.py").replace(
            "In fm.py", "In model.py"
        )
        for requirement in payload["implementation_requirements"]
    ]
    if not payload["target_files"] or any(
        not path.startswith("candidate/") or ".." in path.split("/")
        for path in payload["target_files"]
    ):
        raise ValueError("Research plan targets must remain under candidate/")
    command = payload["command"]
    if len(command) == 1 and isinstance(command[0], str):
        command = shlex.split(command[0])
    if command and command[0] == "candidate/train.py":
        command = ["python"] + command
    if len(command) >= 3 and command[1:3] == ["-m", "candidate.train"]:
        command = [command[0], "candidate/train.py"] + command[3:]
    if len(command) >= 2 and command[1].startswith("candidate/"):
        original_entrypoint = command[1]
        command[1] = "candidate/train.py"
        if "candidate/train.py" not in payload["target_files"]:
            payload["target_files"].append("candidate/train.py")
        payload["implementation_requirements"].append(
            f"Expose {original_entrypoint} through the safety-controlled candidate/train.py entrypoint."
        )
    if len(command) < 2 or command[1] != "candidate/train.py":
        raise ValueError(f"Research plan command must run candidate/train.py; got {command!r}")
    payload["command"] = command
    return payload


def parse_proposal(raw_text: str) -> ExperimentProposal:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    payload = json.loads(text)
    # Some coding models redundantly populate both supported change formats despite the
    # contract. Prefer deterministic full-file materialisation when those updates are
    # structurally usable; otherwise retain the concise patch. This avoids a second LLM call
    # for a mechanical formatting defect while keeping ExperimentProposal.validate strict.
    if str(payload.get("patch", "")).strip() and payload.get("file_updates"):
        updates = payload["file_updates"]
        declared = set(payload.get("target_files", []))
        update_paths = {
            update.get("path") for update in updates if isinstance(update, dict)
        }
        if updates and None not in update_paths and update_paths.issubset(declared):
            payload["patch"] = ""
        else:
            payload["file_updates"] = []
    proposal = ExperimentProposal(**payload)
    proposal.validate()
    return proposal


def redact_secrets(value: str) -> str:
    """Remove configured credentials before error text reaches persistent logs."""
    cleaned = str(value)
    for name in ("SOCLAAS_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    return cleaned
