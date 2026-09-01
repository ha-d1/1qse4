"""Autonomous inspect, experiment, reflect, and finalize campaign controller."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

import numpy as np

from agent_codex import CodexCLIClient
from agent_prompts import (
    ACTION_NOTES,
    CANDIDATE_CONSTRAINTS,
    IMMUTABLE_RULES,
    PROTECTED_PATHS,
    SYSTEM_PROMPT,
    build_generate_prompt,
    build_repair_prompt,
)
from agent_sandbox import SandboxError, WorktreeSandbox
from agent_state import CampaignStateError, CampaignStore
from baseline import run_random
import data
from evaluate import evaluate
from experiment import (
    dependency_versions,
    make_result,
    prepare_candidate_data,
    profile_data,
    write_result,
)


CANONICAL_FILES = (
    "video_features_basic_pure.csv",
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
TERMINAL_ITERATION_STATES = {"accepted", "rejected", "failed"}
TERMINAL_CAMPAIGN_STATES = {
    "target_met", "target_unmet", "non_reproducible", "finalization_failed", "preflight_failed",
}
EPSILON = 0.002
PUBLISHED_FM_PRIMARY = 0.6016
MAX_INSPECTION_ROUNDS = 3
MAX_INSPECTION_BYTES = 64 * 1024
MAX_SEARCH_MATCHES = 100
MAX_CANDIDATE_REPAIRS = 3
MAX_FAILURE_FEEDBACK_BYTES = 16 * 1024
MAX_SOURCE_CONTEXT_BYTES = 256 * 1024


class ProtocolError(ValueError):
    pass


def _exact_keys(value, expected, form):
    if not isinstance(value, dict):
        raise ProtocolError(f"{form} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        unknown = sorted(actual - set(expected))
        missing = sorted(set(expected) - actual)
        details = []
        if unknown:
            details.append(f"unknown keys {unknown}")
        if missing:
            details.append(f"missing keys {missing}")
        raise ProtocolError(f"Invalid {form}: " + "; ".join(details))


def _required_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")


def validate_inspection_request(value):
    if not isinstance(value, dict) or "kind" not in value:
        raise ProtocolError("Inspection request must be an object with a kind")
    kind = value["kind"]
    if kind == "file":
        _exact_keys(value, ("kind", "path", "start", "end"), "file inspection")
        _required_string(value["path"], "file inspection path")
        for name in ("start", "end"):
            number = value[name]
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                raise ProtocolError(f"file inspection {name} must be a positive integer")
        if value["end"] < value["start"]:
            raise ProtocolError("file inspection end must be at least start")
    elif kind == "search":
        _exact_keys(value, ("kind", "path", "pattern"), "search inspection")
        _required_string(value["path"], "search inspection path")
        _required_string(value["pattern"], "search inspection pattern")
        if len(value["pattern"]) > 512:
            raise ProtocolError("search inspection pattern exceeds 512 characters")
        try:
            re.compile(value["pattern"])
        except re.error as exc:
            raise ProtocolError(f"Invalid search regular expression: {exc}") from None
    elif kind == "data_profile":
        _exact_keys(value, ("kind",), "data_profile inspection")
    else:
        raise ProtocolError(f"Unknown inspection kind: {kind!r}")
    return value


def validate_action(value):
    if not isinstance(value, dict) or "action" not in value:
        raise ProtocolError("Model action must be an object with an action")
    action = value["action"]
    if action == "inspect":
        _exact_keys(value, ("action", "requests"), "inspect action")
        requests = value["requests"]
        if not isinstance(requests, list) or not requests:
            raise ProtocolError("inspect requests must be a non-empty list")
        if len(requests) > 20:
            raise ProtocolError("inspect action exceeds 20 requests")
        for item in requests:
            validate_inspection_request(item)
    elif action == "experiment":
        _exact_keys(value, ("action", "name", "hypothesis", "source", "config", "expected_effect"),
                    "experiment action")
        for name in ("name", "hypothesis", "source", "expected_effect"):
            _required_string(value[name], f"experiment {name}")
        if not isinstance(value["config"], dict):
            raise ProtocolError("experiment config must be a JSON object")
    elif action == "finish":
        _exact_keys(value, ("action", "reason"), "finish action")
        _required_string(value["reason"], "finish reason")
    else:
        raise ProtocolError(f"Unknown model action: {action!r}")
    return value


def validate_reflection(value):
    _exact_keys(value, ("diagnosis", "evidence", "next_hypothesis", "stop"), "reflection")
    _required_string(value["diagnosis"], "reflection diagnosis")
    _required_string(value["next_hypothesis"], "reflection next_hypothesis")
    if not isinstance(value["evidence"], list) or not all(
            isinstance(item, str) for item in value["evidence"]):
        raise ProtocolError("reflection evidence must be a list of strings")
    if not isinstance(value["stop"], bool):
        raise ProtocolError("reflection stop must be a boolean")
    return value


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value




def _metrics_equal(left, right, tolerance=1e-8):
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(a - b) > tolerance:
                return False
        elif a != b:
            return False
    return True


class CampaignController:
    def __init__(self, client, sandbox, store, clock=time.monotonic, recorded_baseline=None):
        self.client = client
        self.sandbox = sandbox
        self.store = store
        self.clock = clock
        self.recorded_baseline = recorded_baseline
        self.state = None
        self.profile = None
        self.splits = None
        self.candidate_data_views = None
        self._session_started = None
        self._elapsed_before = 0.0

    def _elapsed(self):
        if self._session_started is None:
            return self._elapsed_before
        return self._elapsed_before + max(0.0, self.clock() - self._session_started)

    def _save(self):
        self.state["elapsed_seconds"] = self._elapsed()
        self.store.save(_plain(self.state))

    @staticmethod
    def _validate_limits(limits):
        required = {
            "max_iterations", "max_hours", "max_model_calls", "run_timeout", "memory_gb", "threads",
        }
        if set(limits) != required:
            raise ValueError(f"limits must contain exactly {sorted(required)}")
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_iterations", "max_model_calls", "run_timeout", "threads"):
            if int(limits[name]) != limits[name]:
                raise ValueError(f"{name} must be an integer")
            limits[name] = int(limits[name])
        limits["max_hours"] = float(limits["max_hours"])
        limits["memory_gb"] = float(limits["memory_gb"])
        return limits

    def _load_and_profile(self, data_dir):
        data_dir = Path(data_dir).resolve(strict=True)
        if not data_dir.is_dir():
            raise CampaignStateError(f"Data path is not a directory: {data_dir}")
        for filename in CANONICAL_FILES:
            path = data_dir / filename
            if path.is_symlink() or not path.is_file():
                raise CampaignStateError(f"Required canonical CSV is missing or unsafe: {path}")
        splits = data.load(data_dir)
        empty = [name for name in ("train", "valid", "test") if not splits.get(name)]
        if empty:
            raise CampaignStateError(f"Required dataset splits are empty: {empty}")
        first = profile_data(data_dir)
        second = profile_data(data_dir)
        if first != second:
            raise CampaignStateError("Dataset profile/fingerprint is not stable across repeated reads")
        self.splits = splits
        self.profile = first
        return data_dir

    def _prepare_candidate_views(self):
        self.candidate_data_views = prepare_candidate_data(
            self.state["data_dir"],
            self.store.campaign_dir / "candidate-data",
            self.state["data_fingerprint"],
        )

    def run(self, data_dir, limits):
        limits = self._validate_limits(dict(limits))
        with self.store.lock():
            if self.store.exists() or self.store.campaign_dir.exists():
                raise CampaignStateError(
                    f"Campaign {self.store.campaign_id!r} already exists; use agent.py resume")
            data_dir = self._load_and_profile(data_dir)
            if self.sandbox.ref_exists(self.sandbox.base_ref):
                raise CampaignStateError(
                    f"Campaign {self.store.campaign_id!r} already has Git state; use agent.py resume")
            base_commit = self.sandbox.create_base_snapshot()
            initial = {
                "base_commit": base_commit,
                "best_commit": base_commit,
                "best_config": {},
                "best_ensemble_metrics": None,
                "best_score_hashes": {},
                "best_exploratory_primary": None,
                "target_primary": None,
                "data_dir": str(data_dir),
                "data_fingerprint": self.profile["fingerprint_sha256"],
                "data_profile": self.profile,
                "codex": dict(self.client.identity),
                "limits": limits,
                "counters": {
                    "model_calls": 0,
                    "completed_iterations": 0,
                    "consecutive_model_failures": 0,
                    "consecutive_no_improvement": 0,
                },
                "baseline": None,
                "iterations": [],
                "elapsed_seconds": 0.0,
                "status": "preflight",
                "stop_reason": None,
            }
            self.state = self.store.initialize(initial)
            self._session_started = self.clock()
            self._elapsed_before = 0.0
            try:
                self._prepare_candidate_views()
                self._prepare_baseline()
                return self._campaign_loop()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if self.state["status"] == "preflight":
                    self.state["status"] = "preflight_failed"
                    self.state["stop_reason"] = f"{type(exc).__name__}: {exc}"
                    self._save()
                raise

    def resume(self):
        with self.store.lock():
            self.state = self.store.load()
            if self.state["codex"] != self.client.identity:
                raise CampaignStateError(
                    "Resume requires the same Codex CLI version and model")
            data_dir = self._load_and_profile(self.state["data_dir"])
            if self.profile["fingerprint_sha256"] != self.state["data_fingerprint"]:
                raise CampaignStateError("Dataset fingerprint changed; refusing to resume")
            self._elapsed_before = float(self.state.get("elapsed_seconds", 0.0))
            self._session_started = self.clock()
            if self.state["status"] in TERMINAL_CAMPAIGN_STATES:
                return self.state
            self._prepare_candidate_views()
            for iteration in self.state["iterations"]:
                if iteration["status"] not in TERMINAL_ITERATION_STATES:
                    self.sandbox.remove_worktree(iteration["number"])
            self.sandbox.remove_named_worktree("baseline")
            self.sandbox.remove_named_worktree("final")
            if self.state["baseline"] is None:
                self._prepare_baseline()
            return self._campaign_loop()

    def _baseline_result(self, base_commit, seed, scores, config, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        score_path = output_dir / "scores.npy"
        np.save(score_path, np.asarray(scores), allow_pickle=False)
        score_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
        rows = self.splits["valid"]
        metrics = evaluate([row[1] for row in rows], [row[6] for row in rows], scores)
        result = make_result(base_commit, "valid", seed, config, status="ok", metrics=_plain(metrics),
                             rows=len(rows), runtime_seconds=0.0, score_sha256=score_hash, error=None)
        write_result(output_dir / "result.json", result)
        return result

    def _prepare_baseline(self):
        base_commit = self.state["base_commit"]
        baseline_dir = self.store.campaign_dir / "baseline"
        config = {}
        if self.recorded_baseline is not None:
            random_result = _plain(self.recorded_baseline["random_test"])
            config = dict(self.recorded_baseline.get("config", {}))
            score_arrays = self.recorded_baseline["scores"]
            if len(score_arrays) != 3:
                raise CampaignStateError("Recorded baseline must contain three seed score arrays")
            seed_results = [
                self._baseline_result(base_commit, seed, np.asarray(scores), config,
                                      baseline_dir / "seeds" / str(seed))
                for seed, scores in enumerate(score_arrays)
            ]
        else:
            random_result = _plain(run_random(self.splits, seed=0)["test"])
            primary = random_result["primary"]
            if not 0.474 <= primary <= 0.476:
                raise CampaignStateError(
                    f"Random seed-0 test sanity primary {primary:.6f} is outside [0.474, 0.476]")
            worktree = self.sandbox.start_named_worktree("baseline", self.sandbox.base_ref)
            try:
                seed_results = []
                for seed in (0, 1, 2):
                    output = baseline_dir / "seeds" / str(seed)
                    result, _ = self.sandbox.run_candidate(
                        worktree, "solution", base_commit, self.state["data_dir"], "valid", seed,
                        config, output, baseline_dir / f"stdout-{seed}.log",
                        baseline_dir / f"stderr-{seed}.log",
                        timeout=self.state["limits"]["run_timeout"],
                        memory_gb=self.state["limits"]["memory_gb"],
                        threads=self.state["limits"]["threads"],
                        candidate_data_dir=self.candidate_data_views["valid"])
                    if result["status"] != "ok":
                        raise CampaignStateError(f"FM baseline seed {seed} failed: {result['error']}")
                    seed_results.append(result)
            finally:
                self.sandbox.remove_named_worktree("baseline")
        arrays = [np.load(baseline_dir / "seeds" / str(seed) / "scores.npy", allow_pickle=False)
                  for seed in (0, 1, 2)]
        ensemble = np.mean(np.stack(arrays), axis=0)
        rows = self.splits["valid"]
        ensemble_metrics = _plain(evaluate([row[1] for row in rows],
                                           [row[6] for row in rows], ensemble))
        self.state["baseline"] = {
            "random_seed_0_test": random_result,
            "fm_seed_results": seed_results,
            "fm_ensemble_metrics": ensemble_metrics,
            "config": config,
        }
        self.state["best_ensemble_metrics"] = ensemble_metrics
        self.state["best_score_hashes"] = {
            str(seed): seed_results[seed]["score_sha256"] for seed in (0, 1, 2)
        }
        self.state["best_config"] = config
        self.state["best_exploratory_primary"] = seed_results[0]["metrics"]["primary"]
        if self.recorded_baseline is not None and "target_primary" in self.recorded_baseline:
            target_primary = self.recorded_baseline["target_primary"]
            if (isinstance(target_primary, bool)
                    or not isinstance(target_primary, (int, float))
                    or not np.isfinite(target_primary)):
                raise CampaignStateError("Recorded target_primary must be finite")
            self.state["target_primary"] = float(target_primary)
        else:
            self.state["target_primary"] = float(
                np.nextafter(PUBLISHED_FM_PRIMARY, np.inf))
        self.state["status"] = "running"
        self._save()

    def _model_profile(self):
        profile = json.loads(json.dumps(self.profile))
        profile.get("splits", {}).get("test", {}).pop("positive_rate", None)
        return profile

    def _system_context(self):
        return {
            "task": "Improve validation primary for within-user ranking of logged exposures.",
            "immutable_rules": IMMUTABLE_RULES,
            "candidate_protocol": (
                "score(splits, data_access, target_split: str, seed: int, config: dict) -> numpy.ndarray"
            ),
            "coding_agent_prompt": SYSTEM_PROMPT,
            "candidate_constraints": CANDIDATE_CONSTRAINTS,
            "protected_paths": PROTECTED_PATHS,
            "profile": self._model_profile(),
            "baseline": {
                "local_fm_ensemble": self.state["baseline"]["fm_ensemble_metrics"],
                "published_fm_primary": PUBLISHED_FM_PRIMARY,
                "target_primary": self.state["target_primary"],
            },
            "dependencies": dependency_versions(),
            "resource_limits": {
                "run_timeout_seconds": self.state["limits"]["run_timeout"],
                "memory_gb": self.state["limits"]["memory_gb"],
                "threads": self.state["limits"]["threads"],
            },
            "research_evidence": [
                "Expanding static basic fields produced no gain; five fields outperformed the expanded set within noise.",
                "FM dimensions 8, 16, and 32 were effectively unchanged; capacity is not the bottleneck.",
                "Train-field CTR and watch-duration residuals raised the three-seed primary from 0.58749 to 0.59315. A within-user pairwise finetune raised it to 0.59688 at pair_blend 0.12; sweeping only that measured component selected pair_blend 0.8 and reached 0.60231. All are already present in current_best_source.",
                "Seed/lower-LR ensembles, user-history residuals, item/time residuals, censored-watch auxiliary training, joint-click auxiliary training, and the attempted listwise softmax failed; do not repeat them.",
                "Improve the existing pairwise objective itself rather than adding another residual or only changing pair_blend: consider more informative positive-negative sampling, hard negatives, field-specific tables, training schedule, or regularization. Preserve the working current source.",
            ],
            "action_schema": {
                "inspect": {
                    "action": "inspect",
                    "requests": [
                        {"kind": "file", "path": "solution.py", "start": 1, "end": 20},
                        {"kind": "search", "path": ".", "pattern": "duration"},
                        {"kind": "data_profile"},
                    ],
                },
                "experiment": {
                    "action": "experiment", "name": "string", "hypothesis": "string",
                    "source": "complete solution.py", "config": {}, "expected_effect": "string",
                },
                "finish": {"action": "finish", "reason": "string"},
            },
            "action_notes": ACTION_NOTES,
        }

    def _history(self):
        history = []
        for iteration in self.state["iterations"]:
            action = iteration.get("action") or {}
            if action.get("action") != "experiment":
                continue
            history.append({
                "number": iteration["number"],
                "name": action.get("name"),
                "hypothesis": action.get("hypothesis"),
                "status": iteration["status"],
                "seed_0_metrics": (iteration.get("seed_results") or {}).get("0", {}).get("metrics"),
                "ensemble_metrics": iteration.get("ensemble_metrics"),
                "reflection": iteration.get("reflection"),
            })
        return history

    def _new_messages(self):
        return [
            {"role": "system", "content": json.dumps(self._system_context(), sort_keys=True)},
            {"role": "user", "content": json.dumps(build_generate_prompt(
                current_best_source=self._read_captured_file("solution.py"),
                prior_experiments=self._history(),
                best_metrics=self.state["best_ensemble_metrics"],
                target_primary=self.state["target_primary"],
            ), sort_keys=True)},
        ]

    def _record_failure(self):
        self.state["counters"]["consecutive_model_failures"] += 1
        self._save()

    def _record_success(self):
        self.state["counters"]["consecutive_model_failures"] = 0
        self._save()

    def _model_call(self, messages, iteration_dir, kind):
        if (self.state["counters"]["model_calls"]
                >= self.state["limits"]["max_model_calls"]):
            self.state["stop_reason"] = "max_model_calls"
            self._save()
            return None
        call_number = self.state["counters"]["model_calls"] + 1
        request_id = f"{self.store.campaign_id}-{call_number:04d}"
        request_record = {"request_id": request_id, "messages": messages}
        if kind == "action":
            request_name, response_name = "request.json", "response.json"
        elif kind == "reflection":
            request_name, response_name = "reflection-request.json", "reflection.json"
        else:
            request_name, response_name = f"{kind}-request.json", f"{kind}-response.json"
        self.store.write_json(iteration_dir / request_name, request_record)
        codex_dir = iteration_dir / "codex"
        self.store.write_json(
            codex_dir / f"{call_number:04d}-{kind}-request.json",
            request_record,
        )
        self.state["counters"]["model_calls"] = call_number
        self._save()
        try:
            response = self.client.complete(messages, request_id)
        except Exception as exc:
            error = {"error": {"type": type(exc).__name__, "message": str(exc)}}
            self.store.write_json(iteration_dir / response_name, error)
            self.store.write_json(
                codex_dir / f"{call_number:04d}-{kind}-response.json",
                error,
            )
            self._record_failure()
            return None
        self.store.write_json(iteration_dir / response_name, response)
        self.store.write_json(
            codex_dir / f"{call_number:04d}-{kind}-response.json",
            response,
        )
        return response

    @staticmethod
    def _safe_repo_path(path):
        if path == ".":
            return "."
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ProtocolError(f"Inspection path must be repository-relative: {path!r}")
        return candidate.as_posix().rstrip("/")

    def _repository_files(self):
        output = self.sandbox._git(
            ["ls-tree", "-r", "-z", self.sandbox.best_ref]).stdout.decode("utf-8", "replace")
        files = {}
        for record in output.split("\0"):
            if not record:
                continue
            metadata, path = record.split("\t", 1)
            mode, kind, _ = metadata.split(" ", 2)
            if kind == "blob" and mode != "120000":
                files[path] = mode
        return files

    def _read_captured_file(self, path):
        output = self.sandbox._git(["show", f"{self.sandbox.best_ref}:{path}"]).stdout
        if len(output) > 256 * 1024:
            raise ProtocolError(f"Inspection file exceeds 256 KiB: {path!r}")
        try:
            return output.decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError(f"Inspection file is not UTF-8 text: {path!r}") from None

    def _perform_inspections(self, requests, remaining_bytes):
        files = self._repository_files()
        results = []
        for request_value in requests:
            kind = request_value["kind"]
            if kind == "data_profile":
                results.append({"kind": kind, "profile": self._model_profile()})
                continue
            path = self._safe_repo_path(request_value["path"])
            if kind == "file":
                if path not in files:
                    raise ProtocolError(f"Inspection file is not in the captured repository: {path!r}")
                lines = self._read_captured_file(path).splitlines()
                start, end = request_value["start"], request_value["end"]
                selected = [f"{index}:{lines[index - 1]}" for index in range(
                    start, min(end, len(lines)) + 1)]
                results.append({"kind": kind, "path": path, "start": start, "end": end,
                                "content": "\n".join(selected)})
                continue
            expression = re.compile(request_value["pattern"])
            prefix = path + "/"
            targets = (list(files) if path == "." else
                       [name for name in files if name == path or name.startswith(prefix)])
            if not targets:
                raise ProtocolError(f"Inspection search path is not in the captured repository: {path!r}")
            matches = []
            for filename in sorted(targets):
                for line_number, line in enumerate(self._read_captured_file(filename).splitlines(), 1):
                    if expression.search(line):
                        matches.append({"path": filename, "line": line_number, "text": line[:500]})
                        if len(matches) >= MAX_SEARCH_MATCHES:
                            break
                if len(matches) >= MAX_SEARCH_MATCHES:
                    break
            results.append({"kind": kind, "path": path, "pattern": request_value["pattern"],
                            "matches": matches, "truncated": len(matches) >= MAX_SEARCH_MATCHES})
        encoded = json.dumps({"inspection_results": results}, sort_keys=True).encode("utf-8")
        if remaining_bytes <= 0:
            return "", 0
        if len(encoded) > remaining_bytes:
            truncated = (b"[inspection context truncated]\n" + encoded)[:remaining_bytes]
            value = truncated.decode("utf-8", "ignore")
            return value, len(value.encode("utf-8"))
        return encoded.decode("utf-8"), len(encoded)

    def _request_action(self, iteration):
        number = iteration["number"]
        iteration_dir = self.store.iteration_dir(number)
        messages = iteration.get("messages") or self._new_messages()
        inspection_rounds = iteration.get("inspection_rounds", 0)
        inspection_bytes = iteration.get("inspection_bytes", 0)
        correction_attempts = iteration.get("correction_attempts", 0)
        while True:
            response = self._model_call(messages, iteration_dir, "action")
            if response is None:
                correction_attempts += 1
                iteration["correction_attempts"] = correction_attempts
                if self.state.get("stop_reason") or correction_attempts > 2:
                    iteration["error"] = (
                        self.state.get("stop_reason") or "action transport failed")
                    self._save()
                    return None
                self._save()
                continue
            try:
                action = validate_action(response)
                if iteration.get("requires_experiment") and action["action"] != "experiment":
                    raise ProtocolError("A patch correction must return an experiment action")
                if (action["action"] == "finish"
                        and self.state["best_ensemble_metrics"]["primary"]
                        < self.state["target_primary"]):
                    raise ProtocolError(
                        "finish is not allowed before best primary reaches target_primary")
            except ProtocolError as exc:
                self._record_failure()
                correction_attempts += 1
                iteration["correction_attempts"] = correction_attempts
                messages.extend([
                    {"role": "assistant", "content": json.dumps(response, sort_keys=True)},
                    {"role": "user", "content": f"Protocol validation error: {exc}. Return a corrected action."},
                ])
                iteration["messages"] = messages
                self._save()
                if correction_attempts > 2:
                    iteration["error"] = str(exc)
                    return None
                continue
            if action["action"] == "inspect":
                if inspection_rounds >= MAX_INSPECTION_ROUNDS:
                    error = "Inspection round limit is 3 per iteration"
                    self._record_failure()
                    correction_attempts += 1
                    messages.extend([
                        {"role": "assistant", "content": json.dumps(action, sort_keys=True)},
                        {"role": "user", "content": error + "; return an experiment or finish action."},
                    ])
                    if correction_attempts > 2:
                        iteration["error"] = error
                        return None
                    continue
                try:
                    context, used = self._perform_inspections(
                        action["requests"], MAX_INSPECTION_BYTES - inspection_bytes)
                except ProtocolError as exc:
                    self._record_failure()
                    correction_attempts += 1
                    messages.extend([
                        {"role": "assistant", "content": json.dumps(action, sort_keys=True)},
                        {"role": "user", "content": f"Inspection validation error: {exc}"},
                    ])
                    if correction_attempts > 2:
                        iteration["error"] = str(exc)
                        return None
                    continue
                inspection_rounds += 1
                inspection_bytes += used
                messages.extend([
                    {"role": "assistant", "content": json.dumps(action, sort_keys=True)},
                    {"role": "user", "content": context},
                ])
                iteration.update({
                    "messages": messages,
                    "inspection_rounds": inspection_rounds,
                    "inspection_bytes": inspection_bytes,
                    "correction_attempts": correction_attempts,
                })
                self._record_success()
                continue
            iteration["action"] = action
            iteration["messages"] = messages
            self._save()
            return action

    @staticmethod
    def _replacement_patch(worktree, source):
        original = (Path(worktree) / "solution.py").read_text(encoding="utf-8")
        if not source.endswith("\n"):
            source += "\n"
        diff_lines = difflib.unified_diff(
            original.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile="a/solution.py",
            tofile="b/solution.py",
        )
        patch_lines = ["diff --git a/solution.py b/solution.py\n"]
        for line in diff_lines:
            if line.endswith("\n"):
                patch_lines.append(line)
            else:
                patch_lines.extend((line + "\n", "\\ No newline at end of file\n"))
        return "".join(patch_lines)

    def _obtain_proposal(self, iteration):
        number = iteration["number"]
        iteration_dir = self.store.iteration_dir(number)
        worktree = self.sandbox.start_iteration(number)
        while True:
            action = iteration.get("action") or self._request_action(iteration)
            if action is None:
                self.sandbox.remove_worktree(number)
                iteration["status"] = "failed"
                iteration.setdefault("error", self.state.get("stop_reason") or "model_protocol_failure")
                self._save()
                return None
            if action["action"] == "finish":
                self.state["stop_reason"] = "model_finish: " + action["reason"]
                iteration["status"] = "failed"
                iteration["error"] = "model_finish"
                self.sandbox.remove_worktree(number)
                self._record_success()
                return None
            self.store.write_text(iteration_dir / "proposal.py", action["source"])
            patch = self._replacement_patch(worktree, action["source"])
            self.store.write_text(iteration_dir / "proposal.patch", patch)
            try:
                commit = self.sandbox.apply_and_commit(number, patch)
            except SandboxError as exc:
                self._record_failure()
                attempts = iteration.get("patch_correction_attempts", 0) + 1
                iteration["patch_correction_attempts"] = attempts
                iteration["error"] = str(exc)
                if attempts > 2:
                    iteration["status"] = "failed"
                    self._save()
                    self.sandbox.remove_worktree(number)
                    return None
                messages = iteration["messages"] + [
                    {"role": "assistant", "content": json.dumps(action, sort_keys=True)},
                    {"role": "user", "content": (
                        f"Source validation error: {exc}. Return a new corrected experiment action. "
                        "Do not repeat the invalid source or inspect. Return the complete corrected "
                        "solution.py in the source field with no Markdown, diff, commentary, TODO, "
                        "placeholder, or ellipsis."
                    )},
                ]
                iteration["messages"] = messages
                iteration["requires_experiment"] = True
                iteration["action"] = None
                self._save()
                continue
            iteration["candidate_commit"] = commit
            iteration["requires_experiment"] = False
            iteration["status"] = "patched"
            iteration["error"] = None
            self._record_success()
            return worktree

    @staticmethod
    def _tail_text(path, limit=MAX_FAILURE_FEEDBACK_BYTES):
        path = Path(path)
        if not path.is_file():
            return ""
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(-limit, os.SEEK_END)
            value = handle.read().decode("utf-8", "replace")
        if size > limit:
            return "[earlier output trimmed]\n" + value
        return value

    def _candidate_failure_feedback(self, iteration, seed, result):
        iteration_dir = self.store.iteration_dir(iteration["number"])
        stdout = iteration_dir / ("stdout.log" if seed == 0 else f"stdout-seed-{seed}.log")
        stderr = iteration_dir / ("stderr.log" if seed == 0 else f"stderr-seed-{seed}.log")
        return {
            "seed": seed,
            "result": _plain(result),
            "stdout_tail": self._tail_text(stdout),
            "stderr_tail": self._tail_text(stderr),
        }

    @staticmethod
    def _source_context(worktree):
        source = (Path(worktree) / "solution.py").read_text(encoding="utf-8")
        encoded = source.encode("utf-8")
        if len(encoded) <= MAX_SOURCE_CONTEXT_BYTES:
            return source
        return (
            encoded[:MAX_SOURCE_CONTEXT_BYTES].decode("utf-8", "ignore")
            + "\n# [source context truncated]\n"
        )

    def _repair_failed_candidate(self, iteration, worktree, seed, result):
        repairs = iteration.setdefault("repair_history", [])
        if len(repairs) >= MAX_CANDIDATE_REPAIRS:
            iteration["error"] = (
                f"candidate still failed after {MAX_CANDIDATE_REPAIRS} repair attempts")
            self._save()
            return False
        repair_number = len(repairs) + 1
        previous_action = dict(iteration["action"])
        previous_commit = iteration["candidate_commit"]
        feedback = self._candidate_failure_feedback(iteration, seed, result)
        record = {
            "number": repair_number,
            "status": "requested",
            "failed_commit": previous_commit,
            "failure": feedback,
            "errors": [],
        }
        repairs.append(record)
        iteration["failure_feedback"] = feedback
        self._save()
        messages = [
            {"role": "system", "content": json.dumps(self._system_context(), sort_keys=True)},
            {"role": "user", "content": json.dumps(build_repair_prompt(
                previous_action=previous_action,
                previous_source=self._source_context(worktree),
                failure=feedback,
                repair_attempt=repair_number,
            ), sort_keys=True)},
        ]
        corrections = 0
        iteration_dir = self.store.iteration_dir(iteration["number"])
        while corrections <= 2:
            response = self._model_call(
                messages, iteration_dir, f"repair-{repair_number}")
            if response is None:
                if self.state.get("stop_reason"):
                    record["status"] = "failed"
                    self._save()
                    return False
                corrections += 1
                continue
            try:
                action = validate_action(response)
                if action["action"] != "experiment":
                    raise ProtocolError("A candidate repair must return an experiment action")
                if action["name"] != previous_action["name"]:
                    raise ProtocolError(
                        "A candidate repair must keep the original experiment name")
                if action["hypothesis"] != previous_action["hypothesis"]:
                    raise ProtocolError(
                        "A candidate repair must keep the original hypothesis")
                self.store.write_text(
                    iteration_dir / f"repair-{repair_number}.py", action["source"])
                patch = self._replacement_patch(worktree, action["source"])
                self.store.write_text(
                    iteration_dir / f"repair-{repair_number}.patch", patch)
                commit = self.sandbox.apply_and_commit(iteration["number"], patch)
            except (ProtocolError, SandboxError) as exc:
                self._record_failure()
                corrections += 1
                record["errors"].append(str(exc))
                messages.extend([
                    {"role": "assistant", "content": json.dumps(response, sort_keys=True)},
                    {"role": "user", "content": (
                        f"Repair validation error: {exc}. Return complete corrected solution.py "
                        "source against the same previous_source. Keep the exact original name "
                        "and hypothesis. Do not repeat the invalid source."
                    )},
                ])
                self._save()
                continue
            record.update({
                "status": "applied",
                "action": action,
                "candidate_commit": commit,
            })
            iteration["action"] = action
            iteration["candidate_commit"] = commit
            iteration["error"] = None
            self._record_success()
            return True
        record["status"] = "failed"
        iteration["error"] = "candidate repair protocol failed"
        self._save()
        return False

    def _run_seed(self, iteration, worktree, seed):
        iteration_dir = self.store.iteration_dir(iteration["number"])
        output_dir = iteration_dir / "seeds" / str(seed)
        result_path = output_dir / "result.json"
        if result_path.is_file():
            with result_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            if (result.get("candidate_commit") == iteration["candidate_commit"]
                    and result.get("seed") == seed and result.get("config") == iteration["action"]["config"]):
                return result
        stdout = iteration_dir / ("stdout.log" if seed == 0 else f"stdout-seed-{seed}.log")
        stderr = iteration_dir / ("stderr.log" if seed == 0 else f"stderr-seed-{seed}.log")
        result, _ = self.sandbox.run_candidate(
            worktree, "solution", iteration["candidate_commit"], self.state["data_dir"],
            "valid", seed, iteration["action"]["config"], output_dir, stdout, stderr,
            timeout=self.state["limits"]["run_timeout"],
            memory_gb=self.state["limits"]["memory_gb"],
            threads=self.state["limits"]["threads"],
            candidate_data_dir=self.candidate_data_views["valid"])
        return result

    def _ensemble_metrics(self, iteration, seeds=(0, 1, 2)):
        iteration_dir = self.store.iteration_dir(iteration["number"])
        arrays = [np.load(iteration_dir / "seeds" / str(seed) / "scores.npy", allow_pickle=False)
                  for seed in seeds]
        scores = np.mean(np.stack(arrays), axis=0)
        rows = self.splits["valid"]
        return _plain(evaluate([row[1] for row in rows], [row[6] for row in rows], scores))

    def _execute_candidate(self, iteration, worktree):
        iteration["status"] = "running"
        iteration.setdefault("seed_results", {})
        iteration.setdefault("repair_history", [])
        self._save()
        while True:
            iteration["seed_results"] = {}
            iteration["ensemble_metrics"] = None
            seed_zero = self._run_seed(iteration, worktree, 0)
            iteration["seed_results"]["0"] = seed_zero
            candidate_failed = seed_zero["status"] != "ok"
            confirmation = False
            primary = None
            failed_seed = 0
            failed_result = seed_zero
            if not candidate_failed:
                primary = seed_zero["metrics"]["primary"]
                threshold = self.state["best_exploratory_primary"]
                confirmation = primary > threshold
            if confirmation:
                for seed in (1, 2):
                    result = self._run_seed(iteration, worktree, seed)
                    iteration["seed_results"][str(seed)] = result
                    if result["status"] != "ok":
                        candidate_failed = True
                        failed_seed = seed
                        failed_result = result
                        break
                if not candidate_failed:
                    iteration["ensemble_metrics"] = self._ensemble_metrics(iteration)
            if candidate_failed and self._repair_failed_candidate(
                    iteration, worktree, failed_seed, failed_result):
                continue
            if primary is not None:
                self.state["best_exploratory_primary"] = max(
                    self.state["best_exploratory_primary"], primary)
            break
        iteration["confirmation_attempted"] = confirmation
        iteration["candidate_failed"] = candidate_failed
        iteration["status"] = "evaluated"
        self._save()

    def _reflection_messages(self, iteration):
        results = iteration.get("seed_results", {})
        return [
            {"role": "system", "content": (
                "Return exactly one JSON object with diagnosis:string, evidence:list[string], "
                "next_hypothesis:string, stop:boolean. Diagnose only the supplied execution "
                "evidence. Set stop=true only when the confirmed ensemble reached target_primary."
            )},
            {"role": "user", "content": json.dumps(_plain({
                "hypothesis": iteration["action"]["hypothesis"],
                "expected_effect": iteration["action"]["expected_effect"],
                "cumulative_diff_summary": self.sandbox.diff_summary(iteration["candidate_commit"]),
                "seed_results": results,
                "ensemble_metrics": iteration.get("ensemble_metrics"),
                "failure_feedback": iteration.get("failure_feedback"),
                "target_primary": self.state["target_primary"],
                "delta_from_best": (
                    iteration.get("ensemble_metrics", {}).get("primary", float("-inf"))
                    - self.state["best_ensemble_metrics"]["primary"]
                    if iteration.get("ensemble_metrics") else None
                ),
                "delta_from_baseline": (
                    iteration.get("ensemble_metrics", {}).get("primary", float("-inf"))
                    - self.state["baseline"]["fm_ensemble_metrics"]["primary"]
                    if iteration.get("ensemble_metrics") else None
                ),
                "prior_experiments": self._history()[:-1],
            }), sort_keys=True)},
        ]

    def _reflect(self, iteration):
        messages = self._reflection_messages(iteration)
        iteration_dir = self.store.iteration_dir(iteration["number"])
        corrections = 0
        while True:
            response = self._model_call(messages, iteration_dir, "reflection")
            if response is None:
                corrections += 1
                if self.state.get("stop_reason") or corrections > 2:
                    iteration["reflection_error"] = (
                        self.state.get("stop_reason") or "reflection transport failed")
                    self._save()
                    return False
                continue
            try:
                reflection = validate_reflection(response)
            except ProtocolError as exc:
                self._record_failure()
                corrections += 1
                messages.extend([
                    {"role": "assistant", "content": json.dumps(response, sort_keys=True)},
                    {"role": "user", "content": f"Reflection validation error: {exc}"},
                ])
                if corrections > 2:
                    iteration["reflection_error"] = str(exc)
                    self._save()
                    return False
                continue
            iteration["reflection"] = reflection
            iteration["status"] = "reflected"
            self._record_success()
            return True

    def _complete_iteration(self, iteration):
        previous_best = self.state["best_ensemble_metrics"]["primary"]
        ensemble = iteration.get("ensemble_metrics")
        accepted = (not iteration.get("candidate_failed") and ensemble is not None
                    and ensemble["primary"] > previous_best)
        if accepted:
            self.sandbox.promote(iteration["candidate_commit"], self.state["best_commit"])
            self.state["best_commit"] = iteration["candidate_commit"]
            self.state["best_config"] = iteration["action"]["config"]
            self.state["best_ensemble_metrics"] = ensemble
            self.state["best_score_hashes"] = {
                seed: result["score_sha256"]
                for seed, result in iteration["seed_results"].items()
            }
            iteration["status"] = "accepted"
            improvement = ensemble["primary"] - previous_best
        else:
            iteration["status"] = "failed" if iteration.get("candidate_failed") else "rejected"
            improvement = 0.0
        iteration["improvement_to_confirmed_best"] = improvement
        counters = self.state["counters"]
        counters["completed_iterations"] += 1
        if improvement > EPSILON:
            counters["consecutive_no_improvement"] = 0
        else:
            counters["consecutive_no_improvement"] += 1
        if (iteration.get("reflection", {}).get("stop")
                and self.state["best_ensemble_metrics"]["primary"]
                >= self.state["target_primary"]):
            self.state["stop_reason"] = "model_reflection_stop"
        self.sandbox.remove_worktree(iteration["number"])
        self._save()

    def _process_iteration(self, iteration):
        status = iteration["status"]
        if status == "requested":
            worktree = self._obtain_proposal(iteration)
            if worktree is None:
                return
            status = "patched"
        if status in ("patched", "running"):
            worktree = self.sandbox.start_iteration(
                iteration["number"], iteration["candidate_commit"])
            self._execute_candidate(iteration, worktree)
            status = "evaluated"
        if status == "evaluated":
            if not self._reflect(iteration):
                iteration["reflection"] = {
                    "diagnosis": "Model reflection was unavailable; controller used execution evidence.",
                    "evidence": [
                        "Candidate acceptance is determined by trusted seed and ensemble metrics."
                    ],
                    "next_hypothesis": "Continue from the confirmed best candidate.",
                    "stop": False,
                }
                iteration["status"] = "reflected"
                self._save()
            status = "reflected"
        if status == "reflected":
            self._complete_iteration(iteration)

    def _policy_stop(self):
        best = self.state.get("best_ensemble_metrics")
        target = self.state.get("target_primary")
        if best is not None and target is not None and best["primary"] >= target:
            self.state["stop_reason"] = "target_primary_met"
            self._save()
            return True
        if self.state.get("stop_reason"):
            return True
        counters = self.state["counters"]
        limits = self.state["limits"]
        if counters["completed_iterations"] >= limits["max_iterations"]:
            self.state["stop_reason"] = "max_iterations"
        elif self._elapsed() >= limits["max_hours"] * 3600:
            self.state["stop_reason"] = "max_hours"
        elif counters["model_calls"] >= limits["max_model_calls"]:
            self.state["stop_reason"] = "max_model_calls"
        elif counters["consecutive_no_improvement"] >= 3:
            self.state["stop_reason"] = "three_consecutive_no_improvement"
        if self.state.get("stop_reason"):
            self._save()
            return True
        return False

    def _campaign_loop(self):
        while not self._policy_stop():
            incomplete = next((item for item in reversed(self.state["iterations"])
                               if item["status"] not in TERMINAL_ITERATION_STATES), None)
            if incomplete is None:
                number = len(self.state["iterations"]) + 1
                incomplete = {
                    "number": number,
                    "status": "requested",
                    "action": None,
                    "candidate_commit": None,
                    "seed_results": {},
                    "ensemble_metrics": None,
                    "reflection": None,
                    "error": None,
                }
                self.state["iterations"].append(incomplete)
                self._save()
            self._process_iteration(incomplete)
        return self._finalize()

    def _finalize(self):
        self.state["status"] = "finalizing"
        self._save()
        final_dir = self.store.campaign_dir / "final"
        worktree = None
        try:
            self.store.write_bytes(self.store.campaign_dir / "best.patch", self.sandbox.best_patch())
            worktree = self.sandbox.start_named_worktree("final", self.state["best_commit"])
            regenerated = []
            for seed in (0, 1, 2):
                output = final_dir / "valid" / str(seed)
                result, _ = self.sandbox.run_candidate(
                    worktree, "solution", self.state["best_commit"], self.state["data_dir"],
                    "valid", seed, self.state["best_config"], output,
                    final_dir / f"valid-{seed}.stdout.log", final_dir / f"valid-{seed}.stderr.log",
                    timeout=self.state["limits"]["run_timeout"],
                    memory_gb=self.state["limits"]["memory_gb"],
                    threads=self.state["limits"]["threads"],
                    candidate_data_dir=self.candidate_data_views["valid"])
                regenerated.append(result)
            hashes_match = all(
                result["status"] == "ok"
                and result["score_sha256"] == self.state["best_score_hashes"].get(str(seed))
                for seed, result in enumerate(regenerated)
            )
            if hashes_match:
                arrays = [np.load(final_dir / "valid" / str(seed) / "scores.npy", allow_pickle=False)
                          for seed in (0, 1, 2)]
                rows = self.splits["valid"]
                metrics = _plain(evaluate(
                    [row[1] for row in rows], [row[6] for row in rows],
                    np.mean(np.stack(arrays), axis=0)))
                hashes_match = _metrics_equal(metrics, self.state["best_ensemble_metrics"])
            else:
                metrics = None
            if not hashes_match:
                submission = self.store.campaign_dir / "submission.csv"
                try:
                    submission.unlink()
                except FileNotFoundError:
                    pass
                self.state["status"] = "non_reproducible"
                self.state["stop_reason"] = "best validation scores or metrics did not reproduce"
                self.store.write_json(self.store.campaign_dir / "final.json", {
                    "best_commit": self.state["best_commit"],
                    "config": self.state["best_config"],
                    "validation_metrics": self.state["best_ensemble_metrics"],
                    "regenerated_validation_metrics": metrics,
                    "dependency_versions": dependency_versions(),
                    "status": "non_reproducible",
                })
                self._save()
                return self.state

            test_paths = []
            test_results = []
            for seed in (0, 1, 2):
                output = final_dir / "test" / str(seed)
                result, _ = self.sandbox.run_candidate(
                    worktree, "solution", self.state["best_commit"], self.state["data_dir"],
                    "test", seed, self.state["best_config"], output,
                    final_dir / f"test-{seed}.stdout.log", final_dir / f"test-{seed}.stderr.log",
                    timeout=self.state["limits"]["run_timeout"],
                    memory_gb=self.state["limits"]["memory_gb"],
                    threads=self.state["limits"]["threads"],
                    candidate_data_dir=self.candidate_data_views["test"])
                if result["status"] != "ok":
                    raise CampaignStateError(f"Final test seed {seed} failed: {result['error']}")
                test_results.append(result)
                test_paths.append(output / "scores.npy")
            submission = self.store.campaign_dir / "submission.csv"
            submit_outcome = self.sandbox.run_submit(
                worktree, self.state["data_dir"], test_paths, submission,
                final_dir / "submit.stdout.log", final_dir / "submit.stderr.log",
                timeout=self.state["limits"]["run_timeout"],
                memory_gb=self.state["limits"]["memory_gb"],
                threads=self.state["limits"]["threads"])
            if submit_outcome["returncode"] != 0 or submit_outcome["timed_out"]:
                raise CampaignStateError("Final submission generation failed")
            check_outcome = self.sandbox.run_submission_check(
                worktree, self.state["data_dir"], submission,
                final_dir / "check.stdout.log", final_dir / "check.stderr.log",
                timeout=self.state["limits"]["run_timeout"],
                memory_gb=self.state["limits"]["memory_gb"],
                threads=self.state["limits"]["threads"])
            if check_outcome["returncode"] != 0 or check_outcome["timed_out"]:
                raise CampaignStateError("Final submit.py --check failed")
            budget_exhausted = self.state["stop_reason"] in {
                "max_iterations", "max_hours", "max_model_calls",
            }
            final_status = ("target_met"
                            if not budget_exhausted
                            and metrics["primary"] >= self.state["target_primary"]
                            else "target_unmet")
            manifest = {
                "best_commit": self.state["best_commit"],
                "config": self.state["best_config"],
                "validation_metrics": metrics,
                "dependency_versions": dependency_versions(),
                "test_seed_results": test_results,
                "status": final_status,
            }
            self.store.write_json(self.store.campaign_dir / "final.json", manifest)
            self.state["status"] = final_status
            self._save()
            return self.state
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self.state["status"] = "finalization_failed"
            self.state["stop_reason"] = f"{type(exc).__name__}: {exc}"
            self._save()
            return self.state
        finally:
            self.sandbox.remove_named_worktree("final")


def format_status(state):
    baseline = state.get("baseline") or {}
    baseline_metrics = baseline.get("fm_ensemble_metrics")
    best_metrics = state.get("best_ensemble_metrics")
    baseline_primary = baseline_metrics.get("primary") if baseline_metrics else None
    best_primary = best_metrics.get("primary") if best_metrics else None
    delta = (best_primary - baseline_primary
             if best_primary is not None and baseline_primary is not None else None)
    current = next((item for item in reversed(state.get("iterations", []))
                    if item["status"] not in TERMINAL_ITERATION_STATES), None)
    lines = [
        f"campaign: {state['campaign_id']}",
        f"status: {state['status']}",
        f"baseline_primary: {baseline_primary if baseline_primary is not None else 'pending'}",
        f"best_primary: {best_primary if best_primary is not None else 'pending'}",
        f"delta_from_baseline: {delta if delta is not None else 'pending'}",
        f"target_primary: {state.get('target_primary')}",
        f"model_calls: {state['counters']['model_calls']}",
        f"completed_iterations: {state['counters']['completed_iterations']}",
        f"current_iteration: {current['number'] if current else 'none'}",
        f"stop_reason: {state.get('stop_reason')}",
    ]
    return "\n".join(lines)


def _limits_from_args(args):
    return {
        "max_iterations": args.max_iterations,
        "max_hours": args.max_hours,
        "max_model_calls": args.max_model_calls,
        "run_timeout": args.run_timeout,
        "memory_gb": args.memory_gb,
        "threads": args.threads,
    }

def _positive_int(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _add_codex_arguments(command_parser):
    command_parser.add_argument("--model", required=True)
    command_parser.add_argument("--codex-bin", default="codex")
    command_parser.add_argument("--codex-prefix-arg", action="append", default=[])
    command_parser.add_argument("--codex-timeout", type=_positive_int, default=900)


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--campaign", required=True)
    run_parser.add_argument("--data_dir", required=True)
    run_parser.add_argument("--max-iterations", type=int, default=12)
    run_parser.add_argument("--max-hours", type=float, default=4)
    run_parser.add_argument("--max-model-calls", type=int, default=60)
    run_parser.add_argument("--run-timeout", type=int, default=900)
    run_parser.add_argument("--memory-gb", type=float, default=8)
    run_parser.add_argument("--threads", type=int, default=4)
    _add_codex_arguments(run_parser)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--campaign", required=True)
    _add_codex_arguments(resume_parser)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--campaign", required=True)

    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    try:
        store = CampaignStore(repo_root / ".agent-runs", args.campaign)
        if args.command == "status":
            print(format_status(store.load()))
            return 0
        client = CodexCLIClient(
            args.codex_bin,
            args.model,
            prefix_args=args.codex_prefix_arg,
            timeout_seconds=args.codex_timeout,
        )
        sandbox = WorktreeSandbox(repo_root, store.campaign_dir, args.campaign,
                                  python_executable=sys.executable)
        controller = CampaignController(client, sandbox, store)
        if args.command == "run":
            state = controller.run(args.data_dir, _limits_from_args(args))
        else:
            state = controller.resume()
        print(format_status(state))
        return 0 if state["status"] in ("target_met", "target_unmet") else 1
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
