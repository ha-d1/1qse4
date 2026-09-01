import csv
import difflib
import hashlib
import json
import shutil
from pathlib import Path
import os
import tempfile
import subprocess
import sys
import unittest
import time
from unittest import mock

import numpy as np
from agent import CampaignController, ProtocolError, format_status, validate_action, validate_reflection
from agent_codex import CodexCLIClient, CodexCLIError
from agent_prompts import (
    PROTECTED_PATHS,
    SYSTEM_PROMPT,
    build_generate_prompt,
    build_repair_prompt,
)
from agent_sandbox import SandboxError, WorktreeSandbox, _run_process
from agent_state import CampaignLockedError, CampaignStateError, CampaignStore

from data import load as load_dataset
from baseline import fit_fm, run_fm
from evaluate import evaluate
from experiment import (
    DataAccess,
    create_submission,
    prepare_candidate_data,
    profile_data,
    run_candidate,
)
from solution import score


VIDEO_FILE = "video_features_basic_pure.csv"
TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"


def synthetic_splits():
    def rows(day, count):
        return [(day, f"u{i % 3}", f"v{i % 5}", f"a{i % 2}", str(i % 2),
                 float(100 + i), i % 2) for i in range(count)]
    return {
        "train": rows(20220410, 24),
        "valid": rows(20220424, 12),
        "test": rows(20220501, 12),
    }


def write_dataset(root):
    root = Path(root)
    videos = [(f"v{i}", f"a{i % 2}") for i in range(5)]
    with (root / VIDEO_FILE).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "author_id", "video_type"])
        writer.writerows((video, author, "NORMAL") for video, author in videos)
    header = ["date", "user_id", "video_id", "tab", "duration_ms", "long_view",
              "is_click", "play_time_ms"]
    with (root / TRAIN_LOG).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in synthetic_splits()["train"]:
            writer.writerow([row[0], row[1], row[2], row[4], row[5], row[6], row[6], row[5]])
    with (root / LATER_LOG).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for split in ("valid", "test"):
            for row in synthetic_splits()[split]:
                writer.writerow([row[0], row[1], row[2], row[4], row[5], row[6], row[6], row[5]])


class CandidateBoundaryTests(unittest.TestCase):
    def test_fit_fm_extraction_and_candidate_boundary(self):
        splits = synthetic_splits()
        model, encoded = fit_fm(splits, epochs=3, bs=8, patience=2, seed=7, verbose=False)
        scores = model.predict(encoded["valid"][0])
        expected = evaluate(encoded["valid"][2], encoded["valid"][1], scores)
        actual = run_fm(splits, epochs=3, bs=8, patience=2, seed=7, verbose=False)
        protocol_scores = score(splits, None, "valid", 7, {
            "epochs": 3,
            "bs": 8,
            "patience": 2,
            "loss": "metadata-only",
            "history_blend": 0.25,
        })
        self.assertEqual(actual["valid"], expected)
        self.assertEqual(protocol_scores.shape, scores.shape)
        self.assertTrue(np.isfinite(protocol_scores).all())


class ExperimentHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        write_dataset(self.data_dir)

    def tearDown(self):
        self.temporary.cleanup()

    def test_data_access_is_column_and_split_limited(self):
        access = DataAccess(self.data_dir, "valid")
        self.assertIn(TRAIN_LOG, access.columns())
        rows = list(access.iter_rows(TRAIN_LOG, ("user_id", "is_click"), "train"))
        self.assertEqual(len(rows), 24)
        with self.assertRaisesRegex(ValueError, "direct CSV"):
            list(access.iter_rows("../" + TRAIN_LOG, ("user_id",)))
        with self.assertRaisesRegex(ValueError, "Unknown columns"):
            list(access.iter_rows(TRAIN_LOG, ("missing",)))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            list(access.iter_rows(LATER_LOG, ("user_id",), "test"))
        with self.assertRaisesRegex(ValueError, "Label column"):
            list(access.iter_rows(TRAIN_LOG, ("long_view",), "train"))

    def test_profile_is_aggregate_and_stable(self):
        first = profile_data(self.data_dir)
        second = profile_data(self.data_dir)
        self.assertEqual(first, second)
        self.assertEqual(first["splits"]["train"]["rows"], 24)
        self.assertEqual(first["splits"]["valid"]["users"], 3)
        self.assertEqual(len(first["fingerprint_sha256"]), 64)
        serialized = json.dumps(first)
        self.assertNotIn('"u0"', serialized)
        self.assertNotIn('"v0"', serialized)

    def test_candidate_run_and_submission_round_trip(self):
        run_dirs = []
        for seed in (0, 1, 2):
            output = self.data_dir / f"test-{seed}"
            result = run_candidate("solution", "abc123", self.data_dir, "test", seed,
                                   {"epochs": 2, "bs": 8, "patience": 1}, output)
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(result["metrics"])
            self.assertEqual(set(result), {
                "status", "candidate_commit", "target_split", "seed", "config", "metrics",
                "rows", "runtime_seconds", "score_sha256", "python_version", "numpy_version",
                "lightgbm_version", "error",
            })
            run_dirs.append(output / "scores.npy")
        submission = self.data_dir / "submission.csv"
        rows = create_submission(self.data_dir, run_dirs, submission)
        self.assertEqual(rows, 12)
        self.assertTrue(submission.is_file())

    def test_candidate_rejects_cardinality_and_nonfinite_scores(self):
        bad_values = (
            (np.zeros(3), "expected 12"),
            (np.full(12, np.nan), "NaN or Inf"),
        )
        for index, (scores, message) in enumerate(bad_values):
            def write_bad_scores(*arguments):
                np.save(arguments[-1], scores, allow_pickle=False)

            with self.subTest(message=message), mock.patch(
                    "experiment._run_isolated_worker",
                    side_effect=write_bad_scores):
                result = run_candidate(
                    "solution", "bad", self.data_dir, "valid", 0, {},
                    self.data_dir / f"bad-{index}")
            self.assertEqual(result["status"], "error")
            self.assertRegex(result["error"]["message"], message)
            self.assertIsNone(result["score_sha256"])

    def test_candidate_cannot_observe_test_labels(self):
        profile = profile_data(self.data_dir)
        views = prepare_candidate_data(
            self.data_dir,
            self.data_dir / "candidate-views",
            profile["fingerprint_sha256"],
        )
        original = load_dataset(self.data_dir)
        masked = load_dataset(views["test"])
        self.assertTrue(any(row[6] for row in original["test"]))
        np.testing.assert_array_equal(
            [row[6] for row in masked["test"]],
            np.zeros(len(masked["test"])),
        )


class CodexCLITransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.fake_script = root / "fake_codex.py"
        self.record_path = root / "calls.jsonl"
        self.secret = "super-secret-transport-value"
        self.fake_script.write_text(r'''
import json
import os
from pathlib import Path
import sys
import time


args = sys.argv[1:]
prompt = None
entries = None
if args and args[0] == "exec":
    entries = os.listdir(".")
    prompt = json.loads(sys.stdin.read())

record_path = os.environ.get("FAKE_CODEX_RECORD")
if record_path:
    record = {
        "args": args,
        "cwd": os.getcwd(),
        "entries": entries,
        "prompt": prompt,
        "credentials": {
            name: os.environ.get(name)
            for name in (
                "AGENT" + "_API_KEY",
                "OPENAI_API_KEY",
                "CODEX_API_KEY",
                "CODEX_ACCESS_TOKEN",
            )
        },
        "paths": {
            name: os.environ.get(name)
            for name in (
                "PWD",
                "TMPDIR",
                "OLDPWD",
                "PYTHONPATH",
                "GIT_DIR",
                "GIT_WORK_TREE",
            )
        },
    }
    with Path(record_path).open("a") as handle:
        handle.write(json.dumps(record) + "\n")

mode = os.environ.get("FAKE_CODEX_MODE", "success")
secret = "super-secret-transport-value"
if args == ["--version"]:
    if mode == "version-exit":
        sys.stderr.write(secret)
        raise SystemExit(11)
    print("codex-cli 0.46.0")
elif args == ["login", "status"]:
    if mode == "login-exit":
        sys.stderr.write(secret)
        raise SystemExit(12)
    if mode == "non-chatgpt":
        print("Logged in using API key: " + secret)
    else:
        print("Logged in using ChatGPT")
elif args and args[0] == "exec":
    if mode == "nonzero":
        sys.stderr.write(secret)
        raise SystemExit(17)
    if mode == "timeout":
        sys.stderr.write(secret)
        sys.stderr.flush()
        time.sleep(60)
    if mode == "oversized":
        sys.stdout.buffer.write(b"x" * (512 * 1024 + 1))
    elif mode == "invalid-utf8":
        sys.stdout.buffer.write(b"\xff")
    else:
        sys.stdout.write(os.environ.get(
            "FAKE_CODEX_RESPONSE",
            '{"action":"finish","reason":"ok"}',
        ))
else:
    sys.stderr.write(secret)
    raise SystemExit(91)
''')

    def tearDown(self):
        self.temporary.cleanup()

    def _environment(self, **values):
        return {"FAKE_CODEX_RECORD": str(self.record_path), **values}

    def _client(self, timeout_seconds=5, **environment):
        with mock.patch.dict(os.environ, self._environment(**environment)):
            return CodexCLIClient(
                sys.executable,
                "fake-model",
                prefix_args=(str(self.fake_script),),
                timeout_seconds=timeout_seconds,
            )

    def _complete(self, client, messages=None, request_id="request-7", **environment):
        if messages is None:
            messages = [{"role": "user", "content": "work"}]
        with mock.patch.dict(os.environ, self._environment(**environment)):
            return client.complete(messages, request_id)

    def _records(self):
        return [
            json.loads(line)
            for line in self.record_path.read_text().splitlines()
        ]

    def test_probes_exec_boundary_and_exact_object_parsing(self):
        messages = [
            {"role": "system", "content": "System first."},
            {"role": "user", "content": "User second."},
            {"role": "assistant", "content": "Assistant third."},
        ]
        credentials = {
            "AGENT" + "_API_KEY": "must-not-be-used",
            "OPENAI_API_KEY": "must-not-be-used",
            "CODEX_API_KEY": "must-not-be-used",
            "CODEX_ACCESS_TOKEN": "must-not-be-used",
        }
        with mock.patch.dict(os.environ, {
                **self._environment(
                    FAKE_CODEX_RESPONSE='  {"action":"finish","reason":"done"}\n',
                ),
                **credentials,
                "OLDPWD": "/repository/old",
                "PYTHONPATH": "/repository/python",
                "GIT_DIR": "/repository/.git",
                "GIT_WORK_TREE": "/repository",
        }):
            client = CodexCLIClient(
                sys.executable,
                "fake-model",
                prefix_args=(str(self.fake_script),),
                timeout_seconds=5,
            )
            result = client.complete(messages, "request-7")

        self.assertEqual(
            client.identity,
            {"version": "0.46.0", "model": "fake-model"},
        )
        self.assertEqual(result, {"action": "finish", "reason": "done"})
        records = self._records()
        self.assertEqual(records[0]["args"], ["--version"])
        self.assertEqual(records[1]["args"], ["login", "status"])
        execution = records[2]
        self.assertEqual(execution["args"], [
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            "fake-model",
            "--color",
            "never",
            "-",
        ])
        self.assertEqual(execution["entries"], [])
        self.assertEqual(execution["prompt"], {
            "instruction": (
                "Act only as the model component of the supplied controller "
                "conversation. Do not inspect the local machine or use tools. "
                "Follow system messages as highest priority and return exactly "
                "one JSON object with no Markdown."
            ),
            "request_id": "request-7",
            "messages": messages,
        })
        for record in records:
            self.assertEqual(record["credentials"], {
                name: None for name in credentials
            })
        self.assertEqual(execution["paths"]["PWD"], execution["cwd"])
        self.assertEqual(execution["paths"]["TMPDIR"], execution["cwd"])
        for name in ("OLDPWD", "PYTHONPATH", "GIT_DIR", "GIT_WORK_TREE"):
            self.assertIsNone(execution["paths"][name])
        self.assertTrue(Path(execution["cwd"]).name.startswith("kuairand-codex-"))
        self.assertFalse(Path(execution["cwd"]).exists())

    def test_missing_executable_and_login_errors_are_secret_safe(self):
        missing = str(Path(self.temporary.name) / "missing-codex")
        with self.assertRaisesRegex(
                CodexCLIError,
                "Codex CLI executable not found: " + missing):
            CodexCLIClient(missing, "fake-model")

        with self.assertRaises(CodexCLIError) as login:
            self._client(FAKE_CODEX_MODE="non-chatgpt")
        self.assertEqual(
            str(login.exception),
            "Codex CLI must be logged in with ChatGPT; run 'codex login'",
        )
        self.assertNotIn(self.secret, str(login.exception))

    def test_probe_and_exec_failures_never_expose_stderr(self):
        for mode, expected in (
                ("version-exit", "version check failed with exit status 11"),
                ("login-exit", "login check failed with exit status 12")):
            with self.subTest(mode=mode), self.assertRaises(CodexCLIError) as error:
                self._client(FAKE_CODEX_MODE=mode)
            self.assertIn(expected, str(error.exception))
            self.assertNotIn(self.secret, str(error.exception))

        client = self._client()
        with self.assertRaises(CodexCLIError) as error:
            self._complete(client, FAKE_CODEX_MODE="nonzero")
        self.assertIn("Codex CLI exited with status 17", str(error.exception))
        self.assertNotIn(self.secret, str(error.exception))

    def test_timeout_is_bounded_and_secret_safe(self):
        client = self._client(timeout_seconds=0.05)
        with self.assertRaises(CodexCLIError) as error:
            self._complete(client, FAKE_CODEX_MODE="timeout")
        self.assertIn("Codex CLI timed out after 0.05 seconds", str(error.exception))
        self.assertNotIn(self.secret, str(error.exception))

    def test_response_rejections_are_specific(self):
        client = self._client()
        for response, expected in (
                ("{not-json", "malformed JSON"),
                ('{"action":"finish"} trailing', "trailing content"),
                ("[]", "must be a JSON object")):
            with self.subTest(response=response), self.assertRaisesRegex(
                    CodexCLIError, expected):
                self._complete(client, FAKE_CODEX_RESPONSE=response)

        with self.assertRaisesRegex(CodexCLIError, "exceeds 512 KiB"):
            self._complete(client, FAKE_CODEX_MODE="oversized")
        with self.assertRaisesRegex(CodexCLIError, "not valid UTF-8"):
            self._complete(client, FAKE_CODEX_MODE="invalid-utf8")

    def test_request_validation_precedes_subprocess_execution(self):
        client = self._client()
        probes = len(self._records())
        invalid_requests = (
            ([], "", "request_id"),
            ("not-a-list", "request", "messages must be a list"),
            ([{"role": "user", "content": "ok", "extra": 1}], "request", "role/content"),
            ([{"role": "user", "content": 3}], "request", "role/content"),
        )
        for messages, request_id, expected in invalid_requests:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                    ValueError, expected):
                client.complete(messages, request_id)
        self.assertEqual(len(self._records()), probes)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.git("init")
        self.git("config", "user.name", "Test User")
        self.git("config", "user.email", "test@example.com")
        (self.repo / ".gitignore").write_text(".agent-worktrees/\n.agent-runs/\n")
        (self.repo / "tracked.txt").write_text("committed\n")
        (self.repo / "solution.py").write_text(
            "import numpy as np\n\ndef score(*args):\n    return np.zeros(2)\n")
        self.git("add", ".")
        self.git("commit", "-m", "base")
        (self.repo / "tracked.txt").write_text("dirty tracked\n")
        (self.repo / "untracked.txt").write_text("dirty untracked\n")
        self.sandbox = WorktreeSandbox(
            self.repo, self.repo / ".agent-runs" / "campaign", "campaign",
            python_executable=sys.executable)

    def tearDown(self):
        for iteration in (1, 2, 3):
            try:
                self.sandbox.remove_worktree(iteration)
            except (AttributeError, SandboxError):
                pass
        self.temporary.cleanup()

    def git(self, *args, cwd=None):
        return subprocess.run(["git", *args], cwd=cwd or self.repo, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_dirty_snapshot_and_proposal_leave_main_checkout_unchanged(self):
        status_before = self.git("status", "--short").stdout
        base = self.sandbox.create_base_snapshot()
        self.assertEqual(self.git("status", "--short").stdout, status_before)
        worktree = self.sandbox.start_iteration(1)
        self.assertEqual((worktree / "tracked.txt").read_text(), "dirty tracked\n")
        self.assertEqual((worktree / "untracked.txt").read_text(), "dirty untracked\n")
        patch = """diff --git a/solution.py b/solution.py
--- a/solution.py
+++ b/solution.py
@@ -1,4 +1,4 @@
 import numpy as np
 
 def score(*args):
-    return np.zeros(2)
+    return np.ones(2)
"""
        proposal = self.sandbox.apply_and_commit(1, patch)
        self.assertNotEqual(proposal, base)
        self.assertEqual(self.sandbox.resolve_ref(self.sandbox.best_ref), base)
        self.sandbox.promote(proposal, base)
        self.assertEqual(self.sandbox.resolve_ref(self.sandbox.best_ref), proposal)
        self.assertIn("np.zeros", (self.repo / "solution.py").read_text())
        self.assertEqual(self.git("status", "--short").stdout, status_before)


    def test_standard_unified_diff_without_final_newline_is_accepted(self):
        base = self.sandbox.create_base_snapshot()
        self.sandbox.start_iteration(1)
        patch = """--- a/solution.py
+++ b/solution.py
@@ -1,1 +1,1 @@
 import numpy as np
 
 def score(*args):
-    return np.zeros(2)
+    return np.ones(2)
"""
        patch = patch.rstrip("\n")
        proposal = self.sandbox.apply_and_commit(1, patch)
        self.assertNotEqual(proposal, base)
        self.assertEqual((self.sandbox.worktree_path(1) / "solution.py").read_text(),
                         "import numpy as np\n\ndef score(*args):\n    return np.ones(2)\n")

    def test_patch_policy_rejects_paths_imports_and_calls(self):
        self.sandbox.create_base_snapshot()
        self.sandbox.start_iteration(1)
        protected = """diff --git a/tracked.txt b/tracked.txt
--- a/tracked.txt
+++ b/tracked.txt
@@ -1 +1 @@
-dirty tracked
+changed
"""
        with self.assertRaisesRegex(SandboxError, "only edit solution"):
            self.sandbox.apply_and_commit(1, protected)
        forbidden_import = """diff --git a/solution.py b/solution.py
--- a/solution.py
+++ b/solution.py
@@ -1,4 +1,5 @@
+import os
 import numpy as np
 
 def score(*args):
     return np.zeros(2)
"""
        with self.assertRaisesRegex(SandboxError, "Forbidden import"):
            self.sandbox.apply_and_commit(1, forbidden_import)
        forbidden_call = """diff --git a/solution.py b/solution.py
--- a/solution.py
+++ b/solution.py
@@ -1,4 +1,5 @@
 import numpy as np
 
 def score(*args):
+    open("leak")
     return np.zeros(2)
"""
        with self.assertRaisesRegex(SandboxError, "Forbidden call"):
            self.sandbox.apply_and_commit(1, forbidden_call)

    def test_timeout_kills_process_group_and_strips_secret(self):
        root = Path(self.temporary.name)
        pid_path = root / "child.pid"
        code = (
            "import os,pathlib,subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); "
            "print(os.environ.get('THIRD_PARTY_API_KEY','missing'), flush=True); time.sleep(60)"
        )
        with mock.patch.dict(os.environ, {"THIRD_PARTY_API_KEY": "never-copy-this"}):
            outcome = _run_process(
                [sys.executable, "-c", code, str(pid_path)], root,
                root / "stdout.log", root / "stderr.log", timeout=0.2,
                memory_gb=1, threads=1, seed=3)
        self.assertTrue(outcome["timed_out"])
        self.assertEqual((root / "stdout.log").read_text().strip(), "missing")
        child_pid = int(pid_path.read_text())
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("candidate child process survived timeout cleanup")


class PromptContractTests(unittest.TestCase):
    def test_generate_prompt_carries_current_source_and_research_history(self):
        prompt = build_generate_prompt(
            current_best_source="def score(*args):\n    return scores\n",
            prior_experiments=[{"name": "dead-end", "status": "rejected"}],
            best_metrics={"primary": 0.6},
            target_primary=0.6016,
        )
        self.assertEqual(
            prompt["current_best_source"],
            "def score(*args):\n    return scores\n",
        )
        self.assertEqual(prompt["prior_experiments"][0]["status"], "rejected")
        self.assertIn("one hypothesis", prompt["instruction"])
        self.assertIn("agent_prompts.py", PROTECTED_PATHS)
        self.assertIn("constant within a user", " ".join(SYSTEM_PROMPT))

    def test_repair_prompt_carries_exact_failed_source_and_evidence(self):
        failure = {
            "result": {"status": "failed", "error": "timeout"},
            "stdout_tail": "partial progress",
            "stderr_tail": "traceback tail",
        }
        prompt = build_repair_prompt(
            previous_action={"name": "pair-loss", "hypothesis": "hard negatives"},
            previous_source="def score():\n    broken()\n",
            failure=failure,
            repair_attempt=2,
        )
        self.assertEqual(prompt["previous_source"], "def score():\n    broken()\n")
        self.assertIs(prompt["failure"], failure)
        self.assertEqual(prompt["repair_attempt"], 2)
        self.assertIn("root cause", prompt["instruction"])
        self.assertIn("no-op fallback", prompt["instruction"])


class CampaignStateTests(unittest.TestCase):
    def test_atomic_state_and_lock_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".agent-runs"
            first = CampaignStore(root, "state-test")
            second = CampaignStore(root, "state-test")
            with first.lock():
                state = first.initialize({"status": "running", "counters": {}})
                state["status"] = "saved"
                first.save(state)
                with self.assertRaises(CampaignLockedError):
                    with second.lock():
                        pass
                stray = first.state_path.with_name(first.state_path.name + ".tmp-crash")
                stray.write_text("{broken")
                self.assertEqual(first.load()["status"], "saved")
            self.assertTrue(stray.exists())

    def test_campaign_paths_reject_invalid_ids_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "campaign_id"):
                CampaignStore(root, "../escape")
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            store = CampaignStore(linked, "campaign")
            with self.assertRaisesRegex(CampaignStateError, "symlink"):
                with store.lock():
                    pass

    def test_action_and_reflection_schemas_are_exact(self):
        experiment = {
            "action": "experiment",
            "name": "duration",
            "hypothesis": "Duration ranks positives",
            "source": "def score(*args):\n    return []\n",
            "config": {},
            "expected_effect": "Higher primary",
        }
        self.assertIs(validate_action(experiment), experiment)
        with self.assertRaisesRegex(ProtocolError, "unknown keys"):
            validate_action({**experiment, "command": "python"})
        with self.assertRaisesRegex(ProtocolError, "positive integer"):
            validate_action({
                "action": "inspect",
                "requests": [{"kind": "file", "path": "solution.py", "start": 0, "end": 2}],
            })
        reflection = {
            "diagnosis": "worked", "evidence": ["primary increased"],
            "next_hypothesis": "finish", "stop": True,
        }
        self.assertIs(validate_reflection(reflection), reflection)
        with self.assertRaisesRegex(ProtocolError, "boolean"):
            validate_reflection({**reflection, "stop": 1})

    def test_stopping_policy_uses_first_reached_limit(self):
        cases = (
            ({"completed_iterations": 2}, "max_iterations"),
            ({"model_calls": 5}, "max_model_calls"),
            ({"consecutive_no_improvement": 3}, "three_consecutive_no_improvement"),
        )
        for changes, expected in cases:
            with self.subTest(expected=expected):
                controller = CampaignController(None, None, None, clock=lambda: 0)
                counters = {
                    "model_calls": 0,
                    "completed_iterations": 0,
                    "consecutive_model_failures": 0,
                    "consecutive_no_improvement": 0,
                }
                counters.update(changes)
                controller.state = {
                    "stop_reason": None,
                    "counters": counters,
                    "limits": {
                        "max_iterations": 2, "max_hours": 1, "max_model_calls": 5,
                        "run_timeout": 30, "memory_gb": 8, "threads": 1,
                    },
                    "best_ensemble_metrics": {"primary": 1.0},
                    "target_primary": 2.0,
                }
                controller._save = lambda: None
                self.assertTrue(controller._policy_stop())
                self.assertEqual(controller.state["stop_reason"], expected)

        controller = CampaignController(None, None, None, clock=lambda: 0)
        controller.state = {
            "stop_reason": "max_model_calls",
            "counters": {
                "model_calls": 5, "completed_iterations": 0,
                "consecutive_model_failures": 3, "consecutive_no_improvement": 0,
            },
            "limits": {
                "max_iterations": 2, "max_hours": 1, "max_model_calls": 5,
                "run_timeout": 30, "memory_gb": 8, "threads": 1,
            },
            "best_ensemble_metrics": {"primary": 1.0},
            "target_primary": 0.5,
        }
        controller._save = lambda: None
        self.assertTrue(controller._policy_stop())
        self.assertEqual(controller.state["stop_reason"], "target_primary_met")

        controller = CampaignController(None, None, None, clock=lambda: 3601)
        controller._session_started = 0
        controller.state = {
            "stop_reason": None,
            "counters": {
                "model_calls": 0, "completed_iterations": 0,
                "consecutive_model_failures": 3, "consecutive_no_improvement": 0,
            },
            "limits": {
                "max_iterations": 2, "max_hours": 1, "max_model_calls": 5,
                "run_timeout": 30, "memory_gb": 8, "threads": 1,
            },
            "best_ensemble_metrics": {"primary": 1.0},
            "target_primary": 2.0,
        }
        controller._save = lambda: None
        self.assertTrue(controller._policy_stop())
        self.assertEqual(controller.state["stop_reason"], "max_hours")


class FakeClient:
    def __init__(self, responses, identity=None):
        self.identity = dict(identity if identity is not None else {
            "version": "0.46.0",
            "model": "fake-ranking-model",
        })
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, request_id):
        self.calls.append((messages, request_id))
        if not self.responses:
            raise AssertionError("Unexpected model call")
        return self.responses.pop(0)

def prepare_campaign_fixture(root):
    repo = Path(root) / "repo"
    repo.mkdir()
    source_root = Path(__file__).resolve().parent
    maintained = [
        ".gitignore", "agent.py", "agent_codex.py", "agent_prompts.py",
        "agent_sandbox.py", "agent_state.py", "baseline.py", "data.py",
        "evaluate.py", "experiment.py", "solution.py", "submit.py",
        "requirements.txt",
    ]
    for name in maintained:
        shutil.copy2(source_root / name, repo / name)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    data_dir = Path(root) / "data"
    data_dir.mkdir()
    write_dataset(data_dir)
    return repo, data_dir


def replacement_source(repo, function_source):
    original = (Path(repo) / "solution.py").read_text()
    return original[:original.index("def score")] + function_source


def actual_recorded_baseline(data_dir):
    splits = load_dataset(data_dir)
    config = {"epochs": 1, "bs": 8, "patience": 1}
    return {
        "random_test": {
            "GAUC": 0.5, "nDCG@5": 0.45, "primary": 0.475,
            "users": 3, "rows": 12,
        },
        "scores": [score(splits, None, "valid", seed, config) for seed in (0, 1, 2)],
        "config": config,
    }




class CampaignIntegrationTests(unittest.TestCase):
    def test_inspect_experiment_reflect_finish_and_final_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            maintained = [
                ".gitignore", "agent.py", "agent_codex.py", "agent_prompts.py",
                "agent_sandbox.py", "agent_state.py", "baseline.py", "data.py",
                "evaluate.py", "experiment.py", "solution.py", "submit.py",
                "requirements.txt",
            ]
            source_root = Path(__file__).resolve().parent
            for name in maintained:
                shutil.copy2(source_root / name, repo / name)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True,
                           stdout=subprocess.PIPE)
            data_dir = root / "data"
            data_dir.mkdir()
            write_dataset(data_dir)

            original = (repo / "solution.py").read_text()
            prefix = original[:original.index("def score")]
            replacement = prefix + '''def score(splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    \"\"\"Rank rows by observed video duration.\"\"\"
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
'''
            replacement = replacement.rstrip("\n")
            responses = [
                {"action": "inspect", "requests": [{"kind": "data_profile"}]},
                {
                    "action": "experiment",
                    "name": "duration-order",
                    "hypothesis": "Longer videos align with the fixture relevance order.",
                    "source": replacement,
                    "config": {},
                    "expected_effect": "Improve seed-zero and confirmed ensemble primary.",
                },
                {
                    "diagnosis": "Duration ordering improved both ranking metrics.",
                    "evidence": ["Seed zero and the three-seed ensemble exceed the baseline."],
                    "next_hypothesis": "No further fixture experiment is needed.",
                    "stop": False,
                },
            ]
            client = FakeClient(responses)
            store = CampaignStore(repo / ".agent-runs", "integration")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "integration",
                                      python_executable=sys.executable)
            zeros = np.zeros(len(synthetic_splits()["valid"]))
            recorded = {
                "random_test": {
                    "GAUC": 0.5, "nDCG@5": 0.45, "primary": 0.475,
                    "users": 3, "rows": 12,
                },
                "scores": [zeros.copy(), zeros.copy(), zeros.copy()],
                "config": {"epochs": 1, "bs": 8, "patience": 1},
                "target_primary": 0.65,
            }
            status_before = subprocess.run(
                ["git", "status", "--short"], cwd=repo, check=True,
                stdout=subprocess.PIPE, text=True).stdout
            controller = CampaignController(
                client, sandbox, store, recorded_baseline=recorded)
            state = controller.run(data_dir, {
                "max_iterations": 3,
                "max_hours": 1,
                "max_model_calls": 12,
                "run_timeout": 30,
                "memory_gb": 8,
                "threads": 1,
            })
            self.assertEqual(state["status"], "target_met")
            self.assertEqual(state["counters"]["completed_iterations"], 1)
            self.assertEqual(state["iterations"][0]["status"], "accepted")
            self.assertEqual(set(state["iterations"][0]["seed_results"]), {"0", "1", "2"})
            self.assertEqual(state["best_commit"], state["iterations"][0]["candidate_commit"])
            self.assertEqual(state["target_primary"], 0.65)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(state["codex"], client.identity)
            campaign_dir = store.campaign_dir
            iteration_dir = campaign_dir / "iterations" / "0001"
            for name in (
                    "request.json", "response.json", "proposal.py", "proposal.patch",
                    "stdout.log", "stderr.log", "reflection.json"):
                self.assertTrue((iteration_dir / name).is_file(), name)
            self.assertTrue((iteration_dir / "codex").is_dir())
            self.assertTrue(
                (iteration_dir / "codex" / "0001-action-request.json").is_file())
            system_context = json.loads(client.calls[0][0][0]["content"])
            self.assertNotIn("positive_rate", system_context["profile"]["splits"]["test"])
            self.assertIn("pairwise", " ".join(system_context["research_evidence"]))
            self.assertEqual(system_context["protected_paths"][0], "evaluate.py")
            self.assertIn("agent_codex.py", system_context["protected_paths"])
            self.assertIn("traceback tail", " ".join(system_context["coding_agent_prompt"]))
            self.assertIn("zero-masked", " ".join(system_context["coding_agent_prompt"]))
            prompt_contract = " ".join(system_context["coding_agent_prompt"])
            self.assertIn("constant within a user", prompt_contract)
            self.assertIn("one causal hypothesis", prompt_contract)
            self.assertIn("all-zero scores", prompt_contract)
            self.assertIn("timeout or memory-limit", prompt_contract)
            self.assertIn("source", system_context["action_schema"]["experiment"])
            self.assertNotIn("patch", system_context["action_schema"]["experiment"])
            initial_prompt = json.loads(client.calls[0][0][1]["content"])
            self.assertEqual(initial_prompt["current_best_source"], original)
            self.assertIn("target_primary", initial_prompt["instruction"])
            self.assertTrue((campaign_dir / "submission.csv").is_file())
            self.assertTrue((campaign_dir / "best.patch").is_file())
            self.assertTrue((campaign_dir / "final.json").is_file())
            manifest = json.loads((campaign_dir / "final.json").read_text())
            self.assertEqual(manifest["status"], "target_met")
            self.assertEqual(manifest["best_commit"], state["best_commit"])
            self.assertEqual(manifest["config"], state["best_config"])
            self.assertEqual(manifest["validation_metrics"], state["best_ensemble_metrics"])
            self.assertTrue(all(result["metrics"] is None
                                for result in manifest["test_seed_results"]))
            for seed in (0, 1, 2):
                reproduced = json.loads(
                    (campaign_dir / "final" / "valid" / str(seed) / "result.json").read_text())
                self.assertEqual(
                    reproduced["score_sha256"], state["best_score_hashes"][str(seed)])
            self.assertIn(
                "Format and alignment validation passed",
                (campaign_dir / "final" / "check.stdout.log").read_text())
            self.assertIn(b"solution.py", (campaign_dir / "best.patch").read_bytes())
            self.assertIn("delta_from_baseline:", format_status(state))
            status_after = subprocess.run(
                ["git", "status", "--short"], cwd=repo, check=True,
                stdout=subprocess.PIPE, text=True).stdout
            self.assertEqual(status_after, status_before)

    def test_resume_reuses_completed_candidate_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            source_root = Path(__file__).resolve().parent
            maintained = [
                ".gitignore", "agent.py", "agent_codex.py", "agent_sandbox.py", "agent_state.py",
                "baseline.py", "data.py", "evaluate.py", "experiment.py", "solution.py",
                "submit.py", "requirements.txt",
            ]
            for name in maintained:
                shutil.copy2(source_root / name, repo / name)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True,
                           stdout=subprocess.PIPE)
            data_dir = root / "data"
            data_dir.mkdir()
            write_dataset(data_dir)

            original = (repo / "solution.py").read_text()
            prefix = original[:original.index("def score")]
            replacement = prefix + '''def score(splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
'''
            experiment_action = {
                "action": "experiment",
                "name": "resume-duration",
                "hypothesis": "Duration changes row ordering.",
                "source": replacement,
                "config": {},
                "expected_effect": "Exercise candidate subprocess resume.",
            }
            splits = load_dataset(data_dir)
            baseline_config = {"epochs": 1, "bs": 8, "patience": 1}
            baseline_scores = [
                score(splits, None, "valid", seed, baseline_config) for seed in (0, 1, 2)
            ]
            recorded = {
                "random_test": {
                    "GAUC": 0.5, "nDCG@5": 0.45, "primary": 0.475,
                    "users": 3, "rows": 12,
                },
                "scores": baseline_scores,
                "config": baseline_config,
                "target_primary": 2.0,
            }
            store = CampaignStore(repo / ".agent-runs", "resume")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "resume",
                                      python_executable=sys.executable)
            first_client = FakeClient([experiment_action])
            controller = CampaignController(
                first_client, sandbox, store, recorded_baseline=recorded)
            original_run_candidate = sandbox.run_candidate
            interrupted = False

            def interrupt_after_result(*args, **kwargs):
                nonlocal interrupted
                result = original_run_candidate(*args, **kwargs)
                if not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                return result

            sandbox.run_candidate = interrupt_after_result
            with self.assertRaises(KeyboardInterrupt):
                controller.run(data_dir, {
                    "max_iterations": 1,
                    "max_hours": 1,
                    "max_model_calls": 12,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            result_path = store.iteration_dir(1) / "seeds" / "0" / "result.json"
            score_path = store.iteration_dir(1) / "seeds" / "0" / "scores.npy"
            self.assertTrue(result_path.is_file())
            score_hash_before = hashlib.sha256(score_path.read_bytes()).hexdigest()
            result_mtime_before = result_path.stat().st_mtime_ns

            for identity in (
                    {"version": "0.47.0", "model": "fake-ranking-model"},
                    {"version": "0.46.0", "model": "different-model"}):
                with self.subTest(identity=identity), self.assertRaisesRegex(
                        CampaignStateError,
                        "Resume requires the same Codex CLI version and model"):
                    CampaignController(
                        FakeClient([], identity=identity),
                        sandbox,
                        store,
                        recorded_baseline=recorded,
                    ).resume()

            sandbox.run_candidate = original_run_candidate
            resumed_client = FakeClient([
                {
                    "diagnosis": "The completed seed-zero run was recovered.",
                    "evidence": ["The persisted result was reused."],
                    "next_hypothesis": "Finish.",
                    "stop": False,
                },
            ])
            resumed = CampaignController(
                resumed_client, sandbox, store, recorded_baseline=recorded).resume()
            self.assertEqual(resumed["status"], "target_unmet")
            self.assertEqual(result_path.stat().st_mtime_ns, result_mtime_before)
            self.assertEqual(hashlib.sha256(score_path.read_bytes()).hexdigest(), score_hash_before)
            self.assertEqual(len(resumed_client.calls), 1)
            self.assertEqual(resumed["counters"]["completed_iterations"], 1)

    def test_failed_candidate_repairs_same_iteration_from_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, data_dir = prepare_campaign_fixture(root)
            original = (repo / "solution.py").read_text()
            prefix = original[:original.index("def score")]
            hypothesis = "Duration ordering should beat the constant fixture baseline."
            invalid_source = prefix + '''def score(
        splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.full(len(splits[target_split]), np.nan)
'''
            repaired_source = prefix + '''def score(
        splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
'''


            client = FakeClient([
                {
                    "action": "experiment",
                    "name": "repair-duration",
                    "hypothesis": hypothesis,
                    "source": invalid_source,
                    "config": {},
                    "expected_effect": "Exercise runtime repair before measuring duration.",
                },
                {
                    "action": "experiment",
                    "name": "repair-duration",
                    "hypothesis": hypothesis,
                    "source": repaired_source,
                    "config": {},
                    "expected_effect": "Return finite duration scores after repairing the failure.",
                },
                {
                    "diagnosis": "The repaired duration candidate cleared the target.",
                    "evidence": ["All three validation seeds completed with the same ranking."],
                    "next_hypothesis": "No further experiment is required.",
                    "stop": True,
                },
            ])
            rows = synthetic_splits()["valid"]
            zeros = np.zeros(len(rows))
            recorded = {
                "random_test": {
                    "GAUC": 0.5, "nDCG@5": 0.45, "primary": 0.475,
                    "users": 3, "rows": 12,
                },
                "scores": [zeros.copy(), zeros.copy(), zeros.copy()],
                "config": {"epochs": 1, "bs": 8, "patience": 1},
                "target_primary": 0.65,
            }
            store = CampaignStore(repo / ".agent-runs", "repair")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "repair",
                                      python_executable=sys.executable)
            state = CampaignController(
                client, sandbox, store, recorded_baseline=recorded).run(data_dir, {
                    "max_iterations": 2,
                    "max_hours": 1,
                    "max_model_calls": 8,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            iteration = state["iterations"][0]
            self.assertEqual(state["status"], "target_met")
            self.assertEqual(iteration["status"], "accepted")
            self.assertFalse(iteration["candidate_failed"])
            self.assertEqual(len(iteration["repair_history"]), 1)
            repair = iteration["repair_history"][0]
            self.assertEqual(repair["status"], "applied")
            self.assertIn("Candidate scores contain NaN or Inf",
                          repair["failure"]["stderr_tail"])
            repair_prompt = json.loads(client.calls[1][0][1]["content"])
            self.assertEqual(repair_prompt["previous_source"], invalid_source)
            self.assertIn("Candidate scores contain NaN or Inf",
                          repair_prompt["failure"]["stderr_tail"])
            self.assertTrue((store.iteration_dir(1) / "repair-1.patch").is_file())
            self.assertTrue((store.iteration_dir(1) / "repair-1.py").is_file())
            self.assertTrue((store.iteration_dir(1) / "repair-1-request.json").is_file())
            self.assertEqual(state["counters"]["completed_iterations"], 1)

    def test_invalid_reflection_cannot_discard_measured_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, data_dir = prepare_campaign_fixture(root)
            candidate_source = replacement_source(repo, '''def score(
        splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
''')
            experiment = {
                "action": "experiment",
                "name": "reflection-fallback",
                "hypothesis": "Duration clears the fixture target.",
                "source": candidate_source,
                "config": {},
                "expected_effect": "Produce a confirmed winning ensemble.",
            }
            client = FakeClient([
                experiment,
                {"diagnosis": "missing required fields"},
                {"diagnosis": "still missing required fields"},
                {"diagnosis": "third malformed reflection"},
            ])
            rows = synthetic_splits()["valid"]
            zeros = np.zeros(len(rows))
            recorded = {
                "random_test": {
                    "GAUC": 0.5, "nDCG@5": 0.45, "primary": 0.475,
                    "users": 3, "rows": 12,
                },
                "scores": [zeros.copy(), zeros.copy(), zeros.copy()],
                "config": {"epochs": 1, "bs": 8, "patience": 1},
                "target_primary": 0.65,
            }
            store = CampaignStore(repo / ".agent-runs", "reflection-fallback")
            sandbox = WorktreeSandbox(
                repo, store.campaign_dir, "reflection-fallback",
                python_executable=sys.executable)
            state = CampaignController(
                client, sandbox, store, recorded_baseline=recorded).run(data_dir, {
                    "max_iterations": 1,
                    "max_hours": 1,
                    "max_model_calls": 8,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            iteration = state["iterations"][0]
            self.assertEqual(state["status"], "target_met")
            self.assertEqual(iteration["status"], "accepted")
            self.assertIn("missing keys", iteration["reflection_error"])
            self.assertIn("controller used execution evidence",
                          iteration["reflection"]["diagnosis"])
            self.assertEqual(len(client.calls), 4)

    def test_seed_zero_without_exploratory_margin_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, data_dir = prepare_campaign_fixture(root)
            candidate_source = replacement_source(repo, '''def score(
        splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
''')
            client = FakeClient([
                {
                    "action": "experiment",
                    "name": "unconfirmed-duration",
                    "hypothesis": "Match the strongest exploratory seed.",
                    "source": candidate_source,
                    "config": {},
                    "expected_effect": "No margin over the exploratory threshold.",
                },
                {
                    "diagnosis": "Seed zero did not clear the confirmation margin.",
                    "evidence": ["Only seed zero ran."],
                    "next_hypothesis": "Stop.",
                    "stop": True,
                },
            ])
            rows = synthetic_splits()["valid"]
            duration = np.asarray([row[5] for row in rows])
            recorded = {
                "random_test": {
                    "GAUC": 0.5, "nDCG@5": 0.45, "primary": 0.475,
                    "users": 3, "rows": 12,
                },
                "scores": [duration, -duration, np.zeros(len(rows))],
                "config": {"epochs": 1, "bs": 8, "patience": 1},
                "target_primary": 2.0,
            }
            store = CampaignStore(repo / ".agent-runs", "unconfirmed")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "unconfirmed",
                                      python_executable=sys.executable)
            state = CampaignController(
                client, sandbox, store, recorded_baseline=recorded).run(data_dir, {
                    "max_iterations": 1,
                    "max_hours": 1,
                    "max_model_calls": 8,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            iteration = state["iterations"][0]
            self.assertEqual(iteration["status"], "rejected")
            self.assertEqual(set(iteration["seed_results"]), {"0"})
            self.assertFalse(iteration["confirmation_attempted"])
            self.assertEqual(state["best_commit"], state["base_commit"])
            self.assertEqual(state["status"], "non_reproducible")
            self.assertFalse((store.campaign_dir / "submission.csv").exists())
            self.assertTrue((store.campaign_dir / "best.patch").is_file())
            self.assertEqual(
                json.loads((store.campaign_dir / "final.json").read_text())["status"],
                "non_reproducible")


    def test_target_met_wins_at_iteration_budget_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, data_dir = prepare_campaign_fixture(root)
            candidate_source = replacement_source(repo, '''def score(
        splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
''')
            client = FakeClient([
                {
                    "action": "experiment",
                    "name": "budget-duration",
                    "hypothesis": "Duration clears the target.",
                    "source": candidate_source,
                    "config": {},
                    "expected_effect": "Confirmed target improvement.",
                },
                {
                    "diagnosis": "The target was cleared.",
                    "evidence": ["Three-seed primary exceeds target."],
                    "next_hypothesis": "Budget controls final labeling.",
                    "stop": False,
                },
            ])
            rows = synthetic_splits()["valid"]
            zeros = np.zeros(len(rows))
            recorded = {
                "random_test": {
                    "GAUC": 0.5, "nDCG@5": 0.45, "primary": 0.475,
                    "users": 3, "rows": 12,
                },
                "scores": [zeros.copy(), zeros.copy(), zeros.copy()],
                "config": {"epochs": 1, "bs": 8, "patience": 1},
                "target_primary": 0.65,
            }
            store = CampaignStore(repo / ".agent-runs", "budget")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "budget",
                                      python_executable=sys.executable)
            state = CampaignController(
                client, sandbox, store, recorded_baseline=recorded).run(data_dir, {
                    "max_iterations": 1,
                    "max_hours": 1,
                    "max_model_calls": 8,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            self.assertGreaterEqual(
                state["best_ensemble_metrics"]["primary"], state["target_primary"])
            self.assertEqual(state["stop_reason"], "target_primary_met")
            self.assertEqual(state["status"], "target_met")

if __name__ == "__main__":
    unittest.main()
