import csv
import difflib
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shutil
from pathlib import Path
import os
import tempfile
import subprocess
import sys
import unittest
import threading
import time
from unittest import mock

import numpy as np
from agent import CampaignController, ProtocolError, format_status, validate_action, validate_reflection
from agent_api import AgentAPIError, OpenAICompatibleClient, client_from_environment
from agent_sandbox import SandboxError, WorktreeSandbox, _run_process
from agent_state import CampaignLockedError, CampaignStateError, CampaignStore

from data import load as load_dataset
from baseline import fit_fm, run_fm
from evaluate import evaluate
from experiment import DataAccess, create_submission, profile_data, run_candidate
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
    def test_fit_fm_extraction_preserves_scoring(self):
        splits = synthetic_splits()
        model, encoded = fit_fm(splits, epochs=3, bs=8, patience=2, seed=7, verbose=False)
        scores = model.predict(encoded["valid"][0])
        expected = evaluate(encoded["valid"][2], encoded["valid"][1], scores)
        actual = run_fm(splits, epochs=3, bs=8, patience=2, seed=7, verbose=False)
        protocol_scores = score(splits, None, "valid", 7,
                                {"epochs": 3, "bs": 8, "patience": 2})
        self.assertEqual(actual["valid"], expected)
        np.testing.assert_array_equal(protocol_scores, scores)


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
            (lambda *args: np.zeros(3), "expected 12"),
            (lambda *args: np.full(12, np.nan), "NaN or Inf"),
        )
        for index, (scorer, message) in enumerate(bad_values):
            with self.subTest(message=message), mock.patch(
                    "experiment._candidate_module", return_value=scorer):
                result = run_candidate(
                    "solution", "bad", self.data_dir, "valid", 0, {},
                    self.data_dir / f"bad-{index}")
            self.assertEqual(result["status"], "error")
            self.assertRegex(result["error"]["message"], message)
            self.assertIsNone(result["score_sha256"])

    def test_candidate_cannot_observe_test_labels(self):
        def label_scorer(splits, *args):
            return np.asarray([row[6] for row in splits["test"]])

        with mock.patch("experiment._candidate_module", return_value=label_scorer):
            result = run_candidate(
                "solution", "masked", self.data_dir, "test", 0, {},
                self.data_dir / "masked")
        self.assertEqual(result["status"], "ok")
        np.testing.assert_array_equal(
            np.load(self.data_dir / "masked" / "scores.npy"), np.zeros(12))


class APITransportTests(unittest.TestCase):
    def setUp(self):
        self.responses = []
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                owner.requests.append({
                    "path": self.path,
                    "request_id": self.headers.get("X-Request-ID"),
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(length)),
                })
                status, body = owner.responses.pop(0)
                encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.secret = "super-secret-agent-key"
        self.client = OpenAICompatibleClient(
            f"http://127.0.0.1:{self.server.server_port}/v1", "fake-model", self.secret)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_retry_reuses_request_id_and_parses_object(self):
        content = {"action": "finish", "reason": "done"}
        self.responses.extend([
            (429, {"error": "retry"}),
            (200, {"choices": [{"message": {"content": json.dumps(content)}}],
                   "provider_metadata": {"ignored": True}}),
        ])
        with mock.patch("agent_api.time.sleep") as sleep:
            result = self.client.complete([{"role": "user", "content": "work"}], "request-7")
        self.assertEqual(result, content)
        sleep.assert_called_once_with(1)
        self.assertEqual([item["request_id"] for item in self.requests],
                         ["request-7", "request-7"])
        self.assertEqual(self.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(self.requests[0]["body"]["temperature"], 0.2)

    def test_errors_are_malformed_and_secret_safe(self):
        self.responses.append((200, b"{not-json"))
        with self.assertRaises(AgentAPIError) as malformed:
            self.client.complete([], "malformed")
        self.assertNotIn(self.secret, str(malformed.exception))

        self.responses.append((400, {"error": self.secret}))
        with self.assertRaises(AgentAPIError) as rejected:
            self.client.complete([], "rejected")
        self.assertNotIn(self.secret, str(rejected.exception))

    def test_environment_configuration_requires_key_base_and_model(self):
        with self.assertRaisesRegex(ValueError, "AGENT_API_KEY"):
            client_from_environment(environ={})
        client = client_from_environment(
            api_base="https://example.invalid/v1", api_model="model",
            environ={"AGENT_API_KEY": "key-only"})
        self.assertEqual(client.base_url, "https://example.invalid/v1")
        self.assertEqual(client.model, "model")


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


    def test_standard_unified_diff_is_accepted(self):
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
            "print(os.environ.get('AGENT_API_KEY','missing'), flush=True); time.sleep(60)"
        )
        with mock.patch.dict(os.environ, {"AGENT_API_KEY": "never-copy-this"}):
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
            "patch": "diff --git a/solution.py b/solution.py",
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
            ({"api_calls": 5}, "max_api_calls"),
            ({"consecutive_no_improvement": 3}, "three_consecutive_no_improvement"),
            ({"consecutive_model_failures": 3}, "three_consecutive_model_failures"),
        )
        for changes, expected in cases:
            with self.subTest(expected=expected):
                controller = CampaignController(None, None, None, clock=lambda: 0)
                counters = {
                    "api_calls": 0,
                    "completed_iterations": 0,
                    "consecutive_model_failures": 0,
                    "consecutive_no_improvement": 0,
                }
                counters.update(changes)
                controller.state = {
                    "stop_reason": None,
                    "counters": counters,
                    "limits": {
                        "max_iterations": 2, "max_hours": 1, "max_api_calls": 5,
                        "run_timeout": 30, "memory_gb": 8, "threads": 1,
                    },
                    "best_ensemble_metrics": {"primary": 1.0},
                    "target_primary": 0.5,
                }
                controller._save = lambda: None
                self.assertTrue(controller._policy_stop())
                self.assertEqual(controller.state["stop_reason"], expected)
        controller = CampaignController(None, None, None, clock=lambda: 3601)
        controller._session_started = 0
        controller.state = {
            "stop_reason": None,
            "counters": {
                "api_calls": 0, "completed_iterations": 0,
                "consecutive_model_failures": 0, "consecutive_no_improvement": 0,
            },
            "limits": {
                "max_iterations": 2, "max_hours": 1, "max_api_calls": 5,
                "run_timeout": 30, "memory_gb": 8, "threads": 1,
            },
            "best_ensemble_metrics": {"primary": 1.0},
            "target_primary": 0.5,
        }
        controller._save = lambda: None
        self.assertTrue(controller._policy_stop())
        self.assertEqual(controller.state["stop_reason"], "max_hours")


class FakeClient:
    base_url = "https://fake.invalid/v1"
    model = "fake-ranking-model"

    def __init__(self, responses):
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
        ".gitignore", "agent.py", "agent_api.py", "agent_sandbox.py", "agent_state.py",
        "baseline.py", "data.py", "evaluate.py", "experiment.py", "solution.py",
        "submit.py", "requirements.txt",
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


def replacement_patch(repo, function_source):
    original = (Path(repo) / "solution.py").read_text()
    replacement = original[:original.index("def score")] + function_source
    return "diff --git a/solution.py b/solution.py\n" + "".join(difflib.unified_diff(
        original.splitlines(keepends=True), replacement.splitlines(keepends=True),
        fromfile="a/solution.py", tofile="b/solution.py"))


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
                ".gitignore", "agent.py", "agent_api.py", "agent_sandbox.py", "agent_state.py",
                "baseline.py", "data.py", "evaluate.py", "experiment.py", "solution.py",
                "submit.py", "requirements.txt",
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
            patch = "diff --git a/solution.py b/solution.py\n" + "".join(difflib.unified_diff(
                original.splitlines(keepends=True), replacement.splitlines(keepends=True),
                fromfile="a/solution.py", tofile="b/solution.py"))
            responses = [
                {"action": "inspect", "requests": [{"kind": "data_profile"}]},
                {
                    "action": "experiment",
                    "name": "duration-order",
                    "hypothesis": "Longer videos align with the fixture relevance order.",
                    "patch": patch,
                    "config": {},
                    "expected_effect": "Improve seed-zero and confirmed ensemble primary.",
                },
                {
                    "diagnosis": "Duration ordering improved both ranking metrics.",
                    "evidence": ["Seed zero and the three-seed ensemble exceed the baseline."],
                    "next_hypothesis": "No further fixture experiment is needed.",
                    "stop": False,
                },
                {"action": "finish", "reason": "The bounded integration objective is complete."},
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
            }
            status_before = subprocess.run(
                ["git", "status", "--short"], cwd=repo, check=True,
                stdout=subprocess.PIPE, text=True).stdout
            with mock.patch.dict(os.environ, {"AGENT_API_KEY": "integration-secret"}):
                controller = CampaignController(
                    client, sandbox, store, recorded_baseline=recorded)
                state = controller.run(data_dir, {
                    "max_iterations": 3,
                    "max_hours": 1,
                    "max_api_calls": 12,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            self.assertEqual(state["status"], "target_met")
            self.assertEqual(state["counters"]["completed_iterations"], 1)
            self.assertEqual(state["iterations"][0]["status"], "accepted")
            self.assertEqual(set(state["iterations"][0]["seed_results"]), {"0", "1", "2"})
            self.assertEqual(state["best_commit"], state["iterations"][0]["candidate_commit"])
            self.assertAlmostEqual(
                state["target_primary"],
                max(state["baseline"]["fm_ensemble_metrics"]["primary"], 0.6016) + 0.002)
            self.assertEqual(len(client.calls), 4)
            campaign_dir = store.campaign_dir
            iteration_dir = campaign_dir / "iterations" / "0001"
            for name in (
                    "request.json", "response.json", "proposal.patch", "stdout.log",
                    "stderr.log", "reflection.json"):
                self.assertTrue((iteration_dir / name).is_file(), name)
            system_context = json.loads(client.calls[0][0][0]["content"])
            self.assertNotIn("positive_rate", system_context["profile"]["splits"]["test"])
            self.assertIn("pairwise/listwise", " ".join(system_context["research_evidence"]))
            self.assertEqual(system_context["protected_paths"][0], "evaluate.py")
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
            campaign_bytes = b"".join(
                path.read_bytes() for path in campaign_dir.rglob("*") if path.is_file())
            self.assertNotIn(b"integration-secret", campaign_bytes)

    def test_resume_reuses_completed_candidate_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            source_root = Path(__file__).resolve().parent
            maintained = [
                ".gitignore", "agent.py", "agent_api.py", "agent_sandbox.py", "agent_state.py",
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
            patch = "diff --git a/solution.py b/solution.py\n" + "".join(difflib.unified_diff(
                original.splitlines(keepends=True), replacement.splitlines(keepends=True),
                fromfile="a/solution.py", tofile="b/solution.py"))
            experiment_action = {
                "action": "experiment",
                "name": "resume-duration",
                "hypothesis": "Duration changes row ordering.",
                "patch": patch,
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
                    "max_iterations": 3,
                    "max_hours": 1,
                    "max_api_calls": 12,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            result_path = store.iteration_dir(1) / "seeds" / "0" / "result.json"
            score_path = store.iteration_dir(1) / "seeds" / "0" / "scores.npy"
            self.assertTrue(result_path.is_file())
            score_hash_before = hashlib.sha256(score_path.read_bytes()).hexdigest()
            result_mtime_before = result_path.stat().st_mtime_ns

            sandbox.run_candidate = original_run_candidate
            resumed_client = FakeClient([
                {
                    "diagnosis": "The completed seed-zero run was recovered.",
                    "evidence": ["The persisted result was reused."],
                    "next_hypothesis": "Finish.",
                    "stop": False,
                },
                {"action": "finish", "reason": "Resume behavior verified."},
            ])
            resumed = CampaignController(
                resumed_client, sandbox, store, recorded_baseline=recorded).resume()
            self.assertEqual(resumed["status"], "target_unmet")
            self.assertEqual(result_path.stat().st_mtime_ns, result_mtime_before)
            self.assertEqual(hashlib.sha256(score_path.read_bytes()).hexdigest(), score_hash_before)
            self.assertEqual(len(resumed_client.calls), 2)
            self.assertEqual(resumed["counters"]["completed_iterations"], 1)

    def test_failed_candidate_is_reflected_before_stopping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            source_root = Path(__file__).resolve().parent
            maintained = [
                ".gitignore", "agent.py", "agent_api.py", "agent_sandbox.py", "agent_state.py",
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
    return np.full(len(splits[target_split]), np.nan)
'''
            patch = "diff --git a/solution.py b/solution.py\n" + "".join(difflib.unified_diff(
                original.splitlines(keepends=True), replacement.splitlines(keepends=True),
                fromfile="a/solution.py", tofile="b/solution.py"))
            client = FakeClient([
                {
                    "action": "experiment",
                    "name": "invalid-output",
                    "hypothesis": "This intentionally exercises candidate failure handling.",
                    "patch": patch,
                    "config": {},
                    "expected_effect": "The harness rejects non-finite scores.",
                },
                {
                    "diagnosis": "The candidate violated the finite-score contract.",
                    "evidence": ["Seed zero returned a non-finite score error."],
                    "next_hypothesis": "Stop after recording the failure.",
                    "stop": True,
                },
            ])
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
            }
            store = CampaignStore(repo / ".agent-runs", "failed")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "failed",
                                      python_executable=sys.executable)
            state = CampaignController(
                client, sandbox, store, recorded_baseline=recorded).run(data_dir, {
                    "max_iterations": 2,
                    "max_hours": 1,
                    "max_api_calls": 8,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            iteration = state["iterations"][0]
            self.assertEqual(iteration["status"], "failed")
            self.assertTrue(iteration["candidate_failed"])
            self.assertEqual(iteration["seed_results"]["0"]["status"], "error")
            self.assertIn("finite-score contract", iteration["reflection"]["diagnosis"])
            self.assertTrue((store.iteration_dir(1) / "reflection.json").is_file())
            self.assertEqual(state["counters"]["completed_iterations"], 1)

    def test_seed_zero_without_exploratory_margin_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, data_dir = prepare_campaign_fixture(root)
            patch = replacement_patch(repo, '''def score(
        splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
''')
            client = FakeClient([
                {
                    "action": "experiment",
                    "name": "unconfirmed-duration",
                    "hypothesis": "Match the strongest exploratory seed.",
                    "patch": patch,
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
            }
            store = CampaignStore(repo / ".agent-runs", "unconfirmed")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "unconfirmed",
                                      python_executable=sys.executable)
            state = CampaignController(
                client, sandbox, store, recorded_baseline=recorded).run(data_dir, {
                    "max_iterations": 2,
                    "max_hours": 1,
                    "max_api_calls": 8,
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


    def test_budget_exhaustion_remains_target_unmet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, data_dir = prepare_campaign_fixture(root)
            patch = replacement_patch(repo, '''def score(
        splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    del data_access, seed, config
    return np.asarray([row[5] for row in splits[target_split]], dtype=np.float64)
''')
            client = FakeClient([
                {
                    "action": "experiment",
                    "name": "budget-duration",
                    "hypothesis": "Duration clears the target.",
                    "patch": patch,
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
            }
            store = CampaignStore(repo / ".agent-runs", "budget")
            sandbox = WorktreeSandbox(repo, store.campaign_dir, "budget",
                                      python_executable=sys.executable)
            state = CampaignController(
                client, sandbox, store, recorded_baseline=recorded).run(data_dir, {
                    "max_iterations": 1,
                    "max_hours": 1,
                    "max_api_calls": 8,
                    "run_timeout": 30,
                    "memory_gb": 8,
                    "threads": 1,
                })
            self.assertGreaterEqual(
                state["best_ensemble_metrics"]["primary"], state["target_primary"])
            self.assertEqual(state["stop_reason"], "max_iterations")
            self.assertEqual(state["status"], "target_unmet")

if __name__ == "__main__":
    unittest.main()
