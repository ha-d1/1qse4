from __future__ import annotations

import sys
import tempfile
import unittest
import json
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from convergence import ConvergenceState
from development_data import ensure_development_splits, remove_labels
from experiment_runner import ExperimentRunner
from experiment_schema import ExperimentProposal
from gemini_client import GeminiClient
from patch_manager import PatchError, PatchManager
from gemini_client import GeminiClient
from run_logger import RunLogger
from verify_baseline import serialisable_metrics


class DevelopmentDataTests(unittest.TestCase):
    def test_rejects_test_labels(self) -> None:
        with self.assertRaises(ValueError):
            ensure_development_splits({"train": [], "valid": [], "test": []})

    def test_removes_labels(self) -> None:
        rows = [(20220429, "u", "v", "a", "tab", 1000.0, 1)]
        self.assertEqual(remove_labels(rows), [(20220429, "u", "v", "a", "tab", 1000.0)])


class ConvergenceTests(unittest.TestCase):
    def test_stops_after_three_small_improvements(self) -> None:
        state = ConvergenceState(epsilon=0.002, patience=3)
        for score in (0.6000, 0.6010, 0.6015, 0.6016):
            state.observe(score)
        self.assertEqual(state.stop_reason(1.0), "converged")

    def test_large_improvement_resets_counter(self) -> None:
        state = ConvergenceState(epsilon=0.002, patience=3)
        for score in (0.6000, 0.6010, 0.6040):
            state.observe(score)
        self.assertIsNone(state.stop_reason(1.0))


class ProposalTests(unittest.TestCase):
    def test_rejects_changes_outside_candidate(self) -> None:
        proposal = ExperimentProposal(
            "h", "r", ["evaluate.py"], "gain", "risk", "patch", ["python3", "x.py"]
        )
        with self.assertRaises(ValueError):
            proposal.validate()

    def test_gemini_structured_proposal_and_usage(self) -> None:
        class Usage:
            prompt_token_count = 20
            candidates_token_count = 10
            total_token_count = 30

        class Response:
            text = json.dumps(
                {
                    "hypothesis": "Try multiple negatives",
                    "reasoning": "Pairwise ranking benefits from more comparisons",
                    "target_files": ["candidate/model.py"],
                    "expected_effect": "Improve GAUC",
                    "risk": "Longer runtime",
                    "patch": "--- a/candidate/model.py\n+++ b/candidate/model.py\n",
                    "command": ["python3", "candidate/train.py", "--objective", "bpr"],
                }
            )
            usage_metadata = Usage()
            response_id = "mock-response"

        class Models:
            def generate_content(self, **kwargs):
                return Response()

        class Client:
            models = Models()

        result = GeminiClient(client=Client()).propose({"iteration": 1})
        self.assertEqual(result.proposal.target_files, ["candidate/model.py"])
        self.assertEqual(result.usage["total_tokens"], 30)


class PatchManagerTests(unittest.TestCase):
    def test_rejects_protected_file_patch(self) -> None:
        manager = PatchManager(PROJECT_ROOT)
        patch = "--- a/evaluate.py\n+++ b/evaluate.py\n@@ -1 +1 @@\n-old\n+new\n"
        with self.assertRaises(PatchError):
            manager.validate(patch)

    def test_applies_and_rolls_back_candidate_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate").mkdir()
            target = root / "candidate" / "model.py"
            target.write_text("VALUE = 1\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            patch = (
                "--- a/candidate/model.py\n"
                "+++ b/candidate/model.py\n"
                "@@ -1 +1 @@\n"
                "-VALUE = 1\n"
                "+VALUE = 2\n"
            )
            manager = PatchManager(root)
            manager.apply(patch)
            self.assertEqual(target.read_text(), "VALUE = 2\n")
            manager.rollback(patch)
            self.assertEqual(target.read_text(), "VALUE = 1\n")


class RunnerAndLoggerTests(unittest.TestCase):
    def test_runner_uses_argv_without_shell(self) -> None:
        runner = ExperimentRunner(PROJECT_ROOT, timeout_seconds=5)
        result = runner.run(["python3", "-c", "print('ok')"])
        self.assertEqual(result.status, "success")
        self.assertEqual(result.stdout.strip(), "ok")

    def test_logger_creates_structured_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(tmp, "run_test")
            path = logger.log_iteration(1, {"status": "success"})
            self.assertTrue(path.exists())

    def test_numpy_metrics_are_json_serialisable(self) -> None:
        metrics = serialisable_metrics({"primary": np.float32(0.6016), "rows": 10})
        json.dumps(metrics)
        self.assertIsInstance(metrics["primary"], float)


if __name__ == "__main__":
    unittest.main()
