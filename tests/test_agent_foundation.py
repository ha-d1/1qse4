from __future__ import annotations

import sys
import tempfile
import unittest
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from convergence import ConvergenceState
from candidate_preflight import smoke_test, synthetic_splits, synthetic_training_auxiliary
from candidate.model import (
    blend_within_user,
    make_hard_bpr_pairs,
    user_author_history_scores,
)
from candidate.train import save_checkpoint
from data import (
    FIELDS,
    USER_AUTHOR_FIELDS,
    WEEKDAY_FIELDS,
    encode,
    fit_feature_encoder,
    load_train_auxiliary,
    transform_rows,
)
from agent import (
    candidate_sources,
    command_with_checkpoint,
    command_with_verified_data_dir,
    load_prior_experiment_history,
    record_failed_llm_usage,
    reference_api_contracts,
)
from development_data import (
    ensure_development_splits,
    load_training_auxiliary,
    remove_labels,
)
from experiment_runner import ExperimentRunner
from experiment_schema import ExperimentProposal
from llm_common import parse_plan, parse_proposal, redact_secrets
from llm_common import SYSTEM_INSTRUCTION
from llm_factory import create_llm_client
from make_submission import score_unlabelled_rows, within_user_rank_average
from generate_research_report import collect_report
from patch_manager import PatchError, PatchManager
from proposal_materializer import materialize_patch
from research_planner import build_research_context
from run_logger import RunLogger
from run_agent_with_keychain import find_data_dir, load_soclaas_key
from resource_tracker import ResourceTracker
from soclaas_client import SoCLaaSClient
from verify_baseline import serialisable_metrics


class DevelopmentDataTests(unittest.TestCase):
    def test_auxiliary_loader_reads_training_dates_only(self) -> None:
        header = (
            "date,is_click,is_like,is_follow,is_comment,is_forward,play_time_ms,long_view\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "log_standard_4_08_to_4_21_pure.csv").write_text(
                header + "20220408,1,0,0,1,0,1200,1\n"
            )
            (root / "log_standard_4_22_to_5_08_pure.csv").write_text(
                header + "20220422,0,1,1,0,1,500,0\n"
            )
            auxiliary = load_train_auxiliary(root, fields=("is_click", "play_time_ms"))
        np.testing.assert_array_equal(auxiliary["is_click"], [1.0])
        np.testing.assert_array_equal(auxiliary["play_time_ms"], [1200.0])

    def test_auxiliary_loader_rejects_primary_or_unknown_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported auxiliary"):
            load_train_auxiliary("unused", fields=("long_view",))

    def test_training_auxiliary_boundary_checks_alignment(self) -> None:
        train_rows = [
            (20220408, "u1", "v1", "a1", "t", 10.0, 1),
            (20220408, "u2", "v2", "a2", "t", 20.0, 0),
        ]
        with patch(
            "development_data.load_train_auxiliary",
            return_value={"is_click": np.asarray([1.0], dtype=np.float32)},
        ), patch(
            "development_data.load_selected", return_value={"train": train_rows}
        ):
            with self.assertRaisesRegex(ValueError, "not aligned"):
                load_training_auxiliary("unused", fields=("is_click",))

    def test_final_checkpoint_scores_unlabelled_rows(self) -> None:
        train = [
            (20220408, "u1", "v1", "a1", "t", 10.0, 1),
            (20220409, "u1", "v2", "a2", "t", 20.0, 0),
        ]
        target = [(20220429, "u1", "v3", "a1", "t", 15.0)]
        encoder = fit_feature_encoder(train, fields=FIELDS)
        model = SimpleNamespace(
            V=np.zeros((encoder["dimension"], 2), dtype=np.float32),
            W=np.zeros(encoder["dimension"], dtype=np.float32),
            b=np.float32(0.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = save_checkpoint(
                Path(tmp) / "best.npz",
                model,
                {"feature_set": "base", "lr": 0.00025},
            )
            scores, metadata = score_unlabelled_rows(checkpoint, train, target)
        self.assertEqual(len(scores), 1)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(metadata["feature_set"], "base")

    def test_within_user_rank_ensemble_is_group_local(self) -> None:
        combined = within_user_rank_average(
            [
                np.asarray([1.0, 3.0, 100.0, 0.0]),
                np.asarray([4.0, 2.0, -50.0, 50.0]),
            ],
            ["u1", "u1", "u2", "u2"],
        )
        np.testing.assert_allclose(combined, [0.5, 0.5, 0.5, 0.5])

    def test_within_user_rank_ensemble_rejects_misalignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            within_user_rank_average([np.asarray([1.0])], ["u1", "u2"])

    def test_transform_unlabelled_rows_does_not_require_label_column(self) -> None:
        train = [(20220408, "u1", "v1", "a1", "t", 10.0, 1)]
        encoder = fit_feature_encoder(train)
        X, labels, users = transform_rows(
            [(20220429, "u1", "v2", "a1", "t", 12.0)],
            encoder,
            include_labels=False,
        )
        self.assertEqual(X.shape, (1, 5))
        self.assertIsNone(labels)
        self.assertEqual(users, ["u1"])

    def test_checkpoint_persists_weights_and_metadata(self) -> None:
        model = SimpleNamespace(
            V=np.asarray([[1.0, 2.0]], dtype=np.float32),
            W=np.asarray([3.0], dtype=np.float32),
            b=np.float32(4.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(Path(tmp) / "best.npz", model, {"seed": 1})
            with np.load(path) as checkpoint:
                np.testing.assert_array_equal(checkpoint["V"], model.V)
                self.assertEqual(json.loads(str(checkpoint["metadata"]))["seed"], 1)

    def test_candidate_rejects_zero_negative_samples(self) -> None:
        from candidate.model import train_candidate

        with self.assertRaisesRegex(ValueError, "negatives_per_positive"):
            train_candidate(
                {"train": [], "valid": []}, negatives_per_positive=0
            )

    def test_candidate_rejects_misaligned_auxiliary_targets(self) -> None:
        from candidate.model import train_candidate

        with self.assertRaisesRegex(ValueError, "not aligned"):
            train_candidate(
                synthetic_splits(),
                epochs=1,
                auxiliary_targets={"is_click": np.asarray([1.0])},
                auxiliary_task="is_click",
                auxiliary_ratio=0.1,
            )

    def test_candidate_auxiliary_objective_runs_on_train_only_targets(self) -> None:
        from candidate.model import train_candidate

        metrics, _ = train_candidate(
            synthetic_splits(),
            epochs=1,
            auxiliary_targets={"is_click": synthetic_training_auxiliary()["is_click"]},
            auxiliary_task="is_click",
            auxiliary_ratio=0.25,
        )
        self.assertEqual(metrics["auxiliary_task"], "is_click")
        self.assertGreater(metrics["history"][0]["auxiliary_pairs"], 0)

    def test_candidate_auxiliary_ratio_requires_task(self) -> None:
        from candidate.model import train_candidate

        with self.assertRaisesRegex(ValueError, "requires an auxiliary_task"):
            train_candidate(synthetic_splits(), epochs=1, auxiliary_ratio=0.1)

    def test_hard_negative_mining_selects_highest_scoring_candidates(self) -> None:
        class ColumnScoreModel:
            def predict(self, values):
                return values[:, 1].astype(np.float32)

        positive_row = np.asarray([[1, 0]], dtype=np.int32)
        candidate_pairs = [
            (positive_row, np.asarray([[0, value]], dtype=np.int32))
            for value in (2, 9, 4, 6)
        ]
        with patch("candidate.model.make_bpr_pairs", side_effect=candidate_pairs):
            positive, negative = make_hard_bpr_pairs(
                np.zeros((1, 2), dtype=np.int32),
                np.asarray([1.0]),
                ["u"],
                ColumnScoreModel(),
                negatives_per_positive=1,
                candidate_multiplier=4,
                seed=7,
            )
        self.assertEqual(positive.shape, negative.shape)
        self.assertEqual(int(negative[0, 1]), 9)

    def test_candidate_rejects_invalid_hard_negative_pool(self) -> None:
        from candidate.model import train_candidate

        with self.assertRaisesRegex(ValueError, "hard_candidate_multiplier"):
            train_candidate(
                synthetic_splits(), epochs=1, hard_candidate_multiplier=0
            )

    def test_history_scores_ignore_validation_labels(self) -> None:
        train = [
            (20220408, "u1", "v1", "a1", "t", 10.0, 1),
            (20220409, "u1", "v2", "a2", "t", 20.0, 0),
        ]
        valid_a = [(20220415, "u1", "v3", "a1", "t", 15.0, 0)]
        valid_b = [(20220415, "u1", "v3", "a1", "t", 15.0, 1)]
        np.testing.assert_array_equal(
            user_author_history_scores(train, valid_a),
            user_author_history_scores(train, valid_b),
        )

    def test_history_blend_operates_within_user(self) -> None:
        scores = blend_within_user(
            np.asarray([0.0, 1.0, 10.0, 20.0]),
            np.asarray([1.0, 0.0, 0.0, 1.0]),
            ["u1", "u1", "u2", "u2"],
            0.5,
        )
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.isfinite(scores).all())

    def test_weekday_feature_is_optional_and_train_fitted(self) -> None:
        splits = {
            "train": [
                (20220408, "u1", "v1", "a1", "t", 10.0, 1),
                (20220409, "u1", "v2", "a2", "t", 20.0, 0),
            ],
            "valid": [(20220415, "u1", "v1", "a1", "t", 15.0, 1)],
        }
        base, _ = encode(splits, fields=FIELDS)
        temporal, _ = encode(splits, fields=WEEKDAY_FIELDS)
        self.assertEqual(base["train"][0].shape[1], 5)
        self.assertEqual(temporal["train"][0].shape[1], 6)
        self.assertEqual(temporal["valid"][0][0, -1], temporal["train"][0][0, -1])

    def test_user_author_cross_is_optional(self) -> None:
        splits = {
            "train": [
                (20220408, "u1", "v1", "a1", "t", 10.0, 1),
                (20220408, "u1", "v2", "a2", "t", 20.0, 0),
            ],
            "valid": [(20220415, "u1", "v3", "a1", "t", 15.0, 1)],
        }
        encoded, _ = encode(splits, fields=USER_AUTHOR_FIELDS)
        self.assertEqual(encoded["train"][0].shape[1], 6)
        self.assertEqual(encoded["valid"][0][0, -1], encoded["train"][0][0, -1])

    def test_rejects_test_labels(self) -> None:
        with self.assertRaises(ValueError):
            ensure_development_splits({"train": [], "valid": [], "test": []})

    def test_removes_labels(self) -> None:
        rows = [(20220429, "u", "v", "a", "tab", 1000.0, 1)]
        self.assertEqual(remove_labels(rows), [(20220429, "u", "v", "a", "tab", 1000.0)])

    def test_development_loader_requests_only_train_and_valid(self) -> None:
        fake = {"train": [], "valid": []}
        with patch("development_data.load_selected", return_value=fake) as loader:
            from development_data import load_development_splits

            self.assertEqual(load_development_splits("unused"), fake)
        loader.assert_called_once_with("unused", ["train", "valid"])

    def test_final_rows_use_unlabelled_loader(self) -> None:
        rows = [(20220429, "u", "v", "a", "tab", 1000.0)]
        with patch("development_data.load_unlabelled", return_value=rows) as loader:
            from development_data import load_final_prediction_rows

            self.assertEqual(load_final_prediction_rows("unused"), rows)
        loader.assert_called_once_with("unused", "test")

    def test_preflight_fixture_has_only_development_splits(self) -> None:
        splits = synthetic_splits()
        self.assertEqual(set(splits), {"train", "valid"})
        self.assertTrue(all(len(row) == 7 for rows in splits.values() for row in rows))
        grouped = {}
        for row in splits["train"]:
            grouped.setdefault(row[1], []).append(row[6])
        self.assertTrue(all(grouped["all_positive"]))
        self.assertFalse(any(grouped["all_negative"]))
        self.assertEqual(len(grouped["singleton"]), 1)

    def test_preflight_auxiliary_fixture_matches_train_only(self) -> None:
        splits = synthetic_splits()
        auxiliary = synthetic_training_auxiliary()
        self.assertNotIn("long_view", auxiliary)
        self.assertTrue(
            all(len(values) == len(splits["train"]) for values in auxiliary.values())
        )


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

    def test_agent_budget_allows_one_bounded_same_iteration_repair(self) -> None:
        config = json.loads((PROJECT_ROOT / "agent_config.json").read_text())
        self.assertEqual(config["budget"]["max_repair_attempts"], 1)


class ProposalTests(unittest.TestCase):
    def test_coder_contract_routes_candidate_data_and_forbids_valid_label_features(self) -> None:
        self.assertIn("from candidate.data import", SYSTEM_INSTRUCTION)
        self.assertIn("Validation labels are evaluation-only", SYSTEM_INSTRUCTION)

    def test_rejects_changes_outside_candidate(self) -> None:
        proposal = ExperimentProposal(
            "h", "r", ["evaluate.py"], "gain", "risk", "patch", ["python3", "x.py"]
        )
        with self.assertRaises(ValueError):
            proposal.validate()

    @staticmethod
    def proposal_payload() -> dict:
        return {
            "hypothesis": "Try multiple negatives",
            "reasoning": "Pairwise ranking benefits from more comparisons",
            "target_files": ["candidate/model.py"],
            "expected_effect": "Improve GAUC",
            "risk": "Longer runtime",
            "patch": "--- a/candidate/model.py\n+++ b/candidate/model.py\n",
            "command": ["python3", "candidate/train.py", "--objective", "bpr"],
        }

    def test_soclaas_structured_proposal_usage_and_repair_prompt(self) -> None:
        response = SimpleNamespace(
            id="chatcmpl-mock",
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=12, total_tokens=52),
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.proposal_payload())))],
        )

        class Completions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return response

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = SoCLaaSClient(client=client)
        result = adapter.repair({"iteration": 1}, {"error": "bad patch"})
        self.assertEqual(result.proposal.command[1], "candidate/train.py")
        self.assertEqual(result.usage["prompt_tokens"], 40)
        self.assertEqual(result.usage["response_tokens"], 12)
        self.assertEqual(result.interaction_id, "chatcmpl-mock")
        user_message = completions.calls[0]["messages"][1]["content"]
        self.assertIn("bad patch", user_message)
        self.assertIn("Recheck every imported symbol", user_message)
        self.assertEqual(completions.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(completions.calls[0]["max_tokens"], 7000)

    def test_soclaas_wraps_provider_errors(self) -> None:
        class Completions:
            def create(self, **kwargs):
                raise ConnectionError("offline")

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with self.assertRaisesRegex(RuntimeError, "SoCLaaS proposal failed"):
            SoCLaaSClient(client=client, max_attempts=1).propose({"iteration": 1})

    def test_failed_structured_response_preserves_billable_usage(self) -> None:
        response = SimpleNamespace(
            id="bad-json",
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4, total_tokens=13),
            choices=[SimpleNamespace(message=SimpleNamespace(content="{unfinished"))],
        )

        class Completions:
            def create(self, **kwargs):
                return response

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with self.assertRaises(RuntimeError) as caught:
            SoCLaaSClient(client=client, max_attempts=1).propose({"iteration": 1})
        resources = ResourceTracker()
        record_failed_llm_usage(resources, caught.exception)
        self.assertEqual(resources.snapshot()["total_tokens"], 13)

    def test_soclaas_planning_stage_uses_autonomous_model(self) -> None:
        plan = {
            "hypothesis": "Listwise softmax improves top-five ranking",
            "reasoning": "It optimises each user impression set jointly",
            "direction": "listwise objective",
            "target_files": ["candidate/model.py", "candidate/train.py"],
            "implementation_requirements": ["Implement an analytic score gradient"],
            "command": ["python", "candidate/train.py", "--objective", "listwise"],
            "risk": "Higher runtime",
        }
        response = SimpleNamespace(
            id="plan-mock",
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10, total_tokens=40),
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(plan)))],
        )

        class Completions:
            def __init__(self):
                self.call = None

            def create(self, **kwargs):
                self.call = kwargs
                return response

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        result = SoCLaaSClient(client=client, planning_model="qwen3.6:35b").plan({"iteration": 1})
        self.assertEqual(result.plan["direction"], "listwise objective")
        self.assertEqual(result.usage["total_tokens"], 40)
        self.assertEqual(completions.call["model"], "qwen3.6:35b")
        self.assertEqual(completions.call["max_tokens"], 1200)

    def test_compact_reference_contracts_replace_full_protected_sources(self) -> None:
        contracts = reference_api_contracts()
        self.assertIn("make_bpr_pairs", contracts["baseline.py"])
        self.assertIn("load_training_auxiliary", contracts["development_data.py"])
        self.assertLess(sum(map(len, contracts.values())), 2_000)

    def test_plan_rejects_protected_target(self) -> None:
        plan = {
            "hypothesis": "h",
            "reasoning": "r",
            "direction": "d",
            "target_files": ["evaluate.py"],
            "implementation_requirements": ["x"],
            "command": ["python", "candidate/train.py"],
            "risk": "risk",
        }
        with self.assertRaisesRegex(ValueError, "candidate"):
            parse_plan(json.dumps(plan))

    def test_plan_parser_handles_reasoning_wrapper(self) -> None:
        plan = {
            "hypothesis": "h",
            "reasoning": "r",
            "direction": "d",
            "target_files": ["candidate/model.py"],
            "implementation_requirements": ["x"],
            "command": ["python", "candidate/train.py"],
            "risk": "risk",
        }
        wrapped = "Reasoning complete.\n```json\n" + json.dumps(plan) + "\n```"
        self.assertEqual(parse_plan(wrapped)["direction"], "d")

    def test_plan_parser_discards_echoed_request_fields(self) -> None:
        plan = {
            "hypothesis": "h",
            "reasoning": "r",
            "direction": "d",
            "target_files": ["candidate/model.py"],
            "implementation_requirements": ["x"],
            "command": ["python", "candidate/train.py"],
            "risk": "risk",
            "task": "echo",
            "context": {"echo": True},
            "plan_schema": {"echo": True},
        }
        parsed = parse_plan(json.dumps(plan))
        self.assertNotIn("task", parsed)
        self.assertEqual(parsed["hypothesis"], "h")

    def test_plan_parser_normalizes_candidate_script_command(self) -> None:
        plan = {
            "hypothesis": "h",
            "reasoning": "r",
            "direction": "d",
            "target_files": ["candidate/model.py"],
            "implementation_requirements": ["x"],
            "command": ["candidate/train.py", "--objective", "listwise"],
            "risk": "risk",
        }
        self.assertEqual(parse_plan(json.dumps(plan))["command"][0], "python")

    def test_plan_parser_normalizes_fm_file_alias(self) -> None:
        plan = {
            "hypothesis": "h",
            "reasoning": "r",
            "direction": "time features",
            "target_files": ["candidate/fm.py", "candidate/train.py"],
            "implementation_requirements": ["In fm.py, add the time feature"],
            "command": ["python", "candidate/train.py"],
            "risk": "risk",
        }
        parsed = parse_plan(json.dumps(plan))
        self.assertEqual(
            parsed["target_files"], ["candidate/model.py", "candidate/train.py"]
        )
        self.assertIn("In model.py", parsed["implementation_requirements"][0])

    def test_plan_parser_routes_new_candidate_entrypoint_through_train(self) -> None:
        plan = {
            "hypothesis": "h",
            "reasoning": "r",
            "direction": "d",
            "target_files": ["candidate/listnet.py"],
            "implementation_requirements": ["x"],
            "command": ["python", "candidate/listnet.py", "--seed", "0"],
            "risk": "risk",
        }
        parsed = parse_plan(json.dumps(plan))
        self.assertEqual(parsed["command"][1], "candidate/train.py")
        self.assertIn("candidate/train.py", parsed["target_files"])

    def test_parses_fenced_json(self) -> None:
        proposal = parse_proposal("```json\n" + json.dumps(self.proposal_payload()) + "\n```")
        self.assertEqual(proposal.hypothesis, "Try multiple negatives")

    def test_rejects_test_split_command(self) -> None:
        payload = self.proposal_payload()
        payload["command"] += ["--split", "test"]
        with self.assertRaisesRegex(ValueError, "hidden test"):
            ExperimentProposal(**payload).validate()

    def test_provider_factory_defaults_to_soclaas(self) -> None:
        with patch("soclaas_client.SoCLaaSClient.__init__", return_value=None):
            client = create_llm_client(
                {"provider": "soclaas", "model": "qwen3-coder-next", "max_attempts": 1}
            )
        self.assertIsInstance(client, SoCLaaSClient)

    def test_redacts_environment_credentials(self) -> None:
        with patch.dict(os.environ, {"SOCLAAS_API_KEY": "do-not-log-me"}, clear=False):
            self.assertEqual(redact_secrets("error do-not-log-me"), "error [REDACTED]")

    def test_materializes_full_file_update_as_valid_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate").mkdir()
            target = root / "candidate" / "model.py"
            target.write_text("VALUE = 1\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            proposal = ExperimentProposal(
                "h",
                "r",
                ["candidate/model.py"],
                "gain",
                "risk",
                "",
                ["python3", "candidate/train.py"],
                [{"path": "candidate/model.py", "content": "VALUE = 2\n"}],
            )
            proposal.validate()
            generated = materialize_patch(root, proposal)
            manager = PatchManager(root)
            manager.apply(generated)
            self.assertEqual(target.read_text(), "VALUE = 2\n")
            manager.rollback(generated)
            self.assertEqual(target.read_text(), "VALUE = 1\n")

    def test_materializer_allows_unchanged_declared_targets_to_be_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate").mkdir()
            target = root / "candidate" / "model.py"
            target.write_text("VALUE = 1\n")
            proposal = ExperimentProposal(
                "h",
                "r",
                ["candidate/model.py", "candidate/train.py"],
                "gain",
                "risk",
                "",
                ["python3", "candidate/train.py"],
                [{"path": "candidate/model.py", "content": "VALUE = 2\n"}],
            )
            proposal.validate()
            generated = materialize_patch(root, proposal)
            self.assertIn("candidate/model.py", generated)
            self.assertNotIn("candidate/train.py", generated)

    def test_materializer_repairs_whitespace_and_multifile_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate").mkdir()
            first = root / "candidate" / "model.py"
            second = root / "candidate" / "train.py"
            first.write_text("VALUE = 1\n")
            second.write_text("MODE = 'old'\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            proposal = ExperimentProposal(
                "h",
                "r",
                ["candidate/model.py", "candidate/train.py"],
                "gain",
                "risk",
                "",
                ["python3", "candidate/train.py"],
                [
                    {"path": "candidate/model.py", "content": "VALUE = 2  \n    \n"},
                    {"path": "candidate/train.py", "content": "MODE = 'new'"},
                ],
            )
            generated = materialize_patch(root, proposal)
            manager = PatchManager(root)
            manager.apply(generated)
            self.assertEqual(first.read_text(), "VALUE = 2\n")
            self.assertEqual(second.read_text(), "MODE = 'new'\n")

    def test_research_context_marks_plain_bpr_as_completed_user_work(self) -> None:
        context = build_research_context(
            1,
            {"primary": 0.60323},
            [],
            100,
            10,
            reference_sources={"baseline.py": "class FM: pass\n"},
        )
        existing = context["existing_user_work"]
        self.assertIn("four independently sampled negatives", existing["candidate"])
        self.assertIn("Do not rediscover plain BPR", existing["instruction"])
        self.assertAlmostEqual(existing["matched_seed_primary"]["mean"], 0.6038278142611185)
        self.assertEqual(
            context["read_only_reference_sources"]["baseline.py"], "class FM: pass\n"
        )
        self.assertEqual(context["data_contract"]["user_id_type"], "opaque string")
        self.assertEqual(context["completed_autonomous_directions"][0]["decision"], "rejected")
        self.assertNotIn("listwise", " ".join(context["available_directions"]).lower())
        self.assertTrue(
            any(
                "Soft-NDCG" in item["direction"]
                for item in context["completed_autonomous_directions"]
            )
        )
        self.assertNotIn("top-k", " ".join(context["available_directions"]).lower())


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
    def test_research_report_aggregates_tokens_and_manual_interventions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            autonomous = runs / "run_20260101T000000Z"
            autonomous.mkdir()
            (autonomous / "summary.json").write_text(
                json.dumps(
                    {
                        "resources": {
                            "prompt_tokens": 10,
                            "response_tokens": 5,
                            "total_tokens": 15,
                            "llm_calls": 2,
                            "iterations": 1,
                            "wall_clock_seconds": 3.0,
                        }
                    }
                )
            )
            manual = runs / "manual_test"
            manual.mkdir()
            (manual / "result.json").write_text(
                json.dumps(
                    {
                        "resources": {
                            "manual_interventions": 1,
                            "training_runtime_seconds": 4.0,
                        }
                    }
                )
            )
            config = {
                "benchmark": {
                    "name": "KuaiRand-Pure",
                    "label": "long_view",
                    "metrics": ["GAUC", "nDCG@5"],
                    "primary": "mean(GAUC, nDCG@5)",
                    "official_valid_primary": 0.6016,
                    "incumbent_valid_primary": 0.6039,
                    "incumbent_single_seed_primary": 0.6039,
                    "incumbent_three_seed_mean": 0.6038,
                },
                "llm": {
                    "provider": "soclaas",
                    "planning_model": "planner",
                    "model": "coder",
                },
                "safety": {"development_splits": ["train", "valid"]},
            }
            report = collect_report(runs, config)
        self.assertEqual(report["autonomous_agent"]["total_tokens"], 15)
        self.assertEqual(report["other_resources"]["manual_interventions"], 1)

    def test_checkpoint_path_is_forced(self) -> None:
        command = [
            "python",
            "candidate/train.py",
            "--checkpoint-out",
            "wrong.npz",
        ]
        forced = command_with_checkpoint(command, Path("run/candidate.npz"))
        self.assertEqual(forced.count("--checkpoint-out"), 1)
        self.assertEqual(forced[-1], "run/candidate.npz")

    def test_runner_uses_argv_without_shell(self) -> None:
        runner = ExperimentRunner(PROJECT_ROOT, timeout_seconds=5)
        result = runner.run(["python3", "-c", "print('ok')"])
        self.assertEqual(result.status, "success")
        self.assertEqual(result.stdout.strip(), "ok")

    def test_runner_maps_generic_python_to_active_interpreter(self) -> None:
        runner = ExperimentRunner(PROJECT_ROOT, timeout_seconds=5)
        result = runner.run(["python", "-c", "import sys; print(sys.executable)"])
        self.assertEqual(result.status, "success")
        self.assertEqual(Path(result.stdout.strip()).resolve(), Path(sys.executable).resolve())

    def test_candidate_preflight_runs_current_control(self) -> None:
        with patch("builtins.print"):
            smoke_test(["python", "candidate/train.py", "--objective", "bpr"])

    def test_runner_does_not_expose_llm_credentials(self) -> None:
        runner = ExperimentRunner(PROJECT_ROOT, timeout_seconds=5)
        with patch.dict(os.environ, {"SOCLAAS_API_KEY": "do-not-expose"}, clear=False):
            result = runner.run(
                [
                    "python3",
                    "-c",
                    "import os; print(os.environ.get('SOCLAAS_API_KEY', 'absent'))",
                ]
            )
        self.assertEqual(result.stdout.strip(), "absent")

    def test_logger_creates_structured_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(tmp, "run_test")
            path = logger.log_iteration(1, {"status": "success"})
            self.assertTrue(path.exists())

    def test_logger_snapshots_best_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "candidate"
            source.mkdir()
            (source / "model.py").write_text("VALUE = 1\n")
            logger = RunLogger(Path(tmp) / "runs", "run_test")
            snapshot = logger.snapshot_candidate(source, 2)
            self.assertEqual((snapshot / "model.py").read_text(), "VALUE = 1\n")

    def test_numpy_metrics_are_json_serialisable(self) -> None:
        metrics = serialisable_metrics({"primary": np.float32(0.6016), "rows": 10})
        json.dumps(metrics)
        self.assertIsInstance(metrics["primary"], float)

    def test_resource_tracker_accepts_provider_usage_fields(self) -> None:
        resources = ResourceTracker()
        resources.add_llm_usage(prompt_tokens=20, response_tokens=10, total_tokens=30)
        snapshot = resources.snapshot()
        self.assertEqual(snapshot["total_tokens"], 30)
        self.assertEqual(snapshot["llm_calls"], 1)

    def test_prior_run_history_is_available_to_next_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = (
                Path(tmp)
                / "run_20260101T000000Z"
                / "iterations"
                / "iteration_001"
                / "result.json"
            )
            result.parent.mkdir(parents=True)
            result.write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "status": "failed",
                        "accepted": False,
                        "error": "string user id",
                        "attempts": [
                            {
                                "proposal": {"hypothesis": "ListNet"},
                                "preflight": {"status": "failed"},
                            }
                        ],
                    }
                )
            )
            history = load_prior_experiment_history(Path(tmp))
            self.assertEqual(history[0]["hypothesis"], "ListNet")
            self.assertEqual(history[0]["last_preflight"]["status"], "failed")

    def test_generated_data_path_is_replaced_with_verified_path(self) -> None:
        command = ["python", "candidate/train.py", "--data_dir", "data", "--seed", "0"]
        result = command_with_verified_data_dir(command, Path("/verified/data"))
        self.assertNotIn("data", result)
        self.assertEqual(result[-2:], ["--data_dir", "/verified/data"])

    def test_planner_cli_aliases_are_normalized(self) -> None:
        command = [
            "python",
            "candidate/train.py",
            "--dataset",
            "KuaiRand-Pure",
            "--label",
            "long_view",
            "--model",
            "ndcg_bpr",
        ]
        result = command_with_verified_data_dir(command, Path("/verified/data"))
        self.assertNotIn("--dataset", result)
        self.assertNotIn("--label", result)
        self.assertIn("--objective", result)
        self.assertIn("ndcg_bpr", result)

    def test_keychain_loader_captures_key_without_printing_it(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="secret-from-keychain\n", stderr="")
        with patch("run_agent_with_keychain.subprocess.run", return_value=completed) as runner:
            self.assertEqual(load_soclaas_key("student"), "secret-from-keychain")
        command = runner.call_args.args[0]
        self.assertEqual(command[-1], "-w")
        self.assertNotIn("secret-from-keychain", command)

    def test_keychain_launcher_finds_explicit_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            self.assertEqual(find_data_dir(root, str(data_dir)), data_dir.resolve())


if __name__ == "__main__":
    unittest.main()
