"""Git worktree isolation and constrained candidate execution."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import subprocess
import sys
import tempfile
import time

from experiment import make_result, write_result


_CAMPAIGN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_CANDIDATE_RE = re.compile(r"solution(?:_[A-Za-z0-9_]+)?\.py\Z")
_ALLOWED_IMPORTS = {
    "collections", "copy", "dataclasses", "functools", "heapq", "itertools", "math",
    "random", "statistics", "typing", "numpy", "baseline", "data",
}
_FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__"}
_MAX_FILE_BYTES = 256 * 1024
_MAX_CHANGED_LINES = 2000
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024


class SandboxError(ValueError):
    pass


def _secret_stripped_environment():
    environment = {}
    for name, value in os.environ.items():
        upper = name.upper()
        parts = set(upper.split("_"))
        if ({"TOKEN", "SECRET", "PASSWORD"} & parts
                or upper == "API_KEY" or upper.endswith("_API_KEY")
                or upper.endswith("APIKEY") or "ACCESS_KEY" in upper):
            continue
        environment[name] = value
    return environment


def _git_environment(index_file=None):
    environment = _secret_stripped_environment()
    environment.update({
        "GIT_AUTHOR_NAME": "Agent Campaign",
        "GIT_AUTHOR_EMAIL": "agent-campaign@localhost",
        "GIT_COMMITTER_NAME": "Agent Campaign",
        "GIT_COMMITTER_EMAIL": "agent-campaign@localhost",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    })
    if index_file is not None:
        environment["GIT_INDEX_FILE"] = str(index_file)
    return environment


def _safe_child_environment(seed, threads):
    environment = _secret_stripped_environment()
    thread_count = str(threads)
    environment.update({
        "PYTHONHASHSEED": str(seed),
        "OMP_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
    })
    return environment


def _resource_limits(memory_bytes):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_OUTPUT_BYTES, _MAX_OUTPUT_BYTES))
    return apply


def _run_process(argv, cwd, stdout_path, stderr_path, timeout, memory_gb, threads, seed):
    if timeout <= 0 or memory_gb <= 0 or threads <= 0:
        raise ValueError("timeout, memory_gb, and threads must be positive")
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            list(argv), cwd=cwd, env=_safe_child_environment(seed, threads), shell=False,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            preexec_fn=_resource_limits(int(memory_gb * 1024 ** 3)),
        )
        def terminate_group():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group()
            return_code = process.returncode
        except BaseException:
            terminate_group()
            raise
    return {
        "returncode": return_code,
        "timed_out": timed_out,
        "runtime_seconds": time.monotonic() - started,
    }


class WorktreeSandbox:
    def __init__(self, repo_root, campaign_root, campaign_id, python_executable=None):
        if not isinstance(campaign_id, str) or not _CAMPAIGN_RE.fullmatch(campaign_id):
            raise ValueError("campaign_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        self.repo_root = Path(repo_root).resolve(strict=True)
        self.campaign_root = Path(campaign_root)
        self.campaign_id = campaign_id
        self.python_executable = python_executable or sys.executable
        self.ref_root = f"refs/agent-campaigns/{campaign_id}"
        self.base_ref = self.ref_root + "/base"
        self.best_ref = self.ref_root + "/best"
        self.worktree_root = self.repo_root / ".agent-worktrees"
        check = self._git(["rev-parse", "--show-toplevel"]).stdout.decode().strip()
        if Path(check).resolve() != self.repo_root:
            raise ValueError(f"Not a Git repository root: {self.repo_root}")

    def _git(self, arguments, cwd=None, input_bytes=None, check=True, environment=None):
        command = ["git", *arguments]
        completed = subprocess.run(
            command, cwd=cwd or self.repo_root, input=input_bytes, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=False,
            env=environment if environment is not None else _secret_stripped_environment(),
        )
        if check and completed.returncode:
            message = completed.stderr.decode("utf-8", "replace").strip()
            raise SandboxError(f"git {' '.join(arguments[:2])} failed: {message}")
        return completed

    def resolve_ref(self, ref):
        return self._git(["rev-parse", "--verify", ref]).stdout.decode().strip()

    def ref_exists(self, ref):
        return self._git(["show-ref", "--verify", "--quiet", ref], check=False).returncode == 0

    def create_base_snapshot(self):
        if self.ref_exists(self.base_ref) or self.ref_exists(self.best_ref):
            raise SandboxError(f"Campaign {self.campaign_id!r} already has Git refs")
        parent = self.resolve_ref("HEAD")
        with tempfile.TemporaryDirectory(prefix="agent-index-") as temporary:
            index_path = Path(temporary) / "index"
            environment = _git_environment(index_path)
            self._git(["read-tree", "HEAD"], environment=environment)
            self._git(["add", "-A", "--", "."], environment=environment)
            tree = self._git(["write-tree"], environment=environment).stdout.decode().strip()
            commit = self._git(
                ["commit-tree", tree, "-p", parent],
                input_bytes=f"agent campaign {self.campaign_id} base\n".encode(),
                environment=environment,
            ).stdout.decode().strip()
        zero = "0" * 40
        self._git(["update-ref", self.base_ref, commit, zero])
        try:
            self._git(["update-ref", self.best_ref, commit, zero])
        except BaseException:
            self._git(["update-ref", "-d", self.base_ref], check=False)
            raise
        return commit

    @staticmethod
    def _iteration_number(iteration):
        if not isinstance(iteration, int) or iteration <= 0:
            raise ValueError("iteration must be a positive integer")
        return iteration

    def iteration_ref(self, iteration):
        self._iteration_number(iteration)
        return f"{self.ref_root}/iterations/{iteration}"

    def worktree_path(self, iteration):
        self._iteration_number(iteration)
        return self.worktree_root / f"{self.campaign_id}-{iteration:04d}"

    def remove_worktree(self, iteration):
        path = self.worktree_path(iteration)
        if path.exists() or path.is_symlink():
            self._git(["worktree", "remove", "--force", str(path)], check=False)
            if path.exists() or path.is_symlink():
                raise SandboxError(f"Could not remove worktree: {path}")
        self._git(["worktree", "prune"])

    def start_iteration(self, iteration, ref=None):
        path = self.worktree_path(iteration)
        self.remove_worktree(iteration)
        path.parent.mkdir(parents=True, exist_ok=True)
        commit = self.resolve_ref(ref or self.best_ref)
        self._git(["worktree", "add", "--detach", str(path), commit])
        return path

    def named_worktree_path(self, name):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", name):
            raise ValueError("worktree name must be lowercase alphanumeric with optional hyphens")
        return self.worktree_root / f"{self.campaign_id}-{name}"

    def remove_named_worktree(self, name):
        path = self.named_worktree_path(name)
        if path.exists() or path.is_symlink():
            self._git(["worktree", "remove", "--force", str(path)], check=False)
            if path.exists() or path.is_symlink():
                raise SandboxError(f"Could not remove worktree: {path}")
        self._git(["worktree", "prune"])

    def start_named_worktree(self, name, ref):
        path = self.named_worktree_path(name)
        self.remove_named_worktree(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        commit = self.resolve_ref(ref)
        self._git(["worktree", "add", "--detach", str(path), commit])
        return path

    @staticmethod
    def _patch_paths(patch):
        paths = []
        old_paths = []
        new_paths = []
        for line in patch.splitlines():
            if line.startswith("diff --git "):
                try:
                    parts = shlex.split(line)
                except ValueError:
                    raise SandboxError("Malformed diff path header") from None
                if len(parts) != 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
                    raise SandboxError("Malformed diff path header")
                old_path, new_path = parts[2][2:], parts[3][2:]
                if old_path != new_path:
                    raise SandboxError("Candidate patches may not rename files")
                paths.append(new_path)
            elif line.startswith("--- "):
                old_paths.append(line[4:].split("\t", 1)[0])
            elif line.startswith("+++ "):
                new_paths.append(line[4:].split("\t", 1)[0])
        if not paths:
            if len(old_paths) != len(new_paths) or not new_paths:
                raise SandboxError("Patch contains no file changes")
            for old_path, new_path in zip(old_paths, new_paths):
                if old_path == "/dev/null" or new_path == "/dev/null":
                    raise SandboxError("Candidate patches may only modify existing files")
                if old_path.startswith("a/"):
                    old_path = old_path[2:]
                if new_path.startswith("b/"):
                    new_path = new_path[2:]
                if old_path != new_path:
                    raise SandboxError("Candidate patches may not rename files")
                paths.append(new_path)
        if not paths:
            raise SandboxError("Patch contains no file changes")
        return paths

    @staticmethod
    def _validate_paths(worktree, paths):
        for name in paths:
            candidate = Path(name)
            if (candidate.is_absolute() or len(candidate.parts) != 1 or ".." in candidate.parts
                    or not _CANDIDATE_RE.fullmatch(name)):
                raise SandboxError(f"Candidate patch may only edit solution.py or solution_*.py: {name!r}")
            path = worktree / candidate
            if path.is_symlink():
                raise SandboxError(f"Candidate path may not be a symlink: {name!r}")

    @staticmethod
    def _validate_python(worktree):
        lightgbm_available = importlib.util.find_spec("lightgbm") is not None
        paths = [worktree / "solution.py", *sorted(worktree.glob("solution_*.py"))]
        for path in paths:
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise SandboxError(f"Candidate path is not a regular file: {path.name}")
            if path.stat().st_size > _MAX_FILE_BYTES:
                raise SandboxError(f"Candidate file exceeds 256 KiB: {path.name}")
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
            except (SyntaxError, UnicodeDecodeError) as exc:
                raise SandboxError(f"Invalid candidate Python in {path.name}: {exc}") from None
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level or node.module is None:
                        raise SandboxError(f"Relative imports are forbidden in {path.name}")
                    imports = [node.module]
                else:
                    imports = []
                for imported in imports:
                    root = imported.split(".", 1)[0]
                    allowed = (root in _ALLOWED_IMPORTS or root.startswith("solution_")
                               or (root == "lightgbm" and lightgbm_available))
                    if not allowed:
                        raise SandboxError(f"Forbidden import {imported!r} in {path.name}")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in _FORBIDDEN_CALLS:
                        raise SandboxError(f"Forbidden call {node.func.id!r} in {path.name}")
                if (isinstance(node, ast.Attribute) and node.attr.startswith("__")
                        and node.attr.endswith("__")):
                    raise SandboxError(f"Dunder attribute access is forbidden in {path.name}")

    def apply_and_commit(self, iteration, patch):
        if not isinstance(patch, str):
            raise SandboxError("patch must be a string")
        if "GIT binary patch" in patch or "Binary files " in patch:
            raise SandboxError("Binary patches are forbidden")
        if re.search(r"^(?:rename from|rename to|similarity index) ", patch, re.MULTILINE):
            raise SandboxError("Candidate patches may not rename files")
        if re.search(r"^(?:new file mode|old mode|new mode) 120000$", patch, re.MULTILINE):
            raise SandboxError("Candidate patches may not create symlinks")
        worktree = self.worktree_path(iteration)
        if not worktree.is_dir():
            raise SandboxError(f"Iteration worktree does not exist: {worktree}")
        paths = self._patch_paths(patch)
        self._validate_paths(worktree, paths)
        encoded = patch.encode("utf-8")
        self._git(["apply", "--check", "--recount", "-"], cwd=worktree, input_bytes=encoded)
        numstat = self._git(["apply", "--numstat", "--recount", "-"], cwd=worktree,
                            input_bytes=encoded).stdout.decode("utf-8", "replace")
        changed = 0
        for line in numstat.splitlines():
            additions, deletions, _ = line.split("\t", 2)
            if additions == "-" or deletions == "-":
                raise SandboxError("Binary patches are forbidden")
            changed += int(additions) + int(deletions)
        if changed > _MAX_CHANGED_LINES:
            raise SandboxError(f"Candidate patch changes {changed} lines; limit is 2000")
        try:
            self._git(["apply", "--recount", "-"], cwd=worktree, input_bytes=encoded)
            self._validate_paths(worktree, paths)
            self._validate_python(worktree)
        except BaseException:
            self._git(["reset", "--hard", "HEAD"], cwd=worktree, check=False)
            self._git(["clean", "-fd"], cwd=worktree, check=False)
            raise
        self._git(["add", "-A", "--", *paths], cwd=worktree)
        environment = _git_environment()
        self._git(["commit", "--no-gpg-sign", "-m", f"agent proposal {iteration}"],
                  cwd=worktree, environment=environment)
        commit = self.resolve_ref_at(worktree, "HEAD")
        ref = self.iteration_ref(iteration)
        old = self.resolve_ref(ref) if self.ref_exists(ref) else "0" * 40
        self._git(["update-ref", ref, commit, old])
        return commit

    def resolve_ref_at(self, worktree, ref):
        return self._git(["rev-parse", "--verify", ref], cwd=worktree).stdout.decode().strip()

    def promote(self, commit, expected_best):
        self._git(["update-ref", self.best_ref, commit, expected_best])
        return commit

    def best_patch(self):
        return self._git(["diff", "--binary", self.base_ref, self.best_ref]).stdout

    def diff_summary(self, commit):
        return self._git(["diff", "--stat", self.base_ref, commit]).stdout.decode("utf-8", "replace")

    def run_candidate(self, worktree, candidate_module, candidate_commit, data_dir, target_split,
                      seed, config, output_dir, stdout_path, stderr_path, timeout=900,
                      memory_gb=8, threads=4):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            self.python_executable, "experiment.py", "run",
            "--candidate-module", candidate_module,
            "--candidate-commit", candidate_commit,
            "--data_dir", str(Path(data_dir).resolve()),
            "--target-split", target_split,
            "--seed", str(seed),
            "--config-json", json.dumps(config, separators=(",", ":"), sort_keys=True),
            "--output-dir", str(output_dir.resolve()),
        ]
        outcome = _run_process(argv, worktree, stdout_path, stderr_path, timeout,
                               memory_gb, threads, seed)
        result_path = output_dir / "result.json"
        if not result_path.is_file():
            if outcome["timed_out"]:
                error_type = "TimeoutError"
                message = f"Candidate exceeded {timeout} second timeout"
            else:
                error_type = "ChildProcessError"
                message = f"Candidate process exited with status {outcome['returncode']} before result.json"
            result = make_result(candidate_commit, target_split, seed, config,
                                 runtime_seconds=outcome["runtime_seconds"],
                                 error={"type": error_type, "message": message})
            write_result(result_path, result)
        else:
            with result_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
        return result, outcome

    def run_submit(self, worktree, data_dir, score_paths, output_path, stdout_path, stderr_path,
                   timeout=900, memory_gb=8, threads=4):
        argv = [
            self.python_executable, "experiment.py", "submit", "--data_dir",
            str(Path(data_dir).resolve()), "--scores", *(str(Path(path).resolve()) for path in score_paths),
            "--output", str(Path(output_path).resolve()),
        ]
        return _run_process(argv, worktree, stdout_path, stderr_path, timeout,
                            memory_gb, threads, seed=0)

    def run_submission_check(self, worktree, data_dir, output_path, stdout_path, stderr_path,
                             timeout=900, memory_gb=8, threads=4):
        argv = [
            self.python_executable, "submit.py", str(Path(output_path).resolve()),
            "--data_dir", str(Path(data_dir).resolve()), "--split", "test", "--check",
        ]
        return _run_process(argv, worktree, stdout_path, stderr_path, timeout,
                            memory_gb, threads, seed=0)
