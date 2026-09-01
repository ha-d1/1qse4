"""Trusted profiling, candidate execution, and submission harness."""
import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import data
from evaluate import evaluate
from submit import read_submission, write_submission


_CANDIDATE_RE = re.compile(r"solution(?:_[A-Za-z0-9_]+)?\Z")
_PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)
_TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
_LATER_LOGS = (
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
)
_SERVING_LOG_COLUMNS = {
    "date", "user_id", "video_id", "hourmin", "time_ms", "duration_ms", "tab", "is_rand",
}
_SCREEN_TRAIN_RANGE = (20220408, 20220418)
_SCREEN_VALID_RANGE = (20220419, 20220421)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def dependency_versions():
    try:
        import lightgbm
    except ImportError:
        lightgbm_version = None
    else:
        lightgbm_version = lightgbm.__version__
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "lightgbm_version": lightgbm_version,
    }


def make_result(candidate_commit, target_split, seed, config, status="error", metrics=None,
                rows=0, runtime_seconds=0.0, score_sha256=None, error=None):
    versions = dependency_versions()
    return {
        "status": status,
        "candidate_commit": candidate_commit,
        "target_split": target_split,
        "seed": seed,
        "config": config,
        "metrics": metrics,
        "rows": rows,
        "runtime_seconds": runtime_seconds,
        "score_sha256": score_sha256,
        "python_version": versions["python_version"],
        "numpy_version": versions["numpy_version"],
        "lightgbm_version": versions["lightgbm_version"],
        "error": error,
    }


def write_result(path, result):
    _atomic_json(path, result)


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_sanitized_log(source, destination, keep_outcomes):
    with Path(source).open(newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            raise ValueError(f"Log CSV has no date header: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for line, record in enumerate(reader, start=2):
                try:
                    date = int(record["date"])
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid date in {source} line {line}") from None
                if not keep_outcomes(date):
                    for name in reader.fieldnames:
                        if name not in _SERVING_LOG_COLUMNS:
                            record[name] = "0"
                writer.writerow(record)


def prepare_candidate_data(data_dir, output_root, fingerprint):
    """Create immutable candidate views with every holdout outcome masked."""
    data_dir = Path(data_dir).resolve(strict=True)
    output_root = Path(output_root)
    manifest_path = output_root / "manifest.json"
    expected_manifest = {
        "data_fingerprint": fingerprint,
        "screen_train_range": list(_SCREEN_TRAIN_RANGE),
        "screen_valid_range": list(_SCREEN_VALID_RANGE),
        "version": 1,
    }
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError(f"Unsafe candidate data path: {output_root}")
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = None
        if current != expected_manifest:
            raise ValueError("Existing candidate data view does not match this dataset")
        return {name: output_root / name for name in ("screen", "valid", "test")}

    temporary = output_root.with_name(output_root.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        for view in ("screen", "valid", "test"):
            (temporary / view).mkdir(parents=True)
        for source in sorted(data_dir.iterdir(), key=lambda item: item.name):
            if source.suffix.lower() != ".csv" or source.is_symlink() or not source.is_file():
                continue
            if source.name == _TRAIN_LOG:
                _write_sanitized_log(
                    source, temporary / "screen" / source.name,
                    lambda date: _SCREEN_TRAIN_RANGE[0] <= date <= _SCREEN_TRAIN_RANGE[1])
                for view in ("valid", "test"):
                    _link_or_copy(source, temporary / view / source.name)
            elif source.name in _LATER_LOGS:
                for view in ("screen", "valid", "test"):
                    _write_sanitized_log(source, temporary / view / source.name, lambda _date: False)
            else:
                for view in ("screen", "valid", "test"):
                    _link_or_copy(source, temporary / view / source.name)
        _atomic_json(temporary / "manifest.json", expected_manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {name: output_root / name for name in ("screen", "valid", "test")}


class DataAccess:
    """Read-only access to sanitized CSV columns within one evaluation view."""

    def __init__(self, data_dir, target_split, evaluation_split=None):
        if target_split not in ("valid", "test"):
            raise ValueError(f"Unknown target split: {target_split!r}")
        evaluation_split = evaluation_split or target_split
        if evaluation_split not in ("screen", "valid", "test"):
            raise ValueError(f"Unknown evaluation split: {evaluation_split!r}")
        expected_target = "test" if evaluation_split == "test" else "valid"
        if target_split != expected_target:
            raise ValueError(
                f"Target {target_split!r} does not match evaluation {evaluation_split!r}")
        self.data_dir = Path(data_dir).resolve(strict=True)
        if not self.data_dir.is_dir():
            raise ValueError(f"Data directory is not a directory: {data_dir}")
        self.target_split = target_split
        self.evaluation_split = evaluation_split
        self._headers = self._read_headers()

    def _read_headers(self):
        headers = {}
        for path in sorted(self.data_dir.iterdir(), key=lambda item: item.name):
            if path.suffix.lower() != ".csv" or path.is_symlink() or not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
            if header is None:
                header = []
            if not header or any(not name for name in header) or len(set(header)) != len(header):
                raise ValueError(f"Invalid CSV header in {path.name}")
            headers[path.name] = tuple(header)
        return headers

    def columns(self):
        return dict(self._headers)

    def _resolve(self, filename):
        if not isinstance(filename, str) or not filename:
            raise ValueError("filename must be a non-empty string")
        if (filename != Path(filename).name or "/" in filename or "\\" in filename
                or not filename.lower().endswith(".csv")):
            raise ValueError(f"Only direct CSV filenames are allowed: {filename!r}")
        if filename not in self._headers:
            raise ValueError(f"Unknown CSV file: {filename!r}")
        path = self.data_dir / filename
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self.data_dir:
            raise ValueError(f"Unsafe CSV path: {filename!r}")
        return path

    def _split_range(self, split):
        if split not in ("train", self.target_split):
            raise ValueError(f"Split {split!r} is unavailable for target {self.target_split!r}")
        if self.evaluation_split == "screen":
            return _SCREEN_TRAIN_RANGE if split == "train" else _SCREEN_VALID_RANGE
        return data.SPLITS[split]

    def iter_rows(self, filename, columns, split=None):
        path = self._resolve(filename)
        if not isinstance(columns, tuple) or not all(isinstance(name, str) for name in columns):
            raise ValueError("columns must be a tuple of column names")
        unknown = sorted(set(columns) - set(self._headers[filename]))
        if unknown:
            raise ValueError(f"Unknown columns for {filename}: {unknown}")
        if data.LABEL in columns:
            raise ValueError(f"Label column {data.LABEL!r} is not available through DataAccess")
        has_date = "date" in self._headers[filename]
        if has_date and split is None:
            raise ValueError(f"An explicit split is required for dated CSV {filename!r}")
        if not has_date and split is not None:
            raise ValueError(f"CSV file {filename!r} has no date column")
        lo, hi = self._split_range(split) if split is not None else (None, None)

        def rows():
            with path.open(newline="", encoding="utf-8") as handle:
                for line, record in enumerate(csv.DictReader(handle), start=2):
                    if split is not None:
                        try:
                            date = int(record["date"])
                        except (TypeError, ValueError):
                            raise ValueError(f"Invalid date in {filename} line {line}") from None
                        if not lo <= date <= hi:
                            continue
                    yield tuple(record[name] for name in columns)

        return rows()


def _percentile_map(values):
    if not values:
        return {f"p{p}": None for p in _PERCENTILES}
    result = np.percentile(np.asarray(values, dtype=np.float64), _PERCENTILES)
    return {f"p{p}": float(value) for p, value in zip(_PERCENTILES, result)}


def _canonical_values(row, duration_edges):
    return (row[1], row[2], row[3], row[4], str(int(np.searchsorted(duration_edges, row[5]))))


def _fingerprint(splits):
    digest = hashlib.sha256()
    for split in ("train", "valid", "test"):
        digest.update(split.encode("ascii") + b"\0")
        for row in splits[split]:
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def profile_data(data_dir):
    splits = data.load(data_dir)
    train_durations = [row[5] for row in splits["train"]]
    duration_edges = (np.quantile(np.asarray(train_durations, dtype=np.float64),
                                  np.linspace(0, 1, 11)[1:-1])
                      if train_durations else np.asarray([], dtype=np.float64))
    train_vocab = [set() for _ in data.FIELDS]
    for row in splits["train"]:
        for index, value in enumerate(_canonical_values(row, duration_edges)):
            train_vocab[index].add(value)

    split_profiles = {}
    for split, rows in splits.items():
        users = {row[1] for row in rows}
        videos = {row[2] for row in rows}
        exposures = {}
        observed = [set() for _ in data.FIELDS]
        unknown = [0] * len(data.FIELDS)
        for row in rows:
            exposures[row[1]] = exposures.get(row[1], 0) + 1
            for index, value in enumerate(_canonical_values(row, duration_edges)):
                observed[index].add(value)
                if value not in train_vocab[index]:
                    unknown[index] += 1
        feature_fields = {}
        for index, field in enumerate(data.FIELDS):
            feature_fields[field] = {
                "cardinality": len(observed[index]),
                "train_cardinality": len(train_vocab[index]),
                "unknown_rate": (unknown[index] / len(rows)) if rows else 0.0,
            }
        dates = [row[0] for row in rows]
        split_profiles[split] = {
            "rows": len(rows),
            "users": len(users),
            "videos": len(videos),
            "positive_rate": (sum(row[6] for row in rows) / len(rows)) if rows else 0.0,
            "date_range": [min(dates), max(dates)] if dates else None,
            "duration_ms_percentiles": _percentile_map([row[5] for row in rows]),
            "exposures_per_user_percentiles": _percentile_map(list(exposures.values())),
            "fields": feature_fields,
        }
    headers = DataAccess(data_dir, "valid").columns()
    return {
        "splits": split_profiles,
        "available_csv_headers": {name: list(columns) for name, columns in headers.items()},
        "fingerprint_sha256": _fingerprint(splits),
    }


def _evaluation_view(splits, evaluation_split):
    if evaluation_split == "screen":
        train = [
            row for row in splits["train"]
            if _SCREEN_TRAIN_RANGE[0] <= row[0] <= _SCREEN_TRAIN_RANGE[1]
        ]
        target = [
            row for row in splits["train"]
            if _SCREEN_VALID_RANGE[0] <= row[0] <= _SCREEN_VALID_RANGE[1]
        ]
        if not train or not target:
            raise ValueError("Internal screen split is empty")
        return {
            "train": train,
            "valid": [row[:6] + (0,) for row in target],
            "test": [row[:6] + (0,) for row in splits["test"]],
        }, "valid", target
    if evaluation_split not in ("valid", "test"):
        raise ValueError(f"Unknown evaluation split: {evaluation_split!r}")
    target_key = evaluation_split
    target = splits[target_key]
    return {
        "train": splits["train"],
        "valid": [row[:6] + (0,) for row in splits["valid"]],
        "test": [row[:6] + (0,) for row in splits["test"]],
    }, target_key, target


def _candidate_module(name):
    if not isinstance(name, str) or not _CANDIDATE_RE.fullmatch(name):
        raise ValueError("candidate_module must be 'solution' or match 'solution_*'")
    module = importlib.import_module(name)
    module_path = Path(module.__file__).resolve()
    if module_path.parent != Path(__file__).resolve().parent or module_path.name != name + ".py":
        raise ValueError(f"Candidate module is outside the candidate worktree: {module_path}")
    score = getattr(module, "score", None)
    if not callable(score):
        raise ValueError(f"Candidate module {name!r} has no callable score")
    return score


def _restrict_candidate_filesystem(read_paths, write_path):
    """Apply a fail-closed Landlock policy before importing candidate code."""
    create_ruleset = 444
    add_rule = 445
    restrict_self = 446
    create_version = 1
    rule_path_beneath = 1
    access_execute = 1 << 0
    access_write_file = 1 << 1
    access_read_file = 1 << 2
    access_read_dir = 1 << 3
    access_remove_dir = 1 << 4
    access_remove_file = 1 << 5
    access_make_char = 1 << 6
    access_make_dir = 1 << 7
    access_make_reg = 1 << 8
    access_make_sock = 1 << 9
    access_make_fifo = 1 << 10
    access_make_block = 1 << 11
    access_make_sym = 1 << 12
    access_refer = 1 << 13
    access_truncate = 1 << 14
    base_access = (access_execute | access_write_file | access_read_file | access_read_dir
                   | access_remove_dir | access_remove_file | access_make_char | access_make_dir
                   | access_make_reg | access_make_sock | access_make_fifo | access_make_block
                   | access_make_sym)

    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneathAttr(ctypes.Structure):
        _fields_ = [
            ("allowed_access", ctypes.c_uint64),
            ("parent_fd", ctypes.c_int32),
            ("reserved", ctypes.c_uint32),
        ]

    libc = ctypes.CDLL(None, use_errno=True)

    def syscall(number, *arguments):
        result = libc.syscall(number, *arguments)
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return result

    abi = syscall(create_ruleset, 0, 0, create_version)
    if abi < 3:
        raise RuntimeError(f"Landlock ABI {abi} lacks required truncate protection")
    handled_access = base_access | access_refer | access_truncate
    ruleset = RulesetAttr(handled_access)
    ruleset_fd = syscall(create_ruleset, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    read_access = access_execute | access_read_file | access_read_dir
    write_access = handled_access

    def add_path(path, allowed_access):
        path = Path(path)
        if not path.exists():
            return
        descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            access = allowed_access
            if not path.is_dir():
                access &= ~access_read_dir
            rule = PathBeneathAttr(access, descriptor, 0)
            syscall(add_rule, ruleset_fd, rule_path_beneath, ctypes.byref(rule), 0)
        finally:
            os.close(descriptor)

    try:
        for path in read_paths:
            add_path(path, read_access)
        add_path(write_path, write_access)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        syscall(restrict_self, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)


def _candidate_worker(candidate_module, candidate_data_dir, evaluation_split, seed, config,
                      score_path):
    candidate_data_dir = Path(candidate_data_dir).resolve(strict=True)
    score_path = Path(score_path).resolve()
    score_path.parent.mkdir(parents=True, exist_ok=True)
    worktree = Path(__file__).resolve().parent
    runtime_paths = {
        worktree, candidate_data_dir, Path(sys.prefix), Path(sys.base_prefix),
        Path("/usr"), Path("/lib"), Path("/lib64"), Path("/etc/ld.so.cache"),
        Path("/dev/null"), Path("/dev/urandom"), Path("/proc/cpuinfo"),
        Path("/sys/devices/system/cpu"),
    }
    _restrict_candidate_filesystem(sorted(runtime_paths, key=str), score_path.parent)
    safe_splits = data.load(candidate_data_dir)
    candidate_splits, target_split, expected_rows = _evaluation_view(
        safe_splits, evaluation_split)
    scorer = _candidate_module(candidate_module)
    scores = np.asarray(scorer(
        candidate_splits,
        DataAccess(candidate_data_dir, target_split, evaluation_split),
        target_split,
        seed,
        config,
    ))
    if scores.ndim != 1:
        raise ValueError(f"Candidate scores must be one-dimensional; got shape {scores.shape}")
    if len(scores) != len(expected_rows):
        raise ValueError(f"Candidate returned {len(scores)} scores; expected {len(expected_rows)}")
    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError(f"Candidate scores must be numeric; got dtype {scores.dtype}")
    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores contain NaN or Inf")
    np.save(score_path, scores, allow_pickle=False)


def _run_isolated_worker(candidate_module, candidate_data_dir, evaluation_split, seed, config,
                         score_path):
    unshare = shutil.which("unshare")
    if unshare is None:
        raise RuntimeError("Candidate isolation requires the unshare executable")
    command = [
        unshare,
        "--user", "--map-root-user", "--net", "--pid", "--fork", "--kill-child",
        sys.executable, str(Path(__file__).resolve()), "_worker",
        "--candidate-module", candidate_module,
        "--candidate-data-dir", str(Path(candidate_data_dir).resolve()),
        "--evaluation-split", evaluation_split,
        "--seed", str(seed),
        "--config-json", json.dumps(config, separators=(",", ":"), sort_keys=True),
        "--score-path", str(Path(score_path).resolve()),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parent, env=environment,
                               stdin=subprocess.DEVNULL, check=False)
    if completed.returncode:
        raise RuntimeError(f"Isolated candidate worker exited with status {completed.returncode}")


def run_candidate(candidate_module, candidate_commit, data_dir, evaluation_split, seed, config,
                  output_dir, candidate_data_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = make_result(candidate_commit, evaluation_split, seed, config)
    temporary_data = None
    worker_dir = output_dir / f".worker-{os.getpid()}"
    try:
        if evaluation_split not in ("screen", "valid", "test"):
            raise ValueError("evaluation_split must be 'screen', 'valid', or 'test'")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        splits = data.load(data_dir)
        _, _, expected_rows = _evaluation_view(splits, evaluation_split)
        if candidate_data_dir is None:
            temporary_data = tempfile.TemporaryDirectory(prefix="candidate-data-")
            views = prepare_candidate_data(
                data_dir, Path(temporary_data.name) / "views", _fingerprint(splits))
            candidate_data_dir = views[evaluation_split]
        candidate_data_dir = Path(candidate_data_dir).resolve(strict=True)
        if worker_dir.exists():
            shutil.rmtree(worker_dir)
        worker_dir.mkdir()
        worker_score_path = worker_dir / "scores.npy"
        _run_isolated_worker(
            candidate_module, candidate_data_dir, evaluation_split, seed, config,
            worker_score_path)
        scores = np.load(worker_score_path, allow_pickle=False)
        if scores.ndim != 1:
            raise ValueError(f"Candidate scores must be one-dimensional; got shape {scores.shape}")
        if len(scores) != len(expected_rows):
            raise ValueError(f"Candidate returned {len(scores)} scores; expected {len(expected_rows)}")
        if not np.issubdtype(scores.dtype, np.number):
            raise ValueError(f"Candidate scores must be numeric; got dtype {scores.dtype}")
        if not np.isfinite(scores).all():
            raise ValueError("Candidate scores contain NaN or Inf")
        score_path = output_dir / "scores.npy"
        np.save(score_path, scores, allow_pickle=False)
        score_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
        metrics = None
        if evaluation_split != "test":
            metrics = evaluate(
                [row[1] for row in expected_rows], [row[6] for row in expected_rows], scores)
        result = make_result(
            candidate_commit, evaluation_split, seed, config, status="ok", metrics=metrics,
            rows=len(scores), runtime_seconds=time.monotonic() - started,
            score_sha256=score_hash, error=None)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        result = make_result(
            candidate_commit, evaluation_split, seed, config,
            runtime_seconds=time.monotonic() - started,
            error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        if worker_dir.exists():
            shutil.rmtree(worker_dir)
        if temporary_data is not None:
            temporary_data.cleanup()
    write_result(output_dir / "result.json", result)
    return result


def normalize_within_user(user_ids, scores):
    """Map each seed to stable within-user percentile ranks before ensembling."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(user_ids):
        raise ValueError("Scores must be one-dimensional and row-aligned with user IDs")
    groups = {}
    for index, user_id in enumerate(user_ids):
        groups.setdefault(user_id, []).append(index)
    normalized = np.empty(len(values), dtype=np.float64)
    for indices in groups.values():
        if len(indices) == 1:
            normalized[indices[0]] = 0.5
            continue
        local = values[indices]
        order = np.argsort(local, kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and local[order[end]] == local[order[start]]:
                end += 1
            average_rank = (start + end - 1) / 2
            ranks[order[start:end]] = average_rank / (len(order) - 1)
            start = end
        normalized[np.asarray(indices)] = ranks
    return normalized


def create_submission(data_dir, score_paths, output_path):
    if len(score_paths) != 3:
        raise ValueError("Exactly three score arrays are required (seeds 0, 1, and 2)")
    splits = data.load(data_dir)
    rows = splits["test"]
    user_ids = [row[1] for row in rows]
    arrays = []
    for path in score_paths:
        scores = np.load(path, allow_pickle=False)
        if scores.ndim != 1 or len(scores) != len(rows):
            raise ValueError(f"Score array {path} is not row-aligned with {len(rows)} test rows")
        if not np.isfinite(scores).all():
            raise ValueError(f"Score array {path} contains NaN or Inf")
        arrays.append(normalize_within_user(user_ids, scores))
    ensemble = np.mean(np.stack(arrays), axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    try:
        write_submission(temporary, rows, ensemble)
        checked = read_submission(temporary, rows)
        if len(checked) != len(rows) or not np.isfinite(np.asarray(checked)).all():
            raise ValueError("Submission round-trip validation failed")
        os.replace(temporary, output_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return len(rows)


def _json_object(value):
    def reject_constant(constant):
        raise ValueError(f"Non-finite JSON constant {constant!r} is forbidden")

    try:
        parsed = json.loads(value, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("config must be a JSON object")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--data_dir", required=True)
    profile_parser.add_argument("--output")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--candidate-module", default="solution")
    run_parser.add_argument("--candidate-commit", required=True)
    run_parser.add_argument("--data_dir", required=True)
    run_parser.add_argument(
        "--evaluation-split", required=True, choices=("screen", "valid", "test"))
    run_parser.add_argument("--candidate-data-dir", required=True)
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--config-json", default="{}", type=_json_object)
    run_parser.add_argument("--output-dir", required=True)

    worker_parser = subparsers.add_parser("_worker")
    worker_parser.add_argument("--candidate-module", required=True)
    worker_parser.add_argument("--candidate-data-dir", required=True)
    worker_parser.add_argument(
        "--evaluation-split", required=True, choices=("screen", "valid", "test"))
    worker_parser.add_argument("--seed", required=True, type=int)
    worker_parser.add_argument("--config-json", required=True, type=_json_object)
    worker_parser.add_argument("--score-path", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--data_dir", required=True)
    submit_parser.add_argument("--scores", nargs=3, required=True)
    submit_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "profile":
        profile = profile_data(args.data_dir)
        if args.output:
            _atomic_json(args.output, profile)
        else:
            json.dump(profile, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        return 0
    if args.command == "_worker":
        _candidate_worker(
            args.candidate_module, args.candidate_data_dir, args.evaluation_split,
            args.seed, args.config_json, args.score_path)
        return 0
    if args.command == "run":
        result = run_candidate(
            args.candidate_module, args.candidate_commit, args.data_dir,
            args.evaluation_split, args.seed, args.config_json, args.output_dir,
            candidate_data_dir=args.candidate_data_dir)
        return 0 if result["status"] == "ok" else 1
    rows = create_submission(args.data_dir, args.scores, args.output)
    print(f"Wrote and validated {args.output}: {rows:,d} rows (split=test, seeds=0,1,2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
