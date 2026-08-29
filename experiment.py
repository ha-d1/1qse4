"""Trusted profiling, candidate execution, and submission harness."""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

import data
from evaluate import evaluate
from submit import read_submission, write_submission


_CANDIDATE_RE = re.compile(r"solution(?:_[A-Za-z0-9_]+)?\Z")
_PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)


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


class DataAccess:
    """Read-only, column-limited access to direct dataset CSV files."""

    def __init__(self, data_dir, target_split):
        if target_split not in data.SPLITS:
            raise ValueError(f"Unknown target split: {target_split!r}")
        self.data_dir = Path(data_dir).resolve(strict=True)
        if not self.data_dir.is_dir():
            raise ValueError(f"Data directory is not a directory: {data_dir}")
        self.target_split = target_split
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

    def iter_rows(self, filename, columns, split=None):
        path = self._resolve(filename)
        if not isinstance(columns, tuple) or not all(isinstance(name, str) for name in columns):
            raise ValueError("columns must be a tuple of column names")
        unknown = sorted(set(columns) - set(self._headers[filename]))
        if unknown:
            raise ValueError(f"Unknown columns for {filename}: {unknown}")
        if data.LABEL in columns:
            raise ValueError(f"Label column {data.LABEL!r} is not available through DataAccess")
        if split is not None:
            if split not in ("train", self.target_split):
                raise ValueError(f"Split {split!r} is unavailable for target {self.target_split!r}")
            if "date" not in self._headers[filename]:
                raise ValueError(f"CSV file {filename!r} has no date column")
            lo, hi = data.SPLITS[split]
        else:
            lo = hi = None

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


def run_candidate(candidate_module, candidate_commit, data_dir, target_split, seed, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = make_result(candidate_commit, target_split, seed, config)
    try:
        if target_split not in ("valid", "test"):
            raise ValueError("target_split must be 'valid' or 'test'")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        splits = data.load(data_dir)
        expected_rows = splits[target_split]
        expected_count = len(expected_rows)
        expected_users = tuple(row[1] for row in expected_rows)
        expected_labels = tuple(row[6] for row in expected_rows)
        candidate_splits = splits
        if target_split == "test":
            candidate_splits = dict(splits)
            candidate_splits["test"] = [row[:6] + (0,) for row in expected_rows]
        scorer = _candidate_module(candidate_module)
        scores = np.asarray(scorer(
            candidate_splits, DataAccess(data_dir, target_split), target_split, seed, config))
        if scores.ndim != 1:
            raise ValueError(f"Candidate scores must be one-dimensional; got shape {scores.shape}")
        if len(scores) != expected_count:
            raise ValueError(f"Candidate returned {len(scores)} scores; expected {expected_count}")
        if not np.issubdtype(scores.dtype, np.number):
            raise ValueError(f"Candidate scores must be numeric; got dtype {scores.dtype}")
        if not np.isfinite(scores).all():
            raise ValueError("Candidate scores contain NaN or Inf")
        score_path = output_dir / "scores.npy"
        np.save(score_path, scores, allow_pickle=False)
        score_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
        metrics = None
        if target_split == "valid":
            metrics = evaluate(expected_users, expected_labels, scores)
        result = make_result(candidate_commit, target_split, seed, config, status="ok",
                             metrics=metrics, rows=len(scores),
                             runtime_seconds=time.monotonic() - started,
                             score_sha256=score_hash, error=None)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        result = make_result(candidate_commit, target_split, seed, config,
                             runtime_seconds=time.monotonic() - started,
                             error={"type": type(exc).__name__, "message": str(exc)})
    write_result(output_dir / "result.json", result)
    return result


def create_submission(data_dir, score_paths, output_path):
    if len(score_paths) != 3:
        raise ValueError("Exactly three score arrays are required (seeds 0, 1, and 2)")
    splits = data.load(data_dir)
    rows = splits["test"]
    arrays = []
    for path in score_paths:
        scores = np.load(path, allow_pickle=False)
        if scores.ndim != 1 or len(scores) != len(rows):
            raise ValueError(f"Score array {path} is not row-aligned with {len(rows)} test rows")
        if not np.isfinite(scores).all():
            raise ValueError(f"Score array {path} contains NaN or Inf")
        arrays.append(scores)
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
    parsed = json.loads(value)
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
    run_parser.add_argument("--target-split", required=True, choices=("valid", "test"))
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--config-json", default="{}", type=_json_object)
    run_parser.add_argument("--output-dir", required=True)

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
    if args.command == "run":
        result = run_candidate(args.candidate_module, args.candidate_commit, args.data_dir,
                               args.target_split, args.seed, args.config_json, args.output_dir)
        return 0 if result["status"] == "ok" else 1
    rows = create_submission(args.data_dir, args.scores, args.output)
    print(f"Wrote and validated {args.output}: {rows:,d} rows (split=test, seeds=0,1,2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
