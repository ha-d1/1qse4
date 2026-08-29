# Repository Guidelines

## Project Overview

KuaiRand-Pure Starter Kit: a small Python/NumPy baseline for **within-user ranking** of already logged video exposures. It trains simple rankers, evaluates fixed offline metrics, and produces row-aligned competition submissions. This is a research/competition toolkit, not a deployable service.

## Architecture & Data Flow

The codebase is deliberately flat: root-level modules import one another directly (for example, `from data import load`), with direct CLI entry points rather than an application package.

`data.load(data_dir)` reads video metadata and interaction logs, preserves source row order, and partitions records by fixed dates. `data.encode(splits)` learns categorical vocabularies from the training split only, allocates per-field unknown IDs, and returns encoded feature matrices, labels, user IDs, and the total feature dimension. A baseline scores rows; `evaluate.evaluate(user_ids, labels, scores)` groups and ranks rows per user, then returns `GAUC`, `nDCG@5`, and `primary`.

Submission flow uses the same loader/order contract: `submit.py` writes or validates `row_id,user_id,video_id,score` rows against the loaded split. Do not use `(user_id, video_id)` as a key—duplicates exist. `row_id` is the required stable identity.

**Fixed boundaries:** do not modify `evaluate.py` or task/split/metric conventions. Rank only each user's logged exposures; do not retrieve from the complete catalogue.

## Key Directories

- `./` — all maintained Python modules and entry points; there is no `src/` package layout.
- `KuaiRand-Pure/data/` — downloaded local dataset, default CLI input path; ignored by Git.
- `.venv/`, `venv/`, `.python/` — local Python environments; ignored by Git.

## Development Commands

```bash
# Install the only declared dependency (Python 3.9+)
python3 -m pip install -r requirements.txt

# Download and extract the dataset in the repository root
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz

# Run the official NumPy baseline
python3 baseline.py --model fm

# Deterministic evaluation-harness sanity check
python3 baseline.py --model random

# Generate, validate, and score a submission
python3 submit.py --make --split test submission.csv
python3 submit.py --check --split test submission.csv
python3 submit.py --score --split valid submission.csv
```

All CLIs default to `--data_dir ./KuaiRand-Pure/data`; pass `--data_dir <path>` for another location. There are no configured build, lint, format, type-check, or test commands.

## Code Conventions & Common Patterns

- Keep modules directly importable from the repository root. Use `snake_case` for functions/variables and uppercase constants; `FM` is the existing model class.
- Use `argparse` for CLI interfaces. Preserve existing direct `python3 <entrypoint>.py` workflows rather than adding a command/task-runner layer.
- Data records in `data.py` are positional tuples: `(date, user_id, video_id, author_id, tab, duration_ms, label)`. Preserve this order or migrate every index-based consumer together.
- Keep encoding train-only: unseen category values must use the field-specific unknown IDs created by `data.encode()`.
- Model state is mutable NumPy arrays. `baseline.FM` owns parameters and optimizer moments; early stopping copies/restores array state. Avoid unnecessary array copies in training/scoring paths.
- This project has no async/concurrency or dependency-injection/state-management framework. Prefer explicit function arguments and local NumPy state over introducing framework patterns.
- Let loader/training failures surface normally. For user-controlled submission input, follow `submit.read_submission()` and raise precise `ValueError` diagnostics.
- `ablation_features.py` is an ad-hoc executable: it reads `sys.argv` and runs at import time. Do not import it as a library module.

## Important Files

- `data.py` — canonical dataset contract: date splits, row loading, train-only categorical encoding.
- `baseline.py` — primary CLI; popularity/random/FM baselines, NumPy factorization machine, training and early stopping.
- `evaluate.py` — fixed official metric implementation. `evaluate()` returns `GAUC`, `nDCG@5`, `primary`, user count, and row count.
- `submit.py` — canonical submission writer, checker, and local scorer; owns the strict CSV contract.
- `ablation_features.py` — standalone feature-ablation experiment, separate from the canonical load/encode path.
- `baseline_scores.json` — benchmark/reference scores and tolerance context.
- `requirements.txt` — only declared dependency: `numpy>=1.23`.
- `README.md` — authoritative setup, task constraints, baseline expectations, and research notes.
- `.gitignore` — excludes datasets, environments, `.env`, submissions, logs, and Python cache artifacts.

## Runtime/Tooling Preferences

Use Python 3.9+ with the pip-compatible `requirements.txt`; NumPy is the only base dependency. No lockfile, package manager selection, build system, task runner, CI configuration, formatter, linter, or type checker is configured.

Keep downloaded data, `.env`, virtual environments, generated submission CSVs, and logs local. Do not add heavy model dependencies by default. Optional external experiments may use other frameworks, but keep the default kit runnable with its declared NumPy-only dependency set.

## Testing & QA

No automated test suite or test framework is configured. Validate observable ranking/submission behavior with the repository's own tools and a local dataset:

- Run `python3 baseline.py --model random`; the documented test primary sanity value is approximately `0.475 ± 0.001`.
- Before any submission, run `python3 submit.py --check --split test submission.csv`. It validates the exact header, contiguous row IDs, row count/order, user/video alignment, numeric scores, and finite values.
- Use `python3 submit.py --score --split valid submission.csv` to validate and evaluate a validation submission.

Metric invariants worth preserving in targeted future tests: `primary = (GAUC + nDCG@5) / 2`; GAUC excludes users with a single label class and is positive-count weighted; nDCG@5 gives zero-positive users zero; loader row order controls submission identity. Use explicit seeds for reproducible baseline/ablation experiments.
