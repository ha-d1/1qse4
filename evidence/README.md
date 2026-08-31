# Track 2 Evidence Package

This directory is the compact, Git-tracked audit trail for the autonomous recommender-system
research agent. It contains no API key, hidden-test labels, or hidden-test metric. Raw local runs
remain ignored because they contain large generated patches and transient outputs.

## Contents

- `manifest.json`: benchmark and evidence-integrity summary.
- `resource_report.json`: aggregate Qwen usage, runtime, iterations, failures, and interventions.
- `experiment_index.json`: compact outcome and reflection for every autonomous iteration.
- `sustained_run.json`: the latest run containing at least three iterations in one process.
- `checkpoint_manifest.json`: SHA-256 checksums and metadata for the eight accepted models.
- `checkpoints/`: five base and three hour-aware four-negative BPR FM checkpoints.

## Reproduce the validation submission

From the repository root, after downloading KuaiRand-Pure:

```bash
.venv/bin/python make_submission.py \
  --checkpoint evidence/checkpoints/seed0.npz \
  --checkpoint evidence/checkpoints/seed1.npz \
  --checkpoint evidence/checkpoints/seed2.npz \
  --checkpoint evidence/checkpoints/seed3.npz \
  --checkpoint evidence/checkpoints/seed4.npz \
  --checkpoint evidence/checkpoints/hour_seed0.npz \
  --checkpoint evidence/checkpoints/hour_seed1.npz \
  --checkpoint evidence/checkpoints/hour_seed2.npz \
  --data-dir /path/to/KuaiRand-Pure/data \
  --split valid \
  --output valid_submission.csv
.venv/bin/python submit.py --check --split valid \
  --data_dir /path/to/KuaiRand-Pure/data valid_submission.csv
```

Run `python build_evidence_package.py` after new experiments to refresh this package. Run
`python verify_evidence_package.py` to verify checkpoint hashes, reflection coverage, and the
sustained-run requirement. Use `--split test` only once for the final unlabelled prediction.
