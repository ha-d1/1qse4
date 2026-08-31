# TechJam Track 2 Final Submission Checklist

The accepted recommender is a within-user rank ensemble of five Factorization Machines,
each trained with four independently sampled BPR negatives per positive and learning rate
`0.00025` (seeds 0–4). Its validation primary score is `0.604539156`; the best individual
seed scored `0.604003906`.

## Before the one final hidden-test prediction

- Run the complete unit suite and `verify_baseline.py`.
- Confirm `git diff -- evaluate.py` is empty.
- Confirm the five seed checkpoints named in `research_run_report.md` exist.
- Generate a validation prediction file with `make_submission.py` (the safe default).
- Check it with `submit.py --check --split valid valid_submission.csv`.
- Regenerate `research_run_report.md` and review token, runtime, iteration, GPU, and
  manual-intervention counts.
- Freeze the chosen code and checkpoint. Do not select another model using test results.

## Final test prediction — run once

Only after every check above passes, run `make_submission.py` with the explicit
`--split test` option. Then run `submit.py --check --split test submission.csv`.
This creates scores from unlabelled test rows; it does not evaluate hidden-test labels.

## Package contents

- Source code and dependency list, excluding `.env`, API keys, downloaded data, caches,
  and rejected checkpoints.
- The accepted checkpoint and final checked `submission.csv` if the platform requests them.
- The generated research run report and a concise method/ablation discussion.
- Reproducibility commands, environment details, and a statement that GPU usage was zero.
- An explicit description of Qwen's role: Qwen selected and implemented bounded candidate
  experiments through SoCLaaS; the harness preflighted, ran, measured, accepted/rolled back,
  and logged them. Qwen is the research agent, not the recommender being trained.

Do not modify `evaluate.py`, and do not put `SOCLAAS_API_KEY` in any submission artifact.
