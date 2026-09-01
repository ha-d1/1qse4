# KuaiRand-Pure Within-User Video Ranking

## Project overview

This project ranks the video exposures that KuaiRand users already received. It does not retrieve from the full catalogue. For every user, the model assigns a score to each logged exposure; the official evaluator then ranks those exposures within that user. The optimized label is `long_view`.

The solution combines a compact factorization machine with train-only behavioral residuals:

1. A NumPy FM learns user, video, author, tab, and duration-bucket interactions.
2. Smoothed train-field priors provide robust cold-start corrections.
3. Train-only watch-duration ratios model how strongly users consumed each video.
4. A within-user pairwise table update improves ordering of positive and negative exposures.
5. An optional LightGBM LambdaRank residual consumes the same categorical fields and optional numerical/context/history features.
6. Three seed predictions are normalized within user before averaging. This is deliberately identical to submission-time scoring.

The implementation also includes controlled experiments for train-only temporal FM hyperparameter selection, hard-negative pair sampling, and aggregate numerical/context/history features. These switches are off by default because the combined experiment was weaker than the measured winner; keeping them available makes the comparisons reproducible without changing the submitted default.

## How it addresses the problem

KuaiRand is a logged-exposure ranking problem, not catalogue retrieval. The pipeline preserves source row order and duplicate `(user_id, video_id)` exposures. `row_id` is treated as the stable submission identity. Categorical vocabularies are learned from the training split only, and unseen values receive field-specific unknown IDs.

Validation uses the official `evaluate.py` implementation. It reports GAUC and nDCG@5, with `primary = (GAUC + nDCG@5) / 2`. All model fitting and hyperparameter experiments use training data; validation labels are never passed into the candidate scorer. The optional temporal screen uses earlier training dates to select an FM learning rate and epoch count, then refits on the complete official training split.

## Development tools

- OpenAI Codex CLI for the bounded inspect → experiment → reflect coding loop.
- Linux/WSL2 terminal and standard Git workflow.
- Python 3.14.7 for the recorded campaign.
- NumPy-based local experiments and JSON/CSV run artifacts.
- No Colab or Jupyter notebook was required.

## APIs used

- `evaluate.evaluate(...)` and the repository's `submit.py` contract for local benchmark evaluation and submission validation.
- LightGBM's native `train` API with the `lambdarank` objective when LightGBM is installed.
- XGBoost's native `train` API with `rank:ndcg` for a benchmark comparison.
- OpenAI Codex CLI authentication was used for agent-assisted development. No runtime OpenAI inference API, Google Maps API, or other external serving API is used by the ranker. The solution performs no runtime network access or dependency installation.

## Libraries and frameworks

- Python standard library.
- NumPy (`requirements.txt`) for encoding-compatible FM training and numerical scoring.
- Optional LightGBM (`requirements-agent.txt`) for LambdaRank residual training.
- XGBoost 3.4.1 was installed locally for the ranking-objective benchmark, but its tested configuration was not promoted to the final scorer.
- No PyTorch, TensorFlow, scikit-learn, CatBoost, or Hugging Face model is required.

## Datasets and assets

- **KuaiRand-Pure**, downloaded from the official Zenodo archive linked in `README.md`.
- Standard exposure logs from 20220408–20220508, video metadata, and user metadata supplied by the dataset.
- No manually labelled data, scraped assets, or external media assets were added.

## Results

The confirmed three-seed validation ensemble is:

| Model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM baseline | 0.6674 | 0.5357 | 0.6016 |
| NumPy ensemble | 0.668875 | 0.535752 | 0.602314 |
| LightGBM residual ensemble | 0.668594 | 0.536175 | 0.602384 |
| XGBoost `rank:ndcg` benchmark | 0.618973 | 0.516990 | 0.567982 |

The selected LightGBM result is an absolute primary improvement of `0.000784` over the published official primary (`0.6016`). Relative to the repository's confirmed NumPy ensemble, it improves primary by `0.000071`.

## Reproduction

```bash
python3 -m pip install -r requirements-agent.txt
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
python3 baseline.py --model random
python3 agent.py run --campaign kuairand-v1 --data_dir ./KuaiRand-Pure/data --model MODEL_NAME
```

To reproduce a candidate directly, use `experiment.py run` for seeds 0, 1, and 2, then combine the resulting arrays with `experiment.py submit`. The exact commands and the final submission validation command are in `RESULTS_SUMMARY.md`.

## Limitations and future work

- The model is an offline logged-exposure ranker; it does not solve catalogue retrieval or online exploration.
- The strongest confirmed gain is small and validation-only. It has not been verified against hidden leaderboard labels.
- Aggregate history features are available but sequence-aware interest models such as DIN/SIM are not implemented.
- The temporal screen is intentionally conservative and opt-in because the combined screen/features/hard-negative experiment regressed validation.
- XGBoost's tested ranking objective underperformed LightGBM on the available validation split.
- More time would go to sequence modeling, calibrated multi-objective watch-duration training, richer author/video interactions, and multiple independent temporal validation windows.
