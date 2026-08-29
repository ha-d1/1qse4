# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** torch, pandas, and sklearn are not required.

## Data

Download from https://kuairand.com (direct Zenodo link; no registration required):

```bash
# Run in the Starter Kit directory; extraction produces ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly when the data is elsewhere.

`--model` can be `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, for checking the evaluation code).
FM takes about 40 seconds in total (single-threaded CPU).

## Agentic Experiment Loop

The optional controller snapshots the current checkout into isolated Git worktrees, profiles the
dataset, reproduces three FM validation seeds, and then runs an inspect → experiment → reflect loop.
The default candidate boundary remains NumPy-only; preinstalled LightGBM is available to candidates
when installed from `requirements-agent.txt`. The controller never installs packages.

```bash
export AGENT_API_KEY=...
export AGENT_API_BASE=https://provider.example/v1
export AGENT_MODEL=model-name
python3 agent.py run --campaign kuairand-v1 --data_dir ./KuaiRand-Pure/data
python3 agent.py status --campaign kuairand-v1
python3 agent.py resume --campaign kuairand-v1
```

Campaign state, patches, structured run records, logs, and the checked final submission are written
under `.agent-runs/<campaign>/`. Candidate worktrees under `.agent-worktrees/` are disposable. The
main checkout is not modified. `resume` requires the original dataset fingerprint, API base, model,
and persisted limits.

## Task Definition (the conventions are fixed; do not change them)

| | |
|---|---|
| Task | **Within-user ranking** — rank only the exposures shown to each user in the evaluation split; do not retrieve from the full catalog |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = their average** |
| Data splits | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Users with zero positives | nDCG is recorded as 0.0 and included in the average; GAUC includes only users with `0 < positive count < exposure count`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity for binary labels) |

See `evaluate.py` for the implementation; all conventions are documented in its header comments.

## Baseline Ladder

Scores on the test set. **The FM row is the one to beat.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, for self-checking) | 0.4996 | 0.4511 | 0.4753 |
| item popularity（trivial） | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The true metric range: the nDCG@5 ceiling is 0.729, not 1.0

Among the 23,875 users in the test set:

| | Share | Impact on the metric |
|---|---|---|
| All-negative users (none of the user's exposures are `long_view`) | **27.1%** | nDCG is always **0** and no model can improve it; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Users with meaningful distinctions | **63.7%** | The effective sample for GAUC |

Therefore, even using the true labels as prediction scores (an oracle with perfect ranking) only achieves:

| | random | FM baseline | **oracle ceiling** | Range already captured by FM |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Use the oracle as the denominator when tracking progress.** It is misleading to see 0.5946 and assume that the model is still far from a perfect score of 1.0 — the baseline has already captured 30% of the usable range, and the remaining headroom is 0.27 rather than 0.41.

Across five random seeds, FM has a standard deviation of **0.0008**. Therefore, use **ε = 0.002 (≈2.5σ), N = 3** as the convergence criterion:
three consecutive validation iterations with a primary-score improvement of no more than 0.002 indicate convergence.

> Self-check: if running your evaluation code with `--model random` does not produce a primary score of approximately 0.475 (±0.001), the harness is broken. Fix it first.

## Submission Format

CSV with a header, one row corresponding to each row in the evaluation split:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Starts at 0 and increases consecutively; corresponds to the row order of `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, filter by date, and preserve file order) |
| `user_id` / `video_id` | Redundant fields used only for alignment checks |
| `score` | The score assigned to the row by your model; any real number is valid and only relative ordering matters; NaN / Inf are not allowed |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the evaluation split —
> 3.06% of test rows are duplicate pairs, with up to 12 occurrences. Therefore, it cannot be used as the primary key.

Generate and validate:

```bash
python3 submit.py --make  --split test  submission.csv    # Generate an example submission with the official FM baseline
python3 submit.py --check --split test  submission.csv    # Validate format and alignment
python3 submit.py --score --split valid submission.csv    # Validate and score (local valid split available)
```

`--check` rejects an incorrect header, wrong row count, gaps in `row_id`, misaligned `user_id`/`video_id` values, and non-numeric or NaN/Inf `score` values. **Run `--check` yourself before submitting.**

## Where to Start Making Changes

The ordering below is **experimentally tested**, not speculation. Dead ends already tested by the organizers are marked directly so you do not repeat them.

### Already Tested: These Two Approaches Produced No Gain

| Tested approach | Result |
|---|---|
| **Add static features** — connect all 13 CWM feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** with 5 fields; indistinguishable within noise, and even slightly lower |
| **Increase model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887; virtually unchanged |

Reason: the `user_id × video_id` interaction already captures most of the learnable signal. Coarse buckets such as `follow_user_num_range` are redundant beside `user_id`, and 1.14 million rows cannot support much greater capacity. **The bottleneck is not features or capacity.**

⚠️ Also note: **the first-order contribution of purely user-side features to the score is always 0.** Because ranking is performed within each user, any term that is constant within a user does not change the within-group ordering (experimentally, `item_pop × user bias` produces scores identical to plain `item_pop`). User-side features can only work through **interaction terms with item-side features**.

### Not Yet Explored: The Headroom Should Be Here

Ordered by our estimated likelihood (**the organizers have not tested these; they are for you to explore**):

1. **Change the loss function.** The current loss is pointwise logloss, but the metrics (GAUC / nDCG) are **ranking metrics**. Switch to pairwise (BPR) or listwise (softmax over each user's exposures) to align the objective with evaluation. This is the approach we consider most likely to work.
2. **User history sequences.** The current features **do not use behavioral sequences at all**. Each KuaiRand user has hundreds to thousands of interactions in train, leaving DIN / SIM-style interest modeling completely unexplored.
3. **Multi-objective learning.** The logs also contain `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, and `play_time_ms`, which can be used as auxiliary tasks for the primary `long_view` task.
4. **Model watch duration.** This is precisely CWM's contribution: it models watch duration as **censored regression** (when a video finishes, the true viewing duration is truncated, so a one-sided loss is used instead of squared error). This is a research-rich direction.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since experiments show that capacity is not the bottleneck, **prioritize this after items 1–4**.
6. **Time features and distribution shift.** `hourmin`, `date`, and the shift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` contains random-exposure logs (1.18 million rows). It can serve as an additional unbiased validation set to check whether the model is overfitting biased traffic.

## Using Your Own Model (Including CWM)

`evaluate.py` is completely decoupled from the model; it only requires three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the `user_id` for each row in the evaluation split
- `labels`: the row's `long_view` value (0/1)
- `scores`: the score assigned to the row by your model (any real number; only relative ordering matters)

Therefore, you can completely replace `baseline.py` with PyTorch, LightGBM, or CWM's [xDeepFM](https://github.com/hyz20/CWM),
as long as you ultimately pass `scores` to `evaluate()`. **The scoring conventions are defined exclusively by `evaluate.py`.**

> Note when using CWM: it depends on `torch==1.6.0` (a 2020-era version that may be difficult to install on newer GPUs),
> and its loss optimizes counterfactual watch time while the evaluation label is a reconstructed `long_view2`.
> It is research code for a watch-duration debiasing paper and can serve as **advanced reference material**, but it is not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation + all evaluation conventions. **Do not modify.** |
| `data.py` | Data loading, official splits, and feature encoding. Modify this when adding features. |
| `baseline.py` | Three baselines. FM is the one to beat. |
| `baseline_scores.json` | Officially published scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate submission files. |
| `ablation_features.py` | Feature ablation experiment that reproduces the “adding features produces no gain” numbers. |
| `solution.py` | Model-editable `score(...)` candidate entry point; initially reproduces the FM. |
| `experiment.py` | Trusted data profiler, candidate runner, score validator, and ensemble submission writer. |
| `agent.py` | Campaign CLI and inspect → experiment → reflect controller. |
| `agent_api.py` | Standard-library OpenAI-compatible chat-completions client. |
| `agent_sandbox.py` | Git worktree, patch-policy, resource-limit, and subprocess boundary. |
| `agent_state.py` | Locked atomic campaign state and artifact persistence. |
| `test_agent.py` | Deterministic temporary-data/repository integration and fault tests. |
