# KuaiRand-Pure Run and Iteration Log

## Scope

This log covers the recorded autonomous campaign `kuairand-recover-20260831-5`. The campaign used the trusted Starter Kit harness, three validation seeds, and the fixed `evaluate.py` metrics. It started from the confirmed NumPy ensemble primary `0.6023136846579729` and kept the best source unless a candidate improved the confirmed three-seed ensemble.

Campaign evidence source: `.agent-runs/kuairand-recover-20260831-5/state.json` and its `iterations/` artifacts.

## Baseline entering the campaign

| GAUC | nDCG@5 | Primary | Rows | Users |
|---:|---:|---:|---:|---:|
| 0.6688752613 | 0.5357521080 | 0.6023136847 | 124,909 | 22,377 |

The official published FM reference is primary `0.6016`.

## Iteration 1 — hard-negative pairwise finetune

**Hypothesis.** The existing pairwise table update sampled mostly random within-user negatives. Selecting low-scoring positives and high-scoring negatives from the FM should focus updates on ranking errors.

**Configuration.** `pair_blend=0.8`, `pair_epochs=4`, `pair_lr=0.02`, `pair_max_pairs=8`, `pair_reg=0.0015`, plus hard-negative pool settings.

**Code diff.** `solution.py`: 23 insertions and 12 deletions, replacing random pair selection with score-directed hard pairs.

**Result.** Seed 0 completed without an execution error:

| GAUC | nDCG@5 | Primary |
|---:|---:|---:|
| 0.5771681902 | 0.5009508035 | 0.5390594969 |

**Decision/recovery.** Rejected. The candidate was far below the confirmed best. The reflection identified overfitting or excessive update strength and proposed a weaker schedule rather than retaining this candidate.

## Iteration 2 — decayed pairwise updates

**Hypothesis.** A larger early pairwise update followed by decay could learn reliable preferences while limiting late-epoch disruption.

**Configuration.** `pair_blend=0.8`, `pair_decay=0.6`, `pair_epochs=4`, `pair_lr=0.04`, `pair_max_pairs=6`, `pair_reg=0.0015`.

**Code diff.** `solution.py`: 6 insertions and 4 deletions in the recorded cumulative change for this step.

**Result.** All three seeds completed successfully. The normalized ensemble was:

| GAUC | nDCG@5 | Primary |
|---:|---:|---:|
| 0.6680126162 | 0.5363543156 | 0.6021834659 |

**Decision/recovery.** Rejected because it was `0.0001302187` below the confirmed best. No runtime or data error occurred; the next hypothesis reduced update strength and increased regularization.

## Iteration 3 — mixed semi-hard pairwise update

**Hypothesis.** A minority of semi-hard pairs might retain the generalization of random sampling while adding useful ranking-error information.

**Code diff and error.** The proposed patch did not apply cleanly to `solution.py`. The harness reported patch failures at source lines 246 and 253, supplied corrective feedback, and allowed three correction attempts. No candidate was executed and no metrics were recorded for this iteration.

**Decision/recovery.** Marked failed at the patch boundary. The controller continued to the next hypothesis instead of treating an invalid patch as a model result.

## Iteration 4 — regularized balanced pairwise update

**Hypothesis.** More pair coverage with a lower decaying learning rate and stronger regularization could stabilize sparse field-table corrections.

**Configuration.** `pair_blend=0.8`, `pair_epochs=4`, `pair_lr=0.018`, `pair_max_pairs=12`, `pair_reg=0.003`.

**Code diff.** `solution.py`: 13 insertions and 10 deletions in the recorded proposal.

**Result.** All three seeds completed successfully. The normalized ensemble was:

| GAUC | nDCG@5 | Primary |
|---:|---:|---:|
| 0.6681860005 | 0.5361404236 | 0.6021632121 |

**Decision/recovery.** Rejected because it was `0.0001504726` below the confirmed best. No execution error occurred.

## Follow-up library experiments

After the campaign, the implementation added controlled switches for temporal FM selection, LightGBM numerical/context/history features, and hard-negative pair sampling. A combined exploratory run completed without an execution error but produced:

| GAUC | nDCG@5 | Primary |
|---:|---:|---:|
| 0.6585703143 | 0.5322458928 | 0.5954081035 |

The combined setting was not selected. The switches remain available but default off so the confirmed winner is reproducible.

XGBoost 3.4.1 was benchmarked with native `rank:ndcg` on the five encoded fields:

| GAUC | nDCG@5 | Primary |
|---:|---:|---:|
| 0.6189733 | 0.5169896 | 0.5679815 |

It was not promoted.

## Autonomy and resources

- Completed candidate iterations: **3**, with **4 hypotheses attempted**, out of the competition's 50-iteration cap.
- Controller configuration: maximum 12 iterations, maximum 60 model calls, 4-hour wall-clock limit, 8 GB memory limit, 4 worker threads.
- Recorded completed iterations: **3**.
- Model calls: **9**.
- Recorded campaign wall-clock: **1,500.266 seconds** (`25m 0.3s`).
- GPU use: **0 GPU-hours**; experiments ran on CPU.
- Manual code/hypothesis interventions inside the autonomous loop: **0**. The controller handled candidate rejection and the invalid patch recovery automatically.
- LLM token consumption: **not recorded by the Starter Kit metadata**. The archived state records model-call count but not input/output token counts; no fabricated token total is reported.
