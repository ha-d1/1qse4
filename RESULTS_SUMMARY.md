# Final Submission and Results Summary

## Required benchmark

Dataset: **KuaiRand-Pure**  
Task: within-user ranking of already logged exposures  
Label: `long_view`  
Selected source: `solution.py` at commit `91a5b07`  
Default config: `{}`  
Runtime LightGBM: `4.7.0`  
Runtime NumPy: `2.5.2`  
Runtime Python: `3.14.7`

### Validation-best results

| Model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM baseline | 0.667400 | 0.535700 | 0.601600 |
| Confirmed NumPy ensemble | 0.6688752613 | 0.5357521080 | 0.6023136847 |
| **Selected LightGBM residual ensemble** | **0.6685937039** | **0.5361749933** | **0.6023843486** |

The primary score is `mean(GAUC, nDCG@5)`.

Absolute improvement over the official baseline:

- GAUC: `+0.0011937039`
- nDCG@5: `+0.0004749933`
- Primary: `+0.0007843486`

Absolute improvement over the confirmed NumPy ensemble: `+0.0000706639` primary.

The hidden test labels are unavailable, so no test GAUC or nDCG is reported. The final test output is produced with the same scorer and is checked only for schema, row alignment, finite scores, and row count.

No KuaiRand-1k or KuaiRand-27k bonus benchmark was attempted.

## Final model output

Tracked artifact: [`final_submission.csv`](final_submission.csv)

- Split: `test`
- Rows: `170,588`
- Header and alignment: validated with `submit.py --check`
- File line count including header: `170,589`
- SHA-256: `5776d3f239f50bdf218dc84c20e53c2c0ee1884fd04974673095b510f8bf04b0`

The three final test score runs completed successfully with `170,588` rows each:

| Seed | Runtime seconds | Score-array SHA-256 |
|---:|---:|---|
| 0 | 357.1289 | `91f415a36247f7a0306cd25f0fc0edbaf68bcd3b77932b9f93830f5a559a8bb9` |
| 1 | 363.1939 | `c3401907258635761db2e458c1f1854141588224a4e80072d901d3cd5768321e` |
| 2 | 368.6806 | `7e3a5e26201c0863cce65dd0c1fc750607b8501b8ed4ff669631c314913a5248` |

## Reproduction

Install the declared dependencies and download the data:

```bash
python3 -m pip install -r requirements-agent.txt
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

Run the autonomous campaign or the direct candidate harness:

```bash
python3 agent.py run \
  --campaign kuairand-v1 \
  --data_dir ./KuaiRand-Pure/data \
  --model MODEL_NAME

python3 submit.py --check --split test final_submission.csv
```

For a new direct three-seed output, run `experiment.py run` for seeds `0`, `1`, and `2`, save each `scores.npy`, and combine them:

```bash
python3 experiment.py submit \
  --data_dir ./KuaiRand-Pure/data \
  --scores seed0/scores.npy seed1/scores.npy seed2/scores.npy \
  --output final_submission.csv
```

`experiment.py submit` applies within-user normalization before averaging and performs a round-trip submission check. `submit.py --check` then verifies the exact public schema and source-row alignment.

## Resources and autonomy

The archived autonomous campaign was `kuairand-recover-20260831-5`:

- Completed iterations: **3**, with **4 hypotheses attempted**, out of the 50-iteration competition cap.
- Model calls: **9**.
- Wall-clock: **1,500.266 seconds** (`25m 0.3s`).
- GPU-hours: **0**; CPU-only execution.
- Controller limits: 12 iterations, 60 model calls, 4-hour wall-clock, 8 GB memory, 4 worker threads.
- Manual code/hypothesis interventions inside the autonomous loop: **0**.
- LLM token total: **not recorded**. The Starter Kit state persists model-call count but not input/output token usage, so no token number is fabricated.

## Limitations

- Test labels are hidden; the validation-best number is the evidence-based selection criterion.
- The ranker scores only logged exposures and does not solve catalogue retrieval, exploration, or online serving.
- Aggregate context/history and temporal selection are implemented as opt-in experiments. Their combined validation run scored `0.5954081035`, so the default remains the confirmed LightGBM residual ensemble.
- XGBoost `rank:ndcg` was benchmarked but scored only `0.5679815` primary on validation.
- The final artifact depends on the exact dataset row order and the required `row_id` contract.
