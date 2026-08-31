# TechJam Track 2 Research Run Report

## Selected recommender

- Model: Five-seed within-user rank ensemble of four-negative BPR FMs
- Validation primary: 0.604539156
- Best single-seed validation primary: 0.604003906
- Matched three-seed validation mean: 0.603827814
- All five checkpoints present: True

## Autonomous agent usage

- SoCLaaS planning model: qwen3.6:35b
- SoCLaaS coding model: qwen3-coder-next
- Runs / iterations: 25 / 25
- Proposal attempts / failed attempts: 34 / 31
- Accepted autonomous iterations: 0
- LLM calls: 50
- Prompt / response / total tokens: 357596 / 125991 / 483587
- Autonomous wall-clock seconds: 1663.471

## Other resources and safety

- Logged manual experiment records: 5
- Logged manual interventions: 4
- Manual training runtime seconds: 1117.268
- GPU hours: 0.000
- Development labels used: train and validation only
- Hidden-test evaluation performed: no
- `evaluate.py` modified: no
