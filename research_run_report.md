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
- Runs / iterations: 33 / 39
- Proposal attempts / failed attempts: 51 / 47
- Accepted autonomous iterations: 0
- LLM calls: 89
- Prompt / response / total tokens: 647483 / 214537 / 862020
- Autonomous wall-clock seconds: 2548.900

## Other resources and safety

- Logged manual experiment records: 5
- Logged manual interventions: 4
- Manual training runtime seconds: 1117.268
- GPU hours: 0.000
- Development labels used: train and validation only
- Hidden-test evaluation performed: no
- `evaluate.py` modified: no
