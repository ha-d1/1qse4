# TechJam Track 2 Research Run Report

## Selected recommender

- Model: nine-checkpoint rank ensemble of five base, three hour-aware, and one session-aware four-negative BPR FMs
- Validation primary: 0.605132461
- Best single-seed validation primary: 0.604691982
- Matched three-seed validation mean: 0.604618271
- All nine checkpoints present: True

## Autonomous agent usage

- SoCLaaS planning model: qwen3.6:35b
- SoCLaaS coding model: qwen3-coder-next
- Runs / iterations: 39 / 53
- Proposal attempts / failed attempts: 62 / 52
- Accepted autonomous iterations: 0
- LLM calls: 118
- Prompt / response / total tokens: 822275 / 246769 / 1069044
- Autonomous wall-clock seconds: 4176.815

## Other resources and safety

- Logged manual experiment records: 14
- Logged manual interventions: 4
- Manual training runtime seconds: 2456.513
- GPU hours: 0.000
- Development labels used: train and validation only
- Hidden-test evaluation performed: no
- `evaluate.py` modified: no
