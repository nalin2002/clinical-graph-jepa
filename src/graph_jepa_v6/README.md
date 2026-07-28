# `graph_jepa_v6` source map

- `config.py`: v6 note, confidence, ranking, and scoring configuration.
- `data.py`: v5-compatible graph conversion plus localized note features.
- `model.py`: trusted/uncertain/weak LLM supervision and candidate ranking.
- `patches.py`: neutral shared-core patch API.
- `pretrain.py`: note-aware masked-patch JEPA pretraining CLI.
- `finetune.py`: note-aware graph revision and ranking fine-tuning CLI.
- `training.py`: v6 checkpoint I/O and combined loss training loop.
- `score.py`: v6 data conversion plus revision scoring and candidate generation.
- `evaluate.py`: native filtered leave-one-out evaluation.
- `evaluate_llm.py`: optional LLM-assisted review evaluation.
- `evaluate_loo_v12_jepa_llm.py`: paper-v12 compatibility evaluation.

The package depends on `fawkes_core`, not on any earlier Graph-JEPA version.
