# `graph_jepa_v5` source map

- `config.py`: checkpoint-compatible model, training, and scoring dataclasses.
- `data.py`: raw graph normalization, confidence/artifact features, and PyG conversion.
- `model.py`: v5 confidence supervision and hidden-edge candidate ranking.
- `patches.py`: neutral shared-core patch API.
- `pretrain.py`: masked-patch JEPA pretraining CLI.
- `finetune.py`: revision and candidate-ranking fine-tuning CLI.
- `training.py`: v5 checkpoint I/O and epoch loop.
- `score.py`: edge annotation, candidate addition, and optional pruning.
- `evaluate.py`: native filtered leave-one-out evaluation.
- `evaluate_llm.py`: optional LLM-assisted review evaluation.
- `evaluate_loo_v12_jepa_llm.py`: compatibility evaluation against the paper-v12 workflow.

The package depends on `fawkes_core`, not on any earlier Graph-JEPA version.
