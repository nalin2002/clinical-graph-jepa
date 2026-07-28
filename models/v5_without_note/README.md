# Raw Graph-JEPA v5 without notes

## Artifact

- Final checkpoint: `graph_jepa_v5.pt`
- Pretraining checkpoint: `graph_jepa_v5_pretrain.pt`
- Frozen text encoder: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- Input width: 768
- Architecture: two-layer GINE, 160-dimensional node/patch latents, eight patches
- Training: 60 pretraining epochs plus 90 candidate-ranking/revision epochs

This checkpoint does not consume note embeddings. It is the model compatible
with the packaged raw JSONL.

## Evaluate

From the suite root:

```bash
uv run python -m graph_jepa_v5.evaluate \
  --checkpoint models/v5_without_note/graph_jepa_v5.pt \
  --data jsonl \
  --jsonl-path data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl \
  --candidate-mode schema \
  --device cpu \
  --output outputs/v5_loo.json
```

## Reproduce training

```bash
uv run python -m graph_jepa_v5.pretrain \
  --data jsonl \
  --jsonl-path data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl \
  --encoder sapbert \
  --epochs 60 \
  --lr 0.0008 \
  --batch_size 16 \
  --device cpu \
  --out outputs/v5_retrained

uv run python -m graph_jepa_v5.finetune \
  --checkpoint outputs/v5_retrained/graph_jepa_v5_pretrain.pt \
  --data jsonl \
  --jsonl-path data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl \
  --epochs 90 \
  --llm-confidence-negatives \
  --clinical-artifact-filters \
  --llm-negative-weight 0.6 \
  --device cpu \
  --out outputs/v5_retrained
```

Use a CUDA device for practical full training. Exact checkpoint reproduction
also requires the original split/order and software/hardware determinism.
