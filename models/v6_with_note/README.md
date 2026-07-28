# MIMIC Graph-JEPA v6 with entity-grounded notes

## Artifact

- Final checkpoint: `graph_jepa_v6.pt`
- Pretraining checkpoint: `graph_jepa_v6_pretrain.pt`
- Entity encoder: 768-dimensional SapBERT
- Note input: 768-dimensional Clinical-ModernBERT embedding
- Input width: 1536
- Grounding: endpoints of edges with `labels.prov_in_note`
- Training: 60 pretraining epochs plus 50 revision/ranking epochs

## Required data

Faithful evaluation and training require a JSONL with `note_embedding` and
provenance labels. The packaged raw JSONL has neither. Use the private embedded
dataset associated with this checkpoint.

## Evaluate

```bash
uv run python -m graph_jepa_v6.evaluate \
  --checkpoint models/v6_with_note/graph_jepa_v6.pt \
  --data jsonl \
  --jsonl-path /path/to/fawkes_training_graph_full_embedded_260615.jsonl \
  --candidate-mode schema \
  --device cpu \
  --output outputs/v6_loo.json
```

## Reproduce training

```bash
uv run python -m graph_jepa_v6.pretrain \
  --data jsonl \
  --jsonl-path /path/to/fawkes_training_graph_full_embedded_260615.jsonl \
  --encoder sapbert \
  --note-embedding-dim 768 \
  --note-ground-by prov \
  --epochs 60 \
  --lr 0.0008 \
  --batch_size 16 \
  --device cuda \
  --out outputs/v6_retrained

uv run python -m graph_jepa_v6.finetune \
  --checkpoint outputs/v6_retrained/graph_jepa_v6_pretrain.pt \
  --data jsonl \
  --jsonl-path /path/to/fawkes_training_graph_full_embedded_260615.jsonl \
  --epochs 50 \
  --llm-confidence-negatives \
  --clinical-artifact-filters \
  --llm-negative-weight 0.6 \
  --device cuda \
  --out outputs/v6_retrained
```
