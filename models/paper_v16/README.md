# Paper entity-note v16

Read the [local paper PDF](../../paper/clinical_jepa.pdf), the
[OpenReview record](https://openreview.net/forum?id=HXsMPubPqE), and the
[paper-to-code implementation map](../../docs/PAPER_CODE_MAP.md).

## Artifact

- Checkpoint: `fawkes_trainer_jepa_entity_note_v16_260615.pt`
- Implementation: `src/paper_v16/trainer.py`
- Local checkpoint evaluator: `src/paper_v16/evaluate.py`
- Graph encoder: two-layer, four-head TransformerConv with 128-dimensional nodes
- Entity representation: learned MD5 hash bucket
- Note representation: 768-dimensional Clinical-ModernBERT vector localized by provenance
- Readout: frozen encoder plus DistMult/InfoNCE

## Evaluate

```bash
USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 PRUNE_NO_EVIDENCE=1 \
uv run python -m paper_v16.evaluate \
  --checkpoint models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt \
  --data /path/to/fawkes_training_graph_full_embedded_260615.jsonl \
  --device cpu \
  --output outputs/paper_v16_loo.json
```

## Retrain Option B with notes

```bash
DATA_PATH=/path/to/fawkes_training_graph_full_embedded_260615.jsonl \
USE_NOTE=1 GROUND_BY=prov PUSH=0 \
uv run python -m paper_v16.trainer
```

## Retrain Option A without notes on the packaged raw data

The raw data has no evidence-label vectors, so evidence pruning must be disabled:

```bash
DATA_PATH=data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl \
USE_NOTE=0 PRUNE_NO_EVIDENCE=0 PUSH=0 \
uv run python -m paper_v16.trainer
```

This is an executable Option-A experiment, but it is not an exact reproduction
of the checkpoint's original embedded/evidence-scored training dataset.
