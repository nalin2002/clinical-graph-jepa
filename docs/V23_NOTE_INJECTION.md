# Fawkes v23: discharge-note injection ablation

v23 isolates how the discharge note is pooled and injected into the graph. The
v22 baseline copies one global 768-dimensional Clinical ModernBERT mean onto
every provenance-grounded entity. v23 adds two modes while leaving the dataset,
grounding, patch masking, Graph Transformer, MLP readout, split and schedules
unchanged.

| Mode | Entity note representation | Role |
|---|---|---|
| `mean` | Stored v22 global mean | Exact v22 baseline |
| `uniform` | Token-count-weighted mean over local spans | Parameterized preprocessing control |
| `attention` | Entity-conditioned four-head attention over local spans | Experimental v23 arm |

Clinical ModernBERT remains frozen. Its contextual token states are averaged in
non-overlapping 32-token spans and stored once in a private safetensors sidecar.
The preprocessor fails if a note would be truncated and verifies that the
token-count-weighted span average agrees with the stored v22 mean embedding.

## Build the note-memory artifact

```bash
hf jobs uv run --detach --flavor a100-large --timeout 2h \
  --secrets HF_TOKEN \
  --env OUTPUT_REPO=nalin9/fawkes-training-note-memory-v23-260808 \
  scripts/build_v23_note_memory.py
```

## Smoke test

Push the source commit first and substitute its full SHA:

```bash
hf jobs uv run --detach --flavor t4-small --timeout 1h \
  --secrets HF_TOKEN \
  --env SOURCE_REF=<full-commit-sha> \
  --env NOTE_INJECTION=attention \
  --env JEPA_EPOCHS=2 --env READOUT_EPOCHS=2 --env LOO_CAP=200 \
  --env PUSH=1 --env OUTPUT_REPO=nalin9/fawkes-v23-attention-smoke \
  scripts/fawkes_v23_hf_job.py
```

## Paired experiment

The confirmatory run uses seeds 42--51 and the fixed v22 split seed 42:

```bash
SOURCE_REF=<full-commit-sha> ./submit-v23-note-attention.sh
```

This submits `mean`, `uniform` and `attention` on the same A10G flavor. The
`mean` arm is the exact v22 pooling path retrained on matching hardware, avoiding
a cross-GPU numerical confound. Aggregate all arms:

```bash
python print_v23_note_attention_table.py
```

The primary endpoint is paired leave-one-out MRR over the same 8,283 held-edge
queries. Report mean and sample standard deviation across seeds, paired 95%
confidence intervals, win counts, Hits@1/3/10, batch-mask AUC/AP/MRR and the four
inferred-relation breakdowns stored in each checkpoint.
