# Architecture

## Shared modular pipeline

The v5 and v6 packages use this sequence:

```text
entity text -> frozen text encoder -> node vectors -> typed GNN
            -> graph patches -> online/EMA patch encoders
            -> JEPA loss + edge-revision/ranking losses
```

`fawkes_core` contains the stable schema, JSON adapters, SapBERT/BGE encoders,
patch construction, GNN/patch-transformer primitives, schema-aware revision,
and shared scoring/training helpers. It is a neutral implementation module, not
another model version.

## Raw v5 without notes

- Node input: 768-dimensional SapBERT vector.
- GNN: two GINE layers, hidden/latent dimension 160.
- Patches: eight balanced BFS patches with positional features.
- JEPA: one context patch predicts four target patches through online and EMA
  target encoders, with VICReg variance/covariance regularization.
- Fine-tuning: schema-aware edge revision plus hard candidate ranking.
- Checkpoint input width: 768.

## v6 with entity-grounded notes

v6 preserves the v5 graph architecture and appends a 768-dimensional
Clinical-ModernBERT note vector to selected entity nodes:

```text
SapBERT entity vector 768 + localized note vector 768 = 1536
```

With `note_ground_by=prov`, both endpoints of an edge whose labels contain
`prov_in_note` receive the note vector. Other nodes receive zeros. v6 also uses
LLM-confidence supervision and clinical-artifact filters during fine-tuning.

## Paper entity-note v16

Paper-v16 is a separate single-file lineage:

- Node identity: learned MD5-hash bucket embedding.
- Node type: learned type embedding.
- Numeric input: six demographic slots plus an optional 768-dimensional note.
- GNN: two four-head `TransformerConv` layers, hidden dimension 128.
- Pretraining: randomly masked nodes predicted against an EMA target encoder.
- Readout: frozen encoder plus DistMult/InfoNCE edge recovery.
- Main evaluation: filtered leave-one-out edge recovery and oracle cascade.

Its `v16` label is an experiment revision number and must not be interpreted as
newer than modular v6.
