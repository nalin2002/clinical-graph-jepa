# v24 ACI evaluation artifact

This directory contains the evaluation-only v24 ACI-Bench results and the exact
checkpoint used to produce them.

- `fawkes_entity_note_v23_mean_sp42_s42.pt` is the v23 `mean` checkpoint.
- The v23 `mean` checkpoint is documented as the exact v22 global-mean pipeline:
  patch masking, MLP readout, frozen encoder, and 768-d Clinical ModernBERT
  note features.
- SHA-256: `b9de4d93e0f60f309def98d5f20b38fc1002e0102e244c8eef548b05401f0f01`
- The ACI graph data and Clinical ModernBERT embedding source are not included;
  they remain local evaluation inputs subject to their applicable data-use
  restrictions.

The HTML and JSON files report graph-only, global-note, and provenance-grounded
entity-note ablations over the same checkpoint and query population.
