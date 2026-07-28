# `paper_v16` source map

Paper references: [local PDF](../../paper/clinical_jepa.pdf),
[OpenReview](https://openreview.net/forum?id=HXsMPubPqE), and the
[paper-to-code map](../../docs/PAPER_CODE_MAP.md).

- `trainer.py`: the requested standalone
  `fawkes_trainer_jepa_entity_note_v16_wmatbooth_260723.py`, packaged as an
  importable module and extended with optional `DATA_PATH` local JSONL loading.
- `evaluate.py`: checkpoint-only local leave-one-out evaluator.

The standalone trainer intentionally retains its environment-driven experiment
configuration and architecture so the packaged checkpoint remains compatible.
