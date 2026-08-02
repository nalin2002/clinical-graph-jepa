# `fawkes` — the paper implementation

**This package is the implementation behind the paper**, despite the repository
being named `clinical-graph-jepa` and the sibling package being named
`clinical_jepa`. The name is the original author's. `clinical_jepa` is the
modular revision pipeline and is *not* the paper implementation. See
[`docs/LINEAGE.md`](../../docs/LINEAGE.md).

Paper references: [local PDF](../../paper/clinical_jepa.pdf),
[OpenReview](https://openreview.net/forum?id=HXsMPubPqE), and the
[paper-to-code map](../../docs/PAPER_CODE_MAP.md). For the beginner-oriented
walkthrough of hashed entities, demographic/note input, the GNN, JEPA, DistMult
and the evaluations, read the
[full model guide](../../models/fawkes-entity-note/README.md).

Was `paper_v16`. The v16 (260615) entity-grounded-note experiment: a JEPA world
model over per-admission clinical knowledge graphs, with the Clinical-ModernBERT
note vector localized onto the entity nodes the note actually grounds.

## Modules

| Module | Contents |
| --- | --- |
| `config.py` | `Config` — every experiment knob, and `from_env()` |
| `data.py` | vocabularies, `score_vec`, `load_full_dataset`, `to_data` |
| `model.py` | `Encoder`, `JEPA`, `DistMult`, `Scorer` |
| `steps.py` | the per-batch tensor work: `jepa_step`, `readout_step`, negative sampling |
| `train.py` | the experiment run: `set_seed`, data prep/split, the two training phases, reporting, `main` |
| `evaluate.py` | `evaluate`, `loo_evaluate`, `cascade_evaluate`, `eir_uplift_eval`, the shared ranking helpers, and the checkpoint CLI |

## Configuration

The original `trainer.py` read roughly thirty `os.environ` values at module
import and computed `NUMERIC_DIM` — a tensor shape — from them. Environment
variables therefore had to be set before Python imported the module, and two
configurations could not coexist in one process.

Every one of those globals is now a `Config` field. The environment variable
names, defaults and parsing rules are unchanged, so the documented invocations
still work exactly as before:

```bash
USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 PRUNE_NO_EVIDENCE=1 PUSH=0 \
  fawkes-train
```

> [!CAUTION]
> `fawkes-train` accepts **no command-line options** — the experiment is
> configured from the environment. `--help` prints usage and exits without
> training, and an unrecognized flag is an error; both are guarded by
> `build_arg_parser`. A **bare `fawkes-train` begins a full training run**, and
> `PUSH` defaults to `1`, so on completion it uploads the checkpoint to the
> Hugging Face repository named by `OUTPUT_REPO`. Set `PUSH=0` unless you intend
> to publish.

and so does building a config directly, which the env-global version could not do:

```python
from fawkes.config import Config
from fawkes.model import Encoder

with_note = Config()                    # the released v16 experiment
without_note = Config(use_note=False)   # both usable in one process
```

Importing this package reads no environment variables and has no side effects;
`tests/test_fawkes.py` asserts that with an AST walk.

## Evaluating a released checkpoint

```bash
USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 PRUNE_NO_EVIDENCE=1 \
  fawkes-eval \
    --checkpoint models/fawkes-entity-note/fawkes_trainer_jepa_entity_note_v16_260615.pt \
    --data data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl \
    --device cpu
```

`run()` still refuses to proceed when `use_note` / `ground_by` / `embed_dim` /
`use_scores` disagree with the checkpoint's saved config.

## What changed in the split, and what did not

No numerics, no defaults, no tensor operations, and no environment-variable
names changed. `tests/test_fawkes.py` gates every module against
the `baseline/*.json` files recorded from `paper_v16`
before the split — see `baseline/README.md` and `baseline/COVERAGE.md`. Two
deliberate exceptions:

- **New checkpoints are named `fawkes_entity_note.pt` / `fawkes_no_note.pt`**
  (plan §5.3), replacing the hardcoded `fawkes_trainer_jepa_entity_note_v16_260615.pt`.
  Existing checkpoint files keep their names and load unchanged.
- **`CUBLAS_WORKSPACE_CONFIG` is set in `set_seed`**, not at import. What matters
  is that it precedes the first cuBLAS handle; `set_seed` runs at the top of
  `main`, before any tensor work.

`NOTE` and `HAS_NOTE` remain in the vocabulary but are unused: v15 put the note
on a per-admission NOTE node, and v16 retired it in favour of localizing the same
vector onto grounded entity nodes.
