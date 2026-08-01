# Agent D — Phase 3: `clinical_jepa.train` and `evaluate`

Wave 2. Runs concurrently with Agent E (`score.py`). Your scopes are disjoint.

## Read first

- `docs/RESTRUCTURE_PLAN.md` — §4.2, §5.2, §5.3, and Phase 3 in full.
- `baseline/README.md` — **especially the exact `v5_loo.json` / `v6_loo.json`
  invocation.** Your gate is reproducing those files, so you must reproduce the
  command that made them.
- `old_src/fawkes_core/training.py`, `old_src/graph_jepa_v{5,6}/training.py`,
  `old_src/graph_jepa_v{5,6}/{evaluate,pretrain,finetune}.py`.
- `src/clinical_jepa/` — Agent A's completed Phase 1+2 work. Read it before
  writing; it is what you build on.

## You own — the only files you may create or edit

```
src/clinical_jepa/train/{__init__,loop,pretrain,finetune}.py
src/clinical_jepa/evaluate.py
tests/test_clinical_jepa_train.py
```

## Do not touch

`old_src/` — the oracle for every gate. `src/clinical_jepa/{schema,config,
encoders,model,losses}.py` and `src/clinical_jepa/graph/` — Agent A's finished
work; if you believe one is wrong, report it, do not edit it.
`src/clinical_jepa/score.py` — **Agent E is writing it right now.**
`src/fawkes/`, `src/benchmarks/`, `tests/conftest.py`, `pyproject.toml`,
`README.md`, `models/`, `baseline/`.

## What Agent A already did that you need to know

**Symbols moved out of the model module into `losses.py`.** The old
`evaluate.py` and `training.py` imported several of these from `.model` or
`fawkes_core.model_base`; they are now in `clinical_jepa/losses.py`:

```
_confidence_supervision_masks   _schema_edge_masks   _allowed_target_type_indices
sanitized_graph_data   confidence_sanitized_graph_data   pretrain_sanitized_graph_data
```

**`PatientGraphDataset` is at `clinical_jepa/graph/tensors.py`.** Agent A deleted
the dead `fawkes_core/data.py` copy — there were four; the live one survives.

**`build_checkpoint_encoder` is in `clinical_jepa/encoders.py`**, not
`train/loop.py` where plan §4.2 maps it. This is a deliberate deviation: both
your code and Agent E's need it, and it does nothing but dispatch encoder
construction. Import it from `encoders`. Do not redefine it.

**`GraphJEPAv3.edge_loss` is gone.** Agent A dropped it as an active trap — it
delegated to `revision_loss`, which on the flattened class is v5's
schema+confidence version, not v3's plain BCE. If you find a caller, that is a
finding worth reporting, not a reason to resurrect it.

## Checkpoint paths — a trap

Plan §3 shows `models/clinical-jepa-no-note/`, but **Phase 7 does that rename and
it has not happened.** The directories today are `models/v5_without_note/` and
`models/v6_with_note/`, with sidecars `config_v5.json` / `config_v6.json`.

Plan §5.2 requires argument-parser defaults to point at files that exist. So
point them at the **current** names and leave a comment that Phase 7 updates
them. A default pointing at the post-Phase-7 layout would be broken on arrival.

Checkpoint **save** names do follow the new convention now — §5.3, derived from
`cfg.model.use_note_embeddings`, replacing the hardcoded per-package
`PRETRAIN_CHECKPOINT_NAME` / `FINAL_CHECKPOINT_NAME` constants that were a
duplication vector between v5 and v6:

| Stage | Variant | Filename |
| --- | --- | --- |
| pretrain | no note | `clinical_jepa_no_note_pretrain.pt` |
| final | no note | `clinical_jepa_no_note.pt` |
| pretrain | localized note | `clinical_jepa_note_pretrain.pt` |
| final | localized note | `clinical_jepa_note.pt` |

## One evaluator, not two

`graph_jepa_v5/evaluate.py` (347 lines) and `graph_jepa_v6/evaluate.py` (353) are
the same file with `s/v5/v6/`. Merge them into one `evaluate.py` whose behavior
branches on config, exactly as Agent A did for the model. Same for the three
`training.py` copies into `train/loop.py`, and the `pretrain.py` / `finetune.py`
pairs.

## Gate

The new evaluator reproduces `baseline/v5_loo.json` and `baseline/v6_loo.json`
**exactly** — every metric, every per-relation breakdown, every count. Not
"within tolerance." Plan Phase 3 is explicit about this, and the baselines were
verified byte-reproducible in Phase 0, so exactness is achievable.

Use the invocation recorded in `baseline/README.md` verbatim: `--data synthetic
--synthetic-graphs 256 --synthetic-min-nodes 8 --synthetic-max-nodes 28 --device
cpu --cap 40000 --candidate-mode schema --start-graph 0`, against
`models/v5_without_note/graph_jepa_v5.pt` and `models/v6_with_note/graph_jepa_v6.pt`.

Expected: v5 MRR=0.605 H@1=0.343, v6 MRR=0.621 H@1=0.366, both over 1,755 edges
from 256 graphs.

Prefer a differential assertion (old evaluator vs new, same process, same input)
over comparing floats parsed from JSON — both trees are importable, which is the
whole point of the parallel build.

## Stop and report — do not work around

- **The gate fails.** Never adjust an expected value, loosen to a tolerance, or
  regenerate `baseline/*.json`. A mismatch is information about your merge.
- **v5 and v6 evaluate/training differ somewhere the line counts do not predict.**
  Report it before merging past it.
- You need a symbol that is in neither `old_src` nor Agent A's modules.
- You need to write a file outside your ownership list.

## Do not commit

Leave your work in the tree. A human runs the gate and commits at the wave
barrier, once both Wave 2 agents are done.
