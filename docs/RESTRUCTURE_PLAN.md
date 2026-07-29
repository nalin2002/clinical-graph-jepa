# Repository restructuring plan

Status: **approved, not started**. This document is the specification to build
against. It records what the code does today, what it should become, and the
verification gate that must pass before each step is considered done.

Nothing in this plan changes model numerics. Every gate is an equality check
against the current implementation.

---

## 1. Decisions

These are settled. The rest of the document assumes them.

| # | Decision |
| --- | --- |
| 1 | Two implementation packages plus one comparison package. The paper implementation (today `paper_v16`) becomes **`fawkes`**, keeping its original author's name. The modular pipeline (today `fawkes_core` + `graph_jepa_v5` + `graph_jepa_v6`) becomes **`clinical_jepa`**. **`benchmarks`** holds the cross-lineage evaluators. |
| 2 | All three released `.pt` checkpoints are available locally, so every differential gate below is live. |
| 3 | `evaluate_loo_v12_jepa_llm.py` is retained — it drives the three-way comparison — and moves to `benchmarks/`. |
| 4 | Migration is a **parallel build**, not an in-place refactor. `src/` is renamed `old_src/` and the new tree is written into a fresh `src/`. `old_src/` is deleted only at the very end. |
| 5 | Existing checkpoint **files keep their current names**. New checkpoints written by the new code follow the new convention in §5.3. |

### 1.1 Known naming hazard, and how it is mitigated

The package named `clinical_jepa` is **not** the paper implementation. The
package named `fawkes` **is**. This inverts what the repository name
(`clinical-graph-jepa`) and the manuscript filename (`paper/clinical_jepa.pdf`)
suggest. The naming reflects authorship, which is deliberate, but it will
mislead a first-time reader unless it is stated loudly and repeatedly.

Required mitigations, all of which are deliverables of Phase 7:

- `src/fawkes/__init__.py` docstring opens with: *"The Clinical Graph-JEPA paper
  implementation. Compatible with the released paper checkpoint."*
- `src/clinical_jepa/__init__.py` docstring opens with: *"The modular
  patch-based Graph-JEPA revision pipeline. This is **not** the paper
  implementation — see the `fawkes` package."*
- `README.md` states the mapping in its first table, above the architecture
  comparison.
- `docs/LINEAGE.md` (new) is the single authority on old-name → new-name.
- `docs/PAPER_CODE_MAP.md` is rewritten; every current reference to
  `paper_v16` becomes `fawkes` and every reference to `graph_jepa_v5/v6`
  becomes `clinical_jepa`.

---

## 2. What the code does today

Findings from a full read of `src/` (13,240 lines). These motivate the target
structure; each one maps to a phase below.

### 2.1 `graph_jepa_v5` and `graph_jepa_v6` are the same package, twice

| File | v5 lines | v6 lines | differing lines |
| --- | ---: | ---: | ---: |
| `model.py` | 726 | 727 | **12** |
| `evaluate_llm.py` | 811 | 817 | 14 |
| `evaluate.py` | 347 | 353 | 18 |
| `training.py` | 311 | 317 | 24 |
| `evaluate_loo_v12_jepa_llm.py` | 945 | 951 | 68 |
| `finetune.py` | 276 | 280 | 20 |
| `score.py` | 122 | 127 | 25 |
| `pretrain.py` | 111 | 134 | 43 |
| `config.py` | 139 | 167 | 31 |
| `data.py` | 108 | 313 | 201 |
| `patches.py` | 3 | 3 | 2 |
| `__init__.py` | 20 | 19 | 15 |

Of 8,127 combined lines roughly 3,800 are byte-identical. `graph_jepa_v6/model.py`
ends with:

```python
GraphJEPAv5 = GraphJEPAv6
```

The two model classes are one class body. The entire substantive difference
between the models is four config fields — `base_in_dim`, `use_note_embeddings`,
`note_embedding_dim`, `note_ground_by` — and the note-append branch in
`data.py`. Everything else is `s/v5/v6/` applied to a copy.

`graph_jepa_v6/config.py` already loads v5 checkpoints: `from_dict` defaults
`use_note_embeddings` to `False` and derives `base_in_dim` when those keys are
absent. The backward-compatibility path for the merge is already written and
shipped.

### 2.2 `fawkes_core` is not version-neutral

It is the v3 and v4 lineages flattened into one directory.

| File | Actually contains |
| --- | --- |
| `model_base.py` | `GraphJEPAv3` — the base architecture |
| `revision.py` | `GraphJEPAv4(GraphJEPAv3)` — schema-aware revision losses |
| `score_base.py` | the v3 scoring CLI |
| `score_revision.py` | the v4 scoring CLI; re-exports 15 private functions from `score_base` by module-level assignment |
| `data_graph.py` | docstring: "PyG graph tensor conversion for Graph-JEPA v3" |
| `config.py` | docstring: "Configuration dataclasses for Graph-JEPA v4" |
| `training.py` | docstring: "Shared training helpers for Graph-JEPA v4 scripts" |

Real inheritance chain: `GraphJEPAv3 → GraphJEPAv4 → GraphJEPAv5/v6`. The
version numbers are **architecture layer names**, not repository versions, which
is why they cannot be removed by renaming alone. `GraphJEPAv3` and `GraphJEPAv4`
are never instantiated by any shipped path — no v3 or v4 checkpoints exist in
this repository. They survive only as base classes.

### 2.3 The two lineages are already tangled

`graph_jepa_v{5,6}/evaluate_loo_v12_jepa_llm.py` — 1,896 lines across the two
copies — defines `LooEncoder`, `LooDistMult`, `LooMLPScorer`, and its own
`LOO_NODE_TYPES` / `LOO_RELATION_CANONICAL` / `LOO_SCORE_FEATS` vocabularies,
built on `TransformerConv` with hashed entity buckets.

That is a third copy of the paper-lineage architecture, living inside the
modular packages. `LOO_NUMERIC_DIM = 6` where the paper trainer uses
`6 + EMBED_DIM`, which is consistent with it being the **no-note variant of the
same architecture**. Any clean split of the two pipelines has to resolve this
file first.

### 2.4 Cross-package monkeypatching

`graph_jepa_v6/score.py`:

```python
def _install_v6_data_conversion() -> None:
    _v3.to_graph_data = to_graph_data
    _v4.to_graph_data = to_graph_data
```

This mutates two `fawkes_core` modules globally at call time. It is import-order
dependent, invisible to anyone reading `fawkes_core`, and makes it impossible
for v5 and v6 scoring to coexist in one process.

### 2.5 The paper trainer configures itself from environment globals at import

`paper_v16/trainer.py` reads roughly 30 `os.environ` values at module import
time. `NUMERIC_DIM` — a tensor shape — is computed at import. `evaluate.py`
then compares `trainer.USE_NOTE` against the checkpoint's saved config.

Consequences: environment variables must be set before Python imports the
module (hence the env-prefixed commands in the README), and two configurations
cannot coexist in one process, which makes the trainer effectively untestable.

### 2.6 Endpoint-key normalization exists three times with different coverage

`edge["source"]` vs `edge["source_id"]` is handled in:

- `fawkes_core/data.py::_adapt_mimic_edge` (the JSONL builder path)
- `graph_jepa_v6/data.py::_graph_with_aliases` (v6 only)
- `fawkes_core/score_base.py::_looks_like_mimic_subkg` (the scoring path)

`fawkes_core/schema.py::PatientGraph.from_pipeline_json` copies edge dicts
verbatim and does not normalize. `graph_jepa_v5/data.py` indexes
`edge["source_id"]` unguarded. The result is that the v5 and v6 scoring paths
have different tolerance for the same input file. This must be resolved
deliberately in Phase 4, not silently during a move.

### 2.7 Smaller items

- **Undeclared dependencies.** `openai` and `python-dotenv` are imported by both
  `evaluate_llm.py` copies and appear nowhere in `pyproject.toml`.
  (`wandb` and `FlagEmbedding` are correctly declared as extras.)
- **Fossil guard, duplicated.** `scripts/smoke_check.py::check_independence` and
  `tests/test_suite.py::test_v5_v6_have_no_historical_package_imports` both
  AST-walk for imports of `graph_jepa_v2/v3/v4` — packages that do not exist in
  this repository. The guard enforces nothing.
- **Four names for one project.** Distribution `fawkes-three-model-suite`, repo
  `clinical-graph-jepa`, package `fawkes_core`, README title
  "Clinical Graph-JEPA".
- **No console entry points.** 18 `if __name__ == "__main__"` blocks, all invoked
  as `python -m ...`.
- **3 tests for 13,240 lines**, one of which cannot pass on a clean tree because
  `.gitignore` excludes `*.pt` and `data/**/*.jsonl`.

### 2.8 The one fact that makes this restructure safe

`fawkes_core/training.py::save_checkpoint` writes:

```python
torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, ckpt_path)
```

where `Config.to_dict` is `asdict(self)`. **Checkpoints contain no pickled
first-party classes** — only plain nested dicts and tensors. Renaming modules and
classes therefore cannot break checkpoint loading. `state_dict` keys derive from
attribute paths (`context_node_encoder.input_proj.weight`), not class names, so
they survive class renames but **not** attribute renames — which is why every
model phase below gates on `state_dict` key-set equality.

---

## 3. Target structure

```text
clinical-graph-jepa/
├── src/
│   ├── fawkes/                     THE PAPER IMPLEMENTATION (was paper_v16)
│   │   ├── __init__.py             docstring: "the paper implementation"
│   │   ├── config.py               env globals -> dataclass; env still honored
│   │   ├── data.py                 to_data, load_full_dataset, score_vec
│   │   ├── model.py                Encoder, JEPA, DistMult, Scorer
│   │   ├── train.py                jepa_step, readout_step, main
│   │   ├── evaluate.py             loo_evaluate, cascade_evaluate, eir_uplift_eval
│   │   └── README.md
│   │
│   ├── clinical_jepa/              THE MODULAR PIPELINE (was fawkes_core + v5 + v6)
│   │   ├── __init__.py             docstring: "NOT the paper implementation"
│   │   ├── schema.py
│   │   ├── config.py               merged v4 + v5 + v6 Config
│   │   ├── encoders.py
│   │   ├── graph/
│   │   │   ├── builders.py         Mimic/Jsonl/AciBench builders + THE alias normalizer
│   │   │   ├── tensors.py          one to_graph_data; note append behind a flag
│   │   │   └── patches.py
│   │   ├── model.py                flattened GraphJEPA
│   │   ├── losses.py               revision BCE + candidate ranking
│   │   ├── train/
│   │   │   ├── loop.py
│   │   │   ├── pretrain.py
│   │   │   └── finetune.py
│   │   ├── evaluate.py
│   │   ├── score.py                4 modules, 1,173 lines -> 1
│   │   └── README.md
│   │
│   └── benchmarks/                 CROSS-LINEAGE; the only importer of both
│       ├── __init__.py
│       ├── llm_ranker.py           ChatRanker + API client (currently duplicated)
│       ├── vs_llm.py               was evaluate_llm.py
│       └── vs_loo_baseline.py      was evaluate_loo_v12_jepa_llm.py (three-way)
│
├── old_src/                        the current tree, verbatim; deleted in Phase 8
│
├── baseline/                       NEW: pinned outputs = the regression oracle
│   ├── v5_loo.json  v6_loo.json  paper_loo.json
│   └── v5_keys.json v6_keys.json paper_keys.json
│
├── models/                         same shape, renamed dirs; *.pt still gitignored
│   ├── MANIFEST.json
│   ├── clinical-jepa-no-note/          {README.md, config.json, config_pretrain.json, *.pt}
│   ├── clinical-jepa-localized-note/
│   ├── fawkes-entity-note/
│   └── fawkes-loo-baseline/            benchmark input, not a released artifact
│
├── docs/
│   ├── ARCHITECTURE.md  DATA.md  EVALUATION.md  PAPER_CODE_MAP.md
│   ├── LINEAGE.md                  NEW: old version numbers -> new names
│   ├── RESTRUCTURE_PLAN.md         this file
│   └── assets/
│
├── scripts/  tests/  paper/  data/
└── pyproject.toml  README.md  CLAUDE.md  AGENTS.md  uv.lock
```

`models/<name>/` stays colocated — guide, config, and checkpoint in one
directory. For a released research artifact, "everything about checkpoint X in
one place" is worth more than splitting by file category, and it is far less
churn than moving configs and guides into separate trees.

---

## 4. Name mapping

This table is the source for `docs/LINEAGE.md`.

### 4.1 Packages

| Today | New | Note |
| --- | --- | --- |
| `src/paper_v16/` | `src/fawkes/` | The paper implementation. Original author's name retained. |
| `src/fawkes_core/` | `src/clinical_jepa/` | merged |
| `src/graph_jepa_v5/` | `src/clinical_jepa/` | merged — no-note variant |
| `src/graph_jepa_v6/` | `src/clinical_jepa/` | merged — localized-note variant |
| — | `src/benchmarks/` | new; cross-lineage comparison |

### 4.2 Modules

| Today | New |
| --- | --- |
| `fawkes_core/model_base.py` | `clinical_jepa/model.py` |
| `fawkes_core/revision.py` | `clinical_jepa/losses.py` |
| `graph_jepa_v{5,6}/model.py` | merged into `clinical_jepa/model.py` |
| `fawkes_core/data.py` | `clinical_jepa/graph/builders.py` |
| `fawkes_core/data_graph.py` | merged into `clinical_jepa/graph/tensors.py` |
| `graph_jepa_v{5,6}/data.py` | merged into `clinical_jepa/graph/tensors.py` |
| `fawkes_core/patches.py` | `clinical_jepa/graph/patches.py` |
| `graph_jepa_v{5,6}/patches.py` | deleted (3-line star re-export) |
| `fawkes_core/config.py` + `graph_jepa_v{5,6}/config.py` | `clinical_jepa/config.py` |
| `fawkes_core/training.py` + `graph_jepa_v{5,6}/training.py` | `clinical_jepa/train/loop.py` |
| `graph_jepa_v{5,6}/pretrain.py` | `clinical_jepa/train/pretrain.py` |
| `graph_jepa_v{5,6}/finetune.py` | `clinical_jepa/train/finetune.py` |
| `graph_jepa_v{5,6}/evaluate.py` | `clinical_jepa/evaluate.py` |
| `fawkes_core/score_base.py` + `score_revision.py` + `graph_jepa_v{5,6}/score.py` | `clinical_jepa/score.py` |
| `fawkes_core/encoders.py` | `clinical_jepa/encoders.py` |
| `fawkes_core/schema.py` | `clinical_jepa/schema.py` |
| `graph_jepa_v{5,6}/evaluate_llm.py` | `benchmarks/vs_llm.py` |
| `graph_jepa_v{5,6}/evaluate_loo_v12_jepa_llm.py` | `benchmarks/vs_loo_baseline.py` |
| — | `benchmarks/llm_ranker.py` (extracted from the two `evaluate_llm.py` copies) |
| `paper_v16/trainer.py` | split into `fawkes/{config,data,model,train,evaluate}.py` |
| `paper_v16/evaluate.py` | folded into `fawkes/evaluate.py` |

### 4.3 Symbols

| Today | New |
| --- | --- |
| `GraphJEPAv3` | `GraphJEPA` (`clinical_jepa/model.py`) |
| `GraphJEPAv4` | class dissolved; losses move to `clinical_jepa/losses.py` |
| `GraphJEPAv5`, `GraphJEPAv6` | both dissolved into `GraphJEPA` |
| `_v3`, `_v4` import aliases | deleted |
| `_install_v6_data_conversion` | deleted; converter passed as an argument |
| `LooEncoder`, `LooDistMult`, `LooMLPScorer` | stay in `benchmarks/vs_loo_baseline.py` (see §7.1 for optional consolidation) |

### 4.4 Model directories

| Today | New |
| --- | --- |
| `models/v5_without_note/` | `models/clinical-jepa-no-note/` |
| `models/v6_with_note/` | `models/clinical-jepa-localized-note/` |
| `models/paper_v16/` | `models/fawkes-entity-note/` |
| — | `models/fawkes-loo-baseline/` (benchmark input) |
| `config_v5.json`, `config_v6.json` | `config.json` (directory already scopes it) |
| `config_v5_pretrain.json`, `config_v6_pretrain.json` | `config_pretrain.json` |

---

## 5. Checkpoints

### 5.1 Existing files keep their names

`graph_jepa_v5.pt`, `graph_jepa_v5_pretrain.pt`, `graph_jepa_v6.pt`,
`graph_jepa_v6_pretrain.pt`, and `fawkes_trainer_jepa_entity_note_v16_260615.pt`
are unchanged. A checkpoint is a historical artifact and its filename is
legitimate provenance metadata. Keeping the names also means every `sha256` in
`models/MANIFEST.json` stays byte-stable, which makes the directory rename
provably content-neutral.

`MANIFEST.json` is updated for the new directory paths only. Checksums do not
change.

### 5.2 Loading

New code must load the existing filenames without special-casing. Checkpoint
paths are CLI arguments, so this is automatic — but the *default* values in
argument parsers change to the new directory layout, and the defaults must point
at files that exist.

### 5.3 Saving

New checkpoints written by the new code use these names. The two hardcoded
per-package constants (`PRETRAIN_CHECKPOINT_NAME`, `FINAL_CHECKPOINT_NAME`) were
a duplication vector between v5 and v6; they are replaced by a single
variant-derived name.

`clinical_jepa`, derived from `cfg.model.use_note_embeddings`:

| Stage | Variant | Filename |
| --- | --- | --- |
| pretrain | no note | `clinical_jepa_no_note_pretrain.pt` |
| final | no note | `clinical_jepa_no_note.pt` |
| pretrain | localized note | `clinical_jepa_note_pretrain.pt` |
| final | localized note | `clinical_jepa_note.pt` |

Config sidecars are `config_pretrain.json` and `config.json` in the same
directory.

`fawkes` — replaces the hardcoded `fawkes_trainer_jepa_entity_note_v16_260615.pt`
at `paper_v16/trainer.py:704`:

| Variant | Filename |
| --- | --- |
| `USE_NOTE=1` | `fawkes_entity_note.pt` |
| `USE_NOTE=0` | `fawkes_no_note.pt` |

The HF upload path (`OUTPUT_REPO`, `api.upload_file`) uses the same name. No
date stamp: git history and `MANIFEST.json` already record provenance, and a
date in the filename was what made the old name unreadable.

### 5.4 The fourth checkpoint

`benchmarks/vs_loo_baseline.py` needs `fawkes_jepa_loo_eval_v12_260615.pt`
(`DEFAULT_LOO_FILENAME`), fetched from HF via `--loo-repo-id` or supplied via
`--loo-checkpoint`. It is **not** in `models/MANIFEST.json` today.

Action, in Phase 6: place it at
`models/fawkes-loo-baseline/fawkes_jepa_loo_eval_v12_260615.pt`, add it to
`MANIFEST.json` under a `benchmark_inputs` key (kept separate from the released
model artifacts), and point the argument parser default there.

Without this file the three-way comparison cannot run and its gate cannot pass.

---

## 6. Migration

`src/` is renamed `old_src/` and the new tree is written fresh into `src/`.
Both stay importable for the duration of the migration.

### 6.1 Why this works cleanly here

Every package is being renamed, so **there are no module-name collisions**
between the old tree and the new one:

```
old_src/  fawkes_core   graph_jepa_v5   graph_jepa_v6   paper_v16
src/      clinical_jepa                                 fawkes  benchmarks
```

Both directories can sit on `sys.path` simultaneously. That turns every gate
from "compare against a JSON file recorded weeks ago" into **differential
testing**: run old and new in the same process on the same input and assert
equality. This is a materially stronger guarantee than a pinned baseline, and
it is available for free because of the rename.

`old_src/` is excluded from the built distribution — `[tool.setuptools.packages.find]`
already has `where = ["src"]` — so packaging is unaffected throughout.

### 6.2 Test harness

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "old_src"]   # old_src removed in Phase 8
addopts = "-q"
```

`pythonpath` is a native pytest ini option (pytest >= 7; this project requires
>= 8), so no new dependency and no `sys.path` manipulation in `conftest.py`.

`tests/conftest.py` provides the differential helpers and the skip guard:

```python
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models"
DATA = ROOT / "data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl"

requires_checkpoints = pytest.mark.skipif(
    not (CKPT / "clinical-jepa-no-note").glob("*.pt"),
    reason="released checkpoints not present in the working tree",
)


def assert_pyg_equal(a, b, path=""):
    """Assert two PyG Data objects are elementwise identical."""
    ka, kb = set(a.keys()), set(b.keys())
    assert ka == kb, f"{path}: key mismatch {ka ^ kb}"
    for key in sorted(ka):
        va, vb = a[key], b[key]
        if torch.is_tensor(va):
            assert torch.equal(va, vb), f"{path}.{key}: tensor mismatch"
        else:
            assert va == vb, f"{path}.{key}: {va!r} != {vb!r}"
```

Every gate below is expressed as a test using these helpers, committed with the
phase it gates. Tests that need checkpoints carry `@requires_checkpoints` so a
clean clone gets a **green suite with honest skips** rather than a hard failure.

---

## 7. Phases

Each phase ends with a gate. A phase is not done until its gate passes. If a
gate fails, the correct response is to find out why — never to adjust the
expected value.

### Phase 0 — Freeze the baseline and split the tree

1. Confirm the three `.pt` files and the dataset JSONL are in the working tree.
2. Record the baseline from the **current** code:
   ```
   python -m graph_jepa_v5.evaluate ... --output baseline/v5_loo.json
   python -m graph_jepa_v6.evaluate ... --output baseline/v6_loo.json
   USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 PRUNE_NO_EVIDENCE=1 \
     python -m paper_v16.evaluate ... --output baseline/paper_loo.json
   ```
3. Dump `sorted(model.state_dict().keys())` for each checkpoint to
   `baseline/{v5,v6,paper}_keys.json`.
4. Record the exact command line and environment used for each, in
   `baseline/README.md`. A metric without its invocation is not reproducible.
5. `git mv src old_src`; create an empty `src/`.
6. Add `pythonpath = ["src", "old_src"]` and `tests/conftest.py`.
7. Commit `baseline/` — it is a tracked artifact, not scratch output. Note that
   `.gitignore` currently excludes `outputs/`, so `baseline/` is a deliberately
   different directory.

**Gate:** `pytest` passes with all three current tests green (they now import
from `old_src`), and all six `baseline/*.json` files exist.

---

### Phase 1 — `clinical_jepa` foundation

Build `schema.py`, `config.py`, `encoders.py`, and `graph/{builders,tensors,patches}.py`.

- One `to_graph_data`. Note append is a branch on `use_note_embeddings`, not a
  separate function in a separate package.
- One endpoint-alias normalizer, at the graph-loading boundary
  (`graph/builders.py`). §2.6 is resolved here: decide explicitly whether the
  scoring path should tolerate `source`/`target` keys, and make both variants
  behave the same way. Record the decision in the module docstring.
- `config.py` merges the three `Config` classes. `from_dict` keeps v6's
  backward-compatibility defaults verbatim — that is what lets one class load
  both released checkpoints.

**Gate**, as a committed test:

```python
@requires_checkpoints
def test_tensors_match_old_v5_and_v6():
    for old_mod, use_note in ((old_v5_data, False), (old_v6_data, True)):
        for graph in graphs:          # all 400 records
            assert_pyg_equal(new_to_graph_data(graph, enc, use_note_embeddings=use_note),
                             old_mod.to_graph_data(graph, enc))
```

Plus: `Config.from_dict` on the shipped `config_v5.json` and `config_v6.json`
produces objects whose `to_dict()` round-trips equal to the old classes'.

---

### Phase 2 — `clinical_jepa.model` and `losses`

Flatten `GraphJEPAv3 → GraphJEPAv4 → GraphJEPAv5` into one `GraphJEPA`. The
intermediate classes are never instantiated by any shipped path, so the
hierarchy carries cost with no benefit.

**Attribute paths must not change.** `state_dict` keys derive from them.

**Gate:**

1. `sorted(GraphJEPA(cfg).state_dict().keys())` equals `baseline/{v5,v6}_keys.json`
   for the respective configs.
2. Both released checkpoints `load_state_dict(..., strict=True)` without error.
3. A forward pass on a fixed seeded input produces tensors bit-identical to
   `old_src`'s `GraphJEPAv5`/`GraphJEPAv6` on the same input.

Gate 1 is the one that actually catches a bad flatten — a renamed submodule
attribute shows up there and nowhere else until inference silently degrades.

---

### Phase 3 — `clinical_jepa.train` and `evaluate`

Move `train/{loop,pretrain,finetune}.py` and `evaluate.py`. One evaluator, not
two. Argument-parser defaults point at the new `models/` layout. Checkpoint save
names follow §5.3.

**Gate:** the new evaluator reproduces `baseline/v5_loo.json` and
`baseline/v6_loo.json` **exactly** — every metric, every per-relation
breakdown, every count. Not "within tolerance."

---

### Phase 4 — `clinical_jepa.score`

Merge `score_base.py`, `score_revision.py`, `graph_jepa_v5/score.py`, and
`graph_jepa_v6/score.py` — 1,173 lines across four files — into one `score.py`.

- Delete `_install_v6_data_conversion`. The graph converter becomes a parameter.
- Delete the 15 module-level re-export assignments in `score_revision.py`.
- Apply the §2.6 decision from Phase 1 consistently across both variants.

**Gate:** scoring a fixed input graph with each checkpoint produces byte-identical
output JSON before and after, for both the KEEP/REVIEW/PRUNE path and the
candidate-addition path. Because v5 and v6 currently differ here, run the old
side of the comparison for **both** and confirm the new unified behavior matches
the one chosen in Phase 1 — and that the divergence is documented, not silently
resolved.

---

### Phase 5 — `fawkes` (the paper implementation)

Split `paper_v16/trainer.py` (719 lines) at its existing seams — line 256
("shared encoder"), 327 ("downstream readout"), 423 ("EIR scoring") — into
`config.py`, `data.py`, `model.py`, `train.py`, `evaluate.py`. Fold
`paper_v16/evaluate.py` into `fawkes/evaluate.py`.

Convert the import-time environment globals (§2.5) into a dataclass built by a
`from_env()` classmethod, so the existing `USE_NOTE=1 GROUND_BY=prov ...`
invocations keep working while the module becomes importable without side
effects and testable with two configurations in one process.

**This is the highest-risk phase.** `src/paper_v16/README.md` currently states
the trainer "intentionally retains its environment-driven experiment
configuration and architecture so the packaged checkpoint remains compatible."
Splitting it risks silent reproducibility drift in the one artifact that backs
the paper. Change no numerics, no defaults, no tensor operations, and no
environment-variable names.

**Gate:**

1. Loading the released checkpoint and running `loo_evaluate` reproduces
   `baseline/paper_loo.json` to within `1e-6` on every metric.
2. `Encoder().state_dict()` and `DistMult().state_dict()` key sets equal
   `baseline/paper_keys.json`.
3. A test asserts that importing `fawkes` with no environment variables set
   raises no error and reads no `os.environ` at module scope.

If gate 1 or 2 cannot be met, stop: keep `trainer.py` verbatim inside the
`fawkes` package and do the rename only. An unverifiable refactor of the paper
artifact is not worth the readability gain.

---

### Phase 6 — `benchmarks`

- Extract the duplicated LLM client (`ChatRanker`, `_api_key`, `_api_base`,
  `_node_label`) into `benchmarks/llm_ranker.py`.
- `evaluate_llm.py` → `benchmarks/vs_llm.py`.
- `evaluate_loo_v12_jepa_llm.py` → `benchmarks/vs_loo_baseline.py`, retaining the
  three-way comparison (`_print_three_way`).
- Place the v12 LOO baseline checkpoint per §5.4 and update the parser default.
- Replace the fossil `check_independence` guard with a real import-boundary test.

**Gate:**

1. `benchmarks/vs_loo_baseline.py` reproduces the old three-way output for the
   non-LLM rows exactly (the LLM row is non-deterministic; seed it or assert on
   the LOO and `clinical_jepa` rows only, and say so in the test docstring).
2. An AST-based boundary test asserting that no module under `src/clinical_jepa/`
   imports `fawkes`, and no module under `src/fawkes/` imports `clinical_jepa`.
   `benchmarks` is the only package permitted to import both.

Gate 2 is what actually enforces the two-pipeline separation. The guard it
replaces looks for long-deleted `graph_jepa_v2/v3/v4` and enforces nothing.

#### 7.1 Optional: consolidate `LooEncoder` into `fawkes.model.Encoder`

`LooEncoder` appears to be the no-note variant of the fawkes architecture:
`TransformerConv`, hashed entity buckets, DistMult readout, `LOO_NUMERIC_DIM = 6`
where the paper trainer uses `6 + EMBED_DIM`. `_infer_layers(encoder_state)`
already adapts to the checkpoint's layer count.

Once Phase 5 gives `fawkes` a real config object, `LooEncoder` may be
replaceable by `fawkes.model.Encoder(cfg)` with `use_note=False, numeric_dim=6`,
deleting roughly 200 lines and removing the third copy of the architecture.

This is **optional and gated**, not assumed. Attempt it only after Phase 5
passes, and only if: loading `fawkes_jepa_loo_eval_v12_260615.pt` into
`fawkes.model.Encoder` succeeds with `strict=True`, and the three-way comparison
output is unchanged. If the state dicts do not match, leave `LooEncoder` where it
is and note in `docs/LINEAGE.md` that the v12 baseline is a distinct architecture.

---

### Phase 7 — Packaging and documentation

- Rename the distribution `fawkes-three-model-suite` → `clinical-graph-jepa`,
  matching the repository. Update `description`.
- Add `[project.scripts]`, replacing 18 `python -m` invocations:
  `clinical-jepa-train`, `clinical-jepa-eval`, `clinical-jepa-score`,
  `fawkes-train`, `fawkes-eval`, `benchmark-vs-llm`, `benchmark-vs-loo`.
- Declare the missing dependencies under a new extra:
  `llm = ["openai>=1.0", "python-dotenv>=1.0"]`.
- Rename the `models/` subdirectories per §4.4; update `MANIFEST.json` paths.
  Checksums do not change.
- Write `docs/LINEAGE.md` from §4 — including *why* the version numbers were
  never one sequence.
- Apply the §1.1 anti-confusion mitigations: both `__init__.py` docstrings, the
  README's first table, and a rewritten `docs/PAPER_CODE_MAP.md`.
- Update `README.md` (repository tree, reader-orientation table, all example
  commands), `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/DATA.md`, and
  the three model guides under `models/`.
- Expand `tests/` from one `test_suite.py` into per-module files. Keep every
  differential gate from Phases 1–6 as a permanent regression test.
- Delete the duplicated fossil guard from `scripts/smoke_check.py`.

**Gate:** `pip install -e .` succeeds; every console script runs `--help`; every
command in `README.md` executes as written; `pytest` is green.

---

### Phase 8 — Remove `old_src`

1. Drop `old_src` from `pythonpath` in `pyproject.toml`.
2. Run the full suite. Differential tests that import `old_src` now fail — convert
   each to assert against `baseline/*.json` instead, which is exactly what those
   files are for.
3. `git rm -r old_src`.

**Gate:** `pytest` green with `pythonpath = ["src"]` only, on a clean clone,
with checkpoint-dependent tests skipping rather than failing.

---

## 8. Expected outcome

| | Today | After |
| --- | ---: | ---: |
| `src/` lines | 13,240 | ~9,000 |
| Packages | 4, tangled | 3, boundary-tested |
| Copies of the model architecture | 3 (`v5`, `v6`, `LooEncoder`) | 2, or 1 if §7.1 lands |
| Files named for a version | 8 | 0 |
| Cross-package monkeypatches | 2 | 0 |
| Scoring modules | 4 | 1 |
| Endpoint-alias normalizers | 3 | 1 |
| Undeclared dependencies | 2 | 0 |
| Console entry points | 0 | 7 |
| Tests | 3, one unrunnable | ~15, green on clean clone |

---

## 9. Explicitly out of scope

Per `CLAUDE.md` §3 and §10:

- **No model-variant registry or plugin system.** Two variants and one config
  flag. A registry would be dead flexibility.
- **No configuration framework** (Hydra, OmegaConf). The dataclass config works
  and is already checkpoint-serializable — which is precisely the property that
  makes this whole restructure safe.
- **No unification of the two schemas.** `clinical_jepa.schema.NodeType` and
  `fawkes`'s `NODE_TYPES` genuinely differ (the paper's includes `NOTE` and
  `PROCUREMENT`). Forcing them together would break checkpoint compatibility in
  both directions. They stay separate — that is the point of two packages.
- **No behavior changes bundled into moves.** Every gate is an equality check.
  Anything that looks like a bug (§2.6 especially) gets its own commit, its own
  test, and an explicit note — never smuggled into a rename.
- **No retraining.** No phase produces a new checkpoint.

---

## 10. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Flattening the class hierarchy renames a submodule attribute and silently changes `state_dict` keys | Medium | Phase 2 gate 1 compares key sets against `baseline/*_keys.json` |
| Splitting the paper trainer changes numerics | Medium | Phase 5 gate: `1e-6` metric reproduction; abort to rename-only if it fails |
| The §2.6 v5/v6 scoring divergence is resolved silently and changes real output | Medium | Phase 1 requires an explicit, documented decision; Phase 4 gates on both old behaviors |
| The v12 baseline checkpoint is unobtainable | Medium | Phase 6 gate 1 is blocked; §7.1 becomes impossible. Confirm availability before starting Phase 6 |
| `benchmarks` LLM comparison is non-deterministic and cannot be gated | High | Expected. Gate on the LOO and `clinical_jepa` rows only, stated in the test docstring |
| `git log --follow` breaks across the parallel build | High | Accepted, and inherent to decision 4. `docs/LINEAGE.md` is the durable record. `old_src` is created with `git mv`, so the old files keep their history until Phase 8 |
| Someone reads `clinical_jepa` as the paper implementation | High | §1.1 mitigations |

---

## 11. Open items

1. **v12 baseline checkpoint.** Confirm `fawkes_jepa_loo_eval_v12_260615.pt` is
   obtainable before Phase 6 begins.
2. **§2.6 decision.** Which endpoint-alias behavior is correct — v5's strict
   `source_id` requirement or v6's permissive normalization? Needed in Phase 1.
3. **`baseline/` retention.** Keep it as a permanent regression oracle after
   Phase 8, or delete it once `old_src` is gone? Recommendation: keep. It is
   small, and it is the only remaining evidence that the restructure preserved
   the released models' behavior.
4. **`docs/EVALUATION.md` results table.** It currently reports a v6 vs
   paper-v16 comparison with a caveat that the underlying result files are
   absent. Phase 0 produces exactly those files. Consider replacing the table
   with audited numbers from `baseline/` during Phase 7.
