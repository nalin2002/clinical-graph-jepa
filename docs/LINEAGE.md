# Lineage — old names to new names

This document is the single authority on what each package, module, symbol, and
model directory used to be called. `git log --follow` does not survive the
restructure (the new tree was built in parallel with the old one rather than
renamed in place), so this file is the durable record of provenance.

Source: `docs/RESTRUCTURE_PLAN.md` §4, plus measurements taken in Phase 0.

---

## Read this first: the package names invert what you expect

**`fawkes` is the paper implementation. `clinical_jepa` is not.**

The repository is called `clinical-graph-jepa` and the manuscript is
`paper/clinical_jepa.pdf`, so `clinical_jepa` looks like the package behind the
paper. It is not. The published checkpoint and the reported numbers come from
`fawkes`, which keeps its original author's name.

| Package | What it actually is |
| --- | --- |
| `src/fawkes/` | **The paper implementation.** Was `paper_v16`. Loads the released paper checkpoint; reproduces the published metrics. |
| `src/clinical_jepa/` | **Not the paper implementation.** The modular patch-based Graph-JEPA revision pipeline. Was `fawkes_core` + `graph_jepa_v5` + `graph_jepa_v6`. |
| `src/benchmarks/` | New. Cross-lineage comparison; the only package permitted to import both of the above. |

If you are looking for the code behind the paper, read `src/fawkes/`.

This is a naming hazard, not an accident, and it is mitigated deliberately: both
package docstrings state it, `README.md` states it in its first table, and
`tests/test_import_boundaries.py` enforces that the two never import each other.

---

## Why the version numbers were never one sequence

The old tree had `fawkes_core`, `graph_jepa_v5`, `graph_jepa_v6`, and
`paper_v16`. Those numbers do not form a release history, and reading them as
one is the second most common way to misunderstand this repository.

`fawkes_core` was never version-neutral despite its name. It was the v3 and v4
lineages flattened into one directory:

| Old file | Actually contained |
| --- | --- |
| `fawkes_core/model_base.py` | `GraphJEPAv3` — the base architecture |
| `fawkes_core/revision.py` | `GraphJEPAv4(GraphJEPAv3)` — schema-aware revision losses |
| `fawkes_core/data_graph.py` | docstring: "PyG graph tensor conversion for Graph-JEPA v3" |
| `fawkes_core/config.py` | docstring: "Configuration dataclasses for Graph-JEPA v4" |

The real inheritance chain was:

```
GraphJEPAv3  (fawkes_core/model_base.py:507)
  └── GraphJEPAv4  (fawkes_core/revision.py:213)
        ├── GraphJEPAv5  (graph_jepa_v5/model.py:358)
        └── GraphJEPAv6  (graph_jepa_v6/model.py:356)
```

**`v3`, `v4`, `v5`, `v6` are architecture layer names, not repository versions.**
That is exactly why renaming alone could not remove them — they were load-bearing
class identifiers in an inheritance chain, not directory labels. Removing them
required flattening the hierarchy, which is a code change with a `state_dict`
key-set gate behind it, not a `git mv`.

Two further facts about that chain:

- **`GraphJEPAv3` and `GraphJEPAv4` are never instantiated by any shipped path.**
  They survived only as base classes. No v3 or v4 checkpoints exist in this
  repository, and none are obtainable. Flattening them into one `GraphJEPA` cost
  nothing because nothing ever constructed them directly.
- **`v5` and `v6` were one class body.** `graph_jepa_v6/model.py` ended with the
  literal line `GraphJEPAv5 = GraphJEPAv6` (line 711). The entire substantive
  difference between the two models is four config fields — `base_in_dim`,
  `use_note_embeddings`, `note_embedding_dim`, `note_ground_by` — and a
  note-append branch in `data.py`. Everything else was `s/v5/v6/` applied to a
  copy. Phase 0 confirmed this from the other side: `v5_keys.json` and
  `v6_keys.json` are identical, 119 keys each, and the variants differ only in
  two tensor *shapes* (`context_node_encoder.input_proj.weight` and
  `target_node_encoder.input_proj.weight`, 768 vs 1536 input features).

`paper_v16`'s `16` is unrelated to any of the above. It is an experiment number
from a separate line of work, not the successor to `v6`.

---

## 1. Packages

| Today | New | Note |
| --- | --- | --- |
| `src/paper_v16/` | `src/fawkes/` | The paper implementation. Original author's name retained. |
| `src/fawkes_core/` | `src/clinical_jepa/` | merged |
| `src/graph_jepa_v5/` | `src/clinical_jepa/` | merged — no-note variant |
| `src/graph_jepa_v6/` | `src/clinical_jepa/` | merged — localized-note variant |
| — | `src/benchmarks/` | new; cross-lineage comparison |

The pre-restructure tree lives at `old_src/` for the duration of the migration
and is deleted in Phase 8. It is the oracle every gate compares against, so it is
frozen — never edited, not even to import from the new tree.

## 2. Modules

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
| `graph_jepa_v{5,6}/evaluate_loo_v12_jepa_llm.py` | `benchmarks/vs_fawkes.py` — renamed with the baseline arm; see below |
| — | `benchmarks/llm_ranker.py` (extracted from the two `evaluate_llm.py` copies) |
| `paper_v16/trainer.py` | split into `fawkes/{config,data,model,train,evaluate}.py` |
| `paper_v16/evaluate.py` | folded into `fawkes/evaluate.py` |

`benchmarks/llm_ranker.py` holds `ChatRanker`, `_api_base`, `_api_key`,
`_load_dotenv_files`, `_node_text`, and `_node_label`. All six were duplicated
byte-for-byte between the v5 and v6 `evaluate_llm.py` copies, so the dedup needed
no reconciliation; the v6 copy was taken. The module imports no first-party
package and must not start to — `tests/test_import_boundaries.py` asserts it.

## 3. Symbols

| Today | New |
| --- | --- |
| `GraphJEPAv3` | `GraphJEPA` (`clinical_jepa/model.py`) |
| `GraphJEPAv4` | class dissolved; losses move to `clinical_jepa/losses.py` |
| `GraphJEPAv5`, `GraphJEPAv6` | both dissolved into `GraphJEPA` |
| `_v3`, `_v4` import aliases | deleted |
| `_install_v6_data_conversion` | deleted; converter passed as an argument |
| `LooEncoder`, `LooDistMult`, `LooMLPScorer` | deleted; the baseline arm is the v16 paper checkpoint — see "the v12 LOO baseline" below |

Class renames are safe for checkpoints. `save_checkpoint` writes
`{"state_dict": ..., "config": cfg.to_dict()}` where `to_dict` is `asdict`, so
checkpoints contain no pickled first-party classes — only nested dicts and
tensors. `state_dict` keys derive from *attribute* paths
(`context_node_encoder.input_proj.weight`), which means they survive class
renames but **not** attribute renames. That distinction is why every model phase
gates on `state_dict` key-set equality against `baseline/*_keys.json`.

## 4. Model directories

| Today | New |
| --- | --- |
| `models/v5_without_note/` | `models/clinical-jepa-no-note/` |
| `models/v6_with_note/` | `models/clinical-jepa-localized-note/` |
| `models/paper_v16/` | `models/fawkes-entity-note/` |
| `config_v5.json`, `config_v6.json` | `config.json` (the directory already scopes it) |
| `config_v5_pretrain.json`, `config_v6_pretrain.json` | `config_pretrain.json` |

Checkpoint **files** keep their current names — `graph_jepa_v5.pt`,
`graph_jepa_v6.pt`, `fawkes_trainer_jepa_entity_note_v16_260615.pt`, and the two
`*_pretrain.pt`. A checkpoint is a historical artifact and its filename is
legitimate provenance. Keeping the names also keeps every `sha256` in
`models/MANIFEST.json` byte-stable, which is what makes the directory rename
provably content-neutral. New checkpoints written by the new code use the naming
convention in plan §5.3.

These directory renames landed in Phase 7. `models/fawkes-loo-baseline/`, which
plan §4.4 also listed, was **not** created: the v12 checkpoint it was to hold is
no longer used by anything (see "the v12 LOO baseline" below).

---

## Two measurements worth recording

### The v12 LOO baseline is replaced, not ported — and §7.1 landed anyway

Plan §7.1 proposed consolidating `LooEncoder` into `fawkes.model.Encoder`,
**gated on loading `fawkes_jepa_loo_eval_v12_260615.pt` into it with
`strict=True`**. That checkpoint is not present in the working tree and is not
being obtained, so the gate could never be run — and neither could the arm it
served: the old three-way comparison's first arm had no weights to load.

Phase 6b resolved that by substitution rather than by consolidation. **The
baseline arm of `benchmarks/vs_fawkes.py` is now the v16 paper checkpoint**
(`models/fawkes-entity-note/`, whose behaviour Phase 0 pinned exactly), loaded through
`fawkes` itself. `LooEncoder`, `LooDistMult`, `LooMLPScorer` and their `LOO_*`
vocabularies are therefore deleted rather than ported — roughly 200 lines, and
§2.3's third copy of the paper-lineage architecture. §7.1's outcome (one copy of
that architecture, not three) has landed by a different route than the plan
anticipated; `tests/test_benchmarks.py::test_loo_baseline_architecture_is_gone`
keeps it landed.

**What this does not establish.** The v16 checkpoint was never loaded into
`LooEncoder`, and could not be: `LOO_NUMERIC_DIM` was 6 where the v16
checkpoint's config records `numeric_dim: 774` (6 + 768 note dimensions). They
are different configurations of the same architecture *family*, which is exactly
what nobody has verified. The question §7.1 asked is retired, not answered. The
v12 checkpoint is no longer required by anything in this repository.

**What it buys.** A phase that had no numeric gate now has an exact one: the
`fawkes` arm reproduces `baseline/paper_loo_testsplit.json` — MRR 0.418653448
over n=8283 — to a delta of `0.000e+00`, per-relation rows included.

**What it costs.** The arms are no longer paired. `LooEncoder` was written to eat
`clinical_jepa`'s tensors, so one dataset pipeline could feed both model arms and
one `n` column described both. `fawkes` has its own `to_data`, vocabularies and
edge pruning, so each arm now runs its own pipeline over the same admissions and
the comparison aligns on relation name only, with per-arm counts. See the
`benchmarks/vs_fawkes.py` docstring for the measured population difference.

### The paper checkpoint hash differed from the manifest — corrected in Phase 7

`MANIFEST.json` recorded sha256 `fc8c494a…` for
`fawkes_trainer_jepa_entity_note_v16_260615.pt`. The file on disk hashes
`6c21abb2…`. The byte count matched the manifest exactly (5,204,898), and all
four v5/v6 hashes matched, so the manifest was not generally stale.

**Phase 0 established empirically that the file on disk is nonetheless the
published artifact.** Re-running the trainer's test-split evaluation reproduces
the metrics stored *inside* the checkpoint to a delta of exactly `0.000e+00` on
all four metrics, with `n=8283` matching. The hash difference is a `torch.save`
re-serialization, not different weights. See `baseline/README.md` for the
invocation and `baseline/paper_loo_testsplit.json` for the numbers.

`MANIFEST.json` was deliberately left uncorrected until Phase 7 — editing it
earlier would have destroyed the evidence that this discrepancy was investigated
rather than overlooked. **Phase 7 recorded the real hash** and kept the
superseded value alongside it under `sha256_superseded`, with a `note` field
pointing at the evidence, so the correction is auditable rather than silent.
`tests/test_model_artifacts.py` pins both the corrected value and the retained
evidence trail.
