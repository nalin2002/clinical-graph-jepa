# Agent F — Phase 6b: the `benchmarks` ports, with a v16 baseline

Runs alone. Phases 0–5 and 6a are committed; `src/clinical_jepa/` and
`src/fawkes/` are complete and their gates pass.

## Read first

- `docs/RESTRUCTURE_PLAN.md` — Phase 6, §7.1, §2.3, §9.
- `baseline/README.md` — especially the `paper_loo_testsplit.json` section.
- `baseline/reproduce_paper_testsplit.py` — the split logic you will reuse.
- `old_src/graph_jepa_v6/evaluate_llm.py` and
  `old_src/graph_jepa_v6/evaluate_loo_v12_jepa_llm.py`.
- `src/benchmarks/llm_ranker.py` — Agent C's extraction; reuse it, do not
  re-extract.
- `docs/LINEAGE.md` — you will be updating it.

## You own — the only files you may create or edit

```
src/benchmarks/vs_llm.py
src/benchmarks/vs_fawkes.py
tests/test_benchmarks.py
docs/LINEAGE.md            (the v12-deferral paragraph only)
```

## Do not touch

`old_src/` — the oracle. `src/clinical_jepa/`, `src/fawkes/`,
`src/benchmarks/llm_ranker.py` — finished work owned by others; report problems,
do not edit. `pyproject.toml`, `README.md`, `models/`, `baseline/`,
`tests/conftest.py`, `tests/test_import_boundaries.py`.

`benchmarks` is the **only** package permitted to import both `clinical_jepa` and
`fawkes` — `tests/test_import_boundaries.py` enforces this. Your modules may
import both. Nothing you write may cause either of those packages to import the
other.

## Part 1 — `vs_llm.py`

A port of `evaluate_llm.py`, unblocked now that Phase 3 landed. Import rewrites:

```
fawkes_core.schema    -> clinical_jepa.schema
.data                 -> clinical_jepa.graph.tensors   (PatientGraphDataset)
.evaluate             -> clinical_jepa.evaluate
.training             -> clinical_jepa.train.loop
ChatRanker/_api_key/_api_base/_node_label/_node_text/_load_dotenv_files
                      -> benchmarks.llm_ranker
```

The v5 and v6 copies differ by 14 lines. Merge to one, config-driven, as every
prior phase did. Delete the copies of the symbols Agent C already extracted.

## Part 2 — `vs_fawkes.py`, and the substitution that motivates this phase

The old `evaluate_loo_v12_jepa_llm.py` compared three arms: a v12 LOO baseline,
`clinical_jepa`, and an LLM. **The v12 checkpoint is not available and is not
being obtained.** The baseline arm becomes the **v16 paper checkpoint**, which is
present and whose behavior is pinned exactly.

That converts a phase which had no numeric gate into one that has a strong one.

### Delete `LooEncoder`, `LooDistMult`, `LooMLPScorer`

Roughly 200 lines, plus `LOO_NODE_TYPES`, `LOO_RELATION_CANONICAL`,
`LOO_RELATION_ALIASES`, `LOO_NUMERIC_DIM`, `LOO_SCORE_FEATS`,
`DEFAULT_LOO_FILENAME`, `_infer_layers`, and the `--loo-repo-id` /
`--loo-checkpoint` plumbing. This removes plan §2.3's third copy of the
paper-lineage architecture and lands §7.1, which was blocked only by the missing
v12 checkpoint.

**Do not try to load the v16 checkpoint into `LooEncoder`.** It will not fit:
`LOO_NUMERIC_DIM = 6` where v16's config records `numeric_dim: 774` (6 + 768
note dims). They are different configurations of the same architecture family.
Use `fawkes` directly instead — it loads v16 natively and Phase 5 proved it.

### The arms

| arm | pipeline |
| --- | --- |
| fawkes (v16) | `fawkes.data.to_data` → `fawkes.evaluate.loo_evaluate(enc, scorer, graphs, device, cfg, cap)` |
| clinical_jepa | `clinical_jepa.graph.tensors.PatientGraphDataset` → `clinical_jepa.evaluate` |
| LLM | `benchmarks.llm_ranker.ChatRanker`, as before |

### The alignment problem — the hard part of this phase

The old script fed `LooEncoder` from `clinical_jepa`'s `PatientGraphDataset`
(one pipeline, both arms), because `LooEncoder` was written to eat those tensors.
**`fawkes` cannot do that** — it has its own `to_data` with a different feature
layout. So each arm must run its own pipeline over the same JSONL, and the
comparison table aligns on relation name.

Two consequences you must handle explicitly, not paper over:

1. **The arms may evaluate different edge populations.** `fawkes` keeps graphs
   with `num_nodes >= 3 and edge_index.size(1) >= 4` (`trainer.py:601`, mirrored
   in `baseline/reproduce_paper_testsplit.py`); `clinical_jepa` applies its own
   filter. Per-relation `n` will therefore differ between arms.
2. **A comparison table with mismatched `n` is misleading unless labelled.** The
   old `_print_three_way` printed a single `n` column taken from the v6 arm. That
   was safe when one pipeline fed both. It is not safe now. Print each arm's own
   `n`, or print the intersection and say so. Either is acceptable; silently
   reusing one arm's `n` for all three is not.

Verified for you: `clinical_jepa`'s schema covers this dataset completely — every
node type and **100% of relation instances** (19,540/19,540 over the first 300
records) map into `EDGE_TYPE_TO_IDX`. So the comparison is meaningful; you are
not forcing two vocabularies together. Confirm it over the full file yourself.

## Gate

**1. The fawkes arm reproduces `baseline/paper_loo_testsplit.json` exactly.**
Same 400-graph test split — `RandomState(42).permutation`, `TEST_FRAC=0.1`,
filter `>= 3` nodes and `>= 4` edges. Expect `mrr=0.418653448`,
`hits1=0.247494869`, `hits3=0.473741398`, `hits10=0.863575999`, `n=8283`. Phase 0
achieved delta `0.000e+00`, so exact is the standard, not `1e-6`.

This is the numeric gate the v12 version could never have.

**2. The clinical_jepa arm matches `old_src` differentially.** There is no pinned
baseline for it on this dataset (`baseline/v{5,6}_loo.json` are synthetic), so
assert new-vs-old in one process on the same graphs.

**3. The LLM arm is not gated** — it is non-deterministic. Say so in the test
docstring, per plan Phase 6 gate 1 and §10's risk table.

**4. `tests/test_import_boundaries.py` still passes**, and its walked-file count
goes up rather than down.

## Documentation

`docs/LINEAGE.md` currently records the v12 baseline as deferred and
`LooEncoder` as "a distinct architecture pending verification". That is
superseded. Rewrite that paragraph: the baseline arm is now the v16 paper
checkpoint via `fawkes`, §7.1's consolidation has landed by a different route
than the plan anticipated, and the v12 checkpoint is no longer required by
anything. Keep it factual about what was verified — the v16 checkpoint was never
loaded into `LooEncoder`, so do not claim the two architectures were shown
equivalent.

## Stop and report — do not work around

- **Gate 1 misses.** The fawkes arm is a solved problem; a miss means your
  wiring differs from `baseline/reproduce_paper_testsplit.py`, not that the
  tolerance is wrong.
- **The two arms cannot be aligned on a shared relation vocabulary** without
  dropping a material share of edges. Report the coverage number.
- You need to edit `clinical_jepa`, `fawkes`, or `llm_ranker.py`.
- The LLM path needs a live API key to import or construct. It must not — keep
  the client lazy, as Agent C did.

## Do not commit

Leave your work in the tree; a human runs the gate and commits.
