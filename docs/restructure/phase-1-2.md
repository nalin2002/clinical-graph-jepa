# Agent A — Phases 1 and 2: the `clinical_jepa` core

Wave 1. Runs concurrently with Agent B (`fawkes`) and Agent C (`benchmarks`).
They write different files; you will not collide if you stay inside your scope.

## Read first

- `docs/RESTRUCTURE_PLAN.md` — §2.1, §2.2, §2.6, §2.8, §3, §4.2, §4.3, and
  Phases 1 and 2 in full.
- `old_src/fawkes_core/{schema,config,encoders,model_base,revision,data_graph,patches}.py`
- `old_src/graph_jepa_v5/{config,data,model}.py` and the v6 equivalents.
- `baseline/README.md` — how the oracles you gate against were produced.

Read the v5 and v6 pairs side by side. Per §2.1 roughly 3,800 of 8,127 lines are
byte-identical and `graph_jepa_v6/model.py` ends with `GraphJEPAv5 = GraphJEPAv6`.
You are merging one class body that was copied, not reconciling two designs.

## You own — the only files you may create or edit

```
src/clinical_jepa/{__init__,schema,config,encoders,model,losses}.py
src/clinical_jepa/graph/{__init__,builders,tensors,patches}.py
tests/test_clinical_jepa_core.py
```

## Do not touch

Anything under `old_src/` — it is the oracle; editing it invalidates every gate.
`pyproject.toml`, `README.md`, `models/`, `baseline/`, `tests/conftest.py`.
Any other file under `src/` or `tests/` — Agents B and C own those and are
running right now.

## Context you cannot derive from the plan

**Use `--data synthetic` for every gate.** The `--jsonl-path` default
(`data/fawkes_1k_patients/...`, `fawkes_core/training.py:306`) does not exist on
disk. It does not matter: your gates are differential — old code against new code
on identical input — so synthetic is sufficient. It is also deterministic
(`training.py:39` passes `seed=cfg.train.seed`, default `0`), which a real
dataset would not guarantee.

**§2.6 is decided: normalize endpoint aliases at the graph-loading boundary**, in
`graph/builders.py` — v6's `_edge_endpoint` behavior (`old_src/graph_jepa_v6/data.py:105`).

The evidence, which belongs in the module docstring alongside the decision: the
only real dataset present (`data/fawkes-training-graph-embedded-260615/*.jsonl`)
keys its edges `source`/`target`. `old_src/graph_jepa_v5/data.py:75` indexes
`edge["source_id"]` unguarded and raises `KeyError` on it. v5's strict
requirement was written against the now-absent `fawkes_1k_patients` file. The
permissive path is the one that reads the shipped data.

This is a behavior change for the v5 variant, not a move. Per plan §9 it gets its
own commit and its own test, and it must be visible — never smuggled into a
rename.

**Build a fixture for the §2.6 test.** A handful of graphs carrying both key
styles deliberately, plus the mixed case where an edge has `source` but
`target_id`. Do not depend on anything under `data/`. A targeted fixture tests
the divergence better than any real file, and it keeps your tests runnable on a
clean clone.

**Checkpoint directories are still `models/v5_without_note/` and
`models/v6_with_note/`.** Phase 7 renames them. Config sidecars are still
`config_v5.json` / `config_v6.json`.

## Phase 1 gate

The tensor-equality test from the plan, over the full synthetic graph set, for
both `use_note` values:

```python
@requires_checkpoints
def test_tensors_match_old_v5_and_v6():
    for old_mod, use_note in ((old_v5_data, False), (old_v6_data, True)):
        for graph in graphs:
            assert_pyg_equal(new_to_graph_data(graph, enc, use_note_embeddings=use_note),
                             old_mod.to_graph_data(graph, enc))
```

Plus: `Config.from_dict` on the shipped `models/*/config_v{5,6}.json` produces
objects whose `to_dict()` round-trips equal to the old classes'.

Keep v6's `from_dict` backward-compatibility defaults **verbatim** — defaulting
`use_note_embeddings` to `False` and deriving `base_in_dim` when absent is
precisely what lets one merged class load both released checkpoints (§2.1).

## Phase 2 gate

1. `sorted(GraphJEPA(cfg).state_dict().keys())` equals `baseline/v5_keys.json`
   and `baseline/v6_keys.json` for the respective configs.
2. Both released checkpoints `load_state_dict(..., strict=True)` without error.
3. A forward pass on a fixed seeded input is bit-identical to `old_src`'s
   `GraphJEPAv5` / `GraphJEPAv6` on the same input.

**Attribute paths must not change.** Per §2.8, `state_dict` keys derive from
attribute paths (`context_node_encoder.input_proj.weight`), not class names —
they survive class renames but not attribute renames. Flattening
`GraphJEPAv3 → GraphJEPAv4 → GraphJEPA` is safe only if every submodule keeps its
attribute name on the merged class. Gate 1 is the one that catches a bad flatten;
gates 2 and 3 can both pass while a key set has quietly drifted.

## Stop and report — do not work around

- **Any gate fails.** Never adjust an expected value, never loosen a tolerance,
  never introduce a tolerance where the plan says exact. Report the actual diff.
  A failing gate is information about the merge, not an obstacle to it.
- **A `state_dict` key differs from the baseline.** An attribute path moved.
  Report which one and stop — do not rename the baseline to match.
- **v5 and v6 turn out to differ somewhere §2.1 does not predict.** The premise
  is that the difference is four config fields plus the note-append branch. If
  you find a fifth, that is a finding; surface it before merging past it.
- **You need to write a file outside your ownership list.**

## Do not commit

Leave your work in the tree. The wave barrier is a human running `pytest` and
committing once all three Wave 1 agents are done and green.
