# Agent E — Phase 4: `clinical_jepa.score`

Wave 2. Runs concurrently with Agent D (`train/` and `evaluate.py`). Your scopes
are disjoint.

## Read first

- `docs/RESTRUCTURE_PLAN.md` — §2.4, §2.6, §4.2, §4.3, §9, and Phase 4 in full.
- `old_src/fawkes_core/score_base.py` (633 lines),
  `old_src/fawkes_core/score_revision.py` (291),
  `old_src/graph_jepa_v5/score.py` (122), `old_src/graph_jepa_v6/score.py` (127).
- `src/clinical_jepa/graph/builders.py` — **its module docstring carries the
  §2.6 decision and the evidence behind it.** You must apply that same decision.
- `src/clinical_jepa/` generally — Agent A's completed Phase 1+2 work.

## You own — the only files you may create or edit

```
src/clinical_jepa/score.py
tests/test_clinical_jepa_score.py
```

## Do not touch

`old_src/` — the oracle for every gate. `src/clinical_jepa/{schema,config,
encoders,model,losses}.py` and `src/clinical_jepa/graph/` — Agent A's finished
work. `src/clinical_jepa/train/` and `src/clinical_jepa/evaluate.py` —
**Agent D is writing those right now.** `src/fawkes/`, `src/benchmarks/`,
`tests/conftest.py`, `pyproject.toml`, `README.md`, `models/`, `baseline/`.

## The merge

Four modules, 1,173 lines, into one `score.py`. Three specific demolitions the
plan names:

**1. Delete `_install_v6_data_conversion`** (`graph_jepa_v6/score.py:19`). It
mutates two `fawkes_core` modules globally at call time:

```python
_v3.to_graph_data = to_graph_data
_v4.to_graph_data = to_graph_data
```

It is import-order dependent, invisible to anyone reading `fawkes_core`, and
makes it impossible for v5 and v6 scoring to coexist in one process. The graph
converter becomes a **parameter**. Agent A already unified `to_graph_data` in
`clinical_jepa/graph/tensors.py` with the note-append behind a config flag, so
there is one converter to pass.

**2. Delete the 15 module-level re-export assignments** in `score_revision.py`
(the `from . import score_base as _v3` re-export block). They exist only to make
the v4 CLI reuse v3 private helpers; in one merged module they are dead.

**3. Apply the §2.6 decision consistently to both variants.** Read it from
`builders.py`'s docstring — do not re-derive it. It was settled with evidence in
Phase 1: normalize endpoint aliases at the graph-loading boundary. The third copy
of that logic lives in `score_base.py::_looks_like_mimic_subkg`, and folding it in
is explicitly Phase 4's job.

## Dependencies already resolved for you

**`build_checkpoint_encoder` is in `clinical_jepa/encoders.py`**, not
`train/loop.py` where plan §4.2 maps it. It was the *only* symbol the old
`graph_jepa_v6/score.py` needed from `training.py`; moving it removes your
dependency on Agent D entirely. Import it from `encoders`. Do not redefine it,
and do not import anything from `clinical_jepa.train` — if you find you need to,
stop and report rather than creating a race with Agent D.

Everything else you need — `Config`, `GraphJEPA`, `to_graph_data`,
`build_patch_data`, `PatientGraph`, `RELATION_SCHEMA`, `canonical_relation`,
`adapt_mimic_subkg`, `normalize_graph_edges` — is in Agent A's finished modules.
`GraphJEPAv3`/`v4`/`v5`/`v6` are all one `GraphJEPA` now (plan §4.3).

## Gate

Scoring a fixed input graph with each checkpoint produces **byte-identical output
JSON** before and after, for both paths:

- the KEEP / REVIEW / PRUNE path
- the candidate-addition path

Because v5 and v6 genuinely diverge here today, run the old side for **both** and
confirm the new unified behavior matches the one chosen in Phase 1 — and that the
divergence is **documented, not silently resolved**. Plan §9 is explicit: anything
that looks like a bug gets its own commit, its own test, and an explicit note,
never smuggled into a rename.

That means your test must show, not hide, where old-v5 and old-v6 disagreed. A
test asserting only "new matches old-v6" would pass while concealing the change
in v5's behavior. Assert both comparisons and label the intentional difference.

Checkpoints are at `models/v5_without_note/graph_jepa_v5.pt` and
`models/v6_with_note/graph_jepa_v6.pt` — Phase 7 renames those directories, so
point parser defaults at the **current** names with a comment saying so.

## Stop and report — do not work around

- **The gate fails.** Never adjust an expected value or loosen to a tolerance.
- **You cannot make one code path serve both variants** without changing observable
  output for one of them. That is a real finding — report the specific divergence
  rather than picking a winner silently.
- You need to import from `clinical_jepa.train` or edit any file outside your list.

## Do not commit

Leave your work in the tree. A human runs the gate and commits at the wave
barrier, once both Wave 2 agents are done.
