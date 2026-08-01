# Agent C — Phase 6 (partial): `benchmarks` foundation

Wave 1. Runs concurrently with Agent A (`clinical_jepa`). Your scopes are
disjoint; stay inside yours.

## Your scope is smaller than Phase 6

Plan Phase 6 lists four deliverables. **Two of them cannot be done in this wave**,
and attempting them will waste your time and collide with other agents.

Measured import dependencies of the two benchmark modules:

```
old_src/graph_jepa_v6/evaluate_llm.py
    fawkes_core.schema        -> clinical_jepa.schema        (Agent A, in progress)
    .data                     -> clinical_jepa.graph.tensors (Agent A, in progress)
    .evaluate                 -> clinical_jepa.evaluate      (Agent D, not started)
    .training                 -> clinical_jepa.train.loop    (Agent D, not started)

old_src/graph_jepa_v6/evaluate_loo_v12_jepa_llm.py
    all of the above, plus .evaluate_llm
```

So `vs_llm.py` and `vs_loo_baseline.py` both block on Agent D. They are **out of
scope for you** — do not create them, do not stub them.

Note this corrects an earlier assumption that `vs_loo_baseline.py` was
self-contained because it defines its own `LooEncoder`. It defines its own model
but still imports the shared dataset and evaluation helpers.

### You deliver exactly three things

1. `src/benchmarks/__init__.py` and `src/benchmarks/llm_ranker.py`
2. `docs/LINEAGE.md`
3. `tests/test_import_boundaries.py`

## You own — the only files you may create or edit

```
src/benchmarks/__init__.py
src/benchmarks/llm_ranker.py
docs/LINEAGE.md
tests/test_import_boundaries.py
```

## Do not touch

Anything under `old_src/` — it is the oracle for every gate in this restructure.
Anything under `src/clinical_jepa/` — **Agent A is writing there right now.**
`pyproject.toml`, `README.md`, `models/`, `baseline/`, `tests/conftest.py`,
`docs/RESTRUCTURE_PLAN.md`, `scripts/`.

The `openai` / `python-dotenv` declaration (plan §2.7) is Phase 7's job, in
`pyproject.toml`, which you do not own. Both are already installed in the
environment so you can import-test. Mention the gap in your report; do not fix it.

Likewise `scripts/smoke_check.py::check_independence` — the fossil guard your
boundary test replaces. Phase 7 deletes it. Leave it alone.

## 1. `llm_ranker.py`

Extract the duplicated LLM client from the two `evaluate_llm.py` copies. Verified
for you: the helper region is **byte-identical** between the v5 and v6 copies, so
this is a true dedup with no reconciliation needed. Take the v6 copy.

Symbols named by the plan, with their measured closures:

| Symbol | Lines (v6) | Also needs |
| --- | --- | --- |
| `_api_base` | 75-80 | — |
| `_api_key` | 83-104 | `_load_dotenv_files` |
| `_node_label` | 117-119 | `_node_text` |
| `ChatRanker` | 406-490 | `openai.OpenAI` |

None of these import first-party code. Compute the transitive closure yourself
and confirm it before you move anything — if a symbol you pull in turns out to
need `fawkes_core` or `graph_jepa_v*`, stop and report rather than dragging the
dependency into `benchmarks`.

Do **not** edit the `old_src` copies to import from your new module. `old_src` is
frozen. The duplication disappears when Agent D ports `vs_llm.py`.

Gate: `python -c "from benchmarks.llm_ranker import ChatRanker"` succeeds, and an
AST check confirms the module imports no first-party package.

## 2. `docs/LINEAGE.md`

Write it from plan §4 — it is the single authority on old-name → new-name, and
§1.1 makes it a required mitigation for the naming hazard.

Cover §4.1 (packages), §4.2 (modules), §4.3 (symbols), §4.4 (model directories).

Beyond transcribing the tables, explain **why the version numbers were never one
sequence** — plan §2.2 is the source. `GraphJEPAv3 → v4 → v5/v6` are architecture
layer names, not repository versions, which is why renaming alone cannot remove
them. `GraphJEPAv3` and `GraphJEPAv4` are never instantiated by any shipped path;
they survive only as base classes, and no v3 or v4 checkpoints exist here.

State the §1.1 hazard plainly and early: **`fawkes` is the paper implementation;
`clinical_jepa` is not**, despite the repository name and the manuscript
filename. Say it in the first screenful.

Record two things measured in Phase 0 that belong in the lineage record:

- The v12 LOO baseline is deferred, not ported. §7.1's consolidation of
  `LooEncoder` into `fawkes.model.Encoder` was gated on loading
  `fawkes_jepa_loo_eval_v12_260615.pt`, which is not present and is not being
  obtained. Note it as a distinct architecture pending verification.
- `models/paper_v16/`'s checkpoint sha256 differs from `MANIFEST.json` while its
  byte count matches, and Phase 0 established empirically that it is nonetheless
  the published artifact. See `baseline/README.md`.

## 3. `tests/test_import_boundaries.py`

This is plan Phase 6 gate 2, and it is what actually enforces the two-pipeline
separation. AST-walk each module's imports — do not import the packages, so the
test stays meaningful while the tree is half-built.

Assert: no module under `src/clinical_jepa/` imports `fawkes`; no module under
`src/fawkes/` imports `clinical_jepa`; `benchmarks` is the only package permitted
to import both.

The test must pass **now**, while `src/fawkes/` does not yet exist and
`src/clinical_jepa/` is mid-build. Write it so that an absent package is a skip
or a vacuous pass, never a hard failure — and make sure it cannot silently pass
forever by finding nothing. A test that scans zero files and reports green is
worse than no test. Assert on the file count it actually walked.

## Stop and report — do not work around

- A symbol in the `llm_ranker` closure needs first-party code.
- Your boundary test cannot be written without importing the packages.
- You need to write a file outside your ownership list.
- You find yourself wanting to touch `src/clinical_jepa/` for any reason. Agent A
  owns it and is live; a concurrent write there will corrupt its work.

## Do not commit

Leave your work in the tree. A human runs the gate and commits at the wave
barrier, once both Wave 1 agents are done.
