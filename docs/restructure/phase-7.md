# Agent G — Phase 7: packaging and documentation

Runs alone. Phases 0–6 are committed and the suite is green at 83.

## Read first

- `docs/RESTRUCTURE_PLAN.md` — Phase 7, §1.1, §4.4, §5.1, §11.
- `baseline/README.md` — the authority on what was measured and why.
- `docs/restructure/phase-*.md` — the earlier briefs record decisions you are
  now documenting.
- `pyproject.toml`, `README.md`, `docs/`, `models/*/README.md`.

## Already done — do not redo

- **§1.1 docstrings.** Both `src/clinical_jepa/__init__.py` and
  `src/fawkes/__init__.py` already open with the required text. Verify, do not
  rewrite.
- **`docs/LINEAGE.md`.** Written by Agent C, revised by Agent F for the v16
  substitution. Correct it only if you find a factual error.

## You own

Everything not under `src/` or `old_src/`, plus two narrow source edits:

```
pyproject.toml   README.md   scripts/smoke_check.py
docs/{ARCHITECTURE,DATA,EVALUATION,PAPER_CODE_MAP}.md
models/                      (the rename, MANIFEST.json, the three README.md)
tests/                       (reorganization)
src/benchmarks/llm_ranker.py (docstring only — a stale name, see below)
src/clinical_jepa/{evaluate.py,score.py,train/finetune.py}  (checkpoint paths only)
```

Do not otherwise touch `src/`. Do not touch `old_src/` or `baseline/`.

## 1. Packaging

- Rename the distribution `fawkes-three-model-suite` → `clinical-graph-jepa`.
  Update `description` — it still names "Graph-JEPA v5, Graph-JEPA v6, and
  paper-v16", none of which exist now.
- Declare the missing dependencies: `llm = ["openai>=1.0", "python-dotenv>=1.0"]`.
  Both are imported by `benchmarks/llm_ranker.py` and `vs_llm.py` and appear
  nowhere in `pyproject.toml` (plan §2.7).
- Add `[project.scripts]`. The plan lists seven; the last one's module was
  renamed in Phase 6b, so it is `benchmark-vs-fawkes`, not `benchmark-vs-loo`:

  ```
  clinical-jepa-train  clinical-jepa-eval  clinical-jepa-score
  fawkes-train  fawkes-eval  benchmark-vs-llm  benchmark-vs-fawkes
  ```

  Every one must point at a real `main()`. Check each module actually has one.

## 2. The `models/` rename — do this in one commit, carefully

Per §4.4:

```
models/v5_without_note/  -> models/clinical-jepa-no-note/
models/v6_with_note/     -> models/clinical-jepa-localized-note/
models/paper_v16/        -> models/fawkes-entity-note/
config_v5.json           -> config.json           (likewise v6)
config_v5_pretrain.json  -> config_pretrain.json  (likewise v6)
```

`models/fawkes-loo-baseline/` from §4.4 is **not** created — the v12 checkpoint
it was for is no longer used by anything (Phase 6b).

**Checkpoint `.pt` filenames do not change** (§5.1). They are historical
provenance and keeping them makes the rename provably content-neutral.

Use `git mv`. Then update every reference — these are the ones that exist today,
but grep for yourself rather than trusting this list:

```
src/clinical_jepa/evaluate.py:338        --checkpoint default
src/clinical_jepa/train/finetune.py:195  --checkpoint default
src/clinical_jepa/score.py:689-690       help text
tests/conftest.py:31                     requires_checkpoints guard
tests/test_clinical_jepa_score.py:63-64
tests/test_clinical_jepa_core.py:41      comment
scripts/smoke_check.py:40
src/fawkes/README.md                     several
src/benchmarks/vs_fawkes.py              (also names models/paper_v16/)
```

Agent D left `test_checkpoint_defaults_point_at_files_that_exist` specifically to
fail loudly here. If it goes red, that is the safety net working — fix the
defaults, never the test.

### `MANIFEST.json`

Update the paths. **Checksums for v5/v6 do not change** — verify that.

The `paper_v16` entry's `sha256` is *wrong today and deliberately so*. Phase 0
found the file's byte count matches the manifest exactly while its hash does not,
left it untouched as evidence, and proved empirically that the file is
nonetheless the published artifact (reproducing the metrics stored inside it at
delta `0.000e+00`). **Phase 7 is where that gets corrected.** Recompute the hash,
record the real value, and note in `models/fawkes-entity-note/README.md` what
happened and how it was established — a silently corrected checksum destroys the
evidence trail.

## 3. Delete the fossil guard

`scripts/smoke_check.py::check_independence` globs `src/graph_jepa_v5` and
`src/graph_jepa_v6`, both moved to `old_src/` in Phase 0. **It scans zero files
and passes green** — it enforces nothing. `tests/test_import_boundaries.py`
replaced it with a real AST boundary test that asserts on its own walked-file
count. Delete the function and its call site.

## 4. Documentation

Rewrite for the new tree. `README.md` needs its repository tree, the reader
orientation table, and **every example command** updated — the gate is that they
run as written.

`docs/PAPER_CODE_MAP.md`: every `paper_v16` reference becomes `fawkes`, every
`graph_jepa_v5`/`v6` becomes `clinical_jepa` (§1.1).

`docs/EVALUATION.md`: plan §11 item 4 — its results table carries a caveat that
the underlying result files are absent. They exist now, in `baseline/`. Replace
the table with audited numbers, and be precise about which is which:

- `baseline/paper_loo_testsplit.json` — MRR 0.418653, n=8283, the **published**
  number, reproducing the checkpoint's own stored metrics exactly.
- `baseline/paper_loo.json` — MRR 0.440249, n=40000, the shipped evaluator over
  the whole file. **Not** the paper's reported number; do not present it as one.
- `baseline/v{5,6}_loo.json` — MRR 0.605 / 0.621, on *seeded synthetic* graphs,
  not real data. Say so wherever they appear.

**Three findings from Phases 3, 4 and 6b belong in the docs**, because each will
otherwise be rediscovered as a bug:

1. **Training is not bit-reproducible above one thread.** CPU backward reduces
   across intra-op threads in unfixed order (~4.6e-7 drift between two identical
   runs). `torch.set_num_threads(1)` makes it exact. Forward-only evaluation is
   unaffected and reproduces byte-identically.
2. **Scoring is not reproducible run to run.** The patch partition draws from the
   global RNG (`generator=None`), preserved deliberately as behavior rather than
   changed inside a move. Two runs of identical code gave 0.1266 and 0.12737 for
   the same edge. Seed before scoring if you need reproducibility.
3. **The benchmark arms are not paired**, and a relation showing `n=0` for the
   fawkes arm is a filtered-ranking artifact of star topology, not a model
   deficiency. `vs_fawkes.py`'s docstring has the measurements.

## 5. Tests

Expand `tests/test_suite.py` into per-module files. Every differential gate from
Phases 1–6 stays as a permanent regression test — do not drop or weaken one.

Note `test_suite.py::test_v5_v6_have_no_historical_package_imports` is the same
fossil as §3 above and is equally vacuous; the boundary test supersedes it.

## 6. One stale name

`src/benchmarks/llm_ranker.py`'s docstring refers to `vs_loo_baseline.py`, which
was renamed `vs_fawkes.py` in Phase 6b. Docstring only — change nothing else in
that file.

## A decision you must surface, not silently make

`--jsonl-path` defaults to `data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl`,
which **does not exist**. Agent D deliberately left it rather than repoint it at
the 4,000-record embedded dataset, because those are different datasets and
swapping them silently changes what a bare `--data jsonl` run reads.

Recommended: make `--jsonl-path` **required** when `--data jsonl` is selected, so
there is no wrong default. State what you did in your report either way. Do not
repoint it at the embedded dataset without saying so loudly.

## Gate

1. `pip install -e .` succeeds.
2. Every console script runs `--help` successfully.
3. **Every command in `README.md` executes as written.** Run them.
4. `pytest` is green — at least 83, and not by deleting tests.

Note on gate 1: this installs into the shared `~/.venvs/global`. That is
acceptable and reversible, but say so in your report.

## Stop and report

- A console script has no `main()` to point at.
- A README command cannot be made to run without changing behavior.
- The v5/v6 checksums in `MANIFEST.json` do not match after the rename — that
  would mean the rename was not content-neutral, which is a real problem.
- `test_checkpoint_defaults_point_at_files_that_exist` fails and you cannot see
  why from the defaults alone.

## Do not commit

Leave your work in the tree; a human runs the gate and commits.
