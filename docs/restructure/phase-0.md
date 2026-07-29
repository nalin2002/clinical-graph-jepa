# Wave 0 — Phase 0: freeze the baseline and split the tree

Solo. Nothing else runs until this passes its gate.

## Read first

- `docs/RESTRUCTURE_PLAN.md` — §5, §6, §10, §11 and Phase 0 in full.
- `pyproject.toml`, `tests/test_suite.py`, `models/MANIFEST.json`.

## You own

```
baseline/                 (new)
tests/conftest.py         (new)
pyproject.toml            (pythonpath only — no other key)
src/ -> old_src/          (the git mv)
```

Do not create anything under the new `src/`. Agents A, B and C own that and
start after your gate passes.

## Step 0 — build a working environment

There is no `.venv`, `uv` is not installed, and no interpreter on this machine
has torch. **Nothing in this repo runs right now.** Every later step depends on
fixing that, so do it first and confirm it.

`uv.lock` is committed, so `uv sync` is the intended path. If `uv` is
unavailable, `python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]"`
reproduces it from `pyproject.toml` (unpinned — note that in `baseline/README.md`,
because an unpinned env means the baseline numbers are reproducible only against
the versions you record).

Confirm before continuing, and record the resolved versions:

```
python -c "import torch, torch_geometric, transformers"
python -m pytest --version
```

## Step 1 — measure the pre-state, before touching anything

Run `pytest` on the tree as it stands and record the result verbatim in
`baseline/README.md`.

The plan's gate says "all three current tests green." That is an assumption, not
a measurement — nothing has been runnable to check it. If a test is already
failing, that is a pre-existing failure: record it, do not fix it, and do not let
it be attributed to the move.

## Step 2 — inventory what is actually present

Verified state as of this brief:

| Artifact | Status |
| --- | --- |
| `models/v5_without_note/*.pt` (2 files) | present; sha256 matches `MANIFEST.json` |
| `models/v6_with_note/*.pt` (2 files) | present; sha256 matches `MANIFEST.json` |
| `models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt` | present; **sha256 does NOT match `MANIFEST.json`** |
| `data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl` | present, 4,000 records |
| `data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl` | **absent** — this is the `--jsonl-path` default at `fawkes_core/training.py:306` |
| v12 LOO checkpoint | out of scope; the three-way comparison is deferred |

On the paper checkpoint: byte count matches the manifest exactly (5,204,898),
sha256 does not. All four v5/v6 hashes match, so the manifest is not generally
stale. Most likely a `torch.save` re-serialization of identical weights, but that
is unproven. Record the observed hash in `baseline/README.md`. **Do not edit
`MANIFEST.json`** — that is Phase 7's job, and changing it here would destroy the
evidence.

This does not weaken any gate: Phase 5 compares `old_src/paper_v16` against
`src/fawkes` on *this same file*, so it gates correctly regardless of provenance.
What it does mean is that `baseline/paper_loo.json` may not match the numbers
published in the paper. Say so where you record it.

## Step 3 — record the baselines

`baseline/v5_loo.json`, `baseline/v6_loo.json` — run with `--data synthetic`.

Synthetic is deterministic: `fawkes_core/training.py:39` passes
`seed=cfg.train.seed`, default `0` (`config.py:51`). Use the default
`--synthetic-graphs 256` and pin every argument explicitly on the command line
rather than relying on defaults, since Phase 3 must reproduce this exactly.

Real data is not an option for v5. The only dataset present keys its edges
`source`/`target`; `graph_jepa_v5/data.py:75` indexes `edge["source_id"]`
unguarded and raises `KeyError`. v6 normalizes (`data.py:105`) and would work,
but use synthetic for both so the two baselines are comparable and Phase 3 gates
against one input mode.

`baseline/paper_loo.json` — run against the real dataset. It is the paper
trainer's native input (`trainer.py:119-120` defaults to exactly this repo and
file), and `paper_v16/evaluate.py:94` takes `--data` as a required path.

## Step 4 — dump state_dict keys

`sorted(model.state_dict().keys())` for each checkpoint, to
`baseline/{v5,v6,paper}_keys.json`. These are what Phase 2 gate 1 and Phase 5
gate 2 compare against, and they are the single most load-bearing artifact you
produce — a renamed submodule attribute shows up here and nowhere else until
inference silently degrades.

## Step 5 — `baseline/README.md`

For each of the six files: the exact command line, the environment variables, the
resolved versions of torch / torch-geometric / transformers, and the input mode.
A metric without its invocation is not reproducible.

Also record, explicitly: the paper checkpoint hash discrepancy, the absent
`fawkes_1k_patients` dataset, why v5/v6 use synthetic, and the pre-existing test
result from Step 1.

## Step 6 — split the tree

`git mv src old_src`, then create `src/` with a `.gitkeep` (git does not track
empty directories).

## Step 7 — test harness

In `pyproject.toml`, add to the existing `[tool.pytest.ini_options]`:

```toml
pythonpath = ["src", "old_src"]   # old_src removed in Phase 8
```

Change no other key. The distribution rename, `[project.scripts]`, and the new
`llm` extra are Phase 7.

Write `tests/conftest.py` per plan §6.2, with two corrections:

1. The plan's `requires_checkpoints` is broken. `Path.glob()` returns a
   generator, which is always truthy, so `not (...).glob("*.pt")` is always
   `False` and the skip never fires. Use `not any((...).glob("*.pt"))`.
2. It points at `models/clinical-jepa-no-note`, which does not exist until
   Phase 7 renames the directories. Point it at `models/v5_without_note` and
   leave a comment that Phase 7 updates it.

Its `DATA` constant points at the absent `fawkes_1k_patients` file. Point it at
the real embedded dataset — Track B needs it. Track A's gates use synthetic and
need no data file at all.

## Gate

1. `pytest` passes, with the Step 1 result reproduced — same tests green, same
   pre-existing failures if any, no new ones.
2. All six `baseline/*.json` exist, plus `baseline/README.md`.
3. `old_src/` is importable and the new `src/` is empty.

## Stop and report

- The environment cannot be built. Everything downstream is blocked; this is not
  something to work around.
- A test that the plan expects green is red. Report it; do not fix it.
- Any baseline run is non-deterministic across two invocations. The entire gate
  strategy assumes reproducibility — if synthetic output varies, stop.

## Commit

You are the one wave that does commit, because you are solo and the tree move
must land atomically. One commit for the move and harness, one for `baseline/`.
Later waves leave their work uncommitted for a human wave barrier.
