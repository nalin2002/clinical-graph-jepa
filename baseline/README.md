# Baseline — the regression oracle

Everything here was produced by the **pre-restructure** code, in two rounds:

- **Phase 0** froze the metrics and `state_dict` key lists, before any code moved.
- **Phase 8** recorded everything else, in the last hour the old tree existed —
  the tensors, scored output JSON, LLM prompts, training-step values and CLI
  option sets that the differential gates had been comparing in-process. Those
  are the `old_*` files, written by `record_old_pins.py`.

After Phase 8 there is no other evidence that the restructure preserved the
released models' behaviour, which is why plan §11 item 3 recommends keeping this
directory permanently. `COVERAGE.md` is the companion record: every gate that
lost coverage when the old tree went, and what covers the live half now.

A metric without its invocation is not reproducible, so each command below is
recorded verbatim. If a gate fails, the correct response is to find out why —
**never to regenerate these files.** `record_old_pins.py` cannot in any case be
re-run: it imports `old_src`.

## Environment

Recorded 2026-07-29 in `~/.venvs/global` (the `vglobal` alias), Python 3.12.12:

| Package | Version |
| --- | --- |
| torch | 2.12.1 |
| torch-geometric | 2.8.0.post1 |
| transformers | 5.6.2 |
| numpy | 2.5.0 |
| scikit-learn | 1.9.0 |
| pytest | 9.1.1 |

`pyproject.toml` pins none of these beyond lower bounds, and `uv.lock` was not
used (uv is not installed). **transformers 5.6.2 and numpy 2.5.0 are major
versions ahead of the declared floors** (`>=4.44`, `>=1.26`). The three shipped
tests pass and all baselines below reproduce, but if a future gate fails
inexplicably, an environment drift is the first thing to check.

## Pre-existing test state

`PYTHONPATH=src python -m pytest` — **3 passed**, before any restructuring. No
pre-existing failures. (`src/` was not on the path by default; there is no
`pythonpath` in `[tool.pytest.ini_options]` until Phase 0 adds it.)

## File inventory

| File | Recorded | What it holds |
| --- | --- | --- |
| `v5_loo.json`, `v6_loo.json` | Phase 0 | the two released evaluators' LOO payloads, byte-reproducible |
| `paper_loo.json` | Phase 0 | the shipped paper evaluator's output over the whole file |
| `paper_loo_testsplit.json` | Phase 0 | the published test-split number; reproduces the checkpoint's own metrics exactly |
| `v5_keys.json`, `v6_keys.json`, `paper_keys.json` | Phase 0 | `sorted(state_dict().keys())` |
| `reproduce_paper_testsplit.py` | Phase 0 | the one-off that produced `paper_loo_testsplit.json` |
| `old_clinical_jepa_core.json` | Phase 8 | 512 per-graph tensor digests, the alias fixture, both old `Config.to_dict`s, the old JSONL builder's record, the forward pass and three loss logs per variant |
| `old_clinical_jepa_score.json` | Phase 8 | revision counts, four `_load_graph_for_scoring` outputs, the schema guard, `RELATIONS`/`NEGATED_OR_ABSENT_MARKERS`, the released CLI option set |
| `old_clinical_jepa_score_output/` | Phase 8 | the four scored JSON files the old scorer wrote — the Phase 4 byte gate |
| `old_clinical_jepa_train.json` | Phase 8 | `train_epochs`' parameters after one epoch of each stage, from a seeded initialisation |
| `old_fawkes.json` | Phase 8 | the trainer's 47 module globals in two environments, `jepa_step`/`readout_step`, the three evaluators, and `to_data` over all 4,000 records |
| `old_benchmarks.json` | Phase 8 | the old `clinical_jepa` arm's metrics, 30 sampled queries with full prompt text and ranks, the reply parser, both `_summarize` copies |

## Artifact inventory

| Artifact | Status |
| --- | --- |
| `models/v5_without_note/*.pt` | present; sha256 matches `MANIFEST.json` |
| `models/v6_with_note/*.pt` | present; sha256 matches `MANIFEST.json` |
| `models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt` | present; **sha256 differs from `MANIFEST.json`** — see below |
| `data/fawkes-training-graph-embedded-260615/…jsonl` | present, 4,000 records |
| `data/fawkes_1k_patients/…jsonl` | **absent** — the `--jsonl-path` default at `fawkes_core/training.py:306` |
| v12 LOO checkpoint | out of scope; three-way comparison deferred |

### The paper checkpoint hash

`MANIFEST.json` records `fc8c494a…`; the file on disk hashes `6c21abb2…`, with a
byte count matching the manifest exactly (5,204,898). All four v5/v6 hashes
match, so the manifest is not generally stale.

**This was resolved empirically: the file is the published artifact.** Re-running
the trainer's test-split evaluation reproduces the metrics stored *inside* the
checkpoint to a delta of exactly `0.000e+00` on all four metrics, with `n=8283`
matching (see `paper_loo_testsplit.json`). The hash difference is a `torch.save`
re-serialization, not different weights.

`MANIFEST.json` is deliberately **not** updated here — that is Phase 7. Editing
it now would destroy the evidence.

## Files

### The `old_*` files — Phase 8

```
PYTHONPATH=src:old_src python baseline/record_old_pins.py
```

One command writes all six artifacts plus `old_clinical_jepa_score_output/`. The
script's docstring records what each group is and which gate reads it; the
seeds, thresholds and fixtures are the ones the gates use, duplicated in the
script so it stands on its own.

**Verification, and why the recording direction matters.** Every value is
computed by `old_src`; the new tree is imported only to build inputs and starting
points the old tree could not produce on its own (the seeded synthetic
population, the two randomly-initialised models whose weights the gates copied
from new to old before training, and `benchmarks`' test-split record selection).
The recording is deterministic, so agreement with live `old_src` was checked by
re-recording into a scratch directory and diffing:

```
PYTHONPATH=src:old_src python baseline/record_old_pins.py /tmp/pins
diff -r baseline /tmp/pins        # empty
```

That check was run while `old_src` still existed and cannot be run again. The
suite passing with `pythonpath = ["src", "old_src"]` and then with
`pythonpath = ["src"]` alone, at the same test count, is the other half of the
Phase 8 gate.

**One value could not be pinned literally.** The old `JsonlGraphBuilder` stamps
`extra["_source_path"]` with the file it read, which is a temporary directory.
The pin holds `"<input>:1"`; the gate asserts the real value against its own
input path and substitutes the same placeholder before comparing the rest of
`extra`.

### `v5_loo.json`, `v6_loo.json`

```
PYTHONPATH=src python -m graph_jepa_v5.evaluate \
  --data synthetic --synthetic-graphs 256 --synthetic-min-nodes 8 --synthetic-max-nodes 28 \
  --checkpoint models/v5_without_note/graph_jepa_v5.pt \
  --encoder-cache .cache/graph_jepa_v5/encoder \
  --device cpu --cap 40000 --candidate-mode schema --start-graph 0 \
  --output baseline/v5_loo.json
```

v6 is identical with `graph_jepa_v6`, `models/v6_with_note/graph_jepa_v6.pt`, and
`.cache/graph_jepa_v6/encoder`.

Results: v5 MRR=0.605 H@1=0.343, v6 MRR=0.621 H@1=0.366, both over 1,755 edges
from 256 graphs.

**Why synthetic and not real data.** The only dataset present keys its edges
`source`/`target`; `graph_jepa_v5/data.py:75` indexes `edge["source_id"]`
unguarded and raises `KeyError`. v6 normalizes (`data.py:105`) and would run, but
both use synthetic so the two baselines share an input population and Phase 3
gates against one mode. Synthetic is deterministic — `training.py:39` passes
`seed=cfg.train.seed`, default `0`.

**Determinism verified:** a second v5 run with the identical command produced a
byte-identical JSON.

### `paper_loo.json` — the shipped evaluator's output

```
USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 PRUNE_NO_EVIDENCE=1 \
PYTHONPATH=src python -m paper_v16.evaluate \
  --checkpoint models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt \
  --data data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl \
  --device cpu --output baseline/paper_loo.json
```

MRR=0.440249, H@1=0.263050, n=40000 (the `--cap` ceiling), over the whole file.

**This is not the paper's reported number** and must not be quoted as one. It
gates the Phase 5 refactor of `paper_v16/evaluate.py` — old code against new code
on an identical invocation — and nothing more.

### `paper_loo_testsplit.json` — the published number

MRR=0.418653448, H@1=0.247494869, H@3=0.473741398, H@10=0.863575999, n=8283.
Reproduces `checkpoint["recovery_test_loo"]` exactly (delta `0.000e+00`).

Generated by `reproduce_paper_testsplit.py`, which mirrors `trainer.py:600-610`:
`RandomState(SEED=42).permutation`, first `TEST_FRAC=0.1` as test → **400 graphs**
of 4,000. That 400 is the origin of the "all 400 records" figure in the plan's
Phase 1 gate.

Two reasons the shipped `paper_v16/evaluate.py` cannot produce this:

1. It has **no split argument** — it evaluates whatever file it is given, whole.
2. Its graph filter is `edge_index.size(1) >= 2` (`evaluate.py:61`) where the
   trainer uses `>= 4` (`trainer.py:601`), so the populations differ before the
   split is even applied.

Phase 5 should gate on **both** files: `paper_loo.json` proves the evaluator
refactor is behaviour-preserving, `paper_loo_testsplit.json` proves the model and
data path still reproduce the paper.

### `v5_keys.json`, `v6_keys.json`, `paper_keys.json`

`sorted(state_dict().keys())`. v5/v6 dump the single `state_dict`; paper dumps
`encoder` (31 keys) and `scorer` (1 key) separately, matching its checkpoint
layout.

**`v5_keys.json` and `v6_keys.json` are identical — 119 keys each.** The entire
architectural difference between the variants is two tensor *shapes*:

```
context_node_encoder.input_proj.weight   v5=(160, 768)   v6=(160, 1536)
target_node_encoder.input_proj.weight    v5=(160, 768)   v6=(160, 1536)
```

This matters for Phase 2. Gate 1 (key-set equality) catches a renamed attribute
path, but it is **blind to the v5/v6 variant difference** — a merged `GraphJEPA`
that ignored `use_note_embeddings` entirely would still pass it for both configs.
Gate 2 (`load_state_dict(strict=True)`) is what catches that, because it checks
shapes. Treat gate 2 as load-bearing, not as a formality after gate 1.
