# Agent H — Phase 8: retire `old_src`

Runs alone. Phases 0–7 are committed; the suite is green at 91.

This is the last phase, and it is the only one that **destroys** the oracle every
prior gate was written against. Order matters more here than anywhere else.

## Read first

- `docs/RESTRUCTURE_PLAN.md` — Phase 8, decision 4, §6, §11 item 3.
- `baseline/README.md` — what is pinned today, and what is not.
- All six test files under `tests/`.

## The problem the plan understates

Plan Phase 8 step 2 says: *"Differential tests that import `old_src` now fail —
convert each to assert against `baseline/*.json` instead, which is exactly what
those files are for."*

`baseline/` currently holds **only** LOO metrics, `state_dict` key lists, and the
paper test-split metrics. **24 tests across six files** import `old_src`, and most
check things baseline does not contain — per-graph tensors, scored output JSON,
LLM prompt text and ranks, fawkes training-step values, old parser option sets.

So conversion is not a rewrite of assertions. For most of these tests, **the thing
they compare against does not exist yet and must be recorded before `old_src`
goes away.**

## Order of work — do not deviate

1. **Extend `baseline/` from `old_src`, while it still exists.**
2. Convert each test to assert against the extended baseline.
3. Confirm the whole suite passes *with* `old_src` still on `pythonpath`.
4. Only then drop `old_src` from `pythonpath` and confirm again.
5. Only then `git rm -r old_src`.

Steps 3 and 4 are separate on purpose. A test that silently stopped asserting
anything will still pass at step 5; it is step 3-vs-4 agreement that catches it.

## You own

```
baseline/            (new pinned artifacts + README updates)
tests/               (all files)
pyproject.toml       (pythonpath only)
old_src/             (deletion, last)
docs/LINEAGE.md      (the coverage record, see below)
```

Do not touch anything under `src/`. If a conversion seems to require a source
change, stop and report — that would mean Phase 8 is changing behavior, which it
must not.

## The test taxonomy, and what to do with each

Classify every one of the 24 before touching any. Three kinds:

### (a) Pins current behavior, baseline can hold it → **pin, then convert**

The valuable ones. Record the old side's output now as a tracked artifact, then
assert the new code against that file forever.

Prefer compact artifacts. A stable digest over a large structure is fine and
often better than megabytes of JSON — but the digest must be **reproducible and
diagnosable**: if it mismatches, the failure message has to say more than "hash
differs". Pin per-item digests, not one digest over everything.

Known members: `test_tensors_match_old_v5_and_v6` (512 comparisons),
`test_scoring_output_is_byte_identical` (2 variants × 2 paths — this is Phase 4's
entire gate and is cheap to pin as 4 small JSON files),
`test_config_from_dict_matches_old_v6`, `test_train_epochs_matches_old_loop`,
`test_to_data_matches_trainer`, `test_training_steps_match_trainer`,
`test_batchmask_cascade_and_eir_match_trainer`,
`test_query_sampling_prompts_and_ranks_match_old_evaluate_llm`,
`test_clinical_jepa_arm_matches_old_pipeline`,
`test_parser_keeps_every_released_option` (pin the three old parsers' option sets
as a list).

Note `test_train_epochs_matches_old_loop` trains, so it needs
`torch.set_num_threads(1)` to be exact — Phase 3 measured ~4.6e-7 drift otherwise.
Whatever you pin must be recorded under the same pin.

### (b) Documents what `old_src` used to do → **keep the half that still means something**

These exist to prove a migration decision was deliberate. Once `old_src` is gone,
the historical half is unassertable. For each, ask: *is there a property of the
new code this was really testing?*

- `test_v5_used_to_reject_alias_keys` — the live property is "the new loader
  accepts both spellings", which is already tested. The old `KeyError` becomes a
  docstring statement, not an assertion.
- `test_old_v6_monkeypatch_made_the_variants_mutually_exclusive` — the live
  property is `test_variants_coexist_in_one_process`, which exists and does not
  need `old_src`. Retire the historical half.
- `test_install_v6_data_conversion_is_gone` / `test_reexport_block_is_gone` — the
  "absent from new" half survives; the "present in old" half cannot. Keep the
  half that still runs and say in the docstring what the other half established.
- `test_canonical_input_loads_identically`,
  `test_relation_keyed_canonical_input_also_stops_being_adapted`,
  `test_alias_keyed_input_no_longer_routes_through_the_mimic_adapter`,
  `test_alias_fold_on_the_real_dataset` — these pin the §2.6 behavior change. The
  *new* side's output is worth pinning under (a); the old side's is history.
- `test_unscoreable_graph_still_gets_the_schema_guard` — pin the new behavior;
  the guard property is real and must not regress.

**Do not silently delete a test.** Every retirement is recorded (see below).

### (c) Incidental reference, not a real dependency → **just fix the reference**

`test_import_boundaries.py::test_import_root_extraction_works` uses
`fawkes_core` only as a sample string in a tmp_path fixture. `test_benchmarks.py
::test_checkpoint_defaults_point_at_files_that_exist` may only mention a path.
Check before assuming either way.

## Coverage record — a deliverable, not a footnote

Add a section to `docs/LINEAGE.md` (or a new `baseline/COVERAGE.md`, your call)
listing **every test retired or narrowed in this phase**, and for each: what it
asserted, why it could not survive `old_src`'s deletion, and what now covers the
live half. This is the only durable evidence that the restructure was verified
against the original implementation, and plan §11 item 3's recommendation to keep
`baseline/` permanently rests on it.

Also update `baseline/README.md` for the new artifacts, with the invocation that
produced each — same standard as Phase 0.

## Gate

1. Suite green **with** `pythonpath = ["src", "old_src"]` — the pinned artifacts
   agree with the live old code. This is what proves the pins are correct.
2. Suite green with `pythonpath = ["src"]` only, **same test count and same
   passing set** as step 1 minus exactly the retirements you recorded. A drop you
   cannot account for line by line is a failure.
3. `old_src/` deleted; no reference to it remains anywhere (grep `src/`, `tests/`,
   `scripts/`, `docs/`, `pyproject.toml`).
4. **Checkpoint-dependent tests skip rather than fail when checkpoints are
   absent.** `.gitignore` excludes `*.pt`, so this is the clean-clone contract and
   Phase 8 is the first time it is actually exercised. Verify it properly — e.g.
   temporarily move `models/*/[!R]*.pt` aside, run, confirm skips not errors, and
   put them back. Report the skip count.
5. `pip install -e .` still succeeds and every console script still runs `--help`.

## Stop and report

- A pinned artifact does not reproduce against live `old_src` at step 1. Never
  adjust the pin to match; find out why.
- A conversion would need a change under `src/`.
- The test count drops by more than your recorded retirements.
- Gate 4 produces errors instead of skips — that is a real clean-clone bug and it
  is better found now than by a stranger cloning the repo.

## Do not commit

Leave everything in the tree, `old_src` deletion included, staged or unstaged as
git prefers. A human reviews the coverage record and commits.
