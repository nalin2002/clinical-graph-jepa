# Coverage record — what Phase 8 could not carry across

Phases 1–7 built the new tree beside the old one and gated every merge
**differentially**: old and new imported into one process, run on one input,
asserted equal. Phase 8 deleted the old tree. This file is the record of what
that cost.

It exists because `docs/RESTRUCTURE_PLAN.md` §11 item 3 recommends keeping
`baseline/` permanently as *"the only remaining evidence that the restructure
preserved the released models' behavior"* — and that recommendation is only worth
anything if the gaps in the evidence are written down next to it.

The suite went from **91 tests to 88**: four test cases retired, one new
parametrization added. Every line of that is below.

---

## How the conversion worked

Plan Phase 8 step 2 says the `old_src`-importing tests can be *"converted to
assert against `baseline/*.json`"*. Most could not, as written: `baseline/` held
LOO metrics, `state_dict` key lists and the paper test-split numbers, while the
gates compared per-graph tensors, scored output JSON, LLM prompt text, fawkes
training-step values and the released CLIs' option sets — none of which existed
as an artifact anywhere.

So the old side was **recorded before it was deleted**, by
`baseline/record_old_pins.py`, into six new artifacts:

| Artifact | Feeds |
| --- | --- |
| `old_clinical_jepa_core.json` | `tests/test_clinical_jepa_core.py` |
| `old_clinical_jepa_score.json` | `tests/test_clinical_jepa_score.py` |
| `old_clinical_jepa_score_output/*.json` (4) | the Phase 4 byte gate |
| `old_clinical_jepa_train.json` | `tests/test_clinical_jepa_train.py` |
| `old_fawkes.json` | `tests/test_fawkes.py` |
| `old_benchmarks.json` | `tests/test_benchmarks.py` |

`baseline/README.md` records the invocation and what is in each.

### Digests, and why they are not a weakening

Where the compared object is too large to track — 256 synthetic graphs each
carrying a 17×768 `x`, or 4,000 records each carrying a `numfeat` of
(nodes × 774) — the
pin is a sha256 over `(dtype, shape, raw bytes)` per field, per item
(`tests/conftest.py::digest_fields`). Three properties matter:

- **It is not a tolerance.** A byte digest is exact. It is in fact *stricter*
  than the `torch.equal` the differential used, because it also requires the NaN
  bit patterns to match — `assert_pyg_equal` had to special-case all-NaN
  `edge_llm_confidence` precisely because `torch.equal` cannot compare it.
- **It is diagnosable.** Per item and per field, never one digest over
  everything, so a failure reads `graph_jepa_v6[137]: ['edge_type'] differ`
  rather than "hash differs".
- **It loses the value.** This is the real cost: a differential failure printed
  the two tensors, and a digest failure prints two hex strings. What moved has to
  be recovered by re-running the new code, not read off the assertion.

---

## Retired test cases (4 node IDs)

### `test_clinical_jepa_core.py::test_v5_used_to_reject_alias_keys`

**Asserted.** `graph_jepa_v5.data.to_graph_data` raises `KeyError` on an
edge keyed `source`/`target`, and the merged converter does not. This pinned
plan §2.6 as a deliberate behaviour change rather than an accident.

**Why it could not survive.** The `KeyError` is a property of a module that no
longer exists. Nothing in `src/` can raise it and nothing can be recorded from
it — an exception type is not an artifact.

**What covers the live half.** `test_alias_keyed_edges_are_normalized`, which
converts the `aliased` and `mixed` fixtures successfully and asserts they produce
the same tensors as the `canonical` one *and* the same tensors old v6 produced.
The half that is gone is the demonstration that v5 used to fail; its docstring
now states it.

### `test_clinical_jepa_score.py::test_old_v6_monkeypatch_made_the_variants_mutually_exclusive`

**Asserted.** `graph_jepa_v6.score._install_v6_data_conversion` rebinds
`to_graph_data` on two `fawkes_core` modules and never restores them, so a
process that had run the localized-note CLI once would feed 1536-wide features to
the no-note variant's 768-wide input projection — `RuntimeError: mat1 and mat2
shapes cannot be multiplied`. Plan §2.4's defect, demonstrated by triggering it.

**Why it could not survive.** It is a test *of the old module*. It needed
`_install_v6_data_conversion` to exist in order to install it, and the failure it
provoked is not an output that can be pinned.

**What covers the live half.** `test_variants_coexist_in_one_process`: score
no-note, then localized-note, then no-note again, and assert the first and third
outputs are byte-identical. That is the positive form of the same property and
needs nothing old. `test_install_v6_data_conversion_is_gone` keeps the function
from coming back.

### `test_clinical_jepa_train.py::test_evaluate_matches_old_evaluator[v5]`, `[v6]`

**Asserted.** `clinical_jepa.evaluate.run()` returns a metric dictionary equal to
`graph_jepa_v{5,6}.evaluate.run()`'s on the same argv, each side parsed by its
own parser.

**Why it could not survive.** It calls `run()` on a module that is gone.

**What covers the live half.** `test_evaluate_reproduces_baseline_payload`, which
was always the other half of this gate and is **strictly stronger**:
`baseline/v{5,6}_loo.json` is what the old evaluator wrote on this exact
invocation (`baseline/README.md` records the command), and the comparison is
byte-for-byte over the written payload — every metric, every per-relation row,
the graph count, the recorded invocation fields and the float formatting, none of
which the returned dictionary contained. Nothing is lost except that the two
parsers no longer parse the same argv side by side; the merged parser still has
to accept the recorded invocation verbatim or the test cannot run at all.

---

## Narrowed test cases

These still run and still assert against the old implementation. What changed is
listed per test; where a row says "digests", see the section above.

| Test | Dropped or weakened | Now covered by |
| --- | --- | --- |
| `core::test_config_matches_old_v5_where_old_v5_could_read_it` | `pytest.raises(TypeError)` on `OldV5Config.from_dict(localized-note config)` — the reason only one sidecar is compared | The pin holds v5's `to_dict` for the no-note file **only**, which states the same asymmetry as an artifact. The field-difference assertion is unchanged. |
| `core::test_jsonl_builder_preserves_graph_metadata` | `extra["_source_path"]` was compared against the old builder's value on the same tmp file | Asserted against *this run's* input path, then substituted for a placeholder before the rest of `extra` is compared. Same guarantee in two steps. |
| `core::test_tensors_match_old_v5_and_v6` | Elementwise tensor equality (512 comparisons) | Per-field digests, 8 per graph per variant |
| `core::test_alias_keyed_edges_are_normalized` | Elementwise equality against live old v6 | Per-field digests. The new-vs-canonical half is still elementwise and live. |
| `core::test_forward_matches_old_model` | Elementwise equality of 6 tensors | Per-field digests. The three loss **log dictionaries** are still compared in full and literally — that is where the positive/negative/exclusion counts live. |
| `score::test_scoring_output_is_byte_identical` | The old side no longer runs in-process, so it can no longer be shown that the two paths agree *given identical RNG consumption in one process* | The four files it wrote are tracked under `old_clinical_jepa_score_output/` and the comparison is still `read_bytes() == read_bytes()`. Both sides seed `torch.manual_seed(20260729)` immediately before scoring, so the RNG state is defined, not shared. |
| `score::test_canonical_input_loads_identically`, `test_relation_keyed_canonical_input_also_stops_being_adapted`, `test_alias_keyed_input_no_longer_routes_through_the_mimic_adapter`, `test_alias_fold_on_the_real_dataset` | `old_v4._load_graph_for_scoring` no longer runs; `old_v3._looks_like_mimic_subkg`'s True result is a recorded boolean rather than a call | The old loader's nodes, edges, extra, provenance stamps, endpoints and `node_encoder_keys()` are pinned per fixture. The new side is still computed live and compared in full. |
| `score::test_unscoreable_graph_still_gets_the_schema_guard` | `old_v4.score_graph` on an `OldPatientGraph` | The old `([0.0], ["inconsistent"])` and the old annotated edge dict are pinned; the new side runs live. |
| `score::test_install_v6_data_conversion_is_gone` | `hasattr(old_v6_score, "_install_v6_data_conversion")` — the proof the assertion was about something | Recorded in the retirement entry above. The surviving half is the one that notices a reintroduction. |
| `score::test_reexport_block_is_gone` | `getattr(score_revision, n) is getattr(score_base, n)` for all 15 names — the proof each really was a re-export | The 15 names are still listed and still checked to be defined in `clinical_jepa.score` itself; `RELATIONS` and `NEGATED_OR_ABSENT_MARKERS` are compared against the old module's values, pinned. |
| `score::test_parser_keeps_every_released_option` | Three old parsers built and compared | `record_old_pins.py` asserted the three option sets were identical and wrote one list. A future divergence between the three is unrepresentable — they are gone. |
| `fawkes::test_config_from_env_matches_trainer_globals` | Read the ambient environment on both sides, so it held for **any** configuration | Two recorded environments instead of one: `defaults` (the released configuration) and `non_default`, which is chosen to exercise the int/float/bool/`lower()`/comma-list parsing rules. This is the one place Phase 8 **added** a test case (+1 node) rather than only converting. It also newly asserts `asdict(Config()) == defaults`, which is the class docstring's claim and was previously unchecked. |
| `fawkes::test_to_data_matches_trainer` | Elementwise equality over 4,000 records × 11 fields; `Config.from_env()` → `Config()` | Per-record folded digests plus one digest per field over all 4,000, so a failure names both the record and the field. `Config()` because the pin is the released configuration; reading the ambient environment would make this fail for a stray `USE_NOTE=0` rather than for a drift. `test_config_from_env_matches_trainer_globals` is what covers `from_env`. |
| `fawkes::test_training_steps_match_trainer` | Elementwise equality; `JEPA(cfg)` was unseeded and its weights copied into the old class | `torch.manual_seed(model_init_seed)` before `JEPA(cfg)`, which is how the recorder produced the weights it then copied into `paper_v16.trainer.JEPA`. Same comparison from a reproducible start. Digests for the six tensors; `qsig` compared as an integer. |
| `fawkes::test_batchmask_cascade_and_eir_match_trainer` | `old.evaluate` / `old.cascade_evaluate` / `old.eir_uplift_eval` no longer run | Their return dictionaries are pinned in full, per-relation rows included, and compared with `==`. No digest involved. |
| `train::test_train_epochs_matches_old_loop` | `GraphJEPA(cfg.model)` was unseeded and copied into `GraphJEPAv5`; parameters compared elementwise | Seeded from `model_init_seed`, which is exactly the initialisation the recorder copied into the old loop. Per-parameter digests over all 119 entries. |
| `benchmarks::test_clinical_jepa_arm_matches_old_pipeline` | The old adapter, old dataset class and old evaluator no longer run | Their metric dictionary over the same 120-record test-split slice is pinned and compared with `==`. The new pipeline still runs end to end. |
| `benchmarks::test_query_sampling_prompts_and_ranks_match_old_evaluate_llm` | The old lineage's graph and dataset construction no longer re-derived alongside the new one | Queries, **full prompt text** (30 prompts, ~29 KB) and ranks are pinned as recorded from `graph_jepa_v6/evaluate_llm.py`. The prompts are text and not digests on purpose: the prompt is the interface to the LLM, so a change to it must be readable in the diff. |
| `benchmarks::test_parse_ranking_matches_old`, `test_summarize_merges_two_identical_copies` | The old parser and the two old `_summarize` copies no longer run | Their outputs on the same six replies / same fixture rows are pinned. The two `_summarize` copies are pinned separately, so the merged one is still checked against both. |
| `benchmarks::test_loo_baseline_architecture_is_gone` | `hasattr(old_three_way, ...)` for `LooEncoder`, `LooDistMult`, `LooMLPScorer`, `LOO_NUMERIC_DIM` | The four names are pinned as a recorded list, so the "gone from `src/`" half is still a claim about something that existed. The AST walk over `src/` is unchanged. |

---

## What no longer has any oracle at all

Two things, both stated so they are not mistaken for coverage:

1. **New behaviour of the old modules.** Anything the pins did not record is
   unrecoverable. If a future change to `src/` needs to know what the old code
   did on an input nobody thought to record, the answer is in
   `git log` before this phase's commit — not in `baseline/`.
2. **`baseline/record_old_pins.py` cannot be re-run against the current tree.**
   It imports `old_src`, which this phase deletes. It is kept as the record of
   *how* each pin was produced — the same role
   `baseline/reproduce_paper_testsplit.py` has played since Phase 0 — not as a
   maintained tool. Regenerating a pin is never a repair for a failing gate; see
   `baseline/README.md`.

   It is, however, **still auditable**, and that is the point of keeping it.
   `old_src` survives in git history, so every pin here can be re-derived and
   checked against the original implementation at any time:

   ```bash
   git archive <commit-before-this-one> old_src | tar -x -C /tmp/oldtree
   cp -R /tmp/oldtree/old_src ./old_src          # the subprocess in
                                                 # _trainer_globals hardcodes
                                                 # ROOT/old_src, so it must sit here
   PYTHONPATH=src:old_src python baseline/record_old_pins.py /tmp/pins
   diff -r baseline/old_clinical_jepa_core.json /tmp/pins/old_clinical_jepa_core.json
   rm -rf ./old_src
   ```

   This was run at review time before the phase was committed: all five JSON
   artifacts and the four byte-pinned score outputs came back byte-identical
   against `old_src` restored from `HEAD`. The pins are therefore falsifiable
   rather than merely asserted, which is what distinguishes them from a number
   somebody typed in.
