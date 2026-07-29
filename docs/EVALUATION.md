# Evaluation

## Leave-one-out edge recovery

For each query, one true edge is removed from message passing. The model ranks
the true target against schema-compatible or same-type candidates. Other known
true targets for the same `(source, relation)` query are filtered out.

- MRR: mean reciprocal rank; higher is better.
- Hits@1/3/10: fraction of true targets ranked within the first k positions.
- `--cap`: maximum edge queries, not maximum graphs.

## The two evaluators

`clinical_jepa.evaluate` (`clinical-jepa-eval`) evaluates either variant of the
modular pipeline; the variant comes from the checkpoint's own config, so one
command reads both released checkpoints. Use `--candidate-mode schema` to match
the ranking fine-tuning objective; `--candidate-mode same-type` is the looser
alternative.

`fawkes.evaluate` (`fawkes-eval`) loads the saved encoder and DistMult scorer
and runs the paper model's leave-one-out implementation. Its environment flags
must match the saved checkpoint, because they determine layer dimensions; `run()`
refuses to proceed when `USE_NOTE` / `GROUND_BY` / `EMBED_DIM` / `USE_SCORES`
disagree with the checkpoint's stored config.

Results from the two lineages are not directly comparable unless the graph
records, split, candidate construction, note availability, and query cap are
identical — which, for these two, they are not. See "Comparing the lineages".

## Audited results

Every row below comes from a committed file in `baseline/`, recorded from the
pre-restructure code. `baseline/README.md` records the exact invocation for each
one, because a metric without its invocation is not reproducible. These files
are the regression oracle for the restructure and are never regenerated.

| Result file | Model | Population | MRR | Hits@1 | Hits@3 | Hits@10 | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `paper_loo_testsplit.json` | Fawkes entity-note | Seeded test split, 400 of 4,000 real admissions | **0.418653** | 0.247495 | 0.473741 | 0.863576 | 8,283 |
| `paper_loo.json` | Fawkes entity-note | Shipped evaluator over the whole 4,000-record file | 0.440249 | 0.263050 | 0.500650 | 0.900450 | 40,000 |
| `v5_loo.json` | Clinical-JEPA, no note | Seeded **synthetic** graphs | 0.605425 | 0.343020 | 0.846724 | 1.000000 | 1,755 |
| `v6_loo.json` | Clinical-JEPA, localized note | Seeded **synthetic** graphs | 0.621116 | 0.365812 | 0.848433 | 1.000000 | 1,755 |

### Which of these is the published number

**`paper_loo_testsplit.json` — MRR 0.418653 over n=8,283.** It mirrors the
trainer's split (`RandomState(seed=42).permutation`, first `test_frac=0.1`, with
the trainer's `>= 4` edge filter) and reproduces the metrics stored *inside* the
released checkpoint to a delta of exactly `0.000e+00` on all four metrics.

**`paper_loo.json` is not.** It is the same checkpoint under the shipped
evaluator, which has no split argument and evaluates whatever file it is given,
whole, under a `>= 2` edge filter. The two populations differ before any split
is applied. It exists to gate the evaluator refactor — old code against new code
on an identical invocation — and must not be quoted as a published result.

### The two synthetic rows

`v5_loo.json` and `v6_loo.json` were recorded on **seeded synthetic graphs, not
patient data**, and say nothing about clinical performance. They exist because
the only real dataset present keys its edges `source`/`target`, which the
released no-note evaluator indexed as `source_id` unguarded and so could not
read; using synthetic graphs gave both variants one shared input population.
Synthetic generation is deterministic in `TrainConfig.seed`, and a second run of
the recorded command produced a byte-identical file.

Wherever these two numbers appear, they must be labelled synthetic.

## Comparing the lineages

`benchmark-vs-fawkes` runs both lineages over the same admissions. It does not
make them comparable in the strict sense, and it does not pretend to:

- Each arm runs **its own pipeline**. `fawkes` has its own `to_data`, its own
  node and relation vocabularies, and its own edge pruning; `clinical_jepa`
  keeps every edge whose relation its schema recognizes and then filters
  candidates through typed-schema and LLM-confidence masks.
- The table therefore aligns on **relation name only**, and every row carries
  **its own** `n`, `C` and chance baseline. A single shared `n` would mislabel
  at least one arm.
- A relation showing **`n=0` for one arm is an artifact, not a deficiency**.
  `TAKES_MEDICATION` reports `n=0` for the `fawkes` arm because of filtered
  ranking under star topology in that arm's candidate construction — the
  relation is present in the data and ranked by the other arm.

## Reproducibility

Three measured facts about this codebase:

1. **Training is not bit-reproducible above one thread.** CPU backward reduces
   across intra-op threads in an unfixed order; two identical runs drift about
   `4.6e-7` on the parameters after one epoch. `torch.set_num_threads(1)` makes
   it exact — which is why the training gate tightens to equality rather than
   relaxing to a tolerance. Forward-only evaluation is unaffected and reproduces
   byte-identically at the default thread count.
2. **Scoring is not reproducible run to run.** The patch partition draws from
   the global RNG (`generator=None`), so the structural half of every score
   depends on how much randomness the process consumed earlier. Two runs of
   identical code gave `0.1266` and `0.12737` for the same edge. This is
   preserved as behaviour rather than changed inside a move; seed before scoring
   if you need reproducibility.
3. **The benchmark arms are not paired.** See above.

For a reproducible result, retain the evaluator's output JSON together with:

- checkpoint checksum (`models/MANIFEST.json`);
- dataset checksum and split;
- note/provenance availability;
- candidate mode and filtering rules;
- random seed and query cap; and
- command line plus environment flags.
