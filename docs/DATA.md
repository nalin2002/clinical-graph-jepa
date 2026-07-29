# Data contract

## The dataset in this working tree

`data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl`
contains one graph object per line. It is 234 MB and is **not committed** —
`.gitignore` excludes `data/**/*.jsonl` under the PhysioNet data use agreement.
`_download_manifest.json` in the same directory records its source
(`wmatbooth/fawkes-training-graph-embedded-260615`) and byte count.

Audit result, from `python scripts/audit_data.py`:

| Property | Value |
| --- | ---: |
| Graph records | 4,000 |
| Nodes | 186,334 |
| Edges | 267,952 |
| Records with note text | 4,000 |
| Records with `note_embedding` | 4,000 |
| LLM-derived edges | 85,290 |
| Edges with a `labels` dictionary | 267,952 |
| Edges with `labels.prov_in_note` | 52,670 |

Node types: `PATIENT` (4,000), `MEDICATION` (107,623), `DIAGNOSIS` (63,749),
`PROCEDURE` (5,469), `SERVICE` (4,270), `MICROBIOLOGY` (1,223).

Relations: `TAKES_MEDICATION` (107,623), `HAS_DIAGNOSIS` (63,749),
`MANAGED_FOR` (56,294), `INDICATES` (15,353), `COMPLICATED_BY` (12,081),
`UNDERWENT_PROCEDURE` (5,469), `MANAGED_BY_SERVICE` (4,270), `CONFIRMS` (1,562),
`HAD_MICROBIOLOGY` (1,223), `TARGETS_ORGANISM` (328).

All 267,952 relation instances and all six node types map into
`clinical_jepa`'s vocabularies, so no edge is dropped to make a cross-lineage
comparison work — `tests/test_benchmarks.py` measures this rather than assuming
it.

Each record carries `subject_id`, `hadm_id`, `nodes`, `edges`, `note`,
`note_embedding` (768-d), and demographic fields. Nodes use `id`, `type`,
`name`, `normalized_name`. Edges use `source`, `target`, `relation`,
`confidence`, `evidence`, and a `labels` dictionary that includes
`prov_in_note`.

> [!NOTE]
> Edges are keyed `source`/`target`, **not** `source_id`/`target_id`. The
> released no-note evaluator indexed `edge["source_id"]` unguarded and raised
> `KeyError` on this file; the merged `clinical_jepa` normalizes both spellings
> through one endpoint normalizer, so both variants read it. This is a
> deliberate behaviour change, not a side effect of a move.

## The dataset that is not here

`models/MANIFEST.json` records a checksum for a second file,
`data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl` — 400 admission
graphs, 5,420,960 bytes, sha256 `d343446917cb…`. **It is absent from this
working tree.** Its source is the Hugging Face dataset
`wmatbooth/fawkes-1k-patients-graphs-260615`. The manifest entry is retained so
a copy can be verified if one is obtained.

It contains graph topology but no note text, note embeddings, or
evidence-score vectors, so it can faithfully exercise only the no-note model and
`fawkes` Option A.

**It is not recoverable from that repository.** The Hugging Face dataset holds a
file of the same name whose sha256 is `967ad6cc…` at 22,155,648 bytes — a
different artifact, four times the size. The manifest entry is retained as the
only record of the original, not as an invitation to fetch a substitute.

> [!IMPORTANT]
> `--jsonl-path` **used to default to this absent file** and now defaults to the
> embedded dataset above. A bare `--data jsonl` therefore reads 4,000 MIMIC
> admissions (~1.5 s, ~680 MB peak) instead of failing with "JSONL graph file
> not found".
>
> These are **different datasets**, so any recorded result that predates this
> change and does not name its `--jsonl-path` is ambiguous. Nothing in
> `baseline/` is affected — every `clinical_jepa` gate uses seeded synthetic
> graphs and reads no data file at all.
>
> The embedded dataset carries note text, 768-d note embeddings and
> `labels.prov_in_note`, so unlike the file it replaces it can faithfully
> exercise the localized-note variant as well as the no-note one.

## Compatibility

| Dataset feature | No note | Localized note | Fawkes entity-note |
| --- | --- | --- | --- |
| Raw nodes/edges | Required | Required | Required |
| SapBERT generated at runtime | Required | Required | Not used |
| 768-d note embedding | Not used | Required for faithful evaluation | Required for faithful evaluation |
| Edge `labels.prov_in_note` | Not used | Required for provenance grounding | Required for provenance grounding |
| Edge support labels | Used by confidence fine-tuning | Used by confidence fine-tuning | Used by evidence pruning |

The embedded dataset satisfies every row, which is why `fawkes-eval` reproduces
the published metrics from it exactly.

Run `python scripts/audit_data.py --path YOUR.jsonl` before training or
evaluation. With no `--path` it audits the embedded dataset above.

## Privacy

These artifacts are MIMIC-IV-derived and remain subject to the PhysioNet data
use agreement. Do not upload or publish the JSONL, derived embeddings, or model
artifacts without confirming the applicable permissions. `.gitignore` excludes
`data/**/*.jsonl`, `data/**/*.json`, and `*.pt` for this reason.
