# Data contract

## Packaged dataset

`data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl` contains one
graph object per line. Despite the directory name, this local snapshot contains
400 graph records.

Audit result at packaging time:

| Property | Value |
| --- | ---: |
| Graph records | 400 |
| Records with note text | 0 |
| Records with `note_embedding` | 0 |
| LLM-derived edges | 5,494 |
| Edges with a `labels` dictionary | 0 |

Every record contains `subject_id`, `hadm_id`, `nodes`, and `edges`. Nodes use
`id`, `type`, `name`, and `normalized_name`. Edges use `source`, `target`,
`relation`, `confidence`, and `evidence`.

## Compatibility

| Dataset feature | v5 no-note | v6 with-note | paper-v16 with-note |
| --- | --- | --- | --- |
| Raw nodes/edges | Required | Required | Required |
| SapBERT generated at runtime | Required | Required | Not used |
| 768-d note embedding | Not used | Required for faithful evaluation | Required for faithful evaluation |
| Edge `labels.prov_in_note` | Not used | Required for provenance grounding | Required for provenance grounding |
| Edge support labels | Used by confidence fine-tuning | Used by confidence fine-tuning | Used by evidence pruning |

The packaged raw JSONL is therefore directly appropriate for v5. It is useful
for parser/smoke testing of v6 and paper-v16, but zero-filled missing notes do
not reproduce their reported note-augmented performance.

Run `uv run python scripts/audit_data.py --path YOUR.jsonl` before training or
evaluation.

## Privacy

These artifacts are MIMIC-IV-derived and remain subject to the PhysioNet data
use agreement. Do not upload or publish the JSONL, derived embeddings, or model
artifacts without confirming the applicable permissions.
