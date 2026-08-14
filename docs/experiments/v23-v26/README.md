# Clinical Graph-JEPA v23–v26 documentation

This directory contains the complete experiment documentation for the v23–v26
lineage: methodology, implementation process, measured results, execution
commands, and reproduction notes. The HTML reports are self-contained and can
be opened directly from the repository or GitHub.

## Reports

| Version | Scope | Documentation |
| --- | --- | --- |
| v23 | Note-pooling ablation over ten paired seeds: global mean, uniform spans, and entity-conditioned attention | [v23 report](v23/fawkes-v16-to-v23-report.html) |
| v24 | Evaluation-only ACI-Bench transfer study using graph-only, global-note, and entity-grounded-note conditions | [v24 report](v24/v24-aci-evaluation.html) |
| v25 | TF-IDF node-embedding ablation, note-placement ablations, ACI transfer, and reproducibility bundle | [v25 ablation guide](v25/v25-ablations.html), [v25 consolidated report](v25/fawkes-v17-to-v25-report.html) |
| v26 | Twenty-question structured graph-QA evaluation comparing inference-time note removal with entity-grounded notes | [v26 report](v26/fawkes-v17-to-v26-report.html) |

## Reproduction map

- v23 execution protocol: `docs/V23_NOTE_INJECTION.md` and the commands in the v23 HTML report.
- v24 ACI runner: `scripts/evaluate_v25_aci.py` on the v24 checkpoint and ACI input.
- v25 ablations: `scripts/build_tfidf_embeddings.py` and `scripts/evaluate_v25_aci.py`.
- v26 QA evaluation: `scripts/evaluate_v26_graph_qa.py` and `scripts/compare_v26_graph_qa.py`.

The stored JSON files alongside these reports preserve the machine-readable
results. Large datasets and restricted model/data artifacts are intentionally
not duplicated in this documentation bundle; the reports identify their
source repositories, expected paths, and checksums where applicable.
