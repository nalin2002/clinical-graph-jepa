# Evaluation

## Leave-one-out edge recovery

For each query, one true edge is removed from message passing. The model ranks
the true target against schema-compatible or same-type candidates. Other known
true targets for the same `(source, relation)` query are filtered out.

- MRR: mean reciprocal rank; higher is better.
- Hits@1/3/10: fraction of true targets ranked within the first k positions.
- `--cap`: maximum edge queries, not maximum graphs.

## Modular v5/v6 evaluators

`graph_jepa_v5.evaluate` and `graph_jepa_v6.evaluate` evaluate their native
checkpoint architecture. Use `--candidate-mode schema` to match the modular
hard-negative training objective. The `evaluate_llm.py` and
`evaluate_loo_v12_jepa_llm.py` modules are also retained for LLM-assisted and
paper-v12-compatible workflows.

## Paper-v16 evaluator

`paper_v16.evaluate` loads the saved encoder and DistMult scorer and runs the
original standalone leave-one-out implementation. Its environment flags must
match the saved checkpoint because they determine layer dimensions.

Results from the three models are not directly comparable unless the graph
records, split, candidate construction, note availability, and query cap are
identical.

## Project-reported snapshot

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Modular v6 | 0.865 | 0.779 | 0.950 | 1.000 |
| Paper-v16 | 0.571 | 0.429 | 0.626 | 0.872 |

These values were supplied from the project evaluation and are recorded here
separately from the metrics printed in the paper. The underlying result files
were not present when this table was added, so the dataset path, split, query
count, candidate mode, and runtime configuration cannot yet be audited from the
repository.

For a reproducible result, retain the evaluator's output JSON together with:

- checkpoint checksum;
- dataset checksum and split;
- note/provenance availability;
- candidate mode and filtering rules;
- random seed and query cap; and
- command line plus environment flags.
