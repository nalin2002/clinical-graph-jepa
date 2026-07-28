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
