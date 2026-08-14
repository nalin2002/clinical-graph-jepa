# v26 structured graph-QA evaluation

v26 evaluates 20 fixed questions with two inference conditions on the same
v24/v25 checkpoint:

- Option A: inference-time note removal (the 768-d note channel is zeroed).
- Option B: checkpoint default with entity-grounded Clinical-ModernBERT note features.

This is not a separately retrained no-note model. The exact question set,
per-question outputs, and paired comparison are stored in this directory.

Aggregate results:

| Metric | Option A | Option B | B - A |
| --- | ---: | ---: | ---: |
| Macro group F1, all 20 | 0.7056 | 0.7411 | +0.0356 |
| Question exact match | 0.6500 | 0.6500 | +0.0000 |
| Edge Hits@1 | 0.3639 | 0.3806 | +0.0167 |
| Edge Hits@3 | 0.5281 | 0.5911 | +0.0631 |
| Edge Hits@10 | 0.9136 | 0.9500 | +0.0364 |
| Macro group F1, multi-hop n=7 | 0.1587 | 0.2603 | +0.1016 |

Run `scripts/evaluate_v26_graph_qa.py` twice with and without
`--disable-note`, then run `scripts/compare_v26_graph_qa.py`. The 4,000-graph
input is downloaded from `wmatbooth/fawkes-training-graph-embedded-260615`;
the HTML report contains the exact commands.
