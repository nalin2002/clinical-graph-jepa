"""Create paired QA comparisons from evaluator JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_ci(delta, seed=42, draws=20000):
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(delta, dtype=float), size=(draws, len(delta)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def compare(default_path: Path, ablated_path: Path):
    default = json.loads(default_path.read_text(encoding="utf-8"))
    ablated = json.loads(ablated_path.read_text(encoding="utf-8"))
    a = {x["id"]: x for x in default["items"]}
    b = {x["id"]: x for x in ablated["items"]}
    ids = sorted(set(a) & set(b))
    f1_delta = [a[i]["group_f1"] - b[i]["group_f1"] for i in ids]
    exact_delta = [float(a[i]["exact_match"]) - float(b[i]["exact_match"]) for i in ids]
    multi = {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}
    mid = [i for i in ids if a[i]["template"] in multi]
    mf1_delta = [a[i]["group_f1"] - b[i]["group_f1"] for i in mid]
    mexact_delta = [float(a[i]["exact_match"]) - float(b[i]["exact_match"]) for i in mid]
    def metric_delta(field, ids_):
        values = [float(a[i][field]) - float(b[i][field]) for i in ids_]
        return {"default": float(np.mean([a[i][field] for i in ids_])), "ablation": float(np.mean([b[i][field] for i in ids_])), "delta": float(np.mean(values)), "ci95": bootstrap_ci(values), "wins": int(sum(x > 0 for x in values))}
    return {
        "default": str(default_path),
        "ablation": str(ablated_path),
        "n_questions": len(ids),
        "note_effect": "checkpoint_default_minus_inference_note_ablation",
        "all_questions": {
            "macro_group_f1_default": default["macro_group_f1"],
            "macro_group_f1_ablation": ablated["macro_group_f1"],
            "delta": float(np.mean(f1_delta)),
            "ci95": bootstrap_ci(f1_delta),
            "wins": int(sum(x > 0 for x in f1_delta)),
            "exact_match_delta": float(np.mean(exact_delta)),
            "exact_match_ci95": bootstrap_ci(exact_delta),
            "edge_hits1": metric_delta("hits1", ids),
            "edge_hits3": metric_delta("hits3", ids),
            "edge_hits10": metric_delta("hits10", ids),
            "question_hit1": metric_delta("question_hit1", ids),
            "question_hit3": metric_delta("question_hit3", ids),
            "question_hit10": metric_delta("question_hit10", ids),
        },
        "multi_hop": {
            "n_questions": len(mid),
            "macro_group_f1_default": default["multi_hop_macro_group_f1"],
            "macro_group_f1_ablation": ablated["multi_hop_macro_group_f1"],
            "delta": float(np.mean(mf1_delta)),
            "ci95": bootstrap_ci(mf1_delta),
            "wins": int(sum(x > 0 for x in mf1_delta)),
            "exact_match_delta": float(np.mean(mexact_delta)),
            "exact_match_ci95": bootstrap_ci(mexact_delta),
            "edge_hits1": metric_delta("hits1", mid),
            "edge_hits3": metric_delta("hits3", mid),
            "edge_hits10": metric_delta("hits10", mid),
            "question_hit1": metric_delta("question_hit1", mid),
            "question_hit3": metric_delta("question_hit3", mid),
            "question_hit10": metric_delta("question_hit10", mid),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="default.json:ablation.json pairs")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for pair in args.pairs:
        left, right = pair.split(":", 1)
        rows.append(compare(Path(left), Path(right)))
    result = {"benchmark": "fawkes_graph_qa_20", "comparisons": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["default"], row["multi_hop"])


if __name__ == "__main__":
    main()
