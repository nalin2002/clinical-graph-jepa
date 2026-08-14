"""Evaluate structured graph QA with a Fawkes checkpoint.

This is a graph QA benchmark, not an LLM free-form generation benchmark. Each
question contains gold evidence edges. The evaluator masks all gold edges for
one source/relation group, encodes the remaining graph, ranks same-type target
nodes with the checkpoint readout head, and computes set F1 and exact match.

The ``--disable-note`` mode is an inference-time ablation of a note-trained
checkpoint. It is useful when a separately trained no-note checkpoint is not
available, but it must not be reported as a fully retrained Option A model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FAWKES_SRC = ROOT / "src"
DATA_DEFAULT = None
QUESTIONS_DEFAULT = ROOT / "ablation_artifacts/qa/fawkes_graph_qa_20.json"

if str(FAWKES_SRC) not in sys.path:
    sys.path.insert(0, str(FAWKES_SRC))

from fawkes.config import Config  # noqa: E402
from fawkes.data import add_inverses, load_note_memories, resolve_rel, to_data  # noqa: E402
from fawkes.evaluate import rank_true_tail  # noqa: E402
from fawkes.model import Encoder, build_scorer  # noqa: E402


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = dict(checkpoint.get("config", {}))
    # v22/v23 future knobs were deliberately stored in run_config so the
    # legacy config block stayed byte-compatible with v16.
    saved.update(checkpoint.get("run_config", {}))
    cfg = Config.from_checkpoint(saved)
    # The released v24 checkpoint predates the decoder field in its legacy
    # config block, but its scorer state is the v21-v25 MLP readout.
    cfg.decoder = "mlp"
    encoder = Encoder(cfg)
    scorer = build_scorer(cfg)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    scorer.load_state_dict(checkpoint["scorer"], strict=True)
    return checkpoint, cfg, encoder, scorer


def encode(encoder, graph, edge_index, edge_type, edge_feat, cfg, device):
    if cfg.use_scores:
        ei, et, ef = add_inverses(edge_index, edge_type, edge_feat)
    else:
        ei, et = add_inverses(edge_index, edge_type)
        ef = None
    return encoder(
        graph.node_type,
        graph.entity_id,
        graph.numfeat,
        ei,
        et,
        ef,
        graph.sem_id,
        batch_index=torch.zeros(graph.num_nodes, dtype=torch.long, device=device),
        note_memory=getattr(graph, "note_memory", None),
        note_memory_mask=getattr(graph, "note_memory_mask", None),
        note_span_token_counts=getattr(graph, "note_span_token_counts", None),
        note_grounded=getattr(graph, "note_grounded", None),
    )


def f1(pred: set[int], gold: set[int]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    p = len(pred & gold) / len(pred)
    r = len(pred & gold) / len(gold)
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate_question(raw, item, encoder, scorer, cfg, device, disable_note, note_memories):
    demographics = {str(raw.get("subject_id")): {"gender": raw.get("gender"), "age": raw.get("anchor_age")}}
    graph = to_data(raw, demographics, cfg, note_memories=note_memories)
    if disable_note and cfg.use_note and cfg.numeric_dim > 6:
        graph.numfeat[:, 6:] = 0.0
    if disable_note and getattr(graph, "note_grounded", None) is not None:
        graph.note_grounded[:] = False
    graph = graph.to(device)
    node_by_id = {n["id"]: i for i, n in enumerate(raw["nodes"])}
    grouped = defaultdict(set)
    for evidence in item["evidence_edges"]:
        source = node_by_id[evidence["source"]]
        target = node_by_id[evidence["target"]]
        relation = resolve_rel(evidence["relation"])
        grouped[(source, relation)].add(target)

    question_groups = []
    for (source, relation), gold in sorted(grouped.items()):
        keep = torch.ones(graph.edge_index.size(1), dtype=torch.bool, device=device)
        for i in range(graph.edge_index.size(1)):
            if int(graph.edge_index[0, i]) == source and int(graph.edge_type[i]) == relation and int(graph.edge_index[1, i]) in gold:
                keep[i] = False
        context_ei = graph.edge_index[:, keep]
        context_et = graph.edge_type[keep]
        context_ef = graph.edge_feat[keep]
        hidden = encode(encoder, graph, context_ei, context_et, context_ef, cfg, device)
        gold_tensor = torch.tensor(sorted(gold), dtype=torch.long, device=device)
        target_type = int(graph.node_type[gold_tensor[0]])
        candidates = (graph.node_type == target_type).nonzero(as_tuple=False).flatten()
        candidates = candidates[candidates != source]
        scores = scorer(
            hidden,
            torch.full((candidates.numel(),), source, dtype=torch.long, device=device),
            candidates,
            torch.full((candidates.numel(),), relation, dtype=torch.long, device=device),
        )
        order = torch.argsort(scores, descending=True)
        predicted = set(int(x) for x in candidates[order[: len(gold)]].tolist())
        ranked = [int(x) for x in candidates[order].tolist()]
        ranks = {target: ranked.index(target) + 1 for target in gold}
        question_groups.append({
            "source": source,
            "relation": item["evidence_edges"][0]["relation"],
            "gold": sorted(gold),
            "predicted": sorted(predicted),
            "group_f1": f1(predicted, gold),
            "group_exact": predicted == gold,
            "candidate_count": int(candidates.numel()),
            "gold_ranks": ranks,
            "hits1": float(np.mean([rank <= 1 for rank in ranks.values()])),
            "hits3": float(np.mean([rank <= 3 for rank in ranks.values()])),
            "hits10": float(np.mean([rank <= 10 for rank in ranks.values()])),
            "any_hit1": any(rank <= 1 for rank in ranks.values()),
            "any_hit3": any(rank <= 3 for rank in ranks.values()),
            "any_hit10": any(rank <= 10 for rank in ranks.values()),
        })
    group_f1 = float(np.mean([x["group_f1"] for x in question_groups])) if question_groups else 0.0
    exact = bool(question_groups) and all(x["group_exact"] for x in question_groups)
    return {
        "id": item["id"], "template": item["template"], "group_f1": group_f1,
        "exact_match": exact, "groups": question_groups,
        "hits1": float(np.mean([x["hits1"] for x in question_groups])) if question_groups else 0.0,
        "hits3": float(np.mean([x["hits3"] for x in question_groups])) if question_groups else 0.0,
        "hits10": float(np.mean([x["hits10"] for x in question_groups])) if question_groups else 0.0,
        "question_hit1": all(x["any_hit1"] for x in question_groups) if question_groups else False,
        "question_hit3": all(x["any_hit3"] for x in question_groups) if question_groups else False,
        "question_hit10": all(x["any_hit10"] for x in question_groups) if question_groups else False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--questions", type=Path, default=QUESTIONS_DEFAULT)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--disable-note", action="store_true")
    args = ap.parse_args()

    checkpoint, cfg, encoder, scorer = load_checkpoint(Path(args.checkpoint))
    local_note_memory = ROOT / "outputs/note_memory_v23/fawkes_note_memory_v23.safetensors"
    if local_note_memory.is_file() and cfg.note_injection != "mean":
        cfg.note_memory_path = str(local_note_memory)
    device = torch.device(args.device)
    encoder.to(device).eval()
    scorer.to(device).eval()
    note_memories = load_note_memories(cfg)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))["items"]
    wanted = {(str(x["subject_id"]), str(x["hadm_id"])) for x in questions}
    raw_by_key = {}
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            key = (str(raw["subject_id"]), str(raw["hadm_id"]))
            if key in wanted:
                raw_by_key[key] = raw
    missing = wanted - set(raw_by_key)
    if missing:
        raise RuntimeError(f"questions missing from graph corpus: {sorted(missing)[:5]}")

    results = []
    with torch.no_grad():
        for item in questions:
            raw = raw_by_key[(str(item["subject_id"]), str(item["hadm_id"]))]
            results.append(evaluate_question(raw, item, encoder, scorer, cfg, device, args.disable_note, note_memories))
    summary = {
        "benchmark": "fawkes_graph_qa_20",
        "checkpoint": str(args.checkpoint),
        "checkpoint_config": {**checkpoint.get("config", {}), **checkpoint.get("run_config", {})},
        "option_label": "inference_note_ablation" if args.disable_note else "checkpoint_default",
        "disable_note": args.disable_note,
        "n_questions": len(results),
        "macro_group_f1": float(np.mean([x["group_f1"] for x in results])),
        "question_exact_match": float(np.mean([x["exact_match"] for x in results])),
        "edge_hits1": float(np.mean([x["hits1"] for x in results])),
        "edge_hits3": float(np.mean([x["hits3"] for x in results])),
        "edge_hits10": float(np.mean([x["hits10"] for x in results])),
        "question_hit1": float(np.mean([x["question_hit1"] for x in results])),
        "question_hit3": float(np.mean([x["question_hit3"] for x in results])),
        "question_hit10": float(np.mean([x["question_hit10"] for x in results])),
        "multi_hop_macro_group_f1": float(np.mean([x["group_f1"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "multi_hop_question_exact_match": float(np.mean([x["exact_match"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "multi_hop_edge_hits1": float(np.mean([x["hits1"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "multi_hop_edge_hits3": float(np.mean([x["hits3"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "multi_hop_edge_hits10": float(np.mean([x["hits10"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "multi_hop_question_hit1": float(np.mean([x["question_hit1"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "multi_hop_question_hit3": float(np.mean([x["question_hit3"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "multi_hop_question_hit10": float(np.mean([x["question_hit10"] for x in results if x["template"] in {"primary_chain", "medication_chain", "procedure_chain", "complication_chain"}])),
        "items": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("option_label", "n_questions", "macro_group_f1", "question_exact_match", "multi_hop_macro_group_f1", "multi_hop_question_exact_match")}, indent=2))


if __name__ == "__main__":
    main()
