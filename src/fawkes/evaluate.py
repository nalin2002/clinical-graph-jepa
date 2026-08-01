"""Evaluation: batch-mask recovery, leave-one-out, cascade, and EIR uplift.

Split out of ``paper_v16/trainer.py`` (lines 382-421 and 423-579) with
``paper_v16/evaluate.py`` — the checkpoint-only CLI — folded in at the
bottom.

The four evaluators answer different questions and are not interchangeable:

- ``evaluate``          batch-mask (30% of edges held out together) recovery.
- ``loo_evaluate``      (v12) mask exactly ONE edge, keep the whole rest of the
                        graph. No randomness, so exactly reproducible. This is
                        the honest per-relation measure and what the paper reports.
- ``cascade_evaluate``  (v13) recover the inferred relations in order, adding
                        each one's gold edges to the context before the next.
- ``eir_uplift_eval``   (v11) refiner precision/recall against v8-gold triples.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config
from .data import (NUM_NODE_TYPES, RELATION_CANONICAL, TARGET_RELS, add_inverses, normalize_text,
                   resolve_rel, support_graded, to_data)
from .model import DistMult, Encoder
from .train import buckets, readout_step

logger = logging.getLogger("fawkes_jepa")


@torch.no_grad()
def evaluate(encoder, scorer, loader, device, cfg):
    """Batch-mask edge recovery — the readout's own validation/test metric."""
    encoder.eval()
    scorer.eval()
    pos_batches, neg_batches, nonpat_batches = [], [], []
    reciprocal_ranks = []
    hits = {1: 0, 3: 0, 10: 0}
    num_ranked = 0
    qsig_total = 0
    rel_ranks = defaultdict(list)
    rel_hits = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_n = defaultdict(int)
    rel_candidates = defaultdict(list)
    for batch in loader:
        gen = torch.Generator(device=device).manual_seed(int(batch.gid[0]) & 0x7FFFFFFFFFFFFFFF) if cfg.freeze_eval else None
        step_out = readout_step(encoder, scorer, batch, device, False, cfg, gen=gen)
        if step_out is None:
            continue
        _, pos_scores, neg_scores, nonpat, extra = step_out
        pos_batches.append(pos_scores)
        neg_batches.append(neg_scores)
        nonpat_batches.append(nonpat)
        hidden, held_src, held_dst, held_rel, node_type, batch_index, qsig = extra
        qsig_total = (qsig_total + qsig) & 0xFFFFFFFFFFFF
        order, sorted_buckets = buckets(node_type, batch_index)
        for i in range(held_src.numel()):
            if num_ranked >= cfg.mrr_cap:
                break
            src, dst, rel = int(held_src[i]), int(held_dst[i]), int(held_rel[i])
            bucket = int(batch_index[dst]) * NUM_NODE_TYPES + int(node_type[dst])
            lo = int(torch.searchsorted(sorted_buckets, torch.tensor(bucket, device=device), right=False))
            hi = int(torch.searchsorted(sorted_buckets, torch.tensor(bucket, device=device), right=True))
            candidates = order[lo:hi]
            if candidates.numel() < 2:
                continue
            scores = scorer(hidden, torch.full((candidates.numel(),), src, device=device, dtype=torch.long), candidates,
                            torch.full((candidates.numel(),), rel, device=device, dtype=torch.long))
            rank = int((scores > scores[(candidates == dst).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            reciprocal_ranks.append(1.0 / rank)
            rel_ranks[rel].append(1.0 / rank)
            rel_n[rel] += 1
            rel_candidates[rel].append(int(candidates.numel()))
            for k in hits:
                if rank <= k:
                    hits[k] += 1
                    rel_hits[rel][k] += 1
            num_ranked += 1
    positives = torch.cat(pos_batches).cpu().numpy()
    negatives = torch.cat(neg_batches).cpu().numpy()
    nonpat_mask = torch.cat(nonpat_batches).cpu().numpy()
    all_labels = np.concatenate([np.ones_like(positives), np.zeros_like(negatives)])
    all_scores = np.concatenate([positives, negatives])
    auc, ap = roc_auc_score(all_labels, all_scores), average_precision_score(all_labels, all_scores)
    nonobvious_pos = positives[nonpat_mask]
    if len(nonobvious_pos) > 5:
        nonobvious_labels = np.concatenate([np.ones_like(nonobvious_pos), np.zeros_like(negatives)])
        nonobvious_scores = np.concatenate([nonobvious_pos, negatives])
        auc_nonobvious = roc_auc_score(nonobvious_labels, nonobvious_scores)
    else:
        auc_nonobvious = float('nan')
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float('nan')
    hit_rate = {k: hits[k] / max(num_ranked, 1) for k in hits}
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}
    per_rel = []
    for rel in sorted(rel_n, key=lambda x: -rel_n[x]):
        count = rel_n[rel]
        C = float(np.mean(rel_candidates[rel]))
        per_rel.append({"rel": ID2REL.get(rel, f"rel{rel}"), "n": count, "mrr": float(np.mean(rel_ranks[rel])),
                        "h1": rel_hits[rel][1] / count, "h10": rel_hits[rel][10] / count, "C": C,
                        "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0,
                        "chance_h1": 1.0 / C if C >= 1 else 1.0})
    return {"auc": auc, "ap": ap, "auc_nonobvious": auc_nonobvious, "mrr": mrr, "hits1": hit_rate[1],
            "hits3": hit_rate[3], "hits10": hit_rate[10], "n_mrr": num_ranked, "qsig": qsig_total, "per_rel": per_rel}


@torch.no_grad()
def loo_evaluate(encoder, scorer, graphs, device, cfg, cap=None):
    """(v12) LEAVE-ONE-OUT: mask exactly ONE edge, keep the entire rest of the graph, recover it.

    Filtered ranking (drop other true tails of (u,rel)); no randomness -> exact/reproducible. No leak:
    the masked edge's forward AND inverse are absent (inverses are built only over the kept edges).
    """
    cap = cfg.loo_cap if cap is None else cap
    encoder.eval()
    scorer.eval()
    rel_ranks = defaultdict(list)
    rel_hits = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_n = defaultdict(int)
    rel_candidates = defaultdict(list)
    num_queries = 0
    for graph in graphs:
        if num_queries >= cap:
            break
        graph = graph.to(device)
        edge_index, edge_type, node_type = graph.edge_index, graph.edge_type, graph.node_type
        num_edges = edge_index.size(1)
        if num_edges < 2:
            continue
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        for i in range(num_edges):
            if num_queries >= cap:
                break
            src = int(src_nodes[i])
            dst = int(dst_nodes[i])
            rel = int(edge_type[i])
            if src == dst:
                continue
            keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
            keep_mask[i] = False
            if cfg.use_scores:                                        # (v14) gated encoder needs the observed edges' scores
                kept_edge_index, kept_edge_type, kept_edge_feat = add_inverses(
                    edge_index[:, keep_mask], edge_type[keep_mask], graph.edge_feat[keep_mask])
            else:
                kept_edge_index, kept_edge_type = add_inverses(edge_index[:, keep_mask], edge_type[keep_mask])
                kept_edge_feat = None
            hidden = encoder(node_type, graph.entity_id, graph.numfeat, kept_edge_index, kept_edge_type,
                             kept_edge_feat, graph.sem_id)
            candidates = (node_type == node_type[dst]).nonzero(as_tuple=False).view(-1)
            candidates = candidates[candidates != src]
            other_tails = dst_nodes[(src_nodes == src) & (edge_type == rel)]
            other_tails = other_tails[other_tails != dst]             # filtered: other true tails of (u,rel)
            if other_tails.numel() > 0:
                candidates = candidates[~torch.isin(candidates, other_tails)]
            if candidates.numel() < 2 or int((candidates == dst).sum()) == 0:
                continue
            scores = scorer(hidden, torch.full((candidates.numel(),), src, dtype=torch.long, device=device), candidates,
                            torch.full((candidates.numel(),), rel, dtype=torch.long, device=device))
            rank = int((scores > scores[(candidates == dst).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            rel_ranks[rel].append(1.0 / rank)
            rel_n[rel] += 1
            rel_candidates[rel].append(int(candidates.numel()))
            for k in (1, 3, 10):
                if rank <= k:
                    rel_hits[rel][k] += 1
            num_queries += 1
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}
    per_rel = []
    all_ranks = []
    total_hits = {1: 0, 3: 0, 10: 0}
    total_queries = 0
    for rel in sorted(rel_n, key=lambda x: -rel_n[x]):
        count = rel_n[rel]
        C = float(np.mean(rel_candidates[rel]))
        all_ranks += rel_ranks[rel]
        total_queries += count
        for k in total_hits:
            total_hits[k] += rel_hits[rel][k]
        per_rel.append({"rel": ID2REL.get(rel, f"rel{rel}"), "n": count, "mrr": float(np.mean(rel_ranks[rel])),
                        "h1": rel_hits[rel][1] / count, "h3": rel_hits[rel][3] / count, "h10": rel_hits[rel][10] / count, "C": C,
                        "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0,
                        "chance_h1": 1.0 / C if C >= 1 else 1.0})
    return {"mrr": float(np.mean(all_ranks)) if all_ranks else float('nan'), "hits1": total_hits[1] / max(total_queries, 1),
            "hits3": total_hits[3] / max(total_queries, 1), "hits10": total_hits[10] / max(total_queries, 1),
            "n": total_queries, "per_rel": per_rel}


@torch.no_grad()
def cascade_evaluate(encoder, scorer, graphs, order_ids, device, cfg):
    """(v13) iterative completion: start from BACKBONE-only context, recover the inferred relations in order_ids,
    adding each relation's GOLD edges to the context before the next (oracle cascade). Also computes the FLOOR
    (backbone-only context for every inferred relation). Ceiling = the v12 leave-one-out (all other edges present).
    """
    encoder.eval()
    scorer.eval()
    target_ids = [RELATION_CANONICAL[x] for x in TARGET_RELS]
    target_ids_tensor = torch.tensor(sorted(set(target_ids)), device=device)

    def recover(rel_id, ctx_edge_index, ctx_edge_type, ctx_edge_feat, graph, node_type, edge_index, edge_type, results):
        if cfg.use_scores:                                            # (v14) gated encoder needs the context edges' scores
            enc_edge_index, enc_edge_type, enc_edge_feat = add_inverses(ctx_edge_index, ctx_edge_type, ctx_edge_feat)
        else:
            enc_edge_index, enc_edge_type = add_inverses(ctx_edge_index, ctx_edge_type)
            enc_edge_feat = None
        hidden = encoder(node_type, graph.entity_id.to(device), graph.numfeat.to(device),
                         enc_edge_index, enc_edge_type, enc_edge_feat, graph.sem_id.to(device))
        for j in (edge_type == rel_id).nonzero(as_tuple=False).view(-1).tolist():
            src = int(edge_index[0, j])
            dst = int(edge_index[1, j])
            if src == dst:
                continue
            candidates = (node_type == node_type[dst]).nonzero(as_tuple=False).view(-1)
            candidates = candidates[candidates != src]
            other_tails = edge_index[1][(edge_index[0] == src) & (edge_type == rel_id)]
            other_tails = other_tails[other_tails != dst]
            if other_tails.numel() > 0:
                candidates = candidates[~torch.isin(candidates, other_tails)]
            if candidates.numel() < 2 or int((candidates == dst).sum()) == 0:
                continue
            scores = scorer(hidden, torch.full((candidates.numel(),), src, dtype=torch.long, device=device), candidates,
                            torch.full((candidates.numel(),), rel_id, dtype=torch.long, device=device))
            rank = int((scores > scores[(candidates == dst).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            results[rel_id].append((1.0 / rank, rank, int(candidates.numel())))

    floor = defaultdict(list)
    cascade = defaultdict(list)
    for graph in graphs:
        graph = graph.to(device)
        edge_index = graph.edge_index
        edge_type = graph.edge_type
        node_type = graph.node_type
        if edge_index.size(1) < 2:
            continue
        edge_feat = graph.edge_feat if cfg.use_scores else None        # (v14) per-edge scores for the gated encoder
        backbone_mask = ~torch.isin(edge_type, target_ids_tensor)      # backbone = the non-inferred scaffold
        backbone_edge_index = edge_index[:, backbone_mask]
        backbone_edge_type = edge_type[backbone_mask]
        backbone_edge_feat = edge_feat[backbone_mask] if cfg.use_scores else None
        for rel_id in target_ids:
            recover(rel_id, backbone_edge_index, backbone_edge_type, backbone_edge_feat,
                    graph, node_type, edge_index, edge_type, floor)    # FLOOR: backbone-only context
        ctx_edge_index = backbone_edge_index
        ctx_edge_type = backbone_edge_type
        ctx_edge_feat = backbone_edge_feat                             # CASCADE: add each relation's gold in order
        for rel_id in order_ids:
            recover(rel_id, ctx_edge_index, ctx_edge_type, ctx_edge_feat,
                    graph, node_type, edge_index, edge_type, cascade)
            gold_edges = (edge_type == rel_id).nonzero(as_tuple=False).view(-1)
            if gold_edges.numel() > 0:
                ctx_edge_index = torch.cat([ctx_edge_index, edge_index[:, gold_edges]], 1)
                ctx_edge_type = torch.cat([ctx_edge_type, edge_type[gold_edges]])
                if cfg.use_scores:
                    ctx_edge_feat = torch.cat([ctx_edge_feat, edge_feat[gold_edges]], 0)
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}

    def agg(results):
        out = {}
        for rel_id, rows in results.items():
            if not rows:
                continue
            C = float(np.mean([x[2] for x in rows]))
            out[ID2REL.get(rel_id, str(rel_id))] = {
                "n": len(rows), "mrr": float(np.mean([x[0] for x in rows])),
                "h1": float(np.mean([1.0 if x[1] <= 1 else 0.0 for x in rows])),
                "h10": float(np.mean([1.0 if x[1] <= 10 else 0.0 for x in rows])), "C": C,
                "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0}
        return out

    return {"floor": agg(floor), "cascade": agg(cascade), "order": [ID2REL.get(r, str(r)) for r in order_ids]}


# ---- (v11) EIR scoring method (kg_similarity_scorer.py) + the refiner uplift eval ----

def edge_prf(pred, gold):
    """Directional (src_text, REL, dst_text) exact-triple set P/R/F1."""
    if not gold:
        return (1.0, 1.0, 1.0)
    if not pred:
        return (0.0, 0.0, 0.0)
    true_pos = len(pred & gold)
    precision = true_pos / len(pred)
    recall = true_pos / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


@torch.no_grad()
def eir_uplift_eval(encoder, scorer, pairs, tau, device, cfg):
    # refiner per TEST graph: GOLD = backbone + LLM edges with graded support>=tau; hold out EIR_HOLDOUT of gold LLM
    # edges; RAW = draft - held; REFINED = (keep supported, drop unsupported) + model-ADD (recover held-out top-1).
    encoder.eval()
    scorer.eval()
    graph_metrics = defaultdict(list)
    add_counts = defaultdict(lambda: [0, 0])             # add_counts[REL]=[recovered, held]
    for data, graph in pairs:
        nodes = graph["nodes"]
        names = [(n.get("normalized_name") or n.get("name") or "") for n in nodes]
        id_to_index = {n["id"]: i for i, n in enumerate(nodes)}
        records = []
        for edge in graph["edges"]:
            if edge["source"] not in id_to_index or edge["target"] not in id_to_index:
                continue
            try:
                rel_id = resolve_rel(edge.get("relation"))
            except Exception:
                continue
            src = id_to_index[edge["source"]]
            dst = id_to_index[edge["target"]]
            rel_name = str(edge.get("relation", "")).upper()
            is_llm = (edge.get("evidence") == "llm")
            support = support_graded(edge.get("labels")) if is_llm else 1.0
            records.append({"src": src, "dst": dst, "rel_name": rel_name, "rel_id": rel_id,
                            "is_llm": is_llm, "support": support,
                            "triple": (normalize_text(names[src]), rel_name, normalize_text(names[dst]))})
        gold = set(r["triple"] for r in records if (not r["is_llm"]) or r["support"] >= tau)
        gold_llm = [r for r in records if r["is_llm"] and r["support"] >= tau]
        if not gold or int(data.num_nodes) < 3:
            continue
        gen = torch.Generator().manual_seed(int(data.gid[0]) & 0x7FFFFFFF)
        order = torch.randperm(len(gold_llm), generator=gen).tolist() if gold_llm else []
        held = [gold_llm[i] for i in order[:int(round(cfg.eir_holdout * len(gold_llm)))]]
        held_triples = set(r["triple"] for r in held)
        raw_set = set(r["triple"] for r in records) - held_triples
        kept = [r for r in records if ((not r["is_llm"]) or r["support"] >= tau) and r["triple"] not in held_triples]
        kept_set = set(r["triple"] for r in kept)
        node_type = data.node_type.to(device)
        node_type_list = node_type.tolist()
        if kept:
            kept_edge_index = torch.tensor([[r["src"] for r in kept], [r["dst"] for r in kept]],
                                           dtype=torch.long, device=device)
            kept_edge_type = torch.tensor([r["rel_id"] for r in kept], dtype=torch.long, device=device)
        else:
            kept_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            kept_edge_type = torch.zeros((0,), dtype=torch.long, device=device)
        kept_edge_index, kept_edge_type = add_inverses(kept_edge_index, kept_edge_type)
        hidden = encoder(node_type, data.entity_id.to(device), data.numfeat.to(device),
                         kept_edge_index, kept_edge_type, None, data.sem_id.to(device))
        added = set()
        for record in held:
            candidates = [j for j in range(len(nodes)) if node_type_list[j] == node_type_list[record["dst"]] and j != record["src"]]
            add_counts[record["rel_name"]][1] += 1
            if not candidates:
                continue
            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = scorer(hidden, torch.full((len(candidates),), record["src"], dtype=torch.long, device=device), cand_tensor,
                            torch.full((len(candidates),), record["rel_id"], dtype=torch.long, device=device))
            predicted_dst = candidates[int(scores.argmax())]
            added.add((normalize_text(names[record["src"]]), record["rel_name"], normalize_text(names[predicted_dst])))
            if predicted_dst == record["dst"]:
                add_counts[record["rel_name"]][0] += 1
        refined = kept_set | added
        raw_p, raw_r, raw_f = edge_prf(raw_set, gold)
        ref_p, ref_r, ref_f = edge_prf(refined, gold)
        graph_metrics["rawP"].append(raw_p)
        graph_metrics["rawR"].append(raw_r)
        graph_metrics["rawF"].append(raw_f)
        graph_metrics["refP"].append(ref_p)
        graph_metrics["refR"].append(ref_r)
        graph_metrics["refF"].append(ref_f)
    mean = lambda x: float(np.mean(x)) if x else float('nan')
    out = {k: mean(graph_metrics[k]) for k in ("rawP", "rawR", "rawF", "refP", "refR", "refF")}
    out["n_graphs"] = len(graph_metrics["rawF"])
    out["add"] = {rel_name: {"rec": counts[0], "held": counts[1],
                             "recall": (counts[0] / counts[1] if counts[1] else float('nan'))}
                  for rel_name, counts in add_counts.items()}
    return out


# ---- CLI: evaluate a saved checkpoint on a local clinical-graph JSONL file ----
# Was paper_v16/evaluate.py.

def _load_graphs(path: Path, limit: int | None) -> tuple[list, dict]:
    raw = []
    demographics = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                graph = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            raw.append(graph)
            subject = str(graph.get("subject_id"))
            demographics[subject] = {
                "gender": graph.get("gender"),
                "age": graph.get("anchor_age"),
            }
            if limit is not None and len(raw) >= limit:
                break
    if not raw:
        raise ValueError(f"No graphs found in {path}")
    return raw, demographics


def run(args) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    cfg = Config.from_env()
    expected = {
        "use_note": cfg.use_note,
        "ground_by": cfg.ground_by,
        "embed_dim": cfg.embed_dim,
        "use_scores": cfg.use_scores,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Environment does not match checkpoint configuration: "
            f"{mismatches}. Set USE_NOTE/GROUND_BY/EMBED_DIM/USE_SCORES before launch."
        )

    raw, demographics = _load_graphs(Path(args.data), args.max_graphs)
    graphs = []
    for graph in raw:
        data = to_data(graph, demographics, cfg)
        if data.num_nodes >= 3 and data.edge_index.size(1) >= 2:
            graphs.append(data)
    if not graphs:
        raise ValueError("No evaluable graphs remain after conversion")

    device = torch.device(args.device)
    encoder = Encoder(cfg).to(device)
    scorer = DistMult(cfg).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])
    metrics = loo_evaluate(encoder, scorer, graphs, device, cfg, cap=args.cap)
    print(
        f"[LOO] graphs={len(graphs)} MRR={metrics['mrr']:.3f} "
        f"H@1={metrics['hits1']:.3f} H@10={metrics['hits10']:.3f} "
        f"over {metrics['n']} edges"
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved fawkes checkpoint on a local clinical-graph JSONL file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Clinical graph JSONL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cap", type=int, default=40000)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
