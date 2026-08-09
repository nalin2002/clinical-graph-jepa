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

They share one ranking protocol, factored into the helpers at the top:
candidates are same-type nodes in the same graph, other true tails of
``(src, rel)`` are filtered out, and the rank of the true tail is
``1 + |strictly better candidates|`` (ties favor the true tail).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config
from .data import (NUM_NODE_TYPES, RELATION_CANONICAL, TARGET_RELS, add_inverses, normalize_text,
                   resolve_rel, support_graded, to_data)
from .model import Encoder, build_scorer
from .steps import buckets, readout_step

logger = logging.getLogger("fawkes_jepa")


# ---- the shared ranking protocol ----

def chance_mrr(C):
    """Expected MRR of a uniform ranker over C candidates (harmonic-number approximation)."""
    return (math.log(C) + 0.5772) / C if C > 1 else 1.0


def encode_context_graph(encoder, graph, edge_index, edge_type, edge_feat, cfg):
    """Encode ``graph`` from the given context edges (inverses added here).

    ``edge_feat`` is only consulted when ``cfg.use_scores`` — the (v14) gated
    encoder needs the context edges' scores; the structure-only encoder gets None.
    """
    if cfg.use_scores:
        edge_index, edge_type, edge_feat = add_inverses(edge_index, edge_type, edge_feat)
    else:
        edge_index, edge_type = add_inverses(edge_index, edge_type)
        edge_feat = None
    return encoder(graph.node_type, graph.entity_id, graph.numfeat,
                   edge_index, edge_type, edge_feat, graph.sem_id)


def filtered_candidates(node_type, src, dst, rel, src_nodes, dst_nodes, edge_type):
    """The filtered ranking pool for (src, rel, dst): every node of dst's type
    except the head itself and the OTHER true tails of (src, rel)."""
    candidates = (node_type == node_type[dst]).nonzero(as_tuple=False).view(-1)
    candidates = candidates[candidates != src]
    other_tails = dst_nodes[(src_nodes == src) & (edge_type == rel)]
    other_tails = other_tails[other_tails != dst]
    if other_tails.numel() > 0:
        candidates = candidates[~torch.isin(candidates, other_tails)]
    return candidates


def rank_true_tail(scorer, hidden, src, rel, candidates, dst, device):
    """Score (src, rel, ·) against every candidate; rank of the true tail, ties in its favor."""
    scores = scorer(hidden, torch.full((candidates.numel(),), src, dtype=torch.long, device=device), candidates,
                    torch.full((candidates.numel(),), rel, dtype=torch.long, device=device))
    true_score = scores[(candidates == dst).nonzero(as_tuple=False)[0, 0]]
    return int((scores > true_score).sum().item()) + 1


class RelationStats:
    """Per-relation rank bookkeeping shared by ``evaluate`` and ``loo_evaluate``."""

    def __init__(self):
        self.ranks = defaultdict(list)                       # rel -> [1/rank, ...] in query order
        self.hits = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
        self.n = defaultdict(int)
        self.candidates = defaultdict(list)                  # rel -> [pool size per query]
        self.reciprocal_ranks = []                           # every query, insertion order
        self.total = 0

    def add(self, rel, rank, num_candidates):
        self.reciprocal_ranks.append(1.0 / rank)
        self.ranks[rel].append(1.0 / rank)
        self.n[rel] += 1
        self.candidates[rel].append(num_candidates)
        for k in (1, 3, 10):
            if rank <= k:
                self.hits[rel][k] += 1
        self.total += 1

    def by_count(self):
        """Relation ids, most-queried first (ties keep first-seen order)."""
        return sorted(self.n, key=lambda rel: -self.n[rel])

    def total_hits(self, k):
        return sum(h[k] for h in self.hits.values())

    def per_rel_rows(self, include_h3):
        """The per-relation summary table; ``include_h3`` matches each caller's
        recorded output (the batch-mask evaluator never reported H@3)."""
        id_to_rel = {v: k for k, v in RELATION_CANONICAL.items()}
        rows = []
        for rel in self.by_count():
            count = self.n[rel]
            C = float(np.mean(self.candidates[rel]))
            row = {"rel": id_to_rel.get(rel, f"rel{rel}"), "n": count,
                   "mrr": float(np.mean(self.ranks[rel])), "h1": self.hits[rel][1] / count}
            if include_h3:
                row["h3"] = self.hits[rel][3] / count
            row.update({"h10": self.hits[rel][10] / count, "C": C,
                        "chance_mrr": chance_mrr(C),
                        "chance_h1": 1.0 / C if C >= 1 else 1.0})
            rows.append(row)
        return rows


# ---- batch-mask recovery ----

def _rank_held_edges(scorer, extra, stats, cap, device):
    """Rank each held-out edge of one batch against its same-type, same-graph bucket."""
    hidden, held_src, held_dst, held_rel, node_type, batch_index, _ = extra
    order, sorted_buckets = buckets(node_type, batch_index)
    for i in range(held_src.numel()):
        if stats.total >= cap:
            return
        src, dst, rel = int(held_src[i]), int(held_dst[i]), int(held_rel[i])
        bucket = int(batch_index[dst]) * NUM_NODE_TYPES + int(node_type[dst])
        lo = int(torch.searchsorted(sorted_buckets, torch.tensor(bucket, device=device), right=False))
        hi = int(torch.searchsorted(sorted_buckets, torch.tensor(bucket, device=device), right=True))
        candidates = order[lo:hi]
        if candidates.numel() < 2:
            continue
        rank = rank_true_tail(scorer, hidden, src, rel, candidates, dst, device)
        stats.add(rel, rank, int(candidates.numel()))


def _classification_metrics(pos_batches, neg_batches, nonpat_batches):
    """AUC/AP over held edges vs first negatives, plus the non-PATIENT-head AUC."""
    positives = torch.cat(pos_batches).cpu().numpy()
    negatives = torch.cat(neg_batches).cpu().numpy()
    nonpat_mask = torch.cat(nonpat_batches).cpu().numpy()
    all_labels = np.concatenate([np.ones_like(positives), np.zeros_like(negatives)])
    all_scores = np.concatenate([positives, negatives])
    auc, ap = roc_auc_score(all_labels, all_scores), average_precision_score(all_labels, all_scores)
    nonobvious_pos = positives[nonpat_mask]
    nonobvious_neg = negatives[nonpat_mask]      # one negative per positive -> the mask indexes both
    if len(nonobvious_pos) > 5:
        nonobvious_labels = np.concatenate([np.ones_like(nonobvious_pos), np.zeros_like(nonobvious_neg)])
        nonobvious_scores = np.concatenate([nonobvious_pos, nonobvious_neg])
        auc_nonobvious = roc_auc_score(nonobvious_labels, nonobvious_scores)
    else:
        auc_nonobvious = float('nan')
    return auc, ap, auc_nonobvious


@torch.no_grad()
def evaluate(encoder, scorer, loader, device, cfg):
    """Batch-mask edge recovery — the readout's own validation/test metric."""
    encoder.eval()
    scorer.eval()
    pos_batches, neg_batches, nonpat_batches = [], [], []
    stats = RelationStats()
    qsig_total = 0
    for batch in loader:
        gen = torch.Generator(device=device).manual_seed(int(batch.gid[0]) & 0x7FFFFFFFFFFFFFFF) if cfg.freeze_eval else None
        step_out = readout_step(encoder, scorer, batch, device, False, cfg, gen=gen)
        if step_out is None:
            continue
        _, pos_scores, neg_scores, nonpat, extra = step_out
        pos_batches.append(pos_scores)
        neg_batches.append(neg_scores)
        nonpat_batches.append(nonpat)
        qsig_total = (qsig_total + extra[-1]) & 0xFFFFFFFFFFFF
        _rank_held_edges(scorer, extra, stats, cfg.mrr_cap, device)
    auc, ap, auc_nonobvious = _classification_metrics(pos_batches, neg_batches, nonpat_batches)
    mrr = float(np.mean(stats.reciprocal_ranks)) if stats.reciprocal_ranks else float('nan')
    hit_rate = {k: stats.total_hits(k) / max(stats.total, 1) for k in (1, 3, 10)}
    return {"auc": auc, "ap": ap, "auc_nonobvious": auc_nonobvious, "mrr": mrr, "hits1": hit_rate[1],
            "hits3": hit_rate[3], "hits10": hit_rate[10], "n_mrr": stats.total, "qsig": qsig_total,
            "per_rel": stats.per_rel_rows(include_h3=False)}


# ---- leave-one-out ----

@torch.no_grad()
def loo_evaluate(encoder, scorer, graphs, device, cfg, cap=None):
    """(v12) LEAVE-ONE-OUT: mask exactly ONE edge, keep the entire rest of the graph, recover it.

    Filtered ranking (drop other true tails of (u,rel)); no randomness -> exact/reproducible. No leak:
    the masked edge's forward AND inverse are absent (inverses are built only over the kept edges).
    """
    cap = cfg.loo_cap if cap is None else cap
    encoder.eval()
    scorer.eval()
    stats = RelationStats()
    for graph in graphs:
        if stats.total >= cap:
            break
        graph = graph.to(device)
        edge_index, edge_type, node_type = graph.edge_index, graph.edge_type, graph.node_type
        num_edges = edge_index.size(1)
        if num_edges < 2:
            continue
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        for i in range(num_edges):
            if stats.total >= cap:
                break
            src, dst, rel = int(src_nodes[i]), int(dst_nodes[i]), int(edge_type[i])
            if src == dst:
                continue
            keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
            keep_mask[i] = False
            hidden = encode_context_graph(encoder, graph, edge_index[:, keep_mask], edge_type[keep_mask],
                                          graph.edge_feat[keep_mask] if cfg.use_scores else None, cfg)
            candidates = filtered_candidates(node_type, src, dst, rel, src_nodes, dst_nodes, edge_type)
            if candidates.numel() < 2 or int((candidates == dst).sum()) == 0:
                continue
            rank = rank_true_tail(scorer, hidden, src, rel, candidates, dst, device)
            stats.add(rel, rank, int(candidates.numel()))
    per_rel = stats.per_rel_rows(include_h3=True)
    all_ranks = [rr for rel in stats.by_count() for rr in stats.ranks[rel]]
    return {"mrr": float(np.mean(all_ranks)) if all_ranks else float('nan'),
            "hits1": stats.total_hits(1) / max(stats.total, 1),
            "hits3": stats.total_hits(3) / max(stats.total, 1),
            "hits10": stats.total_hits(10) / max(stats.total, 1),
            "n": stats.total, "per_rel": per_rel}


# ---- cascade ----

def _recover_relation(encoder, scorer, graph, rel_id, ctx_edge_index, ctx_edge_type, ctx_edge_feat,
                      results, device, cfg):
    """Rank every (src, rel_id, dst) edge of ``graph`` given only the context edges."""
    hidden = encode_context_graph(encoder, graph, ctx_edge_index, ctx_edge_type, ctx_edge_feat, cfg)
    edge_index, edge_type, node_type = graph.edge_index, graph.edge_type, graph.node_type
    for j in (edge_type == rel_id).nonzero(as_tuple=False).view(-1).tolist():
        src = int(edge_index[0, j])
        dst = int(edge_index[1, j])
        if src == dst:
            continue
        candidates = filtered_candidates(node_type, src, dst, rel_id, edge_index[0], edge_index[1], edge_type)
        if candidates.numel() < 2 or int((candidates == dst).sum()) == 0:
            continue
        rank = rank_true_tail(scorer, hidden, src, rel_id, candidates, dst, device)
        results[rel_id].append((1.0 / rank, rank, int(candidates.numel())))


def _aggregate_cascade(results):
    """(rr, rank, pool-size) tuples -> the per-relation summary keyed by relation name."""
    id_to_rel = {v: k for k, v in RELATION_CANONICAL.items()}
    out = {}
    for rel_id, rows in results.items():
        if not rows:
            continue
        C = float(np.mean([x[2] for x in rows]))
        out[id_to_rel.get(rel_id, str(rel_id))] = {
            "n": len(rows), "mrr": float(np.mean([x[0] for x in rows])),
            "h1": float(np.mean([1.0 if x[1] <= 1 else 0.0 for x in rows])),
            "h10": float(np.mean([1.0 if x[1] <= 10 else 0.0 for x in rows])), "C": C,
            "chance_mrr": chance_mrr(C)}
    return out


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
    floor = defaultdict(list)
    cascade = defaultdict(list)
    for graph in graphs:
        graph = graph.to(device)
        edge_index, edge_type = graph.edge_index, graph.edge_type
        if edge_index.size(1) < 2:
            continue
        edge_feat = graph.edge_feat if cfg.use_scores else None        # (v14) per-edge scores for the gated encoder
        backbone_mask = ~torch.isin(edge_type, target_ids_tensor)      # backbone = the non-inferred scaffold
        backbone_edge_index = edge_index[:, backbone_mask]
        backbone_edge_type = edge_type[backbone_mask]
        backbone_edge_feat = edge_feat[backbone_mask] if cfg.use_scores else None
        for rel_id in target_ids:                                      # FLOOR: backbone-only context
            _recover_relation(encoder, scorer, graph, rel_id, backbone_edge_index, backbone_edge_type,
                              backbone_edge_feat, floor, device, cfg)
        ctx_edge_index = backbone_edge_index                           # CASCADE: add each relation's gold in order
        ctx_edge_type = backbone_edge_type
        ctx_edge_feat = backbone_edge_feat
        for rel_id in order_ids:
            _recover_relation(encoder, scorer, graph, rel_id, ctx_edge_index, ctx_edge_type,
                              ctx_edge_feat, cascade, device, cfg)
            gold_edges = (edge_type == rel_id).nonzero(as_tuple=False).view(-1)
            if gold_edges.numel() > 0:
                ctx_edge_index = torch.cat([ctx_edge_index, edge_index[:, gold_edges]], 1)
                ctx_edge_type = torch.cat([ctx_edge_type, edge_type[gold_edges]])
                if cfg.use_scores:
                    ctx_edge_feat = torch.cat([ctx_edge_feat, edge_feat[gold_edges]], 0)
    id_to_rel = {v: k for k, v in RELATION_CANONICAL.items()}
    return {"floor": _aggregate_cascade(floor), "cascade": _aggregate_cascade(cascade),
            "order": [id_to_rel.get(r, str(r)) for r in order_ids]}


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


def _edge_records(graph, names, id_to_index):
    """One record per resolvable edge: endpoints, relation, LLM/support status, and
    the normalized (src_text, REL, dst_text) triple used for set matching."""
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
    return records


def _encode_kept_edges(encoder, data, kept, node_type, device):
    """Encode the graph from the kept edges only (held-out LLM edges never seen)."""
    if kept:
        kept_edge_index = torch.tensor([[r["src"] for r in kept], [r["dst"] for r in kept]],
                                       dtype=torch.long, device=device)
        kept_edge_type = torch.tensor([r["rel_id"] for r in kept], dtype=torch.long, device=device)
    else:
        kept_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        kept_edge_type = torch.zeros((0,), dtype=torch.long, device=device)
    kept_edge_index, kept_edge_type = add_inverses(kept_edge_index, kept_edge_type)
    return encoder(node_type, data.entity_id.to(device), data.numfeat.to(device),
                   kept_edge_index, kept_edge_type, None, data.sem_id.to(device))


def _model_added_triples(scorer, hidden, held, names, node_type_list, add_counts, device):
    """Top-1 same-type prediction for each held-out edge; returns the ADDed triples
    and tallies recovered/held per relation into ``add_counts``."""
    added = set()
    for record in held:
        candidates = [j for j in range(len(node_type_list))
                      if node_type_list[j] == node_type_list[record["dst"]] and j != record["src"]]
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
    return added


@torch.no_grad()
def eir_uplift_eval(encoder, scorer, pairs, tau, device, cfg):
    """Refiner uplift per TEST graph: GOLD = backbone + LLM edges with graded support>=tau;
    hold out EIR_HOLDOUT of gold LLM edges; RAW = draft - held;
    REFINED = (keep supported, drop unsupported) + model-ADD (recover held-out top-1)."""
    encoder.eval()
    scorer.eval()
    graph_metrics = defaultdict(list)
    add_counts = defaultdict(lambda: [0, 0])             # add_counts[REL]=[recovered, held]
    for data, graph in pairs:
        nodes = graph["nodes"]
        names = [(n.get("normalized_name") or n.get("name") or "") for n in nodes]
        id_to_index = {n["id"]: i for i, n in enumerate(nodes)}
        records = _edge_records(graph, names, id_to_index)
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
        hidden = _encode_kept_edges(encoder, data, kept, node_type, device)
        added = _model_added_triples(scorer, hidden, held, names, node_type_list, add_counts, device)
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
    # The pinned `config` block cannot carry `decoder`, so until run_config existed a
    # non-DistMult checkpoint failed a strict state_dict load unless the caller happened
    # to set DECODER. Checkpoints that record it now describe their own head.
    run_config = checkpoint.get("run_config") or {}
    if "decoder" in run_config:
        cfg = replace(cfg, decoder=run_config["decoder"])
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
    scorer = build_scorer(cfg).to(device)
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
