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
def evaluate(enc, scorer, loader, device, cfg):
    """Batch-mask edge recovery — the readout's own validation/test metric."""
    enc.eval()
    scorer.eval()
    P, N, NP = [], [], []
    rr = []
    hits = {1: 0, 3: 0, 10: 0}
    nmrr = 0
    qsig_tot = 0
    rel_rr = defaultdict(list)
    rel_hits = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_n = defaultdict(int)
    rel_C = defaultdict(list)
    for b in loader:
        gen = torch.Generator(device=device).manual_seed(int(b.gid[0]) & 0x7FFFFFFFFFFFFFFF) if cfg.freeze_eval else None
        r = readout_step(enc, scorer, b, device, False, cfg, gen=gen)
        if r is None:
            continue
        _, pos, neg, nonpat, extra = r
        P.append(pos)
        N.append(neg)
        NP.append(nonpat)
        h, pu, pv, pr, nt, bt, qsig = extra
        qsig_tot = (qsig_tot + qsig) & 0xFFFFFFFFFFFF
        o, sb = buckets(nt, bt)
        for i in range(pu.numel()):
            if nmrr >= cfg.mrr_cap:
                break
            u, v, r_ = int(pu[i]), int(pv[i]), int(pr[i])
            tb = int(bt[v]) * NUM_NODE_TYPES + int(nt[v])
            lo = int(torch.searchsorted(sb, torch.tensor(tb, device=device), right=False))
            hi = int(torch.searchsorted(sb, torch.tensor(tb, device=device), right=True))
            cand = o[lo:hi]
            if cand.numel() < 2:
                continue
            sc = scorer(h, torch.full((cand.numel(),), u, device=device, dtype=torch.long), cand,
                        torch.full((cand.numel(),), r_, device=device, dtype=torch.long))
            rank = int((sc > sc[(cand == v).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            rr.append(1.0 / rank)
            rel_rr[r_].append(1.0 / rank)
            rel_n[r_] += 1
            rel_C[r_].append(int(cand.numel()))
            for kk in hits:
                if rank <= kk:
                    hits[kk] += 1
                    rel_hits[r_][kk] += 1
            nmrr += 1
    pos = torch.cat(P).cpu().numpy()
    neg = torch.cat(N).cpu().numpy()
    npm = torch.cat(NP).cpu().numpy()
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    sp = pos[npm]
    if len(sp) > 5:
        y2 = np.concatenate([np.ones_like(sp), np.zeros_like(neg)])
        s2 = np.concatenate([sp, neg])
        auc2 = roc_auc_score(y2, s2)
    else:
        auc2 = float('nan')
    mrr = float(np.mean(rr)) if rr else float('nan')
    H = {k: hits[k] / max(nmrr, 1) for k in hits}
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}
    per_rel = []
    for r_ in sorted(rel_n, key=lambda x: -rel_n[x]):
        n = rel_n[r_]
        C = float(np.mean(rel_C[r_]))
        per_rel.append({"rel": ID2REL.get(r_, f"rel{r_}"), "n": n, "mrr": float(np.mean(rel_rr[r_])),
                        "h1": rel_hits[r_][1] / n, "h10": rel_hits[r_][10] / n, "C": C,
                        "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0,
                        "chance_h1": 1.0 / C if C >= 1 else 1.0})
    return {"auc": auc, "ap": ap, "auc_nonobvious": auc2, "mrr": mrr, "hits1": H[1], "hits3": H[3],
            "hits10": H[10], "n_mrr": nmrr, "qsig": qsig_tot, "per_rel": per_rel}


@torch.no_grad()
def loo_evaluate(enc, scorer, graphs, device, cfg, cap=None):
    """(v12) LEAVE-ONE-OUT: mask exactly ONE edge, keep the entire rest of the graph, recover it.

    Filtered ranking (drop other true tails of (u,rel)); no randomness -> exact/reproducible. No leak:
    the masked edge's forward AND inverse are absent (inverses are built only over the kept edges).
    """
    cap = cfg.loo_cap if cap is None else cap
    enc.eval()
    scorer.eval()
    rel_rr = defaultdict(list)
    rel_hits = defaultdict(lambda: {1: 0, 3: 0, 10: 0})
    rel_n = defaultdict(int)
    rel_C = defaultdict(list)
    nq = 0
    for d in graphs:
        if nq >= cap:
            break
        d = d.to(device)
        ei, et, nt = d.edge_index, d.edge_type, d.node_type
        E = ei.size(1)
        if E < 2:
            continue
        ei0, ei1 = ei[0], ei[1]
        for i in range(E):
            if nq >= cap:
                break
            u = int(ei0[i])
            v = int(ei1[i])
            r = int(et[i])
            if u == v:
                continue
            keep = torch.ones(E, dtype=torch.bool, device=device)
            keep[i] = False
            if cfg.use_scores:                                        # (v14) gated encoder needs the observed edges' scores
                oei, oet, oef = add_inverses(ei[:, keep], et[keep], d.edge_feat[keep])
            else:
                oei, oet = add_inverses(ei[:, keep], et[keep])
                oef = None
            h = enc(nt, d.entity_id, d.numfeat, oei, oet, oef, d.sem_id)
            cand = (nt == nt[v]).nonzero(as_tuple=False).view(-1)
            cand = cand[cand != u]
            others = ei1[(ei0 == u) & (et == r)]
            others = others[others != v]                              # filtered: other true tails of (u,rel)
            if others.numel() > 0:
                cand = cand[~torch.isin(cand, others)]
            if cand.numel() < 2 or int((cand == v).sum()) == 0:
                continue
            sc = scorer(h, torch.full((cand.numel(),), u, dtype=torch.long, device=device), cand,
                        torch.full((cand.numel(),), r, dtype=torch.long, device=device))
            rank = int((sc > sc[(cand == v).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            rel_rr[r].append(1.0 / rank)
            rel_n[r] += 1
            rel_C[r].append(int(cand.numel()))
            for kk in (1, 3, 10):
                if rank <= kk:
                    rel_hits[r][kk] += 1
            nq += 1
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}
    per_rel = []
    all_rr = []
    tot = {1: 0, 3: 0, 10: 0}
    tot_n = 0
    for r_ in sorted(rel_n, key=lambda x: -rel_n[x]):
        n = rel_n[r_]
        C = float(np.mean(rel_C[r_]))
        all_rr += rel_rr[r_]
        tot_n += n
        for kk in tot:
            tot[kk] += rel_hits[r_][kk]
        per_rel.append({"rel": ID2REL.get(r_, f"rel{r_}"), "n": n, "mrr": float(np.mean(rel_rr[r_])),
                        "h1": rel_hits[r_][1] / n, "h3": rel_hits[r_][3] / n, "h10": rel_hits[r_][10] / n, "C": C,
                        "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0,
                        "chance_h1": 1.0 / C if C >= 1 else 1.0})
    return {"mrr": float(np.mean(all_rr)) if all_rr else float('nan'), "hits1": tot[1] / max(tot_n, 1),
            "hits3": tot[3] / max(tot_n, 1), "hits10": tot[10] / max(tot_n, 1), "n": tot_n, "per_rel": per_rel}


@torch.no_grad()
def cascade_evaluate(enc, scorer, graphs, order_ids, device, cfg):
    """(v13) iterative completion: start from BACKBONE-only context, recover the inferred relations in order_ids,
    adding each relation's GOLD edges to the context before the next (oracle cascade). Also computes the FLOOR
    (backbone-only context for every inferred relation). Ceiling = the v12 leave-one-out (all other edges present).
    """
    enc.eval()
    scorer.eval()
    TGT = [RELATION_CANONICAL[x] for x in TARGET_RELS]
    tgt_t = torch.tensor(sorted(set(TGT)), device=device)

    def recover(relid, cei, cet, cef, d, nt, ei, et, acc):
        if cfg.use_scores:                                            # (v14) gated encoder needs the context edges' scores
            oei, oet, oef = add_inverses(cei, cet, cef)
        else:
            oei, oet = add_inverses(cei, cet)
            oef = None
        h = enc(nt, d.entity_id.to(device), d.numfeat.to(device), oei, oet, oef, d.sem_id.to(device))
        for j in (et == relid).nonzero(as_tuple=False).view(-1).tolist():
            u = int(ei[0, j])
            v = int(ei[1, j])
            if u == v:
                continue
            cand = (nt == nt[v]).nonzero(as_tuple=False).view(-1)
            cand = cand[cand != u]
            others = ei[1][(ei[0] == u) & (et == relid)]
            others = others[others != v]
            if others.numel() > 0:
                cand = cand[~torch.isin(cand, others)]
            if cand.numel() < 2 or int((cand == v).sum()) == 0:
                continue
            sc = scorer(h, torch.full((cand.numel(),), u, dtype=torch.long, device=device), cand,
                        torch.full((cand.numel(),), relid, dtype=torch.long, device=device))
            rank = int((sc > sc[(cand == v).nonzero(as_tuple=False)[0, 0]]).sum().item()) + 1
            acc[relid].append((1.0 / rank, rank, int(cand.numel())))

    floor = defaultdict(list)
    casc = defaultdict(list)
    for d in graphs:
        d = d.to(device)
        ei = d.edge_index
        et = d.edge_type
        nt = d.node_type
        if ei.size(1) < 2:
            continue
        ef = d.edge_feat if cfg.use_scores else None                   # (v14) per-edge scores for the gated encoder
        bmask = ~torch.isin(et, tgt_t)                                 # backbone = the non-inferred scaffold
        bei = ei[:, bmask]
        bet = et[bmask]
        bef = ef[bmask] if cfg.use_scores else None
        for relid in TGT:
            recover(relid, bei, bet, bef, d, nt, ei, et, floor)        # FLOOR: backbone-only context
        cei = bei
        cet = bet
        cef = bef                                                      # CASCADE: add each relation's gold in order
        for relid in order_ids:
            recover(relid, cei, cet, cef, d, nt, ei, et, casc)
            add = (et == relid).nonzero(as_tuple=False).view(-1)
            if add.numel() > 0:
                cei = torch.cat([cei, ei[:, add]], 1)
                cet = torch.cat([cet, et[add]])
                if cfg.use_scores:
                    cef = torch.cat([cef, ef[add]], 0)
    ID2REL = {v: k for k, v in RELATION_CANONICAL.items()}

    def agg(acc):
        out = {}
        for relid, rows in acc.items():
            if not rows:
                continue
            C = float(np.mean([x[2] for x in rows]))
            out[ID2REL.get(relid, str(relid))] = {
                "n": len(rows), "mrr": float(np.mean([x[0] for x in rows])),
                "h1": float(np.mean([1.0 if x[1] <= 1 else 0.0 for x in rows])),
                "h10": float(np.mean([1.0 if x[1] <= 10 else 0.0 for x in rows])), "C": C,
                "chance_mrr": (math.log(C) + 0.5772) / C if C > 1 else 1.0}
        return out

    return {"floor": agg(floor), "cascade": agg(casc), "order": [ID2REL.get(r, str(r)) for r in order_ids]}


# ---- (v11) EIR scoring method (kg_similarity_scorer.py) + the refiner uplift eval ----

def edge_prf(pred, gold):
    """Directional (src_text, REL, dst_text) exact-triple set P/R/F1."""
    if not gold:
        return (1.0, 1.0, 1.0)
    if not pred:
        return (0.0, 0.0, 0.0)
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


@torch.no_grad()
def eir_uplift_eval(enc, scorer, pairs, tau, device, cfg):
    # refiner per TEST graph: GOLD = backbone + LLM edges with graded support>=tau; hold out EIR_HOLDOUT of gold LLM
    # edges; RAW = draft - held; REFINED = (keep supported, drop unsupported) + model-ADD (recover held-out top-1).
    enc.eval()
    scorer.eval()
    A = defaultdict(list)
    add = defaultdict(lambda: [0, 0])                    # add[REL]=[recovered, held]
    for d, g in pairs:
        nodes = g["nodes"]
        names = [(n.get("normalized_name") or n.get("name") or "") for n in nodes]
        idof = {n["id"]: i for i, n in enumerate(nodes)}
        E = []
        for e in g["edges"]:
            if e["source"] not in idof or e["target"] not in idof:
                continue
            try:
                rid = resolve_rel(e.get("relation"))
            except Exception:
                continue
            u = idof[e["source"]]
            v = idof[e["target"]]
            REL = str(e.get("relation", "")).upper()
            llm = (e.get("evidence") == "llm")
            sup = support_graded(e.get("labels")) if llm else 1.0
            E.append({"u": u, "v": v, "REL": REL, "rid": rid, "llm": llm, "sup": sup,
                      "tri": (normalize_text(names[u]), REL, normalize_text(names[v]))})
        gold = set(e["tri"] for e in E if (not e["llm"]) or e["sup"] >= tau)
        gold_llm = [e for e in E if e["llm"] and e["sup"] >= tau]
        if not gold or int(d.num_nodes) < 3:
            continue
        gen = torch.Generator().manual_seed(int(d.gid[0]) & 0x7FFFFFFF)
        order = torch.randperm(len(gold_llm), generator=gen).tolist() if gold_llm else []
        held = [gold_llm[i] for i in order[:int(round(cfg.eir_holdout * len(gold_llm)))]]
        held_tri = set(e["tri"] for e in held)
        raw_set = set(e["tri"] for e in E) - held_tri
        kept = [e for e in E if ((not e["llm"]) or e["sup"] >= tau) and e["tri"] not in held_tri]
        kept_set = set(e["tri"] for e in kept)
        nt = d.node_type.to(device)
        ntl = nt.tolist()
        if kept:
            oei = torch.tensor([[e["u"] for e in kept], [e["v"] for e in kept]], dtype=torch.long, device=device)
            oet = torch.tensor([e["rid"] for e in kept], dtype=torch.long, device=device)
        else:
            oei = torch.zeros((2, 0), dtype=torch.long, device=device)
            oet = torch.zeros((0,), dtype=torch.long, device=device)
        oei, oet = add_inverses(oei, oet)
        h = enc(nt, d.entity_id.to(device), d.numfeat.to(device), oei, oet, None, d.sem_id.to(device))
        added = set()
        for e in held:
            cand = [j for j in range(len(nodes)) if ntl[j] == ntl[e["v"]] and j != e["u"]]
            add[e["REL"]][1] += 1
            if not cand:
                continue
            ct = torch.tensor(cand, dtype=torch.long, device=device)
            sc = scorer(h, torch.full((len(cand),), e["u"], dtype=torch.long, device=device), ct,
                        torch.full((len(cand),), e["rid"], dtype=torch.long, device=device))
            vpred = cand[int(sc.argmax())]
            added.add((normalize_text(names[e["u"]]), e["REL"], normalize_text(names[vpred])))
            if vpred == e["v"]:
                add[e["REL"]][0] += 1
        refined = kept_set | added
        rp, rr, rf = edge_prf(raw_set, gold)
        fp, fr, ff = edge_prf(refined, gold)
        A["rawP"].append(rp)
        A["rawR"].append(rr)
        A["rawF"].append(rf)
        A["refP"].append(fp)
        A["refR"].append(fr)
        A["refF"].append(ff)
    mean = lambda x: float(np.mean(x)) if x else float('nan')
    out = {k: mean(A[k]) for k in ("rawP", "rawR", "rawF", "refP", "refR", "refF")}
    out["n_graphs"] = len(A["rawF"])
    out["add"] = {k: {"rec": v[0], "held": v[1], "recall": (v[0] / v[1] if v[1] else float('nan'))}
                  for k, v in add.items()}
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
