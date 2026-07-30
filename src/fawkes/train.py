"""Training steps and the experiment entry point.

REAL Graph-JEPA world model for clinical KG refinement (WM@Booth). PHASE 1 =
masked-node latent prediction (BYOL, EMA target) on ONE shared encoder = the
world model; PHASE 2 = frozen-encoder edge recovery readout. Inverse-edge
leakage + test-set-selection bugs fixed. CLAUDE.md: deterministic, [TAG] logging,
NO fallback (unknown type/relation/missing data fails loud). Added 260615.

Split out of ``paper_v16/trainer.py``; see ``docs/LINEAGE.md`` for the
full version history that used to sit at the top of that file.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph

from .config import Config
from .data import (NODE_TYPES, NUM_BASE, NUM_NODE_TYPES, RELATION_ALIASES, RELATION_CANONICAL,
                   TARGET_REL_IDS, TARGET_RELS, add_inverses, has_evidence, load_full_dataset,
                   resolve_rel, support_graded, to_data)
from .model import JEPA, build_scorer

logger = logging.getLogger("fawkes_jepa")


def set_seed(cfg):
    """Seed every RNG and request deterministic kernels.

    ``CUBLAS_WORKSPACE_CONFIG`` was set at module import in the original, with
    the comment "MUST precede import torch". What actually matters is that it is
    set before the first cuBLAS handle is created; this runs at the top of
    ``main`` and before any tensor work, so the effect is unchanged — and it
    keeps the package importable without touching the environment.
    """
    if cfg.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def valid_mask(b, ng, ratio, device, tries=10):
    n = b.size(0)
    for _ in range(tries):
        m = torch.rand(n, device=device) < ratio
        tg = torch.zeros(ng, device=device).scatter_add_(0, b, m.float())
        ct = torch.zeros(ng, device=device).scatter_add_(0, b, (~m).float())
        if bool((tg > 0).all()) and bool((ct > 0).all()):
            return m
    raise RuntimeError(f"[FAILURE] no valid node mask after {tries} tries (ratio={ratio}); graphs too small.")


def jepa_step(model, b, device, cfg):
    b = b.to(device)
    nt, eid, numf, ei, et, bt = b.node_type, b.entity_id, b.numfeat, b.edge_index, b.edge_type, b.batch
    N = nt.size(0)
    ng = int(bt.max().item()) + 1
    tmask = valid_mask(bt, ng, cfg.node_mask, device)
    cmask = ~tmask
    cn = cmask.nonzero(as_tuple=False).view(-1)
    if cfg.use_scores:
        cei, cet, emask = subgraph(cn, ei, edge_attr=et, relabel_nodes=True, num_nodes=N, return_edge_mask=True)
        cef = b.edge_feat[emask]
        cei, cet, cef = add_inverses(cei, cet, cef)
        ch = model.ctx(nt[cn], eid[cn], numf[cn], cei, cet, cef, b.sem_id[cn])
    else:
        cei, cet = subgraph(cn, ei, edge_attr=et, relabel_nodes=True, num_nodes=N)
        cei, cet = add_inverses(cei, cet)
        ch = model.ctx(nt[cn], eid[cn], numf[cn], cei, cet, None, b.sem_id[cn])
    csum = global_mean_pool(ch, bt[cn], size=ng)
    with torch.no_grad():
        if cfg.use_scores:
            fei, fet, fef = add_inverses(ei, et, b.edge_feat)
            th = model.tgt(nt, eid, numf, fei, fet, fef, b.sem_id)
        else:
            fei, fet = add_inverses(ei, et)
            th = model.tgt(nt, eid, numf, fei, fet, None, b.sem_id)
        emb_std = th.std(0).mean()
    tn = tmask.nonzero(as_tuple=False).view(-1)
    tgt = F.normalize(th[tn], dim=-1)
    g2c = torch.full((N,), -1, dtype=torch.long, device=device)
    g2c[cn] = torch.arange(cn.numel(), device=device)
    s_, d_ = ei[0], ei[1]
    m1 = tmask[s_] & cmask[d_]
    m2 = tmask[d_] & cmask[s_]
    tgt_ep = torch.cat([s_[m1], d_[m2]])
    nbr = torch.cat([d_[m1], s_[m2]])
    rels = torch.cat([et[m1], et[m2] + NUM_BASE])
    msg = ch[g2c[nbr]] + model.slot_rel(rels)
    slot = torch.zeros(N, cfg.hid, device=device).index_add_(0, tgt_ep, msg)
    cnt = torch.zeros(N, device=device).index_add_(0, tgt_ep, torch.ones(tgt_ep.numel(), device=device))
    slot = slot / cnt.clamp(min=1).unsqueeze(-1)
    q = slot[tn] + model.ctx.type_emb(nt[tn])
    if cfg.query_entity:
        q = q + model.ctx.entity_emb(eid[tn])
    pred = F.normalize(model.pred(torch.cat([csum[bt[tn]], q], -1)), dim=-1)
    return (2 - 2 * (pred * tgt).sum(-1)).mean(), emb_std


def buckets(nt, bt):
    bk = bt * NUM_NODE_TYPES + nt
    o = torch.argsort(bk)
    return o, bk[o]


def same_type_k(targets, nt, bt, o, sb, K, gen=None):
    tb = bt[targets] * NUM_NODE_TYPES + nt[targets]
    lo = torch.searchsorted(sb, tb, right=False)
    hi = torch.searchsorted(sb, tb, right=True)
    span = (hi - lo).clamp(min=1).float()
    r = torch.rand(targets.numel(), K, device=targets.device, generator=gen)
    pick = (r * span.unsqueeze(1)).long() + lo.unsqueeze(1)
    return o[pick.clamp(max=o.numel() - 1)]


def readout_step(enc, scorer, b, device, train, cfg, gen=None, mask_ratio=None):
    b = b.to(device)
    nt, eid, numf, ei, et, bt = b.node_type, b.entity_id, b.numfeat, b.edge_index, b.edge_type, b.batch
    E = ei.size(1)
    if E < 2:
        return None
    ratio = mask_ratio if mask_ratio is not None else cfg.edge_mask   # (v13) mask schedule: held-edge fraction
    perm = torch.randperm(E, device=device, generator=gen)
    k = max(1, min(E - 1, int(ratio * E)))
    hold = perm[:k]
    obs = perm[k:]
    if cfg.use_scores:
        oei, oet, oef = add_inverses(ei[:, obs], et[obs], b.edge_feat[obs])   # observed-only -> held-out scores never seen (no leak)
    else:
        oei, oet = add_inverses(ei[:, obs], et[obs])
        oef = None
    ctxmgr = torch.enable_grad() if (train and not cfg.freeze_encoder) else torch.no_grad()
    with ctxmgr:
        h = enc(nt, eid, numf, oei, oet, oef, b.sem_id)
    pu, pv, pr = ei[0, hold], ei[1, hold], et[hold]
    o, sb = buckets(nt, bt)
    nvk = same_type_k(pv, nt, bt, o, sb, cfg.neg_k, gen=gen)
    Pn = pu.numel()
    u_rep = pu.unsqueeze(1).expand(Pn, cfg.neg_k).reshape(-1)
    r_rep = pr.unsqueeze(1).expand(Pn, cfg.neg_k).reshape(-1)
    pos = scorer(h, pu, pv, pr).view(Pn, 1)
    neg = scorer(h, u_rep, nvk.reshape(-1), r_rep).view(Pn, cfg.neg_k)
    logits = torch.cat([pos, neg], 1) / cfg.temp
    logits[:, 1:][nvk == pv.unsqueeze(1)] = -1e9
    if cfg.loss == "bce":
        pos1 = pos.squeeze(1)
        neg1 = neg[:, 0]
        logits_bce = torch.cat([pos1, neg1])
        labels = torch.cat([torch.ones_like(pos1), torch.zeros_like(neg1)])
        loss = F.binary_cross_entropy_with_logits(logits_bce, labels)
    else:
        if cfg.target_weight != 1.0:                                 # (v13) upweight the 4 inferred relations
            ce = F.cross_entropy(logits, torch.zeros(Pn, dtype=torch.long, device=device), reduction='none')
            is_t = torch.isin(pr, torch.tensor(sorted(TARGET_REL_IDS), device=device))
            w = torch.where(is_t, torch.full((Pn,), cfg.target_weight, device=device), torch.ones(Pn, device=device))
            loss = (ce * w).sum() / w.sum()
        else:
            loss = F.cross_entropy(logits, torch.zeros(Pn, dtype=torch.long, device=device))
    qsig = (int(hold.sum().item()) * 1000003 + int(nvk.reshape(-1).sum().item()) + E * 7919) if gen is not None else 0
    return loss, pos.squeeze(1).detach(), neg[:, 0].detach(), (nt[pu] != 0).detach(), (h.detach(), pu, pv, pr, nt, bt, qsig)


def build_arg_parser() -> argparse.ArgumentParser:
    """A parser that accepts no options, only so ``--help`` cannot train.

    This experiment is configured entirely from the environment — see
    :meth:`Config.from_env` — so there are deliberately no flags to add. The
    parser exists because ``main`` previously took no ``argv`` and ran
    unconditionally, which meant ``fawkes-train --help`` completed a full
    training run and, with ``PUSH=1``, uploaded the result. Parsing first makes
    ``--help`` print usage and exit, and makes an unrecognized flag an error
    instead of a 3-minute surprise.
    """
    return argparse.ArgumentParser(
        prog="fawkes-train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train the fawkes entity-note Graph-JEPA (the paper implementation).",
        epilog=(
            "Takes no arguments. Configuration comes from environment variables;\n"
            "Config.from_env in fawkes/config.py is the full list. Typical run:\n"
            "\n"
            "  USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 \\\n"
            "    PRUNE_NO_EVIDENCE=1 PUSH=0 fawkes-train\n"
            "\n"
            "CAUTION: a bare `fawkes-train` runs the full two-phase training and,\n"
            "because PUSH defaults to 1, uploads the checkpoint to OUTPUT_REPO on\n"
            "Hugging Face. Set PUSH=0 to keep it local."
        ),
    )


def main(argv=None):
    # Parse before anything else: this is the guard that keeps --help and typos
    # from starting a training run. See build_arg_parser.
    build_arg_parser().parse_args(argv)

    # Deferred: evaluate.py imports readout_step/buckets/same_type_k from this
    # module, so importing it at module scope would be circular.
    from .evaluate import cascade_evaluate, eir_uplift_eval, evaluate, loo_evaluate

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    cfg = Config.from_env()
    logger.info(f"[ENTRY] ENTITY-NOTE v16 | data={cfg.data_repo} use_note={cfg.use_note} ground_by={cfg.ground_by} embed_dim={cfg.embed_dim} numeric_dim={cfg.numeric_dim} use_scores={cfg.use_scores} prune_no_evidence={cfg.prune_no_evidence} mask_schedule={cfg.mask_schedule}[{cfg.mask_lo},{cfg.mask_hi}] target_weight={cfg.target_weight} cascade={cfg.run_cascade} order={cfg.cascade_order} jepa_ep={cfg.jepa_epochs} readout_ep={cfg.readout_epochs} loo_cap={cfg.loo_cap} run_eir={cfg.run_eir} "
                f"hid={cfg.hid} node_mask={cfg.node_mask} edge_mask={cfg.edge_mask} entity_emb={cfg.use_entity_emb} decoder={cfg.decoder} loss={cfg.loss} neg_k={cfg.neg_k} freeze_eval={cfg.freeze_eval} deterministic={cfg.deterministic} val={cfg.val_frac} test={cfg.test_frac} seed={cfg.seed}")
    set_seed(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[GPU] {dev}" + (f" {torch.cuda.get_device_name(0)}" if dev.type == 'cuda' else ""))
    raw, demo = load_full_dataset(cfg)
    if cfg.prune_no_evidence:
        tot_llm = sum(1 for g in raw for e in g.get("edges", []) if e.get("evidence") == "llm")
        no_ev = sum(1 for g in raw for e in g.get("edges", [])
                    if e.get("evidence") == "llm" and not has_evidence(e.get("labels")))
        logger.info(f"[PRUNE] LLM edges={tot_llm} no-evidence(pruned)={no_ev} ({100.0*no_ev/max(tot_llm,1):.1f}%) | backbone + evidenced LLM kept")
    # vocab audit (fail loud)
    rels = set()
    types = set()
    for g in raw:
        for n in g.get("nodes", []):
            types.add(n.get("type"))
        for e in g.get("edges", []):
            rels.add(e.get("relation"))
    ur = sorted(r for r in rels if r and r not in (set(RELATION_CANONICAL) | set(RELATION_ALIASES)))
    ut = sorted(t for t in types if t and t not in NODE_TYPES)
    logger.info(f"[VOCAB] rel={len(rels)} type={len(types)} unknown_rel={len(ur)} unknown_type={len(ut)}")
    if ur or ut:
        raise RuntimeError(f"[FAILURE] unknown relations={ur} types={ut}; no fallback.")
    items = []
    for g in raw:
        d = to_data(g, demo, cfg)
        if d.num_nodes >= 3 and d.edge_index.size(1) >= 4:
            items.append((d, g))   # keep raw graph alongside for the EIR eval
    if cfg.use_note:
        ng = sum(int(d.n_grounded[0]) for d, _ in items)
        n_nodes = sum(int(d.num_nodes) for d, _ in items)
        logger.info(f"[NOTE] ground_by={cfg.ground_by} grounded entities={ng}/{n_nodes} ({100.0*ng/max(n_nodes,1):.1f}%) carry the note vector; the rest get a zero note")
    rng = np.random.RandomState(cfg.seed)
    idx = rng.permutation(len(items))
    nte = int(cfg.test_frac * len(items))
    nval = int(cfg.val_frac * len(items))
    te_pairs = [items[i] for i in idx[:nte]]
    val_pairs = [items[i] for i in idx[nte:nte + nval]]
    tr_pairs = [items[i] for i in idx[nte + nval:]]
    te = [d for d, _ in te_pairs]
    val = [d for d, _ in val_pairs]
    tr = [d for d, _ in tr_pairs]
    logger.info(f"[DATA] graphs={len(items)} -> TRAIN={len(tr)} VAL={len(val)} TEST={len(te)} (seeded split)")

    # ---- PHASE 1: JEPA world-model pretraining ----
    model = JEPA(cfg).to(dev)
    opt = torch.optim.Adam([p for p in model.ctx.parameters() if p.requires_grad]
                           + list(model.pred.parameters()) + list(model.slot_rel.parameters()), lr=cfg.lr)
    logger.info(f"[MODEL] JEPA params={sum(p.numel() for p in model.parameters()):,} edge_dim={model.ctx.edim}")
    tl = DataLoader(tr, batch_size=cfg.batch, shuffle=True)
    for ep in range(1, cfg.jepa_epochs + 1):
        model.ctx.train()
        model.ema = cfg.ema_final - (cfg.ema_final - cfg.ema_base) * (math.cos(math.pi * (ep - 1) / max(cfg.jepa_epochs, 1)) + 1) / 2
        t0 = time.perf_counter()
        tot = ts = nb = 0
        for b in tl:
            opt.zero_grad()
            loss, es = jepa_step(model, b, dev, cfg)
            loss.backward()
            opt.step()
            model.update()
            tot += loss.item()
            ts += es.item()
            nb += 1
        if ep % 10 == 0 or ep == cfg.jepa_epochs:
            logger.info(f"[LATENCY] JEPA epoch={ep}/{cfg.jepa_epochs} loss={tot/nb:.4f} emb_std={ts/nb:.4f} ema={model.ema:.4f} took={(time.perf_counter()-t0)*1000:.0f}ms")

    # ---- PHASE 2: downstream edge-recovery readout on the (frozen) world-model encoder ----
    enc = model.ctx
    if cfg.freeze_encoder:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
    scorer = build_scorer(cfg).to(dev)
    params = list(scorer.parameters()) + ([] if cfg.freeze_encoder else list(enc.parameters()))
    ropt = torch.optim.Adam(params, lr=cfg.lr)
    eb = 1 if cfg.freeze_eval else cfg.batch
    rl = DataLoader(tr, batch_size=cfg.batch, shuffle=True)
    vl = DataLoader(val, batch_size=eb, shuffle=False)
    el = DataLoader(te, batch_size=eb, shuffle=False)
    best = {"auc": 0.0}
    best_state = None
    for ep in range(1, cfg.readout_epochs + 1):
        scorer.train()
        if not cfg.freeze_encoder:
            enc.train()
        t0 = time.perf_counter()
        tot = nb = 0
        for b in rl:
            mr = (cfg.mask_lo + (cfg.mask_hi - cfg.mask_lo) * float(torch.rand(1).item())) if cfg.mask_schedule else None   # (B) sampled context density per step
            r = readout_step(enc, scorer, b, dev, True, cfg, mask_ratio=mr)
            if r is None:
                continue
            ropt.zero_grad()
            r[0].backward()
            ropt.step()
            tot += r[0].item()
            nb += 1
        if ep % 5 == 0 or ep == cfg.readout_epochs:
            vm = evaluate(enc, scorer, vl, dev, cfg)
            logger.info(f"[LATENCY] READOUT epoch={ep}/{cfg.readout_epochs} loss={tot/max(nb,1):.4f} took={(time.perf_counter()-t0)*1000:.0f}ms "
                        f"| VAL auc={vm['auc']:.3f} nonobv={vm['auc_nonobvious']:.3f} MRR={vm['mrr']:.3f} H@1={vm['hits1']:.3f} H@10={vm['hits10']:.3f}")
            if vm["auc"] > best["auc"]:
                best = {**vm, "epoch": ep}
                best_state = copy.deepcopy(scorer.state_dict())

    if best_state is not None:
        scorer.load_state_dict(best_state)
    tm = evaluate(enc, scorer, el, dev, cfg)
    logger.info(f"[RESULT] TEST (val-selected) | auc={tm['auc']:.3f} ap={tm['ap']:.3f} non-obvious_auc={tm['auc_nonobvious']:.3f} "
                f"MRR={tm['mrr']:.3f} Hits@1={tm['hits1']:.3f} Hits@3={tm['hits3']:.3f} Hits@10={tm['hits10']:.3f} (n_mrr={tm['n_mrr']}) quiz_sig={tm['qsig']} frozen={cfg.freeze_eval} scores={cfg.use_scores}")
    logger.info("[PER-REL] ONE shared latent, edge recovery by relation (sorted by edge count):")
    for rr_ in tm["per_rel"]:
        flag = " <== ABOVE chance" if rr_["mrr"] > 1.5 * rr_["chance_mrr"] else (" <== ~chance" if rr_["mrr"] < 1.2 * rr_["chance_mrr"] else "")
        logger.info(f"[PER-REL] {rr_['rel']:<22} n={rr_['n']:<5} C={rr_['C']:.0f}  MRR={rr_['mrr']:.3f} (chance {rr_['chance_mrr']:.3f})  H@1={rr_['h1']:.3f}  H@10={rr_['h10']:.3f}{flag}")

    # ---- (v12) LEAVE-ONE-OUT edge recovery: mask ONE edge, keep the rest, recover it (full context, filtered) ----
    loo = loo_evaluate(enc, scorer, te, dev, cfg)
    logger.info(f"[LOO] leave-one-out (mask 1 edge, full context, filtered) | MRR={loo['mrr']:.3f} H@1={loo['hits1']:.3f} H@3={loo['hits3']:.3f} H@10={loo['hits10']:.3f} over {loo['n']} edges")
    logger.info("[LOO] edge recovery by relation (one edge masked at a time, full surrounding context):")
    for rr_ in loo["per_rel"]:
        flag = " <== ABOVE chance" if rr_["mrr"] > 1.5 * rr_["chance_mrr"] else (" <== ~chance" if rr_["mrr"] < 1.2 * rr_["chance_mrr"] else "")
        logger.info(f"[LOO] {rr_['rel']:<22} n={rr_['n']:<5} C={rr_['C']:.0f}  MRR={rr_['mrr']:.3f} (chance {rr_['chance_mrr']:.3f})  H@1={rr_['h1']:.3f}  H@10={rr_['h10']:.3f}{flag}")
    bm = {x["rel"]: x for x in tm["per_rel"]}
    lo = {x["rel"]: x for x in loo["per_rel"]}
    logger.info("[CONTRAST] 4 inferred LLM edges — 30%-batch-mask MRR -> leave-one-out MRR (gain from keeping full context):")
    for rel in sorted(TARGET_RELS):
        a = bm.get(rel)
        c = lo.get(rel)
        if a and c:
            logger.info(f"[CONTRAST] {rel:<16} {a['mrr']:.3f} -> {c['mrr']:.3f}  ({c['mrr']-a['mrr']:+.3f})  | H@1 {a['h1']:.3f} -> {c['h1']:.3f}")

    # ---- (v13) CASCADE: backbone-only FLOOR -> ordered cascade -> LOO ceiling (does order matter?) ----
    casc = None
    if cfg.run_cascade:
        order_ids = [resolve_rel(x) for x in cfg.cascade_order]
        rev_ids = list(reversed(order_ids))
        c1 = cascade_evaluate(enc, scorer, te, order_ids, dev, cfg)
        c2 = cascade_evaluate(enc, scorer, te, rev_ids, dev, cfg)
        casc = {"forward": c1, "reverse": c2}
        logger.info(f"[CASCADE] order = {' -> '.join(c1['order'])}  (oracle: earlier relations' GOLD edges added to context)")
        logger.info("[CASCADE] per relation: FLOOR (backbone-only) -> CASCADE (this order) -> CEILING (LOO, all else present):")
        for rel in cfg.cascade_order:
            fl = c1["floor"].get(rel)
            ca = c1["cascade"].get(rel)
            ce = lo.get(rel)
            if fl and ca and ce:
                head = ce["mrr"] - fl["mrr"]
                got = ca["mrr"] - fl["mrr"]
                logger.info(f"[CASCADE] {rel:<16} MRR {fl['mrr']:.3f} -> {ca['mrr']:.3f} -> {ce['mrr']:.3f}  (chance {fl['chance_mrr']:.3f}) | H@1 {fl['h1']:.3f} -> {ca['h1']:.3f} -> {ce['h1']:.3f} | recovered {got:+.3f} of {head:+.3f} headroom")
        logger.info(f"[CASCADE-ORDER] reverse = {' -> '.join(c2['order'])} (order-dependence check):")
        for rel in cfg.cascade_order:
            caf = c1["cascade"].get(rel)
            car = c2["cascade"].get(rel)
            if caf and car:
                logger.info(f"[CASCADE-ORDER] {rel:<16} forward {caf['mrr']:.3f}  vs  reverse {car['mrr']:.3f}  ({car['mrr']-caf['mrr']:+.3f})")

    # ---- (v11, optional) EIR batch-holdout uplift — OFF by default (full-gold F1 is hub-dominated; LOO above is the honest measure) ----
    tau = None
    eir = None
    if cfg.run_eir:
        sup_v = [support_graded(e.get("labels")) for _, g in val_pairs for e in g["edges"] if e.get("evidence") == "llm"]
        tau = float(np.quantile(sup_v, 0.5)) if sup_v else 0.5
        logger.info(f"[EIR] gold/keep tau (VAL median graded LLM support) = {tau:.3f} | holdout={cfg.eir_holdout} fuzzy={cfg.eir_fuzzy}")
        eir = eir_uplift_eval(enc, scorer, te_pairs, tau, dev, cfg)
        logger.info(f"[EIR-UPLIFT] TEST edge-triple vs v8-gold ({eir['n_graphs']} graphs) | RAW P={eir['rawP']:.3f} R={eir['rawR']:.3f} F1={eir['rawF']:.3f}")
        logger.info(f"[EIR-UPLIFT] TEST edge-triple vs v8-gold | REFINED P={eir['refP']:.3f} R={eir['refR']:.3f} F1={eir['refF']:.3f}  (F1 {eir['refF']-eir['rawF']:+.3f} | precision {eir['refP']-eir['rawP']:+.3f} = DISCONNECT, recall {eir['refR']-eir['rawR']:+.3f} = model ADD)")
        logger.info("[EIR-ADD] held-out evidence-supported edges recovered (top-1 same-type), the 4 LLM targets:")
        for rel in sorted(TARGET_RELS):
            a = eir["add"].get(rel, {"rec": 0, "held": 0, "recall": float('nan')})
            logger.info(f"[EIR-ADD] {rel:<16} recovered = {a['rec']}/{a['held']} = {(a['recall'] if a['recall']==a['recall'] else 0.0):.3f}")

    ckpt = cfg.checkpoint_name
    torch.save({"encoder": enc.state_dict(), "scorer": scorer.state_dict(),
                "config": cfg.checkpoint_dict(),
                "recovery_test_batchmask": tm, "recovery_test_loo": loo, "cascade": casc, "eir": eir}, ckpt)
    if not cfg.push:
        logger.info(f"[DONE] PUSH=0 — saved local {ckpt}")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(cfg.output_repo, repo_type="model", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=ckpt, path_in_repo=ckpt, repo_id=cfg.output_repo, repo_type="model")
    logger.info(f"[DONE] v16 ENTITY-NOTE -> https://huggingface.co/{cfg.output_repo}/blob/main/{ckpt} | LOO MRR={loo['mrr']:.3f} | use_note={cfg.use_note} ground_by={cfg.ground_by} use_scores={cfg.use_scores}")


if __name__ == "__main__":
    main()
