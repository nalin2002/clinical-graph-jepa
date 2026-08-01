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


def valid_mask(batch_index, num_graphs, ratio, device, tries=10):
    num_nodes = batch_index.size(0)
    for _ in range(tries):
        mask = torch.rand(num_nodes, device=device) < ratio
        targets_per_graph = torch.zeros(num_graphs, device=device).scatter_add_(0, batch_index, mask.float())
        context_per_graph = torch.zeros(num_graphs, device=device).scatter_add_(0, batch_index, (~mask).float())
        if bool((targets_per_graph > 0).all()) and bool((context_per_graph > 0).all()):
            return mask
    raise RuntimeError(f"[FAILURE] no valid node mask after {tries} tries (ratio={ratio}); graphs too small.")


def jepa_step(model, batch, device, cfg):
    batch = batch.to(device)
    node_type, entity_id, numfeat = batch.node_type, batch.entity_id, batch.numfeat
    edge_index, edge_type, batch_index = batch.edge_index, batch.edge_type, batch.batch
    num_nodes = node_type.size(0)
    num_graphs = int(batch_index.max().item()) + 1
    target_mask = valid_mask(batch_index, num_graphs, cfg.node_mask, device)
    context_mask = ~target_mask
    context_nodes = context_mask.nonzero(as_tuple=False).view(-1)
    if cfg.use_scores:
        ctx_edge_index, ctx_edge_type, kept_edges = subgraph(
            context_nodes, edge_index, edge_attr=edge_type, relabel_nodes=True, num_nodes=num_nodes, return_edge_mask=True)
        ctx_edge_feat = batch.edge_feat[kept_edges]
        ctx_edge_index, ctx_edge_type, ctx_edge_feat = add_inverses(ctx_edge_index, ctx_edge_type, ctx_edge_feat)
        context_repr = model.ctx(node_type[context_nodes], entity_id[context_nodes], numfeat[context_nodes],
                                 ctx_edge_index, ctx_edge_type, ctx_edge_feat, batch.sem_id[context_nodes])
    else:
        ctx_edge_index, ctx_edge_type = subgraph(
            context_nodes, edge_index, edge_attr=edge_type, relabel_nodes=True, num_nodes=num_nodes)
        ctx_edge_index, ctx_edge_type = add_inverses(ctx_edge_index, ctx_edge_type)
        context_repr = model.ctx(node_type[context_nodes], entity_id[context_nodes], numfeat[context_nodes],
                                 ctx_edge_index, ctx_edge_type, None, batch.sem_id[context_nodes])
    context_summary = global_mean_pool(context_repr, batch_index[context_nodes], size=num_graphs)
    with torch.no_grad():
        if cfg.use_scores:
            full_edge_index, full_edge_type, full_edge_feat = add_inverses(edge_index, edge_type, batch.edge_feat)
            target_repr = model.tgt(node_type, entity_id, numfeat, full_edge_index, full_edge_type, full_edge_feat, batch.sem_id)
        else:
            full_edge_index, full_edge_type = add_inverses(edge_index, edge_type)
            target_repr = model.tgt(node_type, entity_id, numfeat, full_edge_index, full_edge_type, None, batch.sem_id)
        emb_std = target_repr.std(0).mean()
    target_nodes = target_mask.nonzero(as_tuple=False).view(-1)
    target_emb = F.normalize(target_repr[target_nodes], dim=-1)
    global_to_context = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    global_to_context[context_nodes] = torch.arange(context_nodes.numel(), device=device)
    edge_src, edge_dst = edge_index[0], edge_index[1]
    src_is_target = target_mask[edge_src] & context_mask[edge_dst]
    dst_is_target = target_mask[edge_dst] & context_mask[edge_src]
    masked_endpoints = torch.cat([edge_src[src_is_target], edge_dst[dst_is_target]])
    neighbors = torch.cat([edge_dst[src_is_target], edge_src[dst_is_target]])
    neighbor_rels = torch.cat([edge_type[src_is_target], edge_type[dst_is_target] + NUM_BASE])
    messages = context_repr[global_to_context[neighbors]] + model.slot_rel(neighbor_rels)
    slot = torch.zeros(num_nodes, cfg.hid, device=device).index_add_(0, masked_endpoints, messages)
    msg_count = torch.zeros(num_nodes, device=device).index_add_(
        0, masked_endpoints, torch.ones(masked_endpoints.numel(), device=device))
    slot = slot / msg_count.clamp(min=1).unsqueeze(-1)
    query = slot[target_nodes] + model.ctx.type_emb(node_type[target_nodes])
    if cfg.query_entity:
        query = query + model.ctx.entity_emb(entity_id[target_nodes])
    prediction = F.normalize(model.pred(torch.cat([context_summary[batch_index[target_nodes]], query], -1)), dim=-1)
    return (2 - 2 * (prediction * target_emb).sum(-1)).mean(), emb_std


def buckets(node_type, batch_index):
    bucket_ids = batch_index * NUM_NODE_TYPES + node_type
    order = torch.argsort(bucket_ids)
    return order, bucket_ids[order]


def same_type_k(targets, node_type, batch_index, order, sorted_buckets, num_neg, gen=None):
    target_buckets = batch_index[targets] * NUM_NODE_TYPES + node_type[targets]
    lo = torch.searchsorted(sorted_buckets, target_buckets, right=False)
    hi = torch.searchsorted(sorted_buckets, target_buckets, right=True)
    span = (hi - lo).clamp(min=1).float()
    rand = torch.rand(targets.numel(), num_neg, device=targets.device, generator=gen)
    picks = (rand * span.unsqueeze(1)).long() + lo.unsqueeze(1)
    return order[picks.clamp(max=order.numel() - 1)]


def readout_step(encoder, scorer, batch, device, training, cfg, gen=None, mask_ratio=None):
    batch = batch.to(device)
    node_type, entity_id, numfeat = batch.node_type, batch.entity_id, batch.numfeat
    edge_index, edge_type, batch_index = batch.edge_index, batch.edge_type, batch.batch
    num_edges = edge_index.size(1)
    if num_edges < 2:
        return None
    ratio = mask_ratio if mask_ratio is not None else cfg.edge_mask   # (v13) mask schedule: held-edge fraction
    perm = torch.randperm(num_edges, device=device, generator=gen)
    num_held = max(1, min(num_edges - 1, int(ratio * num_edges)))
    held = perm[:num_held]
    observed = perm[num_held:]
    if cfg.use_scores:
        obs_edge_index, obs_edge_type, obs_edge_feat = add_inverses(
            edge_index[:, observed], edge_type[observed], batch.edge_feat[observed])   # observed-only -> held-out scores never seen (no leak)
    else:
        obs_edge_index, obs_edge_type = add_inverses(edge_index[:, observed], edge_type[observed])
        obs_edge_feat = None
    grad_mode = torch.enable_grad() if (training and not cfg.freeze_encoder) else torch.no_grad()
    with grad_mode:
        hidden = encoder(node_type, entity_id, numfeat, obs_edge_index, obs_edge_type, obs_edge_feat, batch.sem_id)
    held_src, held_dst, held_rel = edge_index[0, held], edge_index[1, held], edge_type[held]
    order, sorted_buckets = buckets(node_type, batch_index)
    neg_dst = same_type_k(held_dst, node_type, batch_index, order, sorted_buckets, cfg.neg_k, gen=gen)
    num_pos = held_src.numel()
    src_expanded = held_src.unsqueeze(1).expand(num_pos, cfg.neg_k).reshape(-1)
    rel_expanded = held_rel.unsqueeze(1).expand(num_pos, cfg.neg_k).reshape(-1)
    pos_scores = scorer(hidden, held_src, held_dst, held_rel).view(num_pos, 1)
    neg_scores = scorer(hidden, src_expanded, neg_dst.reshape(-1), rel_expanded).view(num_pos, cfg.neg_k)
    logits = torch.cat([pos_scores, neg_scores], 1) / cfg.temp
    logits[:, 1:][neg_dst == held_dst.unsqueeze(1)] = -1e9
    if cfg.loss == "bce":
        pos_flat = pos_scores.squeeze(1)
        neg_flat = neg_scores[:, 0]
        logits_bce = torch.cat([pos_flat, neg_flat])
        labels = torch.cat([torch.ones_like(pos_flat), torch.zeros_like(neg_flat)])
        loss = F.binary_cross_entropy_with_logits(logits_bce, labels)
    else:
        if cfg.target_weight != 1.0:                                 # (v13) upweight the 4 inferred relations
            per_edge_ce = F.cross_entropy(logits, torch.zeros(num_pos, dtype=torch.long, device=device), reduction='none')
            is_target = torch.isin(held_rel, torch.tensor(sorted(TARGET_REL_IDS), device=device))
            weights = torch.where(is_target, torch.full((num_pos,), cfg.target_weight, device=device),
                                  torch.ones(num_pos, device=device))
            loss = (per_edge_ce * weights).sum() / weights.sum()
        else:
            loss = F.cross_entropy(logits, torch.zeros(num_pos, dtype=torch.long, device=device))
    qsig = (int(held.sum().item()) * 1000003 + int(neg_dst.reshape(-1).sum().item()) + num_edges * 7919) if gen is not None else 0
    return (loss, pos_scores.squeeze(1).detach(), neg_scores[:, 0].detach(), (node_type[held_src] != 0).detach(),
            (hidden.detach(), held_src, held_dst, held_rel, node_type, batch_index, qsig))


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[GPU] {device}" + (f" {torch.cuda.get_device_name(0)}" if device.type == 'cuda' else ""))
    raw, demographics = load_full_dataset(cfg)
    if cfg.prune_no_evidence:
        total_llm = sum(1 for g in raw for e in g.get("edges", []) if e.get("evidence") == "llm")
        no_evidence = sum(1 for g in raw for e in g.get("edges", [])
                          if e.get("evidence") == "llm" and not has_evidence(e.get("labels")))
        logger.info(f"[PRUNE] LLM edges={total_llm} no-evidence(pruned)={no_evidence} ({100.0*no_evidence/max(total_llm,1):.1f}%) | backbone + evidenced LLM kept")
    # vocab audit (fail loud)
    seen_relations = set()
    seen_types = set()
    for g in raw:
        for n in g.get("nodes", []):
            seen_types.add(n.get("type"))
        for e in g.get("edges", []):
            seen_relations.add(e.get("relation"))
    unknown_relations = sorted(r for r in seen_relations
                               if r and r not in (set(RELATION_CANONICAL) | set(RELATION_ALIASES)))
    unknown_types = sorted(t for t in seen_types if t and t not in NODE_TYPES)
    logger.info(f"[VOCAB] rel={len(seen_relations)} type={len(seen_types)} unknown_rel={len(unknown_relations)} unknown_type={len(unknown_types)}")
    if unknown_relations or unknown_types:
        raise RuntimeError(f"[FAILURE] unknown relations={unknown_relations} types={unknown_types}; no fallback.")
    items = []
    for g in raw:
        d = to_data(g, demographics, cfg)
        if d.num_nodes >= 3 and d.edge_index.size(1) >= 4:
            items.append((d, g))   # keep raw graph alongside for the EIR eval
    if cfg.use_note:
        num_grounded = sum(int(d.n_grounded[0]) for d, _ in items)
        total_nodes = sum(int(d.num_nodes) for d, _ in items)
        logger.info(f"[NOTE] ground_by={cfg.ground_by} grounded entities={num_grounded}/{total_nodes} ({100.0*num_grounded/max(total_nodes,1):.1f}%) carry the note vector; the rest get a zero note")
    rng = np.random.RandomState(cfg.seed)
    shuffled = rng.permutation(len(items))
    num_test = int(cfg.test_frac * len(items))
    num_val = int(cfg.val_frac * len(items))
    test_pairs = [items[i] for i in shuffled[:num_test]]
    val_pairs = [items[i] for i in shuffled[num_test:num_test + num_val]]
    train_pairs = [items[i] for i in shuffled[num_test + num_val:]]
    test_graphs = [d for d, _ in test_pairs]
    val_graphs = [d for d, _ in val_pairs]
    train_graphs = [d for d, _ in train_pairs]
    logger.info(f"[DATA] graphs={len(items)} -> TRAIN={len(train_graphs)} VAL={len(val_graphs)} TEST={len(test_graphs)} (seeded split)")

    # ---- PHASE 1: JEPA world-model pretraining ----
    model = JEPA(cfg).to(device)
    optimizer = torch.optim.Adam([p for p in model.ctx.parameters() if p.requires_grad]
                                 + list(model.pred.parameters()) + list(model.slot_rel.parameters()), lr=cfg.lr)
    logger.info(f"[MODEL] JEPA params={sum(p.numel() for p in model.parameters()):,} edge_dim={model.ctx.edge_dim}")
    jepa_loader = DataLoader(train_graphs, batch_size=cfg.batch, shuffle=True)
    for epoch in range(1, cfg.jepa_epochs + 1):
        model.ctx.train()
        model.ema = cfg.ema_final - (cfg.ema_final - cfg.ema_base) * (math.cos(math.pi * (epoch - 1) / max(cfg.jepa_epochs, 1)) + 1) / 2
        start = time.perf_counter()
        total_loss = total_std = num_batches = 0
        for batch in jepa_loader:
            optimizer.zero_grad()
            loss, emb_std = jepa_step(model, batch, device, cfg)
            loss.backward()
            optimizer.step()
            model.update()
            total_loss += loss.item()
            total_std += emb_std.item()
            num_batches += 1
        if epoch % 10 == 0 or epoch == cfg.jepa_epochs:
            logger.info(f"[LATENCY] JEPA epoch={epoch}/{cfg.jepa_epochs} loss={total_loss/num_batches:.4f} emb_std={total_std/num_batches:.4f} ema={model.ema:.4f} took={(time.perf_counter()-start)*1000:.0f}ms")

    # ---- PHASE 2: downstream edge-recovery readout on the (frozen) world-model encoder ----
    encoder = model.ctx
    if cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
    scorer = build_scorer(cfg).to(device)
    params = list(scorer.parameters()) + ([] if cfg.freeze_encoder else list(encoder.parameters()))
    readout_opt = torch.optim.Adam(params, lr=cfg.lr)
    eval_batch = 1 if cfg.freeze_eval else cfg.batch
    readout_loader = DataLoader(train_graphs, batch_size=cfg.batch, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=eval_batch, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=eval_batch, shuffle=False)
    best = {"auc": 0.0}
    best_state = None
    for epoch in range(1, cfg.readout_epochs + 1):
        scorer.train()
        if not cfg.freeze_encoder:
            encoder.train()
        start = time.perf_counter()
        total_loss = num_batches = 0
        for batch in readout_loader:
            sampled_ratio = (cfg.mask_lo + (cfg.mask_hi - cfg.mask_lo) * float(torch.rand(1).item())) if cfg.mask_schedule else None   # (B) sampled context density per step
            step_out = readout_step(encoder, scorer, batch, device, True, cfg, mask_ratio=sampled_ratio)
            if step_out is None:
                continue
            readout_opt.zero_grad()
            step_out[0].backward()
            readout_opt.step()
            total_loss += step_out[0].item()
            num_batches += 1
        if epoch % 5 == 0 or epoch == cfg.readout_epochs:
            val_metrics = evaluate(encoder, scorer, val_loader, device, cfg)
            logger.info(f"[LATENCY] READOUT epoch={epoch}/{cfg.readout_epochs} loss={total_loss/max(num_batches,1):.4f} took={(time.perf_counter()-start)*1000:.0f}ms "
                        f"| VAL auc={val_metrics['auc']:.3f} nonobv={val_metrics['auc_nonobvious']:.3f} MRR={val_metrics['mrr']:.3f} H@1={val_metrics['hits1']:.3f} H@10={val_metrics['hits10']:.3f}")
            if val_metrics["auc"] > best["auc"]:
                best = {**val_metrics, "epoch": epoch}
                best_state = copy.deepcopy(scorer.state_dict())

    if best_state is not None:
        scorer.load_state_dict(best_state)
    test_metrics = evaluate(encoder, scorer, test_loader, device, cfg)
    logger.info(f"[RESULT] TEST (val-selected) | auc={test_metrics['auc']:.3f} ap={test_metrics['ap']:.3f} non-obvious_auc={test_metrics['auc_nonobvious']:.3f} "
                f"MRR={test_metrics['mrr']:.3f} Hits@1={test_metrics['hits1']:.3f} Hits@3={test_metrics['hits3']:.3f} Hits@10={test_metrics['hits10']:.3f} (n_mrr={test_metrics['n_mrr']}) quiz_sig={test_metrics['qsig']} frozen={cfg.freeze_eval} scores={cfg.use_scores}")
    logger.info("[PER-REL] ONE shared latent, edge recovery by relation (sorted by edge count):")
    for rel_row in test_metrics["per_rel"]:
        flag = " <== ABOVE chance" if rel_row["mrr"] > 1.5 * rel_row["chance_mrr"] else (" <== ~chance" if rel_row["mrr"] < 1.2 * rel_row["chance_mrr"] else "")
        logger.info(f"[PER-REL] {rel_row['rel']:<22} n={rel_row['n']:<5} C={rel_row['C']:.0f}  MRR={rel_row['mrr']:.3f} (chance {rel_row['chance_mrr']:.3f})  H@1={rel_row['h1']:.3f}  H@10={rel_row['h10']:.3f}{flag}")

    # ---- (v12) LEAVE-ONE-OUT edge recovery: mask ONE edge, keep the rest, recover it (full context, filtered) ----
    loo = loo_evaluate(encoder, scorer, test_graphs, device, cfg)
    logger.info(f"[LOO] leave-one-out (mask 1 edge, full context, filtered) | MRR={loo['mrr']:.3f} H@1={loo['hits1']:.3f} H@3={loo['hits3']:.3f} H@10={loo['hits10']:.3f} over {loo['n']} edges")
    logger.info("[LOO] edge recovery by relation (one edge masked at a time, full surrounding context):")
    for rel_row in loo["per_rel"]:
        flag = " <== ABOVE chance" if rel_row["mrr"] > 1.5 * rel_row["chance_mrr"] else (" <== ~chance" if rel_row["mrr"] < 1.2 * rel_row["chance_mrr"] else "")
        logger.info(f"[LOO] {rel_row['rel']:<22} n={rel_row['n']:<5} C={rel_row['C']:.0f}  MRR={rel_row['mrr']:.3f} (chance {rel_row['chance_mrr']:.3f})  H@1={rel_row['h1']:.3f}  H@10={rel_row['h10']:.3f}{flag}")
    batchmask_by_rel = {x["rel"]: x for x in test_metrics["per_rel"]}
    loo_by_rel = {x["rel"]: x for x in loo["per_rel"]}
    logger.info("[CONTRAST] 4 inferred LLM edges — 30%-batch-mask MRR -> leave-one-out MRR (gain from keeping full context):")
    for rel in sorted(TARGET_RELS):
        batchmask_row = batchmask_by_rel.get(rel)
        loo_row = loo_by_rel.get(rel)
        if batchmask_row and loo_row:
            logger.info(f"[CONTRAST] {rel:<16} {batchmask_row['mrr']:.3f} -> {loo_row['mrr']:.3f}  ({loo_row['mrr']-batchmask_row['mrr']:+.3f})  | H@1 {batchmask_row['h1']:.3f} -> {loo_row['h1']:.3f}")

    # ---- (v13) CASCADE: backbone-only FLOOR -> ordered cascade -> LOO ceiling (does order matter?) ----
    cascade_results = None
    if cfg.run_cascade:
        order_ids = [resolve_rel(x) for x in cfg.cascade_order]
        reversed_ids = list(reversed(order_ids))
        forward_cascade = cascade_evaluate(encoder, scorer, test_graphs, order_ids, device, cfg)
        reverse_cascade = cascade_evaluate(encoder, scorer, test_graphs, reversed_ids, device, cfg)
        cascade_results = {"forward": forward_cascade, "reverse": reverse_cascade}
        logger.info(f"[CASCADE] order = {' -> '.join(forward_cascade['order'])}  (oracle: earlier relations' GOLD edges added to context)")
        logger.info("[CASCADE] per relation: FLOOR (backbone-only) -> CASCADE (this order) -> CEILING (LOO, all else present):")
        for rel in cfg.cascade_order:
            floor_row = forward_cascade["floor"].get(rel)
            cascade_row = forward_cascade["cascade"].get(rel)
            ceiling_row = loo_by_rel.get(rel)
            if floor_row and cascade_row and ceiling_row:
                headroom = ceiling_row["mrr"] - floor_row["mrr"]
                recovered = cascade_row["mrr"] - floor_row["mrr"]
                logger.info(f"[CASCADE] {rel:<16} MRR {floor_row['mrr']:.3f} -> {cascade_row['mrr']:.3f} -> {ceiling_row['mrr']:.3f}  (chance {floor_row['chance_mrr']:.3f}) | H@1 {floor_row['h1']:.3f} -> {cascade_row['h1']:.3f} -> {ceiling_row['h1']:.3f} | recovered {recovered:+.3f} of {headroom:+.3f} headroom")
        logger.info(f"[CASCADE-ORDER] reverse = {' -> '.join(reverse_cascade['order'])} (order-dependence check):")
        for rel in cfg.cascade_order:
            forward_row = forward_cascade["cascade"].get(rel)
            reverse_row = reverse_cascade["cascade"].get(rel)
            if forward_row and reverse_row:
                logger.info(f"[CASCADE-ORDER] {rel:<16} forward {forward_row['mrr']:.3f}  vs  reverse {reverse_row['mrr']:.3f}  ({reverse_row['mrr']-forward_row['mrr']:+.3f})")

    # ---- (v11, optional) EIR batch-holdout uplift — OFF by default (full-gold F1 is hub-dominated; LOO above is the honest measure) ----
    tau = None
    eir = None
    if cfg.run_eir:
        val_supports = [support_graded(e.get("labels")) for _, g in val_pairs for e in g["edges"] if e.get("evidence") == "llm"]
        tau = float(np.quantile(val_supports, 0.5)) if val_supports else 0.5
        logger.info(f"[EIR] gold/keep tau (VAL median graded LLM support) = {tau:.3f} | holdout={cfg.eir_holdout} fuzzy={cfg.eir_fuzzy}")
        eir = eir_uplift_eval(encoder, scorer, test_pairs, tau, device, cfg)
        logger.info(f"[EIR-UPLIFT] TEST edge-triple vs v8-gold ({eir['n_graphs']} graphs) | RAW P={eir['rawP']:.3f} R={eir['rawR']:.3f} F1={eir['rawF']:.3f}")
        logger.info(f"[EIR-UPLIFT] TEST edge-triple vs v8-gold | REFINED P={eir['refP']:.3f} R={eir['refR']:.3f} F1={eir['refF']:.3f}  (F1 {eir['refF']-eir['rawF']:+.3f} | precision {eir['refP']-eir['rawP']:+.3f} = DISCONNECT, recall {eir['refR']-eir['rawR']:+.3f} = model ADD)")
        logger.info("[EIR-ADD] held-out evidence-supported edges recovered (top-1 same-type), the 4 LLM targets:")
        for rel in sorted(TARGET_RELS):
            add_row = eir["add"].get(rel, {"rec": 0, "held": 0, "recall": float('nan')})
            logger.info(f"[EIR-ADD] {rel:<16} recovered = {add_row['rec']}/{add_row['held']} = {(add_row['recall'] if add_row['recall']==add_row['recall'] else 0.0):.3f}")

    checkpoint_path = cfg.checkpoint_name
    torch.save({"encoder": encoder.state_dict(), "scorer": scorer.state_dict(),
                "config": cfg.checkpoint_dict(),
                "recovery_test_batchmask": test_metrics, "recovery_test_loo": loo,
                "cascade": cascade_results, "eir": eir}, checkpoint_path)
    if not cfg.push:
        logger.info(f"[DONE] PUSH=0 — saved local {checkpoint_path}")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(cfg.output_repo, repo_type="model", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=checkpoint_path, path_in_repo=checkpoint_path, repo_id=cfg.output_repo, repo_type="model")
    logger.info(f"[DONE] v16 ENTITY-NOTE -> https://huggingface.co/{cfg.output_repo}/blob/main/{checkpoint_path} | LOO MRR={loo['mrr']:.3f} | use_note={cfg.use_note} ground_by={cfg.ground_by} use_scores={cfg.use_scores}")


if __name__ == "__main__":
    main()
