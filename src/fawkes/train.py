"""The experiment entry point: data preparation, the two training phases, and reporting.

REAL Graph-JEPA world model for clinical KG refinement (WM@Booth). PHASE 1 =
masked-node latent prediction (BYOL, EMA target) on ONE shared encoder = the
world model; PHASE 2 = frozen-encoder edge recovery readout. Inverse-edge
leakage + test-set-selection bugs fixed. CLAUDE.md: deterministic, [TAG] logging,
NO fallback (unknown type/relation/missing data fails loud). Added 260615.

Split out of ``paper_v16/trainer.py``; see ``docs/LINEAGE.md`` for the
full version history that used to sit at the top of that file. The per-batch
tensor work lives in ``steps.py``; this module owns the run: load and split the
dataset, pretrain the JEPA world model, train the readout with validation-based
model selection, run the evaluations, and save/push the checkpoint.
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
from torch_geometric.loader import DataLoader

from .config import Config
from .data import (NODE_TYPES, RELATION_ALIASES, RELATION_CANONICAL, TARGET_RELS, has_evidence,
                   load_full_dataset, resolve_rel, support_graded, to_data)
from .evaluate import cascade_evaluate, eir_uplift_eval, evaluate, loo_evaluate
from .model import JEPA, build_scorer
from .steps import jepa_step, readout_step

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


# ---- data preparation ----

def _log_prune_stats(raw):
    """[PRUNE] how many LLM edges the no-evidence filter drops (the filter itself runs in to_data)."""
    total_llm = sum(1 for g in raw for e in g.get("edges", []) if e.get("evidence") == "llm")
    no_evidence = sum(1 for g in raw for e in g.get("edges", [])
                      if e.get("evidence") == "llm" and not has_evidence(e.get("labels")))
    logger.info(f"[PRUNE] LLM edges={total_llm} no-evidence(pruned)={no_evidence} ({100.0*no_evidence/max(total_llm,1):.1f}%) | backbone + evidenced LLM kept")


def _audit_vocab(raw):
    """Fail loud on any relation or node type outside the fixed vocabulary."""
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


def _convert_graphs(raw, demographics, cfg):
    """Raw records -> (PyG Data, raw graph) pairs, dropping graphs too small to mask."""
    items = []
    for g in raw:
        d = to_data(g, demographics, cfg)
        if d.num_nodes >= 3 and d.edge_index.size(1) >= 4:
            items.append((d, g))   # keep raw graph alongside for the EIR eval
    if cfg.use_note:
        num_grounded = sum(int(d.n_grounded[0]) for d, _ in items)
        total_nodes = sum(int(d.num_nodes) for d, _ in items)
        logger.info(f"[NOTE] ground_by={cfg.ground_by} grounded entities={num_grounded}/{total_nodes} ({100.0*num_grounded/max(total_nodes,1):.1f}%) carry the note vector; the rest get a zero note")
    return items


def _split_items(items, cfg):
    """Seeded shuffle -> (train, val, test) pairs; test comes first so its membership
    is independent of the val fraction.

    (v20) The permutation draws on ``data_split_seed``, not ``seed``. They default
    to the same 42, so the published split is unchanged, but holding one while
    varying the other is what separates "a different sample of patients" from "a
    different initialisation" in the Table 1 variance study.
    """
    rng = np.random.RandomState(cfg.data_split_seed)
    shuffled = rng.permutation(len(items))
    num_test = int(cfg.test_frac * len(items))
    num_val = int(cfg.val_frac * len(items))
    test_pairs = [items[i] for i in shuffled[:num_test]]
    val_pairs = [items[i] for i in shuffled[num_test:num_test + num_val]]
    train_pairs = [items[i] for i in shuffled[num_test + num_val:]]
    logger.info(f"[DATA] graphs={len(items)} -> TRAIN={len(train_pairs)} VAL={len(val_pairs)} TEST={len(test_pairs)} (split seed={cfg.data_split_seed})")
    return train_pairs, val_pairs, test_pairs


def prepare_data(cfg):
    """Load, audit, convert and split the dataset. Returns (train, val, test) pairs."""
    raw, demographics = load_full_dataset(cfg)
    if cfg.prune_no_evidence:
        _log_prune_stats(raw)
    _audit_vocab(raw)
    items = _convert_graphs(raw, demographics, cfg)
    return _split_items(items, cfg)


# ---- PHASE 1: JEPA world-model pretraining ----

def _ema_at(cfg, epoch):
    """Cosine ramp of the target-encoder EMA rate from ema_base to ema_final."""
    return cfg.ema_final - (cfg.ema_final - cfg.ema_base) * (math.cos(math.pi * (epoch - 1) / max(cfg.jepa_epochs, 1)) + 1) / 2


def pretrain_jepa(train_graphs, device, cfg):
    """Train the JEPA world model (masked-node latent prediction) and return it."""
    model = JEPA(cfg).to(device)
    optimizer = torch.optim.Adam([p for p in model.ctx.parameters() if p.requires_grad]
                                 + list(model.pred.parameters()) + list(model.slot_rel.parameters()), lr=cfg.lr)
    logger.info(f"[MODEL] JEPA params={sum(p.numel() for p in model.parameters()):,} edge_dim={model.ctx.edge_dim}")
    loader = DataLoader(train_graphs, batch_size=cfg.batch, shuffle=True)
    for epoch in range(1, cfg.jepa_epochs + 1):
        model.ctx.train()
        model.ema = _ema_at(cfg, epoch)
        start = time.perf_counter()
        total_loss = total_std = num_batches = 0
        for batch in loader:
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
    return model


# ---- PHASE 2: edge-recovery readout on the (frozen) world-model encoder ----

def train_readout(model, train_graphs, val_graphs, device, cfg):
    """Train the readout scorer on the world-model encoder; keep the best-VAL-auc pair.

    Returns ``(encoder, scorer)`` with the encoder frozen when
    ``cfg.freeze_encoder``, both rolled back to their best VAL epoch.

    The encoder is snapshotted alongside the scorer only when it trains: under
    ``freeze_encoder`` it never moves, so the scorer alone is the whole model,
    but without it the encoder would otherwise keep training past the selected
    epoch and be returned beside a scorer from a different one — a pair that
    existed at no epoch. That matters for the from-scratch ablation arms.
    """
    encoder = model.ctx
    if cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
    scorer = build_scorer(cfg).to(device)
    params = list(scorer.parameters()) + ([] if cfg.freeze_encoder else list(encoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    eval_batch = 1 if cfg.freeze_eval else cfg.batch
    loader = DataLoader(train_graphs, batch_size=cfg.batch, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=eval_batch, shuffle=False)
    best_auc = 0.0
    best_scorer_state = None
    best_encoder_state = None
    for epoch in range(1, cfg.readout_epochs + 1):
        scorer.train()
        if not cfg.freeze_encoder:
            encoder.train()
        start = time.perf_counter()
        total_loss = num_batches = 0
        for batch in loader:
            sampled_ratio = (cfg.mask_lo + (cfg.mask_hi - cfg.mask_lo) * float(torch.rand(1).item())) if cfg.mask_schedule else None   # (B) sampled context density per step
            step_out = readout_step(encoder, scorer, batch, device, True, cfg, mask_ratio=sampled_ratio)
            if step_out is None:
                continue
            loss = step_out[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        if epoch % 5 == 0 or epoch == cfg.readout_epochs:
            val_metrics = evaluate(encoder, scorer, val_loader, device, cfg)
            logger.info(f"[LATENCY] READOUT epoch={epoch}/{cfg.readout_epochs} loss={total_loss/max(num_batches,1):.4f} took={(time.perf_counter()-start)*1000:.0f}ms "
                        f"| VAL auc={val_metrics['auc']:.3f} nonobv={val_metrics['auc_nonobvious']:.3f} MRR={val_metrics['mrr']:.3f} H@1={val_metrics['hits1']:.3f} H@10={val_metrics['hits10']:.3f}")
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_scorer_state = copy.deepcopy(scorer.state_dict())
                if not cfg.freeze_encoder:
                    best_encoder_state = copy.deepcopy(encoder.state_dict())
    if best_scorer_state is not None:
        scorer.load_state_dict(best_scorer_state)
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
    return encoder, scorer


# ---- reporting ----

def _chance_flag(row):
    """Annotate a per-relation row relative to its chance MRR."""
    if row["mrr"] > 1.5 * row["chance_mrr"]:
        return " <== ABOVE chance"
    if row["mrr"] < 1.2 * row["chance_mrr"]:
        return " <== ~chance"
    return ""


def _log_per_rel(tag, rows):
    for row in rows:
        logger.info(f"[{tag}] {row['rel']:<22} n={row['n']:<5} C={row['C']:.0f}  MRR={row['mrr']:.3f} (chance {row['chance_mrr']:.3f})  H@1={row['h1']:.3f}  H@10={row['h10']:.3f}{_chance_flag(row)}")


def _report_test(test_metrics, cfg):
    logger.info(f"[RESULT] TEST (val-selected) | auc={test_metrics['auc']:.3f} ap={test_metrics['ap']:.3f} non-obvious_auc={test_metrics['auc_nonobvious']:.3f} "
                f"MRR={test_metrics['mrr']:.3f} Hits@1={test_metrics['hits1']:.3f} Hits@3={test_metrics['hits3']:.3f} Hits@10={test_metrics['hits10']:.3f} (n_mrr={test_metrics['n_mrr']}) quiz_sig={test_metrics['qsig']} frozen={cfg.freeze_eval} scores={cfg.use_scores}")
    logger.info("[PER-REL] ONE shared latent, edge recovery by relation (sorted by edge count):")
    _log_per_rel("PER-REL", test_metrics["per_rel"])


def _report_loo(loo, test_metrics):
    """(v12) leave-one-out results, plus the batch-mask -> LOO contrast on the 4 inferred relations."""
    logger.info(f"[LOO] leave-one-out (mask 1 edge, full context, filtered) | MRR={loo['mrr']:.3f} H@1={loo['hits1']:.3f} H@3={loo['hits3']:.3f} H@10={loo['hits10']:.3f} over {loo['n']} edges")
    logger.info("[LOO] edge recovery by relation (one edge masked at a time, full surrounding context):")
    _log_per_rel("LOO", loo["per_rel"])
    batchmask_by_rel = {x["rel"]: x for x in test_metrics["per_rel"]}
    loo_by_rel = {x["rel"]: x for x in loo["per_rel"]}
    logger.info("[CONTRAST] 4 inferred LLM edges — 30%-batch-mask MRR -> leave-one-out MRR (gain from keeping full context):")
    for rel in sorted(TARGET_RELS):
        batchmask_row = batchmask_by_rel.get(rel)
        loo_row = loo_by_rel.get(rel)
        if batchmask_row and loo_row:
            logger.info(f"[CONTRAST] {rel:<16} {batchmask_row['mrr']:.3f} -> {loo_row['mrr']:.3f}  ({loo_row['mrr']-batchmask_row['mrr']:+.3f})  | H@1 {batchmask_row['h1']:.3f} -> {loo_row['h1']:.3f}")


def _run_cascade(encoder, scorer, test_graphs, loo, device, cfg):
    """(v13) backbone-only FLOOR -> ordered cascade -> LOO ceiling (does order matter?)."""
    order_ids = [resolve_rel(x) for x in cfg.cascade_order]
    forward_cascade = cascade_evaluate(encoder, scorer, test_graphs, order_ids, device, cfg)
    reverse_cascade = cascade_evaluate(encoder, scorer, test_graphs, list(reversed(order_ids)), device, cfg)
    loo_by_rel = {x["rel"]: x for x in loo["per_rel"]}
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
    return {"forward": forward_cascade, "reverse": reverse_cascade}


def _run_eir(encoder, scorer, val_pairs, test_pairs, device, cfg):
    """(v11, optional) EIR batch-holdout uplift — OFF by default (full-gold F1 is
    hub-dominated; LOO is the honest measure)."""
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
    return eir


def _save_and_push(encoder, scorer, cfg, test_metrics, loo, cascade_results, eir):
    """Save the checkpoint locally; upload to the Hugging Face OUTPUT_REPO when PUSH=1."""
    checkpoint_path = cfg.checkpoint_name
    torch.save({"encoder": encoder.state_dict(), "scorer": scorer.state_dict(),
                "config": cfg.checkpoint_dict(), "run_config": cfg.run_dict(),
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


# ---- entry point ----

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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    cfg = Config.from_env()
    logger.info(f"[ENTRY] ENTITY-NOTE v18 | data={cfg.data_repo} use_note={cfg.use_note} ground_by={cfg.ground_by} embed_dim={cfg.embed_dim} numeric_dim={cfg.numeric_dim} use_scores={cfg.use_scores} prune_no_evidence={cfg.prune_no_evidence} mask_schedule={cfg.mask_schedule}[{cfg.mask_lo},{cfg.mask_hi}] target_weight={cfg.target_weight} cascade={cfg.run_cascade} order={cfg.cascade_order} jepa_ep={cfg.jepa_epochs} readout_ep={cfg.readout_epochs} loo_cap={cfg.loo_cap} run_eir={cfg.run_eir} "
                f"hid={cfg.hid} mask_strategy={cfg.mask_strategy} jepa_patches={cfg.jepa_patches} node_mask={cfg.node_mask} edge_mask={cfg.edge_mask} entity_emb={cfg.use_entity_emb} decoder={cfg.decoder} loss={cfg.loss} neg_k={cfg.neg_k} freeze_eval={cfg.freeze_eval} deterministic={cfg.deterministic} val={cfg.val_frac} test={cfg.test_frac} seed={cfg.seed} data_split_seed={cfg.data_split_seed}")
    set_seed(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[GPU] {device}" + (f" {torch.cuda.get_device_name(0)}" if device.type == 'cuda' else ""))

    train_pairs, val_pairs, test_pairs = prepare_data(cfg)
    train_graphs = [d for d, _ in train_pairs]
    val_graphs = [d for d, _ in val_pairs]
    test_graphs = [d for d, _ in test_pairs]

    model = pretrain_jepa(train_graphs, device, cfg)
    encoder, scorer = train_readout(model, train_graphs, val_graphs, device, cfg)

    eval_batch = 1 if cfg.freeze_eval else cfg.batch
    test_loader = DataLoader(test_graphs, batch_size=eval_batch, shuffle=False)
    test_metrics = evaluate(encoder, scorer, test_loader, device, cfg)
    _report_test(test_metrics, cfg)

    # (v12) LEAVE-ONE-OUT edge recovery: mask ONE edge, keep the rest, recover it (full context, filtered)
    loo = loo_evaluate(encoder, scorer, test_graphs, device, cfg)
    _report_loo(loo, test_metrics)

    cascade_results = _run_cascade(encoder, scorer, test_graphs, loo, device, cfg) if cfg.run_cascade else None
    eir = _run_eir(encoder, scorer, val_pairs, test_pairs, device, cfg) if cfg.run_eir else None

    _save_and_push(encoder, scorer, cfg, test_metrics, loo, cascade_results, eir)


if __name__ == "__main__":
    main()
