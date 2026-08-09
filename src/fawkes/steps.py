"""The per-batch training steps: JEPA masked-node prediction and edge-recovery readout.

Split out of ``train.py`` so that ``evaluate.py`` (which reuses
``readout_step`` and ``buckets``) no longer has to import the training entry
point — previously a circular import that ``train.main`` worked around with a
deferred import.

Everything here is deterministic given the RNG state: the only random draws are
the node mask in ``jepa_step`` (Bernoulli by default; the BFS patch mask from
``patches.py`` when ``cfg.mask_strategy == "patch"``) and the edge permutation /
negative sampling in ``readout_step``, in that order. ``tests/test_fawkes.py``
pins both steps bit-for-bit against the recorded trainer (under the default
``mask_strategy="random"``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph

from .data import NUM_BASE, NUM_NODE_TYPES, TARGET_REL_IDS, add_inverses
from .patches import patch_mask


# ---- PHASE 1: JEPA masked-node latent prediction ----

def valid_mask(batch_index, num_graphs, ratio, device, tries=10):
    """A random node mask where every graph keeps at least one target AND one context node."""
    num_nodes = batch_index.size(0)
    for _ in range(tries):
        mask = torch.rand(num_nodes, device=device) < ratio
        targets_per_graph = torch.zeros(num_graphs, device=device).scatter_add_(0, batch_index, mask.float())
        context_per_graph = torch.zeros(num_graphs, device=device).scatter_add_(0, batch_index, (~mask).float())
        if bool((targets_per_graph > 0).all()) and bool((context_per_graph > 0).all()):
            return mask
    raise RuntimeError(f"[FAILURE] no valid node mask after {tries} tries (ratio={ratio}); graphs too small.")


def _encode_context(model, batch, context_nodes, num_nodes, cfg):
    """Run the context encoder over the context-only subgraph (target nodes removed)."""
    if cfg.use_scores:
        ctx_edge_index, ctx_edge_type, kept_edges = subgraph(
            context_nodes, batch.edge_index, edge_attr=batch.edge_type, relabel_nodes=True,
            num_nodes=num_nodes, return_edge_mask=True)
        ctx_edge_feat = batch.edge_feat[kept_edges]
        ctx_edge_index, ctx_edge_type, ctx_edge_feat = add_inverses(ctx_edge_index, ctx_edge_type, ctx_edge_feat)
    else:
        ctx_edge_index, ctx_edge_type = subgraph(
            context_nodes, batch.edge_index, edge_attr=batch.edge_type, relabel_nodes=True, num_nodes=num_nodes)
        ctx_edge_index, ctx_edge_type = add_inverses(ctx_edge_index, ctx_edge_type)
        ctx_edge_feat = None
    return model.ctx(batch.node_type[context_nodes], batch.entity_id[context_nodes], batch.numfeat[context_nodes],
                     ctx_edge_index, ctx_edge_type, ctx_edge_feat, batch.sem_id[context_nodes],
                     batch_index=batch.batch[context_nodes],
                     note_memory=getattr(batch, "note_memory", None),
                     note_memory_mask=getattr(batch, "note_memory_mask", None),
                     note_span_token_counts=getattr(batch, "note_span_token_counts", None),
                     note_grounded=(batch.note_grounded[context_nodes]
                                    if hasattr(batch, "note_grounded") else None))


def _encode_targets(model, batch, cfg):
    """Run the EMA target encoder over the FULL graph (caller wraps in no_grad)."""
    if cfg.use_scores:
        full_edge_index, full_edge_type, full_edge_feat = add_inverses(batch.edge_index, batch.edge_type, batch.edge_feat)
    else:
        full_edge_index, full_edge_type = add_inverses(batch.edge_index, batch.edge_type)
        full_edge_feat = None
    return model.tgt(batch.node_type, batch.entity_id, batch.numfeat,
                     full_edge_index, full_edge_type, full_edge_feat, batch.sem_id,
                     batch_index=batch.batch,
                     note_memory=getattr(batch, "note_memory", None),
                     note_memory_mask=getattr(batch, "note_memory_mask", None),
                     note_span_token_counts=getattr(batch, "note_span_token_counts", None),
                     note_grounded=getattr(batch, "note_grounded", None))


def _slot_queries(model, batch, context_repr, target_mask, context_mask, context_nodes, target_nodes, cfg):
    """One query vector per masked node: the mean of (context-neighbor repr + relation
    embedding) over the boundary edges into it, plus its type (and optionally entity) embedding.
    """
    node_type, entity_id = batch.node_type, batch.entity_id
    edge_index, edge_type = batch.edge_index, batch.edge_type
    num_nodes = node_type.size(0)
    device = node_type.device
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
    return query


def jepa_step(model, batch, device, cfg):
    """One BYOL step: predict the EMA-target embedding of each masked node from the context.

    Returns ``(loss, emb_std)`` where ``emb_std`` monitors target-embedding
    collapse (a healthy run keeps it well above zero).
    """
    batch = batch.to(device)
    batch_index = batch.batch
    num_nodes = batch.node_type.size(0)
    num_graphs = int(batch_index.max().item()) + 1
    if cfg.mask_strategy == "patch":                             # (v18) mask whole BFS-balanced patches
        target_mask = patch_mask(batch.ptr, batch.edge_index, cfg.node_mask, cfg.jepa_patches, device)
    else:
        target_mask = valid_mask(batch_index, num_graphs, cfg.node_mask, device)
    context_mask = ~target_mask
    context_nodes = context_mask.nonzero(as_tuple=False).view(-1)
    context_repr = _encode_context(model, batch, context_nodes, num_nodes, cfg)
    context_summary = global_mean_pool(context_repr, batch_index[context_nodes], size=num_graphs)
    with torch.no_grad():
        target_repr = _encode_targets(model, batch, cfg)
        emb_std = target_repr.std(0).mean()
    target_nodes = target_mask.nonzero(as_tuple=False).view(-1)
    target_emb = F.normalize(target_repr[target_nodes], dim=-1)
    query = _slot_queries(model, batch, context_repr, target_mask, context_mask, context_nodes, target_nodes, cfg)
    prediction = F.normalize(model.pred(torch.cat([context_summary[batch_index[target_nodes]], query], -1)), dim=-1)
    return (2 - 2 * (prediction * target_emb).sum(-1)).mean(), emb_std


# ---- PHASE 2: edge-recovery readout ----

def buckets(node_type, batch_index):
    """Sort nodes by (graph, node type) so each bucket is one contiguous span."""
    bucket_ids = batch_index * NUM_NODE_TYPES + node_type
    order = torch.argsort(bucket_ids)
    return order, bucket_ids[order]


def same_type_k(targets, node_type, batch_index, order, sorted_buckets, num_neg, gen=None):
    """For each target node, ``num_neg`` random nodes of the SAME type in the SAME graph."""
    target_buckets = batch_index[targets] * NUM_NODE_TYPES + node_type[targets]
    lo = torch.searchsorted(sorted_buckets, target_buckets, right=False)
    hi = torch.searchsorted(sorted_buckets, target_buckets, right=True)
    span = (hi - lo).clamp(min=1).float()
    rand = torch.rand(targets.numel(), num_neg, device=targets.device, generator=gen)
    picks = (rand * span.unsqueeze(1)).long() + lo.unsqueeze(1)
    return order[picks.clamp(max=order.numel() - 1)]


def _encode_observed(encoder, batch, observed, training, cfg):
    """Encode the graph from the OBSERVED edges only, so held-out edges are never seen (no leak)."""
    if cfg.use_scores:
        obs_edge_index, obs_edge_type, obs_edge_feat = add_inverses(
            batch.edge_index[:, observed], batch.edge_type[observed], batch.edge_feat[observed])
    else:
        obs_edge_index, obs_edge_type = add_inverses(batch.edge_index[:, observed], batch.edge_type[observed])
        obs_edge_feat = None
    grad_mode = torch.enable_grad() if (training and not cfg.freeze_encoder) else torch.no_grad()
    with grad_mode:
        return encoder(batch.node_type, batch.entity_id, batch.numfeat,
                       obs_edge_index, obs_edge_type, obs_edge_feat, batch.sem_id,
                       batch_index=batch.batch,
                       note_memory=getattr(batch, "note_memory", None),
                       note_memory_mask=getattr(batch, "note_memory_mask", None),
                       note_span_token_counts=getattr(batch, "note_span_token_counts", None),
                       note_grounded=getattr(batch, "note_grounded", None))


def _readout_loss(pos_scores, neg_scores, neg_dst, held_dst, held_rel, device, cfg):
    """InfoNCE over (1 positive, neg_k negatives) — or BCE on the first negative.

    Accidental true tails among the negatives are masked out, and with
    ``target_weight`` != 1 the 4 inferred relations are upweighted (v13).
    """
    num_pos = pos_scores.size(0)
    logits = torch.cat([pos_scores, neg_scores], 1) / cfg.temp
    logits[:, 1:][neg_dst == held_dst.unsqueeze(1)] = -1e9
    if cfg.loss == "bce":
        pos_flat = pos_scores.squeeze(1)
        neg_flat = neg_scores[:, 0]
        logits_bce = torch.cat([pos_flat, neg_flat])
        labels = torch.cat([torch.ones_like(pos_flat), torch.zeros_like(neg_flat)])
        return F.binary_cross_entropy_with_logits(logits_bce, labels)
    if cfg.target_weight != 1.0:                                 # (v13) upweight the 4 inferred relations
        per_edge_ce = F.cross_entropy(logits, torch.zeros(num_pos, dtype=torch.long, device=device), reduction='none')
        is_target = torch.isin(held_rel, torch.tensor(sorted(TARGET_REL_IDS), device=device))
        weights = torch.where(is_target, torch.full((num_pos,), cfg.target_weight, device=device),
                              torch.ones(num_pos, device=device))
        return (per_edge_ce * weights).sum() / weights.sum()
    return F.cross_entropy(logits, torch.zeros(num_pos, dtype=torch.long, device=device))


def readout_step(encoder, scorer, batch, device, training, cfg, gen=None, mask_ratio=None):
    """One edge-recovery step: hold out a fraction of edges, encode the rest, score
    each held edge against same-type negatives.

    Returns ``None`` for graphs with fewer than 2 edges, else
    ``(loss, pos_scores, first_neg_scores, nonpatient_mask, extra)`` where
    ``extra = (hidden, held_src, held_dst, held_rel, node_type, batch_index, qsig)``
    carries what the evaluators need for MRR ranking. ``qsig`` is a cheap
    signature of the held/negative draw, non-zero only under a seeded ``gen``.
    """
    batch = batch.to(device)
    node_type, batch_index = batch.node_type, batch.batch
    edge_index, edge_type = batch.edge_index, batch.edge_type
    num_edges = edge_index.size(1)
    if num_edges < 2:
        return None
    ratio = mask_ratio if mask_ratio is not None else cfg.edge_mask   # (v13) mask schedule: held-edge fraction
    perm = torch.randperm(num_edges, device=device, generator=gen)
    num_held = max(1, min(num_edges - 1, int(ratio * num_edges)))
    held = perm[:num_held]
    observed = perm[num_held:]
    hidden = _encode_observed(encoder, batch, observed, training, cfg)
    held_src, held_dst, held_rel = edge_index[0, held], edge_index[1, held], edge_type[held]
    order, sorted_buckets = buckets(node_type, batch_index)
    neg_dst = same_type_k(held_dst, node_type, batch_index, order, sorted_buckets, cfg.neg_k, gen=gen)
    num_pos = held_src.numel()
    src_expanded = held_src.unsqueeze(1).expand(num_pos, cfg.neg_k).reshape(-1)
    rel_expanded = held_rel.unsqueeze(1).expand(num_pos, cfg.neg_k).reshape(-1)
    pos_scores = scorer(hidden, held_src, held_dst, held_rel).view(num_pos, 1)
    neg_scores = scorer(hidden, src_expanded, neg_dst.reshape(-1), rel_expanded).view(num_pos, cfg.neg_k)
    loss = _readout_loss(pos_scores, neg_scores, neg_dst, held_dst, held_rel, device, cfg)
    qsig = (int(held.sum().item()) * 1000003 + int(neg_dst.reshape(-1).sum().item()) + num_edges * 7919) if gen is not None else 0
    return (loss, pos_scores.squeeze(1).detach(), neg_scores[:, 0].detach(), (node_type[held_src] != 0).detach(),
            (hidden.detach(), held_src, held_dst, held_rel, node_type, batch_index, qsig))
