"""The shared encoder (the world model), the JEPA wrapper, and the readout heads.

Split out of ``paper_v16/trainer.py`` lines 256-335 — the "shared
encoder" and "downstream readout" seams.

Every submodule attribute name here is load-bearing: ``state_dict`` keys derive
from attribute paths, not class names, so ``convs``, ``norms``, ``type_emb``,
``entity_emb``, ``num_proj``, ``rel_emb``, ``score_gate`` and ``rel`` must not be
renamed. ``tests/test_fawkes.py`` gates on this against ``baseline/paper_keys.json``.

``score_gate`` is always constructed even when ``use_scores`` is off — the
released checkpoint carries its weights, so making it conditional would break a
``strict=True`` load.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from .data import NUM_NODE_TYPES, NUM_RELATIONS, SCORE_DIM


class SpanNoteInjector(nn.Module):
    """Fuse a graph entity with its admission's local note-span memory.

    ``uniform`` is the parameter-matched control: it uses token-count-weighted
    span averaging. ``attention`` learns entity-conditioned weights. Both use
    the same value projection, output projection and gated residual fusion.
    """

    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.note_injection
        self.heads = cfg.note_attn_heads
        self.head_dim = cfg.hid // self.heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(cfg.hid, cfg.hid, bias=False)
        self.key = nn.Linear(cfg.embed_dim, cfg.hid, bias=False)
        self.value = nn.Linear(cfg.embed_dim, cfg.hid, bias=False)
        self.out = nn.Linear(cfg.hid, cfg.hid, bias=False)
        self.gate = nn.Linear(2 * cfg.hid, cfg.hid)

    def forward(self, hidden, batch_index, memory, memory_mask, span_token_counts, grounded):
        if any(value is None for value in (
                batch_index, memory, memory_mask, span_token_counts, grounded)):
            raise ValueError("[FAILURE] v23 note injection received incomplete note-memory tensors")
        if memory.dim() != 3:
            raise ValueError(f"[FAILURE] note_memory rank {memory.dim()} != 3")
        memory = memory.to(dtype=hidden.dtype)
        batch_index = batch_index.long()
        num_graphs, num_spans, _ = memory.shape
        if batch_index.numel() != hidden.size(0) or int(batch_index.max()) >= num_graphs:
            raise ValueError("[FAILURE] node-to-graph batch index does not match note memory")

        values = self.value(memory).view(num_graphs, num_spans, self.heads, self.head_dim)
        node_values = values[batch_index]
        valid = memory_mask.bool()[batch_index]
        if not bool(valid.any(dim=1).all()):
            raise ValueError("[FAILURE] an admission has no valid note spans")

        if self.mode == "uniform":
            weights = span_token_counts.to(dtype=hidden.dtype)[batch_index] * valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1)
            weights = weights.unsqueeze(-1).expand(-1, -1, self.heads)
        else:
            queries = self.query(hidden).view(-1, self.heads, self.head_dim)
            keys = self.key(memory).view(num_graphs, num_spans, self.heads, self.head_dim)
            scores = (queries.unsqueeze(1) * keys[batch_index]).sum(-1) * self.scale
            scores = scores.masked_fill(~valid.unsqueeze(-1), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)

        context = (weights.unsqueeze(-1) * node_values).sum(dim=1).reshape(-1, hidden.size(-1))
        context = self.out(context)
        gate = torch.sigmoid(self.gate(torch.cat([hidden, context], dim=-1)))
        return context * gate * grounded.to(dtype=hidden.dtype).unsqueeze(-1)


class Encoder(nn.Module):
    """The world model: hashed-entity + type + numeric input, TransformerConv stack."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.type_emb = nn.Embedding(NUM_NODE_TYPES, cfg.hid)
        self.entity_emb = nn.Embedding(cfg.entity_vocab, cfg.hid)
        self.num_proj = nn.Linear(cfg.numeric_dim, cfg.hid)
        self.rel_emb = nn.Embedding(NUM_RELATIONS, cfg.edge_emb)
        self.score_gate = nn.Sequential(
            nn.Linear(SCORE_DIM, cfg.edge_emb), nn.ReLU(), nn.Linear(cfg.edge_emb, 1))   # (v14) v8 evidence -> per-edge gate
        self.note_injector = SpanNoteInjector(cfg) if cfg.note_injection != "mean" else None
        self.edge_dim = cfg.edge_emb                        # (v14) gate scales the relation embedding; no concat -> edge_dim unchanged
        self.convs = nn.ModuleList(
            TransformerConv(cfg.hid, cfg.hid // cfg.heads, heads=cfg.heads, concat=True,
                            edge_dim=self.edge_dim, dropout=0.0)
            for _ in range(cfg.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hid) for _ in range(cfg.layers))

    def forward(self, node_type, entity_id, numfeat, edge_index, edge_type, edge_feat=None, sem_id=None,
                batch_index=None, note_memory=None, note_memory_mask=None,
                note_span_token_counts=None, note_grounded=None):
        entity = self.entity_emb(entity_id) if self.cfg.use_entity_emb else 0
        hidden = self.type_emb(node_type) + entity + self.num_proj(numfeat)
        if self.note_injector is not None:
            hidden = hidden + self.note_injector(
                hidden, batch_index, note_memory, note_memory_mask,
                note_span_token_counts, note_grounded)
        edge_attr = self.rel_emb(edge_type)
        if self.cfg.use_scores:
            if edge_feat is None:
                raise ValueError("[FAILURE] USE_SCORES on but edge_feat not passed to encoder; no fallback.")
            edge_attr = edge_attr * torch.sigmoid(self.score_gate(edge_feat))   # (v14) evidence GATE: weak/no-evidence edges contribute less
        for conv, norm in zip(self.convs, self.norms):
            hidden = F.relu(norm(conv(hidden, edge_index, edge_attr)))
        return hidden


class JEPA(nn.Module):
    """Context encoder + EMA target encoder + slot-conditioned predictor."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ctx = Encoder(cfg)
        self.tgt = copy.deepcopy(self.ctx)
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.pred = nn.Sequential(nn.Linear(2 * cfg.hid, cfg.hid), nn.ReLU(), nn.Linear(cfg.hid, cfg.hid))
        self.slot_rel = nn.Embedding(NUM_RELATIONS, cfg.hid)
        self.ema = cfg.ema_base

    @torch.no_grad()
    def update(self):
        for target_param, context_param in zip(self.tgt.parameters(), self.ctx.parameters()):
            target_param.mul_(self.ema).add_(context_param, alpha=1 - self.ema)


class Scorer(nn.Module):
    """MLP readout head — selected by ``DECODER`` when it is not ``distmult``."""

    def __init__(self, cfg):
        super().__init__()
        self.rel = nn.Embedding(NUM_RELATIONS, cfg.hid)
        self.mlp = nn.Sequential(nn.Linear(3 * cfg.hid, cfg.hid), nn.ReLU(), nn.Linear(cfg.hid, 1))

    def forward(self, hidden, head, tail, relation):
        return self.mlp(torch.cat([hidden[head], hidden[tail], self.rel(relation)], -1)).squeeze(-1)


class DistMult(nn.Module):
    """The default readout head, and the one the released checkpoint carries."""

    def __init__(self, cfg):
        super().__init__()
        self.rel = nn.Embedding(NUM_RELATIONS, cfg.hid)

    def forward(self, hidden, head, tail, relation):
        return (hidden[head] * self.rel(relation) * hidden[tail]).sum(-1)


def build_scorer(cfg):
    """``trainer.py:630`` — ``DistMult() if DECODER=="distmult" else Scorer()``."""
    return DistMult(cfg) if cfg.decoder == "distmult" else Scorer(cfg)
