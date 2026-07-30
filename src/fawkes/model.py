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
        self.edim = cfg.edge_emb                            # (v14) gate scales the relation embedding; no concat -> edge_dim unchanged
        self.convs = nn.ModuleList(
            TransformerConv(cfg.hid, cfg.hid // cfg.heads, heads=cfg.heads, concat=True,
                            edge_dim=self.edim, dropout=0.0)
            for _ in range(cfg.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hid) for _ in range(cfg.layers))

    def forward(self, nt, eid, numf, ei, et, efeat=None, sem_id=None):
        ident = self.entity_emb(eid) if self.cfg.use_entity_emb else 0
        h = self.type_emb(nt) + ident + self.num_proj(numf)
        ea = self.rel_emb(et)
        if self.cfg.use_scores:
            if efeat is None:
                raise ValueError("[FAILURE] USE_SCORES on but efeat not passed to encoder; no fallback.")
            ea = ea * torch.sigmoid(self.score_gate(efeat))   # (v14) evidence GATE: weak/no-evidence edges contribute less
        for c, n in zip(self.convs, self.norms):
            h = F.relu(n(c(h, ei, ea)))
        return h


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
        for pt, pc in zip(self.tgt.parameters(), self.ctx.parameters()):
            pt.mul_(self.ema).add_(pc, alpha=1 - self.ema)


class Scorer(nn.Module):
    """MLP readout head — selected by ``DECODER`` when it is not ``distmult``."""

    def __init__(self, cfg):
        super().__init__()
        self.rel = nn.Embedding(NUM_RELATIONS, cfg.hid)
        self.mlp = nn.Sequential(nn.Linear(3 * cfg.hid, cfg.hid), nn.ReLU(), nn.Linear(cfg.hid, 1))

    def forward(self, h, u, v, r):
        return self.mlp(torch.cat([h[u], h[v], self.rel(r)], -1)).squeeze(-1)


class DistMult(nn.Module):
    """The default readout head, and the one the released checkpoint carries."""

    def __init__(self, cfg):
        super().__init__()
        self.rel = nn.Embedding(NUM_RELATIONS, cfg.hid)

    def forward(self, h, u, v, r):
        return (h[u] * self.rel(r) * h[v]).sum(-1)


def build_scorer(cfg):
    """``trainer.py:630`` — ``DistMult() if DECODER=="distmult" else Scorer()``."""
    return DistMult(cfg) if cfg.decoder == "distmult" else Scorer(cfg)
