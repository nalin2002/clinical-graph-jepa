"""Patch/subgraph Graph-JEPA model for clinical KG graph revision.

One :class:`GraphJEPA`, flattened from the ``GraphJEPAv3 -> GraphJEPAv4 ->
GraphJEPAv5/v6`` chain. The version numbers were architecture layer names, not
repository versions; v3 and v4 were never instantiated by any shipped path and
survived only as base classes, and ``graph_jepa_v6/model.py`` ended with
``GraphJEPAv5 = GraphJEPAv6`` — the two "models" were one class body.

**Every submodule keeps its attribute name.** ``state_dict`` keys derive from
attribute paths (``context_node_encoder.input_proj.weight``), not class names,
so they survive the class rename but would not survive an attribute rename.
Both released checkpoints load into this class with ``strict=True``; the only
architectural difference between the no-note and localized-note variants is the
width of ``{context,target}_node_encoder.input_proj``, which follows from
``ModelConfig.in_dim``.

The loss machinery lives in :mod:`clinical_jepa.losses`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .graph.patches import PatchData, PatchTask, pool_nodes_to_patches, visible_mask
from .losses import (
    _batch_bounds,
    _candidate_distractors_for_positive,
    _clinical_artifact_mask,
    _confidence_supervision_masks,
    _edge_triples,
    _observed_invalid_negatives,
    _reversed_schema_invalid_negatives,
    _sample_revision_negatives,
    _schema_edge_masks,
    _select_hidden_positive_indices,
    _unique_negative_triples,
    _weighted_relation_balanced_bce,
    _with_edge_mask,
)

try:
    from torch_geometric.nn import GATConv, GINEConv
except ImportError:  # pragma: no cover - exercised only without PyG installed.
    GATConv = None
    GINEConv = None


def _mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    *,
    activation: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        activation(),
        nn.Linear(hidden, out_dim),
    )


class TypedMessageLayer(nn.Module):
    """Pure-torch typed message passing layer.

    Each directed KG edge sends a typed message from source to target and a
    separate reverse message from target to source.  That keeps relation
    direction visible while still letting evidence flow both ways in small KGs.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.relation_emb = nn.Embedding(cfg.num_relations, cfg.hidden_dim)
        self.msg = _mlp(2 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.rev_msg = _mlp(2 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.self_lin = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.self_lin(h)

        src = edge_index[0]
        dst = edge_index[1]
        r = self.relation_emb(edge_type)
        out = h.new_zeros(h.shape)
        deg = h.new_zeros((h.size(0), 1))

        fwd = self.msg(torch.cat([h[src], r], dim=-1))
        out.index_add_(0, dst, fwd)
        deg.index_add_(0, dst, h.new_ones((dst.numel(), 1)))

        rev = self.rev_msg(torch.cat([h[dst], r], dim=-1))
        out.index_add_(0, src, rev)
        deg.index_add_(0, src, h.new_ones((src.numel(), 1)))

        return self.self_lin(h) + out / deg.clamp_min(1.0)


class PygMessageLayer(nn.Module):
    """PyTorch Geometric typed message passing layer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if GINEConv is None or GATConv is None:
            raise ImportError(
                "The model was configured with gnn_backend='pyg', but "
                "torch_geometric is not importable. Install torch-geometric or "
                "run with --gnn-backend torch."
            )
        self.cfg = cfg
        self.relation_emb = nn.Embedding(cfg.num_relations, cfg.hidden_dim)
        if cfg.conv == "gine":
            self.conv = GINEConv(
                _mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim),
                edge_dim=cfg.hidden_dim,
            )
        elif cfg.conv == "gat":
            self.conv = GATConv(
                cfg.hidden_dim,
                cfg.hidden_dim,
                heads=1,
                edge_dim=cfg.hidden_dim,
                add_self_loops=False,
            )
        else:
            raise ValueError(f"unknown conv: {cfg.conv!r}")

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        edge_attr = self.relation_emb(edge_type)
        return self.conv(h, edge_index, edge_attr)


class GraphNodeEncoder(nn.Module):
    """Typed-edge GNN used before patch pooling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        if cfg.conv not in {"gine", "gat"}:
            raise ValueError(f"unknown conv: {cfg.conv!r}")
        if cfg.gnn_backend not in {"pyg", "torch"}:
            raise ValueError(f"unknown gnn_backend: {cfg.gnn_backend!r}")
        layer_cls = PygMessageLayer if cfg.gnn_backend == "pyg" else TypedMessageLayer
        self.layers = nn.ModuleList(
            layer_cls(cfg) for _ in range(cfg.num_gnn_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_gnn_layers)
        )
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = layer(h, edge_index, edge_type)
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + residual
        return self.out_proj(h)


class PatchTransformer(nn.Module):
    """Small transformer over patch tokens.

    Masked patches keep their positional signal but use a learned content token.
    Visible patches act as keys/values for context prediction.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(cfg.latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        ff_dim = int(cfg.latent_dim * cfg.patch_mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.patch_heads,
            dim_feedforward=ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.patch_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(
        self,
        content: torch.Tensor,
        pos: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze = content.dim() == 2
        if squeeze:
            content = content.unsqueeze(0)
            pos = pos.unsqueeze(0)
            if visible is not None:
                visible = visible.unsqueeze(0)
            if valid is not None:
                valid = valid.unsqueeze(0)

        tokens = content + pos
        key_padding_mask = None
        if visible is not None:
            if valid is None:
                valid = torch.ones_like(visible)
            visible = visible & valid
            if visible.size(1) > 0:
                no_visible = ~visible.any(dim=1)
                if bool(no_visible.any()):
                    visible = visible.clone()
                    visible[no_visible, 0] = True
            masked_content = self.mask_token.to(content.dtype).expand_as(content)
            tokens = torch.where(visible.unsqueeze(-1), tokens, masked_content + pos)
            key_padding_mask = ~visible
        elif valid is not None:
            key_padding_mask = ~valid
            if key_padding_mask.size(1) > 0:
                no_valid = key_padding_mask.all(dim=1)
                if bool(no_valid.any()):
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[no_valid, 0] = False

        out = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        out = self.out_norm(out)
        return out.squeeze(0) if squeeze else out


class EdgePlausibilityHead(nn.Module):
    """Scores a typed edge ``(z_src, z_tgt, relation)``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.relation_emb = nn.Embedding(cfg.num_relations, cfg.latent_dim)
        self.net = _mlp(3 * cfg.latent_dim, cfg.latent_dim, 1)

    def forward(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        r = self.relation_emb(relation)
        return self.net(torch.cat([z_src, z_tgt, r], dim=-1)).squeeze(-1)


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, decay: float) -> None:
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.mul_(decay).add_(p_o.detach(), alpha=1.0 - decay)
    for b_o, b_t in zip(online.buffers(), target.buffers()):
        b_t.copy_(b_o)


def vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    if z.size(0) < 2:
        zero = z.sum() * 0.0
        return zero, zero
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    cov = (z.T @ z) / max(1, z.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / z.size(1)
    return var_loss, cov_loss


def _pack_patches(
    values: torch.Tensor,
    patch_data: PatchData,
    fill_value: float | bool = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if patch_data.patch_ptr is None:
        return values, None

    ptr = patch_data.patch_ptr.to(values.device)
    sizes = ptr[1:] - ptr[:-1]
    batch_size = max(0, int(ptr.numel()) - 1)
    max_patches = int(sizes.max().item()) if sizes.numel() else 0
    out_shape = (batch_size, max_patches) + tuple(values.shape[1:])
    if values.dtype == torch.bool:
        out = torch.full(
            out_shape,
            bool(fill_value),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        out = values.new_full(out_shape, fill_value)
    valid = torch.zeros((batch_size, max_patches), dtype=torch.bool, device=values.device)

    for graph_idx in range(batch_size):
        lo = int(ptr[graph_idx].item())
        hi = int(ptr[graph_idx + 1].item())
        length = hi - lo
        if length <= 0:
            continue
        out[graph_idx, :length] = values[lo:hi]
        valid[graph_idx, :length] = True
    return out, valid


def _unpack_patches(values: torch.Tensor, patch_data: PatchData) -> torch.Tensor:
    if patch_data.patch_ptr is None:
        return values

    ptr = patch_data.patch_ptr.to(values.device)
    chunks = []
    for graph_idx in range(max(0, int(ptr.numel()) - 1)):
        lo = int(ptr[graph_idx].item())
        hi = int(ptr[graph_idx + 1].item())
        length = hi - lo
        if length > 0:
            chunks.append(values[graph_idx, :length])
    if not chunks:
        return values.new_zeros((0,) + tuple(values.shape[2:]))
    return torch.cat(chunks, dim=0)


class GraphJEPA(nn.Module):
    """Patch-based JEPA with a schema-aware graph-revision edge head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.context_node_encoder = GraphNodeEncoder(cfg)
        self.target_node_encoder = GraphNodeEncoder(cfg)
        self.context_patch_pos = _mlp(cfg.patch_pe_dim, cfg.latent_dim, cfg.latent_dim)
        self.target_patch_pos = _mlp(cfg.patch_pe_dim, cfg.latent_dim, cfg.latent_dim)
        self.context_patch_encoder = PatchTransformer(cfg)
        self.target_patch_encoder = PatchTransformer(cfg)

        self.target_node_encoder.load_state_dict(self.context_node_encoder.state_dict())
        self.target_patch_pos.load_state_dict(self.context_patch_pos.state_dict())
        self.target_patch_encoder.load_state_dict(self.context_patch_encoder.state_dict())
        for module in (
            self.target_node_encoder,
            self.target_patch_pos,
            self.target_patch_encoder,
        ):
            for p in module.parameters():
                p.requires_grad_(False)

        self.predictor = _mlp(2 * cfg.latent_dim, cfg.predictor_hidden, cfg.latent_dim)
        self.edge_head = EdgePlausibilityHead(cfg)

    def update_target(self, decay: float) -> None:
        update_ema(self.context_node_encoder, self.target_node_encoder, decay)
        update_ema(self.context_patch_pos, self.target_patch_pos, decay)
        update_ema(self.context_patch_encoder, self.target_patch_encoder, decay)

    def encode_nodes(self, data) -> torch.Tensor:
        return self.context_node_encoder(data.x, data.edge_index, data.edge_type)

    @torch.no_grad()
    def encode_target_nodes(self, data) -> torch.Tensor:
        self.target_node_encoder.eval()
        return self.target_node_encoder(data.x, data.edge_index, data.edge_type)

    def _context_patches(
        self,
        data,
        patch_data: PatchData,
        visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_z = self.encode_nodes(data)
        content = pool_nodes_to_patches(node_z, patch_data)
        pos = self.context_patch_pos(patch_data.patch_pos.to(node_z.device))
        visible = visible.to(node_z.device)
        if patch_data.patch_ptr is not None:
            content, valid = _pack_patches(content, patch_data)
            pos_dense, _ = _pack_patches(pos, patch_data)
            visible, _ = _pack_patches(visible, patch_data, fill_value=False)
            patches = self.context_patch_encoder(content, pos_dense, visible, valid)
            patches = _unpack_patches(patches, patch_data)
        else:
            patches = self.context_patch_encoder(content, pos, visible)
        return patches, pos

    @torch.no_grad()
    def _target_patches(self, data, patch_data: PatchData) -> torch.Tensor:
        self.target_node_encoder.eval()
        self.target_patch_pos.eval()
        self.target_patch_encoder.eval()
        node_z = self.encode_target_nodes(data)
        content = pool_nodes_to_patches(node_z, patch_data)
        pos = self.target_patch_pos(patch_data.patch_pos.to(node_z.device))
        if patch_data.patch_ptr is not None:
            content, valid = _pack_patches(content, patch_data)
            pos, _ = _pack_patches(pos, patch_data)
            patches = self.target_patch_encoder(content, pos, None, valid)
            return _unpack_patches(patches, patch_data)
        return self.target_patch_encoder(content, pos, None)

    def jepa_loss(
        self,
        data,
        patch_data: PatchData,
        task: PatchTask,
        *,
        var_weight: float,
        cov_weight: float,
    ) -> Tuple[torch.Tensor, dict]:
        if patch_data.num_patches < 2 or task.target_idx.numel() == 0:
            zero = self.predictor[0].weight.sum() * 0.0
            return zero, {
                "jepa_inv": 0.0,
                "jepa_var": 0.0,
                "jepa_cov": 0.0,
                "patch_std": 0.0,
            }

        visible = visible_mask(patch_data.num_patches, task.context_idx).to(data.x.device)
        ctx, pos = self._context_patches(data, patch_data, visible)
        tgt = self._target_patches(data, patch_data)
        target_idx = task.target_idx.to(data.x.device)

        pred_in = torch.cat([ctx[target_idx], pos[target_idx]], dim=-1)
        pred = self.predictor(pred_in)
        target = tgt[target_idx].detach()
        inv = F.smooth_l1_loss(pred, target)
        var_loss, cov_loss = vicreg_terms(ctx[visible])
        loss = inv + var_weight * var_loss + cov_weight * cov_loss
        with torch.no_grad():
            patch_std = ctx.std(dim=0).mean() if ctx.size(0) > 1 else ctx.std()
        return loss, {
            "jepa_inv": float(inv.detach()),
            "jepa_var": float(var_loss.detach()),
            "jepa_cov": float(cov_loss.detach()),
            "patch_std": float(patch_std.detach()),
        }

    def revision_loss(
        self,
        data,
        *,
        mask_ratio: float,
        neg_per_pos: int,
        llm_confidence_negatives: bool = False,
        llm_negative_threshold: float = 0.3,
        llm_positive_threshold: float = 0.8,
        llm_negative_threshold_by_relation: dict[str, float] | None = None,
        llm_positive_threshold_by_relation: dict[str, float] | None = None,
        llm_negative_weight: float = 0.2,
        clinical_artifact_filters: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """Schema-aware joint graph-revision loss.

        Positives are schema-valid observed typed edges the confidence
        supervision trusts. Invalid observed typed edges, their reversals, and
        weak LLM negatives are explicit negatives, joined by hard schema-valid
        absent triples from the same graph. Message passing uses sanitized graph
        structure so the encoder does not propagate through known-bad typed
        directions. Uncertain LLM edges are ignored, not supervised.
        """

        if data.edge_index.size(1) == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "revision_bce": 0.0,
                "revision_pos": 0,
                "revision_neg": 0,
                "revision_schema_neg": 0,
                "revision_invalid_neg": 0,
                "revision_reverse_neg": 0,
                "revision_llm_neg": 0,
                "revision_artifact_neg": 0,
                "revision_llm_ignored": 0,
                "revision_hidden": 0,
            }

        schema_valid, schema_invalid, unconstrained = _schema_edge_masks(
            data,
            allow_unconstrained=False,
        )
        trusted_positive, weak_negative, llm_ignored = _confidence_supervision_masks(
            data,
            enabled=llm_confidence_negatives,
            negative_threshold=llm_negative_threshold,
            positive_threshold=llm_positive_threshold,
            negative_threshold_by_relation=llm_negative_threshold_by_relation,
            positive_threshold_by_relation=llm_positive_threshold_by_relation,
            clinical_artifact_filters=clinical_artifact_filters,
        )
        artifact = _clinical_artifact_mask(
            data,
            enabled=clinical_artifact_filters,
        )
        pos_mask = schema_valid & trusted_positive
        invalid_mask = schema_invalid
        weak_negative_mask = weak_negative & ~schema_invalid
        llm_weak_mask = weak_negative_mask & ~artifact
        artifact_negative_mask = weak_negative_mask & artifact
        message_mask = (schema_valid | unconstrained) & trusted_positive
        pos_idx = pos_mask.nonzero(as_tuple=False).flatten()
        device = data.edge_index.device

        hidden = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=device)
        if pos_idx.numel() > 0 and mask_ratio > 0.0:
            hidden_pos = torch.rand(pos_idx.numel(), device=device) < mask_ratio
            if not bool(hidden_pos.any()):
                hidden_pos[
                    int(torch.randint(pos_idx.numel(), (1,), device=device).item())
                ] = True
            hidden[pos_idx[hidden_pos]] = True

        keep = message_mask & ~hidden
        z = self.context_node_encoder(
            data.x,
            data.edge_index[:, keep],
            data.edge_type[keep],
        )

        if pos_idx.numel() > 0:
            pos_src = data.edge_index[0, pos_mask]
            pos_dst = data.edge_index[1, pos_mask]
            pos_rel = data.edge_type[pos_mask]
            pos_logit = self.edge_head(z[pos_src], z[pos_dst], pos_rel)
        else:
            pos_src = pos_dst = pos_rel = torch.zeros(
                (0,),
                dtype=torch.long,
                device=device,
            )
            pos_logit = z.new_zeros((0,))

        positives = {
            (int(s), int(t), int(r))
            for s, t, r in zip(pos_src.tolist(), pos_dst.tolist(), pos_rel.tolist())
        }
        pos_data = _with_edge_mask(data, pos_mask)
        schema_neg = _sample_revision_negatives(pos_data, neg_per_pos=neg_per_pos)
        invalid_neg = _observed_invalid_negatives(data, invalid_mask)
        reverse_neg = _reversed_schema_invalid_negatives(
            data,
            pos_src,
            pos_dst,
            pos_rel,
        )
        weak_neg = _observed_invalid_negatives(data, weak_negative_mask)
        weak_negative_triples = _edge_triples(data, weak_negative_mask)
        llm_negative_triples = _edge_triples(data, llm_weak_mask)
        artifact_negative_triples = _edge_triples(data, artifact_negative_mask)
        negative_exclusions = _edge_triples(
            data,
            ~(weak_negative_mask | schema_invalid),
        )
        neg_src, neg_dst, neg_rel = _unique_negative_triples(
            [schema_neg, invalid_neg, reverse_neg, weak_neg],
            positives | negative_exclusions,
        )

        if neg_src.numel() > 0:
            neg_logit = self.edge_head(z[neg_src], z[neg_dst], neg_rel)
            neg_weights = torch.tensor(
                [
                    (
                        float(llm_negative_weight)
                        if (int(s), int(t), int(r)) in weak_negative_triples
                        else 1.0
                    )
                    for s, t, r in zip(
                        neg_src.tolist(),
                        neg_dst.tolist(),
                        neg_rel.tolist(),
                    )
                ],
                dtype=neg_logit.dtype,
                device=device,
            )
        else:
            neg_logit = z.new_zeros((0,))
            neg_weights = z.new_zeros((0,))

        if pos_logit.numel() == 0 and neg_logit.numel() == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "revision_bce": 0.0,
                "revision_pos": 0,
                "revision_neg": 0,
                "revision_schema_neg": int(schema_neg[0].numel()),
                "revision_invalid_neg": int(invalid_neg[0].numel()),
                "revision_reverse_neg": int(reverse_neg[0].numel()),
                "revision_llm_neg": 0,
                "revision_artifact_neg": 0,
                "revision_llm_ignored": int(llm_ignored.sum().item()),
                "revision_hidden": int(hidden.sum().item()),
            }

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        relations = torch.cat([pos_rel, neg_rel])
        weights = torch.cat([torch.ones_like(pos_logit), neg_weights])
        loss = _weighted_relation_balanced_bce(logits, labels, relations, weights)
        included_llm_neg = sum(
            1
            for s, t, r in zip(
                neg_src.tolist(),
                neg_dst.tolist(),
                neg_rel.tolist(),
            )
            if (int(s), int(t), int(r)) in llm_negative_triples
        )
        included_artifact_neg = sum(
            1
            for s, t, r in zip(
                neg_src.tolist(),
                neg_dst.tolist(),
                neg_rel.tolist(),
            )
            if (int(s), int(t), int(r)) in artifact_negative_triples
        )
        return loss, {
            "revision_bce": float(loss.detach()),
            "revision_pos": int(pos_logit.numel()),
            "revision_neg": int(neg_logit.numel()),
            "revision_schema_neg": int(schema_neg[0].numel()),
            "revision_invalid_neg": int(invalid_neg[0].numel()),
            "revision_reverse_neg": int(reverse_neg[0].numel()),
            "revision_llm_neg": included_llm_neg,
            "revision_artifact_neg": included_artifact_neg,
            "revision_llm_ignored": int(llm_ignored.sum().item()),
            "revision_hidden": int(hidden.sum().item()),
        }

    def candidate_ranking_loss(
        self,
        data,
        *,
        mask_ratio: float,
        neg_per_pos: int,
        max_pos: int,
        temperature: float,
        llm_confidence_negatives: bool = False,
        llm_negative_threshold: float = 0.3,
        llm_positive_threshold: float = 0.8,
        llm_negative_threshold_by_relation: dict[str, float] | None = None,
        llm_positive_threshold_by_relation: dict[str, float] | None = None,
        clinical_artifact_filters: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """Rank hidden true edges above hard schema-valid candidate distractors."""

        if data.edge_index.size(1) == 0 or neg_per_pos <= 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": 0,
                "ranking_llm_excluded": 0,
                "ranking_artifact_excluded": 0,
            }

        schema_valid, _schema_invalid, unconstrained = _schema_edge_masks(
            data,
            allow_unconstrained=False,
        )
        trusted_positive, weak_negative, llm_ignored = _confidence_supervision_masks(
            data,
            enabled=llm_confidence_negatives,
            negative_threshold=llm_negative_threshold,
            positive_threshold=llm_positive_threshold,
            negative_threshold_by_relation=llm_negative_threshold_by_relation,
            positive_threshold_by_relation=llm_positive_threshold_by_relation,
            clinical_artifact_filters=clinical_artifact_filters,
        )
        artifact = _clinical_artifact_mask(
            data,
            enabled=clinical_artifact_filters,
        )
        positive_indices = (schema_valid & trusted_positive).nonzero(
            as_tuple=False,
        ).flatten()
        llm_excluded = int(((weak_negative & ~artifact) | llm_ignored).sum().item())
        artifact_excluded = int(artifact.sum().item())
        hidden_indices = _select_hidden_positive_indices(
            positive_indices,
            mask_ratio=mask_ratio,
            max_pos=max_pos,
        )
        if hidden_indices.numel() == 0:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": 0,
                "ranking_llm_excluded": llm_excluded,
                "ranking_artifact_excluded": artifact_excluded,
            }

        node_type = getattr(data, "node_type", None)
        if node_type is None:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": int(hidden_indices.numel()),
                "ranking_llm_excluded": llm_excluded,
                "ranking_artifact_excluded": artifact_excluded,
            }
        node_type_cpu = node_type.detach().cpu().tolist()
        src_cpu = data.edge_index[0].detach().cpu().tolist()
        dst_cpu = data.edge_index[1].detach().cpu().tolist()
        rel_cpu = data.edge_type.detach().cpu().tolist()
        existing = {
            (int(s), int(t), int(r))
            for s, t, r in zip(src_cpu, dst_cpu, rel_cpu)
        }
        batch_cpu, ptr_cpu = _batch_bounds(data)

        groups: list[tuple[int, int, int, list[tuple[int, int, int]]]] = []
        for edge_idx in hidden_indices.detach().cpu().tolist():
            edge_idx = int(edge_idx)
            source = int(src_cpu[edge_idx])
            target = int(dst_cpu[edge_idx])
            relation = int(rel_cpu[edge_idx])
            negatives = _candidate_distractors_for_positive(
                data,
                source=source,
                target=target,
                relation=relation,
                node_type_cpu=node_type_cpu,
                existing=existing,
                batch_cpu=batch_cpu,
                ptr_cpu=ptr_cpu,
                limit=neg_per_pos,
            )
            if negatives:
                groups.append((source, target, relation, negatives))

        if not groups:
            zero = self.edge_head.net[0].weight.sum() * 0.0
            return zero, {
                "ranking_ce": 0.0,
                "ranking_pos": 0,
                "ranking_neg": 0,
                "ranking_hidden": int(hidden_indices.numel()),
                "ranking_llm_excluded": llm_excluded,
                "ranking_artifact_excluded": artifact_excluded,
            }

        hidden_mask = torch.zeros(
            data.edge_index.size(1),
            dtype=torch.bool,
            device=data.edge_index.device,
        )
        hidden_mask[hidden_indices] = True
        message_mask = (
            (schema_valid | unconstrained)
            & trusted_positive
            & ~hidden_mask
        )
        z = self.context_node_encoder(
            data.x,
            data.edge_index[:, message_mask],
            data.edge_type[message_mask],
        )

        temperature = max(float(temperature), 1e-6)
        losses = []
        total_neg = 0
        device = data.edge_index.device
        for source, target, relation, negatives in groups:
            pos_rel = torch.tensor([relation], dtype=torch.long, device=device)
            pos_logit = self.edge_head(
                z[source:source + 1],
                z[target:target + 1],
                pos_rel,
            )
            neg_src, neg_dst, neg_rel = zip(*negatives)
            neg_src_t = torch.tensor(neg_src, dtype=torch.long, device=device)
            neg_dst_t = torch.tensor(neg_dst, dtype=torch.long, device=device)
            neg_rel_t = torch.tensor(neg_rel, dtype=torch.long, device=device)
            neg_logit = self.edge_head(z[neg_src_t], z[neg_dst_t], neg_rel_t)
            logits = torch.cat([pos_logit, neg_logit], dim=0).unsqueeze(0)
            target_class = torch.zeros((1,), dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits / temperature, target_class))
            total_neg += len(negatives)

        loss = torch.stack(losses).mean()
        return loss, {
            "ranking_ce": float(loss.detach()),
            "ranking_pos": len(groups),
            "ranking_neg": total_neg,
            "ranking_hidden": int(hidden_indices.numel()),
            "ranking_llm_excluded": llm_excluded,
            "ranking_artifact_excluded": artifact_excluded,
        }

    @torch.no_grad()
    def patch_prediction_energy(
        self,
        data,
        patch_data: PatchData,
        target_patch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized prediction energy for target patches."""
        device = data.x.device
        target_patch_idx = target_patch_idx.to(device)
        visible = torch.ones(patch_data.num_patches, dtype=torch.bool, device=device)
        visible[target_patch_idx] = False
        if not bool(visible.any()):
            visible[target_patch_idx] = True

        ctx, pos = self._context_patches(data, patch_data, visible)
        tgt = self._target_patches(data, patch_data)
        pred = self.predictor(torch.cat([ctx[target_patch_idx], pos[target_patch_idx]], dim=-1))
        energy = torch.norm(pred - tgt[target_patch_idx], dim=-1)
        return energy / math.sqrt(self.cfg.latent_dim)

    @torch.no_grad()
    def encode_graph(self, data, patch_data: PatchData) -> torch.Tensor:
        patches = self._target_patches(data, patch_data)
        if patches.numel() == 0:
            return data.x.new_zeros((self.cfg.latent_dim,))
        return patches.mean(dim=0)


__all__ = [
    "EdgePlausibilityHead",
    "GraphJEPA",
    "GraphNodeEncoder",
    "PatchTransformer",
    "PygMessageLayer",
    "TypedMessageLayer",
    "update_ema",
    "vicreg_terms",
]
