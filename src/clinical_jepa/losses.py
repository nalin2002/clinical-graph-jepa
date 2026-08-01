"""Graph-revision and candidate-ranking loss machinery.

Merged from ``fawkes_core/revision.py`` (the schema-aware v4 losses, whose
``GraphJEPAv4`` class is dissolved into :class:`clinical_jepa.model.GraphJEPA`)
and the free functions of ``graph_jepa_v{5,6}/model.py`` (LLM-confidence
supervision and candidate ranking).

Everything here is a pure function over a PyG ``Data`` object — no ``nn.Module``
and no import of :mod:`clinical_jepa.model`, so the model imports the losses and
not the other way round.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .graph.builders import RELATION_SCHEMA, canonical_relation
from .schema import (
    EDGE_TYPE_TO_IDX,
    IDX_TO_EDGE_TYPE,
    IDX_TO_NODE_TYPE,
    NODE_TYPE_TO_IDX,
)

SCHEMA_UNCONSTRAINED_RELATIONS = {"ASSOCIATED_WITH", "CO_OCCURS_WITH"}


# --------------------------------------------------------------------------- #
# Typed-schema lookups.
# --------------------------------------------------------------------------- #
def _allowed_target_type_indices(src_type_idx: int, rel_idx: int) -> set[int]:
    src_type = IDX_TO_NODE_TYPE.get(int(src_type_idx))
    rel = IDX_TO_EDGE_TYPE.get(int(rel_idx))
    if src_type is None or rel is None:
        return set()
    return {
        NODE_TYPE_TO_IDX[target_type]
        for target_type in RELATION_SCHEMA.get((src_type, rel), set())
        if target_type in NODE_TYPE_TO_IDX
    }


def _allowed_relation_indices(src_type_idx: int, target_type_idx: int) -> set[int]:
    src_type = IDX_TO_NODE_TYPE.get(int(src_type_idx))
    target_type = IDX_TO_NODE_TYPE.get(int(target_type_idx))
    if src_type is None or target_type is None:
        return set()
    return {
        EDGE_TYPE_TO_IDX[relation]
        for (schema_src_type, relation), target_types in RELATION_SCHEMA.items()
        if schema_src_type == src_type
        and target_type in target_types
        and relation in EDGE_TYPE_TO_IDX
    }


def _is_unconstrained_relation(relation: str) -> bool:
    return canonical_relation(relation) in SCHEMA_UNCONSTRAINED_RELATIONS


def _is_schema_valid_type_triple(
    src_type: str | None,
    relation: str | None,
    tgt_type: str | None,
    *,
    allow_unconstrained: bool,
) -> bool:
    relation = canonical_relation(relation or "")
    if allow_unconstrained and _is_unconstrained_relation(relation):
        return True
    return tgt_type in RELATION_SCHEMA.get((src_type, relation), set())


def _schema_edge_masks(
    data,
    *,
    allow_unconstrained: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(valid, invalid, unconstrained)`` masks over ``data.edge_index``."""
    device = data.edge_index.device
    num_edges = int(data.edge_index.size(1))
    if num_edges == 0:
        empty = torch.zeros((0,), dtype=torch.bool, device=device)
        return empty, empty, empty

    node_type = getattr(data, "node_type", None)
    if node_type is None:
        valid = torch.ones((num_edges,), dtype=torch.bool, device=device)
        empty = torch.zeros((num_edges,), dtype=torch.bool, device=device)
        return valid, empty, empty

    src = data.edge_index[0].detach().cpu().tolist()
    dst = data.edge_index[1].detach().cpu().tolist()
    rel = data.edge_type.detach().cpu().tolist()
    node_type_cpu = node_type.detach().cpu().tolist()

    valid_values: list[bool] = []
    unconstrained_values: list[bool] = []
    for s, t, r in zip(src, dst, rel):
        src_type = IDX_TO_NODE_TYPE.get(int(node_type_cpu[int(s)]))
        tgt_type = IDX_TO_NODE_TYPE.get(int(node_type_cpu[int(t)]))
        relation = IDX_TO_EDGE_TYPE.get(int(r))
        unconstrained = bool(relation and _is_unconstrained_relation(relation))
        valid_values.append(
            _is_schema_valid_type_triple(
                src_type,
                relation,
                tgt_type,
                allow_unconstrained=allow_unconstrained,
            )
        )
        unconstrained_values.append(unconstrained)

    valid = torch.tensor(valid_values, dtype=torch.bool, device=device)
    unconstrained = torch.tensor(unconstrained_values, dtype=torch.bool, device=device)
    invalid = ~valid & ~unconstrained
    return valid, invalid, unconstrained


def _with_edge_mask(data, mask: torch.Tensor):
    masked = copy.copy(data)
    masked.edge_index = data.edge_index[:, mask]
    masked.edge_type = data.edge_type[mask]
    return masked


def sanitized_graph_data(data):
    """Return a view with typed-invalid edges removed from message passing."""
    valid, _invalid, unconstrained = _schema_edge_masks(
        data,
        allow_unconstrained=True,
    )
    return _with_edge_mask(data, valid | unconstrained)


# --------------------------------------------------------------------------- #
# Negative sampling for the revision BCE.
# --------------------------------------------------------------------------- #
def _graph_bounds(data, node_idx: int, batch_cpu, ptr_cpu) -> tuple[int, int]:
    if batch_cpu is None or ptr_cpu is None:
        return 0, int(data.num_nodes)
    graph_id = int(batch_cpu[node_idx])
    return int(ptr_cpu[graph_id]), int(ptr_cpu[graph_id + 1])


def _batch_bounds(data):
    batch = getattr(data, "batch", None)
    ptr = getattr(data, "ptr", None)
    if batch is None or ptr is None:
        return None, None
    return batch.detach().cpu().tolist(), ptr.detach().cpu().tolist()


def _append_negative(
    neg_src: list[int],
    neg_dst: list[int],
    neg_rel: list[int],
    existing: set[tuple[int, int, int]],
    seen: set[tuple[int, int, int]],
    candidate: tuple[int, int, int],
) -> bool:
    if candidate in existing or candidate in seen:
        return False
    s, t, _r = candidate
    if s == t:
        return False
    seen.add(candidate)
    neg_src.append(s)
    neg_dst.append(t)
    neg_rel.append(_r)
    return True


def _sample_one(candidates: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    if not candidates:
        return None
    idx = int(torch.randint(len(candidates), (1,)).item())
    return candidates[idx]


def _sample_revision_negatives(
    data,
    *,
    neg_per_pos: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample hard, schema-valid false edges from the same graph as each true edge.

    The sampler mixes three corruptions when available:

    * same endpoints, wrong schema-valid relation;
    * same source/relation, wrong schema-valid target;
    * wrong schema-valid source, same relation/target.

    This teaches the edge head to reject plausible-looking but clinically wrong
    additions, such as ``echocardiogram -RULES_OUT-> heart failure`` when the
    observed edge is ``echocardiogram -CONFIRMS-> heart failure``.
    """
    device = data.edge_index.device
    if neg_per_pos <= 0 or data.edge_index.size(1) == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty, empty

    src_cpu = data.edge_index[0].detach().cpu().tolist()
    dst_cpu = data.edge_index[1].detach().cpu().tolist()
    rel_cpu = data.edge_type.detach().cpu().tolist()
    node_type_cpu = getattr(data, "node_type", None)
    node_type_cpu = node_type_cpu.detach().cpu().tolist() if node_type_cpu is not None else None

    batch = getattr(data, "batch", None)
    ptr = getattr(data, "ptr", None)
    if batch is not None and ptr is not None:
        batch_cpu = batch.detach().cpu().tolist()
        ptr_cpu = ptr.detach().cpu().tolist()
    else:
        batch_cpu = None
        ptr_cpu = None

    existing = {
        (int(s), int(t), int(r))
        for s, t, r in zip(src_cpu, dst_cpu, rel_cpu)
    }
    neg_src: list[int] = []
    neg_dst: list[int] = []
    neg_rel: list[int] = []
    seen: set[tuple[int, int, int]] = set()

    for s, t, r in zip(src_cpu, dst_cpu, rel_cpu):
        s = int(s)
        t = int(t)
        r = int(r)
        lo, hi = _graph_bounds(data, s, batch_cpu, ptr_cpu)

        relation_candidates: list[tuple[int, int, int]] = []
        target_candidates: list[tuple[int, int, int]] = []
        source_candidates: list[tuple[int, int, int]] = []

        if node_type_cpu is not None:
            relation_candidates = [
                (s, t, alt_r)
                for alt_r in _allowed_relation_indices(node_type_cpu[s], node_type_cpu[t])
                if alt_r != r
            ]
            if not relation_candidates:
                relation_candidates = [
                    (s, t, alt_r)
                    for alt_r in EDGE_TYPE_TO_IDX.values()
                    if alt_r != r
                ]

        allowed_targets = (
            _allowed_target_type_indices(node_type_cpu[s], r)
            if node_type_cpu is not None
            else set()
        )
        for candidate in range(lo, hi):
            if candidate in (s, t):
                continue
            if node_type_cpu is None:
                target_candidates.append((s, candidate, r))
                source_candidates.append((candidate, t, r))
                continue
            if not allowed_targets or node_type_cpu[candidate] in allowed_targets:
                target_candidates.append((s, candidate, r))

            candidate_target_types = _allowed_target_type_indices(node_type_cpu[candidate], r)
            if not candidate_target_types or node_type_cpu[t] in candidate_target_types:
                source_candidates.append((candidate, t, r))

        buckets = [relation_candidates, target_candidates, source_candidates]
        for offset in range(neg_per_pos):
            for shift in range(len(buckets)):
                bucket = [
                    candidate
                    for candidate in buckets[(offset + shift) % len(buckets)]
                    if candidate not in existing and candidate not in seen
                ]
                candidate = _sample_one(bucket)
                if candidate is None:
                    continue
                if _append_negative(
                    neg_src,
                    neg_dst,
                    neg_rel,
                    existing,
                    seen,
                    candidate,
                ):
                    break

    if not neg_src:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty, empty
    return (
        torch.tensor(neg_src, dtype=torch.long, device=device),
        torch.tensor(neg_dst, dtype=torch.long, device=device),
        torch.tensor(neg_rel, dtype=torch.long, device=device),
    )


def _triples_to_tensors(
    triples: list[tuple[int, int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not triples:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty, empty
    src, dst, rel = zip(*triples)
    return (
        torch.tensor(src, dtype=torch.long, device=device),
        torch.tensor(dst, dtype=torch.long, device=device),
        torch.tensor(rel, dtype=torch.long, device=device),
    )


def _unique_negative_triples(
    parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    positives: set[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triples: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    device = parts[0][0].device if parts else torch.device("cpu")
    for src_t, dst_t, rel_t in parts:
        device = src_t.device
        for s, t, r in zip(src_t.tolist(), dst_t.tolist(), rel_t.tolist()):
            triple = (int(s), int(t), int(r))
            if triple in positives or triple in seen:
                continue
            seen.add(triple)
            triples.append(triple)
    return _triples_to_tensors(triples, device)


def _observed_invalid_negatives(
    data,
    invalid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if invalid_mask.numel() == 0 or not bool(invalid_mask.any()):
        empty = torch.zeros((0,), dtype=torch.long, device=data.edge_index.device)
        return empty, empty, empty
    return (
        data.edge_index[0, invalid_mask],
        data.edge_index[1, invalid_mask],
        data.edge_type[invalid_mask],
    )


def _is_schema_valid_index_triple(data, s: int, t: int, r: int) -> bool:
    node_type = getattr(data, "node_type", None)
    if node_type is None:
        return True
    src_type = IDX_TO_NODE_TYPE.get(int(node_type[int(s)].item()))
    tgt_type = IDX_TO_NODE_TYPE.get(int(node_type[int(t)].item()))
    relation = IDX_TO_EDGE_TYPE.get(int(r))
    return _is_schema_valid_type_triple(
        src_type,
        relation,
        tgt_type,
        allow_unconstrained=False,
    )


def _reversed_schema_invalid_negatives(
    data,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    pos_rel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triples: list[tuple[int, int, int]] = []
    for s, t, r in zip(pos_src.tolist(), pos_dst.tolist(), pos_rel.tolist()):
        s = int(s)
        t = int(t)
        r = int(r)
        if s == t:
            continue
        if not _is_schema_valid_index_triple(data, t, s, r):
            triples.append((t, s, r))
    return _triples_to_tensors(triples, data.edge_index.device)


def _edge_triples(data, mask: torch.Tensor) -> set[tuple[int, int, int]]:
    return {
        (int(source), int(target), int(relation))
        for source, target, relation in zip(
            data.edge_index[0, mask].tolist(),
            data.edge_index[1, mask].tolist(),
            data.edge_type[mask].tolist(),
        )
    }


# --------------------------------------------------------------------------- #
# Relation-balanced BCE.
# --------------------------------------------------------------------------- #
def _relation_balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    relation: torch.Tensor,
) -> torch.Tensor:
    per_example = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    relation_losses = []
    for rel in relation.unique():
        rel_mask = relation == rel
        label_losses = []
        for label in (0.0, 1.0):
            label_mask = rel_mask & (labels == label)
            if bool(label_mask.any()):
                label_losses.append(per_example[label_mask].mean())
        if label_losses:
            relation_losses.append(torch.stack(label_losses).mean())
    if not relation_losses:
        return per_example.mean()
    return torch.stack(relation_losses).mean()


def _weighted_relation_balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    relation: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if bool(torch.all(weights == 1.0)):
        return _relation_balanced_bce(logits, labels, relation)

    per_example = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    relation_losses = []
    for rel in relation.unique():
        rel_mask = relation == rel
        label_losses = []
        for label in (0.0, 1.0):
            cell_mask = rel_mask & (labels == label)
            if not bool(cell_mask.any()):
                continue
            cell_weights = weights[cell_mask]
            label_losses.append((per_example[cell_mask] * cell_weights).mean())
        if label_losses:
            relation_losses.append(torch.stack(label_losses).mean())
    if not relation_losses:
        return (per_example * weights).mean()
    return torch.stack(relation_losses).mean()


# --------------------------------------------------------------------------- #
# LLM-confidence supervision.
# --------------------------------------------------------------------------- #
def _confidence_supervision_masks(
    data,
    *,
    enabled: bool,
    negative_threshold: float,
    positive_threshold: float,
    negative_threshold_by_relation: dict[str, float] | None = None,
    positive_threshold_by_relation: dict[str, float] | None = None,
    clinical_artifact_filters: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return trusted-positive, weak-negative, and ignored edge masks."""

    num_edges = int(data.edge_index.size(1))
    device = data.edge_index.device
    all_edges = torch.ones((num_edges,), dtype=torch.bool, device=device)
    no_edges = torch.zeros((num_edges,), dtype=torch.bool, device=device)
    artifact = getattr(data, "edge_clinical_artifact", None)
    if (
        not clinical_artifact_filters
        or artifact is None
        or int(artifact.numel()) != num_edges
    ):
        artifact = no_edges
    else:
        artifact = artifact.to(device=device, dtype=torch.bool)

    confidence = getattr(data, "edge_llm_confidence", None)
    if (
        not enabled
        or confidence is None
        or int(confidence.numel()) != num_edges
    ):
        return all_edges & ~artifact, artifact, no_edges

    confidence = confidence.to(device)
    scored = torch.isfinite(confidence)
    is_llm = getattr(data, "edge_is_llm", None)
    if is_llm is None or int(is_llm.numel()) != num_edges:
        is_llm = scored
    else:
        is_llm = is_llm.to(device=device, dtype=torch.bool)

    negative_by_relation = {
        str(relation).upper(): float(threshold)
        for relation, threshold in (negative_threshold_by_relation or {}).items()
    }
    positive_by_relation = {
        str(relation).upper(): float(threshold)
        for relation, threshold in (positive_threshold_by_relation or {}).items()
    }
    relation_indices = data.edge_type.detach().cpu().tolist()
    negative_values = [
        negative_by_relation.get(
            str(IDX_TO_EDGE_TYPE.get(int(relation), "")).upper(),
            float(negative_threshold),
        )
        for relation in relation_indices
    ]
    positive_values = [
        positive_by_relation.get(
            str(IDX_TO_EDGE_TYPE.get(int(relation), "")).upper(),
            float(positive_threshold),
        )
        for relation in relation_indices
    ]
    negative_cutoff = torch.tensor(
        negative_values,
        dtype=confidence.dtype,
        device=device,
    )
    positive_cutoff = torch.tensor(
        positive_values,
        dtype=confidence.dtype,
        device=device,
    )

    scored_llm = is_llm & scored
    low = scored_llm & (confidence <= negative_cutoff)
    high = scored_llm & (confidence >= positive_cutoff) & ~low
    ignored = is_llm & ~low & ~high
    trusted_positive = (~is_llm | high) & ~artifact
    weak_negative = (low | artifact) & ~trusted_positive
    ignored &= ~artifact
    return trusted_positive, weak_negative, ignored


def _clinical_artifact_mask(data, *, enabled: bool) -> torch.Tensor:
    num_edges = int(data.edge_index.size(1))
    artifact = getattr(data, "edge_clinical_artifact", None)
    if not enabled or artifact is None or int(artifact.numel()) != num_edges:
        return torch.zeros(
            (num_edges,),
            dtype=torch.bool,
            device=data.edge_index.device,
        )
    return artifact.to(device=data.edge_index.device, dtype=torch.bool)


def confidence_sanitized_graph_data(
    data,
    *,
    enabled: bool,
    negative_threshold: float,
    positive_threshold: float,
    negative_threshold_by_relation: dict[str, float] | None = None,
    positive_threshold_by_relation: dict[str, float] | None = None,
    clinical_artifact_filters: bool = False,
):
    """Keep only schema-valid, trusted edges for fine-tuning message passing."""

    valid, _invalid, unconstrained = _schema_edge_masks(
        data,
        allow_unconstrained=True,
    )
    trusted_positive, _weak_negative, _ignored = _confidence_supervision_masks(
        data,
        enabled=enabled,
        negative_threshold=negative_threshold,
        positive_threshold=positive_threshold,
        negative_threshold_by_relation=negative_threshold_by_relation,
        positive_threshold_by_relation=positive_threshold_by_relation,
        clinical_artifact_filters=clinical_artifact_filters,
    )
    keep = (valid | unconstrained) & trusted_positive
    masked = _with_edge_mask(data, keep)
    for field in (
        "edge_llm_confidence",
        "edge_is_llm",
        "edge_clinical_artifact",
    ):
        values = getattr(data, field, None)
        if values is not None and int(values.numel()) == int(keep.numel()):
            setattr(masked, field, values[keep])
    return masked


def pretrain_sanitized_graph_data(
    data,
    *,
    negative_threshold: float,
    negative_threshold_by_relation: dict[str, float] | None = None,
):
    """Keep schema-valid edges except LLM weak negatives for pretraining."""

    valid, _invalid, unconstrained = _schema_edge_masks(
        data,
        allow_unconstrained=True,
    )
    _trusted_positive, weak_negative, _ignored = _confidence_supervision_masks(
        data,
        enabled=True,
        negative_threshold=negative_threshold,
        positive_threshold=1.0,
        negative_threshold_by_relation=negative_threshold_by_relation,
    )
    keep = (valid | unconstrained) & ~weak_negative
    masked = _with_edge_mask(data, keep)
    for field in (
        "edge_llm_confidence",
        "edge_is_llm",
        "edge_clinical_artifact",
    ):
        values = getattr(data, field, None)
        if values is not None and int(values.numel()) == int(keep.numel()):
            setattr(masked, field, values[keep])
    return masked


# --------------------------------------------------------------------------- #
# Candidate ranking.
# --------------------------------------------------------------------------- #
def _sample_without_replacement(
    candidates: list[tuple[int, int, int]],
    *,
    limit: int,
) -> list[tuple[int, int, int]]:
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        order = torch.randperm(len(candidates)).tolist()
    else:
        order = torch.randperm(len(candidates))[:limit].tolist()
    return [candidates[int(idx)] for idx in order]


def _candidate_distractors_for_positive(
    data,
    *,
    source: int,
    target: int,
    relation: int,
    node_type_cpu: list[int],
    existing: set[tuple[int, int, int]],
    batch_cpu,
    ptr_cpu,
    limit: int,
) -> list[tuple[int, int, int]]:
    """Build hard schema-valid distractors for one hidden positive edge."""

    relation_candidates = [
        (source, target, alt_relation)
        for alt_relation in _allowed_relation_indices(
            node_type_cpu[source],
            node_type_cpu[target],
        )
        if alt_relation != relation
    ]

    allowed_targets = _allowed_target_type_indices(node_type_cpu[source], relation)
    target_candidates: list[tuple[int, int, int]] = []
    source_candidates: list[tuple[int, int, int]] = []
    lo, hi = _graph_bounds(data, source, batch_cpu, ptr_cpu)
    for candidate in range(lo, hi):
        if candidate in (source, target):
            continue
        if node_type_cpu[candidate] in allowed_targets:
            target_candidates.append((source, candidate, relation))

        candidate_target_types = _allowed_target_type_indices(
            node_type_cpu[candidate],
            relation,
        )
        if node_type_cpu[target] in candidate_target_types:
            source_candidates.append((candidate, target, relation))

    buckets = [target_candidates, source_candidates, relation_candidates]
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    per_bucket = max(1, limit // max(1, len(buckets)))
    for bucket in buckets:
        available = [
            candidate
            for candidate in bucket
            if candidate not in existing and candidate not in seen
        ]
        take = min(per_bucket, limit - len(out))
        for candidate in _sample_without_replacement(available, limit=take):
            seen.add(candidate)
            out.append(candidate)
        if len(out) >= limit:
            return out

    if len(out) < limit:
        remaining = [
            candidate
            for bucket in buckets
            for candidate in bucket
            if candidate not in existing and candidate not in seen
        ]
        for candidate in _sample_without_replacement(
            remaining,
            limit=limit - len(out),
        ):
            seen.add(candidate)
            out.append(candidate)
    return out


def _select_hidden_positive_indices(
    positive_indices: torch.Tensor,
    *,
    mask_ratio: float,
    max_pos: int,
) -> torch.Tensor:
    if positive_indices.numel() == 0 or max_pos <= 0:
        return positive_indices.new_zeros((0,))

    target = int(round(float(mask_ratio) * int(positive_indices.numel())))
    target = max(1, target)
    target = min(target, int(positive_indices.numel()), int(max_pos))
    perm = torch.randperm(int(positive_indices.numel()), device=positive_indices.device)
    return positive_indices[perm[:target]]


__all__ = [
    "SCHEMA_UNCONSTRAINED_RELATIONS",
    "confidence_sanitized_graph_data",
    "pretrain_sanitized_graph_data",
    "sanitized_graph_data",
]
