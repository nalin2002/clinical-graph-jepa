"""Score and revise clinical KG edges with Graph-JEPA.

By default this annotates existing edges only. ``--add-candidates`` appends
high-scoring missing edges, while ``--prune-threshold`` removes low-scoring
existing edges. Both actions use the same graph-revision edge score.

Merged from ``fawkes_core/score_base.py`` (v3), ``fawkes_core/score_revision.py``
(v4, the schema-guarded scorer both released variants actually used),
``graph_jepa_v5/score.py`` and ``graph_jepa_v6/score.py`` — 1,173 lines across
four modules. What changed beyond the move:

**``_install_v6_data_conversion`` is gone** (plan §2.4, §4.3). It rebound
``score_base.to_graph_data`` and ``score_revision.to_graph_data`` to v6's
converter at ``run()`` time, which made the conversion import-order dependent,
invisible to anyone reading ``fawkes_core``, and impossible to undo — so a
no-note and a localized-note checkpoint could not be scored in one process.
:func:`_graph_data` replaces it: the note branch is chosen per call from
``cfg.model``, which is where the variant already lives.

**The 15 module-level re-exports are gone.** ``score_revision.py`` aliased
``_v3._flag``, ``_v3._prune`` and thirteen more so the v4 CLI could reuse v3's
private helpers. In one module they are dead.

**One scorer, not two.** v3's unguarded ``score_graph``/``add_candidate_edges``
existed only for the v3 CLI. Both released checkpoints score through v4's
schema-guarded pair, so that is the pair kept. ``GraphJEPAv3``/``v4``/``v5``/
``v6`` are all :class:`~clinical_jepa.model.GraphJEPA` (plan §4.3).

**Endpoint aliases are normalized at the graph-loading boundary** (plan §2.6),
by the single normalizer in :mod:`clinical_jepa.graph.builders` — read that
module's docstring for the decision and its evidence. The old scoring path
re-derived the rule inside ``_looks_like_mimic_subkg`` and used it as a signal
to route alias-keyed graphs through the MIMIC adapter; that clause is gone and
:func:`_load_graph_for_scoring` normalizes instead. This changes the output for
alias-keyed inputs — a non-MIMIC graph is no longer stamped as a MIMIC one —
which is why it has its own test in ``tests/test_clinical_jepa_score.py``
(``test_alias_keyed_input_no_longer_routes_through_the_mimic_adapter``).

``jepa_source`` keeps its ``fawkes_core_revision_candidate_generation`` literal.
It is a provenance value written into output JSON, not a symbol — renaming it
would silently change every scored graph.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch

from .config import Config
from .encoders import build_checkpoint_encoder
from .graph.builders import (
    RELATION_SCHEMA,
    adapt_mimic_subkg,
    canonical_relation,
    normalize_graph_aliases,
    normalize_graph_edges,
)
from .graph.patches import build_patch_data
from .graph.tensors import to_graph_data
from .model import GraphJEPA
from .schema import EDGE_TYPE_TO_IDX, NODE_TYPE_TO_IDX, PatientGraph


RELATIONS = set(EDGE_TYPE_TO_IDX)

NEGATED_OR_ABSENT_MARKERS = (
    "absent ",
    "no ",
    "denies ",
    "denied ",
    "negative ",
    "without ",
    "unremarkable",
)


def _normalise_relation(relation: str) -> str:
    return relation.strip().upper()


def _update_relation_thresholds(target: dict[str, float], values: list[str] | None) -> None:
    if not values:
        return
    for value in values:
        if "=" not in value:
            raise ValueError(f"relation threshold must be RELATION=VALUE, got {value!r}")
        relation, raw_threshold = value.split("=", 1)
        relation = _normalise_relation(relation)
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation for threshold override: {relation!r}")
        target[relation] = float(raw_threshold)


def _update_disabled_relations(
    disabled: list[str],
    *,
    enable: list[str] | None,
    disable: list[str] | None,
) -> list[str]:
    disabled_set = {_normalise_relation(relation) for relation in disabled}
    for relation in enable or []:
        relation = _normalise_relation(relation)
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation to enable for candidates: {relation!r}")
        disabled_set.discard(relation)
    for relation in disable or []:
        relation = _normalise_relation(relation)
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation to disable for candidates: {relation!r}")
        disabled_set.add(relation)
    return sorted(disabled_set)


def _review_thresholds(relation: str | None, cfg: Config) -> tuple[float, float]:
    if relation is None:
        return cfg.score.inconsistent_threshold, cfg.score.weak_threshold
    relation = _normalise_relation(relation)
    inconsistent = cfg.score.inconsistent_threshold_by_relation.get(
        relation,
        cfg.score.inconsistent_threshold,
    )
    weak = cfg.score.weak_threshold_by_relation.get(
        relation,
        cfg.score.weak_threshold,
    )
    return inconsistent, weak


def _flag(score: float, cfg: Config, relation: str | None = None) -> str:
    inconsistent_threshold, weak_threshold = _review_thresholds(relation, cfg)
    if score < inconsistent_threshold:
        return "inconsistent"
    if score < weak_threshold:
        return "weak"
    return "ok"


def _candidate_threshold(relation: str, cfg: Config) -> float | None:
    relation = _normalise_relation(relation)
    if relation in {_normalise_relation(r) for r in cfg.score.disabled_candidate_relations}:
        return None
    return cfg.score.candidate_threshold_by_relation.get(
        relation,
        cfg.score.candidate_threshold,
    )


def _revision_action(
    score: float,
    cfg: Config,
    *,
    relation: str | None = None,
    candidate: bool = False,
) -> str:
    if candidate:
        return "add_candidate"
    if cfg.score.prune_threshold is not None and score < cfg.score.prune_threshold:
        return "prune_candidate"
    _inconsistent_threshold, weak_threshold = _review_thresholds(relation, cfg)
    if score < weak_threshold:
        return "review"
    return "keep"


def _node_res_ids(node: dict) -> set[str]:
    occurrences = node.get("occurrences") or []
    return {
        str(occ.get("res_id"))
        for occ in occurrences
        if occ.get("res_id")
    }


def _has_res_id_context(src: dict, tgt: dict) -> bool:
    return bool(_node_res_ids(src) or _node_res_ids(tgt))


def _shares_res_id(src: dict, tgt: dict) -> bool:
    src_res = _node_res_ids(src)
    tgt_res = _node_res_ids(tgt)
    if not src_res or not tgt_res:
        return False
    return bool(src_res & tgt_res)


def _is_negated_or_absent(node: dict) -> bool:
    text = " ".join(
        str(node.get(key, "")).lower()
        for key in ("text", "evidence")
    ).strip()
    if not text:
        return False
    return any(marker in text for marker in NEGATED_OR_ABSENT_MARKERS)


def _candidate_context_ok(src: dict, tgt: dict, cfg: Config) -> bool:
    if cfg.score.skip_negated_candidates and _is_negated_or_absent(src):
        return False
    if (
        cfg.score.require_shared_res_id
        and _has_res_id_context(src, tgt)
        and not _shares_res_id(src, tgt)
    ):
        return False
    return True


def _iter_candidate_specs(graph: PatientGraph, cfg: Config) -> List[Tuple[int, int, str]]:
    """Return missing schema-valid ``(src_idx, tgt_idx, relation)`` candidates."""
    existing = {
        (e.get("source_id"), e.get("target_id"), e.get("type"))
        for e in graph.edges
    }
    by_source_type: dict[str, List[Tuple[str, set[str]]]] = {}
    for (src_type, relation), target_types in RELATION_SCHEMA.items():
        by_source_type.setdefault(src_type, []).append((relation, target_types))

    candidates: List[Tuple[int, int, str]] = []
    for s_idx, src in enumerate(graph.nodes):
        src_id = src.get("id")
        if not src_id:
            continue
        for relation, target_types in by_source_type.get(src.get("type"), []):
            for t_idx, tgt in enumerate(graph.nodes):
                tgt_id = tgt.get("id")
                if s_idx == t_idx or not tgt_id or tgt.get("type") not in target_types:
                    continue
                if (src_id, tgt_id, relation) in existing:
                    continue
                if not _candidate_context_ok(src, tgt, cfg):
                    continue
                candidates.append((s_idx, t_idx, relation))
    return candidates


def _append_candidate_edges(
    graph: PatientGraph,
    scored: List[Tuple[float, int, int, str]],
    cfg: Config,
) -> int:
    for rank, (score, s_idx, t_idx, relation) in enumerate(scored, start=1):
        graph.edges.append(
            {
                "source_id": graph.nodes[s_idx]["id"],
                "target_id": graph.nodes[t_idx]["id"],
                "type": relation,
                "evidence": "",
                "turn_id": "",
                "jepa_score": round(float(score), 6),
                "jepa_flag": _flag(score, cfg, relation),
                "jepa_revision_action": _revision_action(
                    score,
                    cfg,
                    relation=relation,
                    candidate=True,
                ),
                "jepa_suggested": True,
                "jepa_unverified": True,
                "jepa_candidate_rank": rank,
                "jepa_source": "fawkes_core_revision_candidate_generation",
                "jepa_schema_valid": True,
            }
        )
    return len(scored)


def _edge_schema_error(graph: PatientGraph, edge: dict, cfg: Config) -> str | None:
    id_to_idx = graph.id_to_index()
    s_idx = id_to_idx.get(edge.get("source_id"))
    t_idx = id_to_idx.get(edge.get("target_id"))
    if s_idx is None:
        return f"missing source node {edge.get('source_id')!r}"
    if t_idx is None:
        return f"missing target node {edge.get('target_id')!r}"

    relation = canonical_relation(edge.get("type"))
    unconstrained = {_normalise_relation(r) for r in cfg.score.schema_unconstrained_relations}
    if relation and _normalise_relation(relation) in unconstrained:
        return None

    src_type = graph.nodes[s_idx].get("type")
    tgt_type = graph.nodes[t_idx].get("type")
    allowed_targets = RELATION_SCHEMA.get((src_type, relation))
    if not allowed_targets:
        return f"no schema rule for {src_type} --{relation}-->"
    if tgt_type not in allowed_targets:
        allowed = ", ".join(sorted(allowed_targets))
        return f"{src_type} --{relation}--> {tgt_type} violates schema target {{{allowed}}}"
    return None


def _apply_schema_guard(
    graph: PatientGraph,
    scores: List[float],
    flags: List[str],
    cfg: Config,
) -> None:
    for idx, edge in enumerate(graph.edges):
        error = _edge_schema_error(graph, edge, cfg)
        if error is None:
            edge["jepa_schema_valid"] = True
            edge.pop("jepa_schema_error", None)
            continue

        edge["jepa_schema_valid"] = False
        edge["jepa_schema_error"] = error
        scores[idx] = min(scores[idx], cfg.score.schema_invalid_score)
        flags[idx] = "inconsistent"


def _graph_data(graph: PatientGraph, encoder, cfg: Config):
    """Convert ``graph`` to tensors for the variant this checkpoint describes.

    This is what replaces ``_install_v6_data_conversion``. The note branch is a
    per-call function of ``cfg.model`` — where the variant already lives — not a
    rebound module global, so a no-note and a localized-note config can be
    scored in the same process without one contaminating the other.
    """
    return to_graph_data(
        graph,
        encoder,
        use_note_embeddings=cfg.model.use_note_embeddings,
        note_embedding_dim=cfg.model.note_embedding_dim,
        note_ground_by=cfg.model.note_ground_by,
    )


def _edge_scorer(
    graph: PatientGraph,
    model: GraphJEPA,
    encoder,
    cfg: Config,
    device: torch.device,
):
    """Return ``score(src_idx, tgt_idx, rel_idx) -> float``, or ``None``.

    ``None`` means the graph produced no patches, so nothing is scoreable.
    Existing-edge scoring and candidate scoring share this so the two cannot
    drift apart; the old tree carried four copies of the body.
    """
    data = _graph_data(graph, encoder, cfg).to(device)
    patch_data = build_patch_data(
        data,
        num_patches=cfg.model.num_patches,
        patch_pe_dim=cfg.model.patch_pe_dim,
        generator=None,
    ).to(device)
    if patch_data.num_patches == 0:
        return None

    z_nodes = model.encode_nodes(data)
    # Cache patch energies because many edges share the same target patch.
    patch_energy: dict[int, float] = {}

    def score(s_idx: int, t_idx: int, rel: int) -> float:
        rel_t = torch.tensor([rel], dtype=torch.long, device=device)
        logit = model.edge_head(z_nodes[s_idx:s_idx + 1], z_nodes[t_idx:t_idx + 1], rel_t)
        p_head = torch.sigmoid(logit).item()

        target_patch = int(patch_data.assignment[t_idx].item())
        if target_patch not in patch_energy:
            idx = torch.tensor([target_patch], dtype=torch.long, device=device)
            patch_energy[target_patch] = float(
                model.patch_prediction_energy(data, patch_data, idx)[0].item()
            )
        structural = math.exp(-patch_energy[target_patch] / cfg.score.energy_temperature)
        return cfg.score.alpha * p_head + (1.0 - cfg.score.alpha) * structural

    return score


@torch.no_grad()
def score_graph(
    graph: PatientGraph,
    model: GraphJEPA,
    encoder,
    cfg: Config,
    device: torch.device,
) -> Tuple[List[float], List[str]]:
    scores = [0.0] * len(graph.edges)
    flags = ["inconsistent"] * len(graph.edges)

    # An unscoreable graph still gets the schema guard: v4 wrapped v3's scorer
    # and ran the guard on whatever came back, including v3's two early returns.
    score = _edge_scorer(graph, model, encoder, cfg, device) if graph.nodes else None
    if score is not None:
        id_to_idx = graph.id_to_index()
        for ei, e in enumerate(graph.edges):
            s = id_to_idx.get(e["source_id"])
            t = id_to_idx.get(e["target_id"])
            rel = EDGE_TYPE_TO_IDX.get(e["type"])
            if s is None or t is None or rel is None:
                continue
            scores[ei] = score(s, t, rel)
            flags[ei] = _flag(scores[ei], cfg, e["type"])

    _apply_schema_guard(graph, scores, flags, cfg)
    return scores, flags


@torch.no_grad()
def add_candidate_edges(
    graph: PatientGraph,
    model: GraphJEPA,
    encoder,
    cfg: Config,
    device: torch.device,
) -> int:
    """Add high-scoring missing edges between existing nodes.

    Candidates are schema-valid triples that are absent from the input graph.
    Added edges are marked ``jepa_suggested`` / ``jepa_unverified`` because this
    scorer does not produce transcript evidence.
    """
    if not graph.nodes or cfg.score.max_candidate_edges <= 0:
        return 0

    candidates = _iter_candidate_specs(graph, cfg)
    if not candidates:
        return 0

    score = _edge_scorer(graph, model, encoder, cfg, device)
    if score is None:
        return 0

    scored: List[Tuple[float, int, int, str]] = []
    for s_idx, t_idx, relation in candidates:
        rel = EDGE_TYPE_TO_IDX.get(relation)
        if rel is None:
            continue
        threshold = _candidate_threshold(relation, cfg)
        if threshold is None:
            continue

        value = score(s_idx, t_idx, rel)
        if value >= threshold:
            scored.append((value, s_idx, t_idx, relation))

    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[:cfg.score.max_candidate_edges]
    return _append_candidate_edges(graph, scored, cfg)


def _iter_inputs(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))
    return [input_path]


def _schema_compatibility_errors(graph: PatientGraph) -> list[str]:
    unknown_node_types = sorted(
        {
            str(node.get("type"))
            for node in graph.nodes
            if node.get("type") not in NODE_TYPE_TO_IDX
        }
    )
    errors: list[str] = []
    if unknown_node_types:
        errors.append(f"unknown node types: {', '.join(unknown_node_types)}")

    unknown_edge_types = sorted(
        {
            str(edge.get("type") or edge.get("relation"))
            for edge in graph.edges
            if (edge.get("type") or edge.get("relation")) not in EDGE_TYPE_TO_IDX
        }
    )
    if unknown_edge_types:
        errors.append(f"unknown edge types: {', '.join(unknown_edge_types)}")

    malformed_edges = sum(
        1
        for edge in graph.edges
        if not {"source_id", "target_id", "type"} <= edge.keys()
    )
    if malformed_edges:
        errors.append(f"{malformed_edges} edges missing source_id/target_id/type")
    return errors


def _looks_like_mimic_subkg(data: dict) -> bool:
    """True if ``data`` needs the MIMIC sub-KG adapter to be scoreable.

    The old version had a third clause::

        any("source" in edge or "target" in edge or "relation" in edge ...)

    Only the first two terms were the scoring path's private copy of the plan
    §2.6 endpoint-alias rule, and those are now handled by the one normalizer in
    :func:`_load_graph_for_scoring`. **The third term was not an alias rule at
    all** — nearly every clinical-graph edge carries a ``relation`` key, so the
    clause resolved to roughly "does this graph have any edges?" and sent almost
    everything to the adapter.

    Dropping the whole clause therefore narrows routing for *every* non-MIMIC
    graph, not only alias-keyed ones. A graph with canonical ``source_id``/
    ``target_id`` endpoints and a ``relation`` key used to be adapted and no
    longer is. That is wider than §2.6 strictly required; it is kept because a
    predicate matching essentially every input was not doing format detection,
    and routing arbitrary graphs through a MIMIC adapter was the actual defect.

    Measured impact (see ``tests/test_clinical_jepa_score.py``): **scores do not
    change.** Encoder-visible ``(type, text)``, edge endpoints, ``relation`` and
    ``type`` are all identical either way, because
    :func:`~clinical_jepa.graph.builders.normalize_graph_aliases` applies the
    same ``text <- name <- normalized_name <- id`` fallback the adapter did. What
    is lost is provenance metadata: ``mimic_type`` on nodes, ``mimic_relation``
    on edges, and ``_method`` / ``_source_path`` / ``_mimic_adapter`` in
    ``extra``.
    """
    if "subject_id" in data or "hadm_ids" in data:
        return True
    mimic_only_node_types = {
        "PATIENT",
        "MEDICATION",
        "LAB_TEST",
        "MICROBIOLOGY",
        "SERVICE",
    }
    return any(
        str(node.get("type", "")).upper() in mimic_only_node_types
        for node in data.get("nodes", [])
    )


def _load_graph_for_scoring(path: Path) -> PatientGraph:
    with open(path, "r") as f:
        data = json.load(f)

    # Plan §2.6: endpoint aliases are normalized here, at the graph-loading
    # boundary, by THE normalizer in graph/builders.py -- see that module's
    # docstring for the decision and the evidence behind it.
    graph = normalize_graph_aliases(PatientGraph.from_pipeline_json(data))
    errors = _schema_compatibility_errors(graph)
    if not errors:
        return normalize_graph_edges(graph)

    if _looks_like_mimic_subkg(data):
        graph = adapt_mimic_subkg(data, source_path=path)
        errors = _schema_compatibility_errors(graph)
        if not errors:
            return normalize_graph_edges(graph)

    raise ValueError(
        "KG is not compatible with the JEPA scoring schema "
        f"({'; '.join(errors)})"
    )


def _validate_checkpoint_relation_capacity(graph: PatientGraph, cfg: Config) -> None:
    required = 0
    for edge in graph.edges:
        rel_idx = EDGE_TYPE_TO_IDX.get(edge.get("type"))
        if rel_idx is not None:
            required = max(required, rel_idx + 1)
    if required > cfg.model.num_relations:
        raise ValueError(
            "checkpoint relation embedding is too small for this graph "
            f"(checkpoint num_relations={cfg.model.num_relations}, "
            f"required>={required}). Retrain Graph-JEPA with the expanded "
            "MIMIC schema before scoring raw MIMIC relations."
        )


def _prune(graph: PatientGraph, threshold: float) -> int:
    before = len(graph.edges)
    graph.edges = [e for e in graph.edges if e.get("jepa_score", 1.0) >= threshold]
    return before - len(graph.edges)


def _annotate_revision_actions(graph: PatientGraph, cfg: Config) -> None:
    for edge in graph.edges:
        if edge.get("jepa_suggested"):
            edge["jepa_revision_action"] = _revision_action(
                float(edge.get("jepa_score", 0.0)),
                cfg,
                relation=edge.get("type"),
                candidate=True,
            )
        else:
            edge["jepa_revision_action"] = _revision_action(
                float(edge.get("jepa_score", 0.0)),
                cfg,
                relation=edge.get("type"),
            )


def _load(checkpoint: str, device: torch.device, encoder_cache: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GraphJEPA(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    encoder = build_checkpoint_encoder(cfg, encoder_cache)
    return model, encoder, cfg


def run(args) -> None:
    device = torch.device(args.device)
    model, encoder, cfg = _load(args.checkpoint, device, args.encoder_cache)
    if args.prune_threshold is not None:
        cfg.score.prune_threshold = args.prune_threshold
    if args.candidate_threshold is not None:
        cfg.score.candidate_threshold = args.candidate_threshold
        cfg.score.candidate_threshold_by_relation = {
            relation: args.candidate_threshold
            for relation in RELATIONS
        }
    _update_relation_thresholds(
        cfg.score.weak_threshold_by_relation,
        args.review_weak_threshold,
    )
    _update_relation_thresholds(
        cfg.score.inconsistent_threshold_by_relation,
        args.review_inconsistent_threshold,
    )
    _update_relation_thresholds(
        cfg.score.candidate_threshold_by_relation,
        args.candidate_threshold_relation,
    )
    cfg.score.disabled_candidate_relations = _update_disabled_relations(
        cfg.score.disabled_candidate_relations,
        enable=args.enable_candidate_relation,
        disable=args.disable_candidate_relation,
    )
    if args.max_candidates is not None:
        cfg.score.max_candidate_edges = args.max_candidates
    if args.allow_cross_res_candidates:
        cfg.score.require_shared_res_id = False
    if args.allow_negated_candidates:
        cfg.score.skip_negated_candidates = False

    input_path = Path(args.input)
    output_path = Path(args.output)
    inputs = _iter_inputs(input_path)
    output_is_dir = input_path.is_dir()
    if output_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)

    for jf in inputs:
        try:
            graph = _load_graph_for_scoring(jf)
            _validate_checkpoint_relation_capacity(graph, cfg)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            if output_is_dir:
                print(f"{jf.name}: skipped (not a KG: {exc})")
                continue
            raise

        scores, flags = score_graph(graph, model, encoder, cfg, device)
        graph.annotate_edges(scores, flags)
        _annotate_revision_actions(graph, cfg)

        added = 0
        if args.add_candidates:
            added = add_candidate_edges(graph, model, encoder, cfg, device)
            _annotate_revision_actions(graph, cfg)

        pruned = 0
        if cfg.score.prune_threshold is not None:
            pruned = _prune(graph, cfg.score.prune_threshold)

        dest = output_path / jf.name if output_is_dir else output_path
        graph.save(dest)
        flagged = sum(1 for f in flags if f != "ok")
        print(
            f"{jf.name}: {len(graph.nodes)} nodes, {len(scores)} edges scored, "
            f"{flagged} flagged"
            + (f", {added} candidates added" if added else "")
            + (f", {pruned} pruned" if pruned else "")
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Revise KG edges with Graph-JEPA scores")
    p.add_argument("--input", required=True, help="KG JSON file or directory")
    # Phase 7 renames these directories to models/clinical-jepa-no-note/ and
    # models/clinical-jepa-localized-note/; the checkpoint filenames do not
    # change (plan §5.1). There is no default because the variant is a choice.
    p.add_argument(
        "--checkpoint",
        required=True,
        help="trained checkpoint .pt, e.g. models/v5_without_note/graph_jepa_v5.pt "
             "or models/v6_with_note/graph_jepa_v6.pt",
    )
    p.add_argument("--output", required=True, help="output file or directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--encoder-cache", default=".cache/clinical_jepa/encoder")
    p.add_argument("--prune-threshold", type=float, default=None)
    p.add_argument(
        "--add-candidates",
        action="store_true",
        help="add high-scoring missing schema-valid edges between existing nodes",
    )
    p.add_argument(
        "--candidate-threshold",
        type=float,
        default=None,
        help="global minimum jepa_score for enabled candidate relations",
    )
    p.add_argument(
        "--candidate-threshold-relation",
        action="append",
        default=None,
        metavar="RELATION=VALUE",
        help="override candidate-add threshold for one relation; repeatable",
    )
    p.add_argument(
        "--review-weak-threshold",
        action="append",
        default=None,
        metavar="RELATION=VALUE",
        help="override existing-edge weak/review threshold for one relation; repeatable",
    )
    p.add_argument(
        "--review-inconsistent-threshold",
        action="append",
        default=None,
        metavar="RELATION=VALUE",
        help="override existing-edge inconsistent threshold for one relation; repeatable",
    )
    p.add_argument(
        "--enable-candidate-relation",
        action="append",
        default=None,
        metavar="RELATION",
        help="enable candidate additions for a relation disabled by the score config",
    )
    p.add_argument(
        "--disable-candidate-relation",
        action="append",
        default=None,
        metavar="RELATION",
        help="disable candidate additions for a relation; repeatable",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="maximum number of candidate edges to add per graph",
    )
    p.add_argument(
        "--allow-cross-res-candidates",
        action="store_true",
        help="allow unified-graph candidates whose endpoints never share a res_id",
    )
    p.add_argument(
        "--allow-negated-candidates",
        action="store_true",
        help="allow absent/negated findings as sources for added candidates",
    )
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
