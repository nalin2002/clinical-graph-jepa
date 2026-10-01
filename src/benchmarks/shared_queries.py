"""Score Fawkes and ``clinical_jepa`` on one fixed Fawkes query manifest.

The existing ``benchmark-vs-fawkes`` command aligns the two lineages by
admission, but each arm still creates its own leave-one-out query population.
This module makes the comparison stricter: Fawkes' paper test split and
filtered-ranking candidate lists define the query manifest, then both models
rank exactly those targets against exactly those candidates.

The context graph is still lineage-native. Fawkes encodes its Fawkes tensor and
``clinical_jepa`` encodes its normalized PatientGraph tensor, but the hidden
edge, candidate IDs, and metric denominator are shared.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from clinical_jepa.encoders import build_checkpoint_encoder
from clinical_jepa.evaluate import _trusted_edge_masks
from clinical_jepa.graph.builders import adapt_mimic_subkg
from clinical_jepa.graph.tensors import PatientGraphDataset
from clinical_jepa.schema import EDGE_TYPE_TO_IDX
from clinical_jepa.train.loop import load_model_checkpoint
from fawkes.config import Config
from fawkes.data import RELATION_CANONICAL
from fawkes.evaluate import (
    _load_graphs,
    encode_context_graph,
    filtered_candidates,
    rank_true_tail,
)
from fawkes.model import Encoder, build_scorer

from .vs_fawkes import DEFAULT_CHECKPOINT, DEFAULT_DATA, DEFAULT_FAWKES_CHECKPOINT
from .vs_fawkes import paper_test_split


RELATION_BY_FAWKES_ID = {value: key for key, value in RELATION_CANONICAL.items()}


@dataclass(frozen=True)
class SharedQuery:
    """One Fawkes-defined LOO query, with raw node IDs as the interchange key."""

    query_id: int
    record_index: int
    split_position: int
    fawkes_edge_index: int
    source_id: str
    target_id: str
    relation: str
    candidate_ids: list[str]


class RankStats:
    """Aggregate ranks into the metric schema used by the existing evaluators."""

    def __init__(self) -> None:
        self.rr: dict[str, list[float]] = defaultdict(list)
        self.hits: dict[str, dict[int, int]] = defaultdict(
            lambda: {1: 0, 3: 0, 10: 0}
        )
        self.candidates: dict[str, list[int]] = defaultdict(list)

    @property
    def n(self) -> int:
        return sum(len(values) for values in self.rr.values())

    def add(self, relation: str, rank: int, candidate_count: int) -> None:
        self.rr[relation].append(1.0 / rank)
        self.candidates[relation].append(candidate_count)
        for k in (1, 3, 10):
            if rank <= k:
                self.hits[relation][k] += 1

    @staticmethod
    def _chance_mrr(candidate_count: float) -> float:
        if candidate_count <= 1:
            return 1.0
        return (math.log(candidate_count) + 0.5772) / candidate_count

    def as_dict(self) -> dict:
        total_n = self.n
        all_rr = [rr for values in self.rr.values() for rr in values]
        per_rel = []
        for relation in sorted(self.rr, key=lambda rel: -len(self.rr[rel])):
            n = len(self.rr[relation])
            mean_candidates = float(np.mean(self.candidates[relation]))
            per_rel.append(
                {
                    "rel": relation,
                    "n": n,
                    "mrr": float(np.mean(self.rr[relation])),
                    "h1": self.hits[relation][1] / n,
                    "h3": self.hits[relation][3] / n,
                    "h10": self.hits[relation][10] / n,
                    "C": mean_candidates,
                    "chance_mrr": self._chance_mrr(mean_candidates),
                    "chance_h1": (
                        1.0 / mean_candidates if mean_candidates >= 1 else 1.0
                    ),
                }
            )
        return {
            "mrr": float(np.mean(all_rr)) if all_rr else float("nan"),
            "hits1": (
                sum(h[1] for h in self.hits.values()) / total_n
                if total_n
                else 0.0
            ),
            "hits3": (
                sum(h[3] for h in self.hits.values()) / total_n
                if total_n
                else 0.0
            ),
            "hits10": (
                sum(h[10] for h in self.hits.values()) / total_n
                if total_n
                else 0.0
            ),
            "n": total_n,
            "per_rel": per_rel,
        }


def _node_ids(raw_graph: dict) -> list[str]:
    return [str(node["id"]) for node in raw_graph.get("nodes", [])]


def build_query_manifest(
    raw: list[dict],
    records: list[int],
    fawkes_graphs: list,
    *,
    cap: int,
) -> list[SharedQuery]:
    """Create the Fawkes paper LOO query list with raw candidate IDs."""

    queries: list[SharedQuery] = []
    for split_position, (record_index, graph) in enumerate(
        zip(records, fawkes_graphs)
    ):
        if len(queries) >= cap:
            break
        node_ids = _node_ids(raw[record_index])
        edge_index = graph.edge_index
        edge_type = graph.edge_type
        node_type = graph.node_type
        if int(edge_index.size(1)) < 2:
            continue

        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        for edge_idx in range(int(edge_index.size(1))):
            if len(queries) >= cap:
                break
            source = int(src_nodes[edge_idx])
            target = int(dst_nodes[edge_idx])
            relation_id = int(edge_type[edge_idx])
            if source == target:
                continue
            candidates = filtered_candidates(
                node_type,
                source,
                target,
                relation_id,
                src_nodes,
                dst_nodes,
                edge_type,
            )
            if candidates.numel() < 2 or int((candidates == target).sum()) == 0:
                continue
            relation = RELATION_BY_FAWKES_ID.get(relation_id, f"rel{relation_id}")
            queries.append(
                SharedQuery(
                    query_id=len(queries),
                    record_index=record_index,
                    split_position=split_position,
                    fawkes_edge_index=edge_idx,
                    source_id=node_ids[source],
                    target_id=node_ids[target],
                    relation=relation,
                    candidate_ids=[node_ids[int(idx)] for idx in candidates.tolist()],
                )
            )
    return queries


def _rank_from_scores(scores: torch.Tensor, target_position: int) -> int:
    true_score = scores[target_position]
    return int((scores > true_score).sum().item()) + 1


@torch.no_grad()
def score_fawkes_queries_with_raw(
    encoder,
    scorer,
    raw: list[dict],
    fawkes_graphs: list,
    queries: list[SharedQuery],
    device: torch.device,
    cfg: Config,
) -> tuple[dict, Counter]:
    """Score the manifest with Fawkes, using raw node IDs for lookup."""

    encoder.eval()
    scorer.eval()
    graphs = [graph.to(device) for graph in fawkes_graphs]
    id_maps = {
        query.record_index: {
            node_id: idx
            for idx, node_id in enumerate(_node_ids(raw[query.record_index]))
        }
        for query in queries
    }
    stats = RankStats()
    skipped: Counter = Counter()
    for query in queries:
        graph = graphs[query.split_position]
        edge_count = int(graph.edge_index.size(1))
        id_to_idx = id_maps[query.record_index]
        source = id_to_idx.get(query.source_id)
        target = id_to_idx.get(query.target_id)
        relation = RELATION_CANONICAL.get(query.relation)
        candidates = [id_to_idx.get(node_id) for node_id in query.candidate_ids]
        if query.fawkes_edge_index >= edge_count:
            skipped["skipped_missing_edge_index"] += 1
            continue
        if source is None or target is None or relation is None:
            skipped["skipped_missing_query_endpoint"] += 1
            continue
        if any(candidate is None for candidate in candidates):
            skipped["skipped_missing_candidate"] += 1
            continue

        keep_mask = torch.ones(edge_count, dtype=torch.bool, device=device)
        keep_mask[query.fawkes_edge_index] = False
        hidden = encode_context_graph(
            encoder,
            graph,
            graph.edge_index[:, keep_mask],
            graph.edge_type[keep_mask],
            graph.edge_feat[keep_mask] if cfg.use_scores else None,
            cfg,
        )
        candidate_t = torch.tensor(candidates, dtype=torch.long, device=device)
        rank = rank_true_tail(
            scorer,
            hidden,
            source,
            relation,
            candidate_t,
            target,
            device,
        )
        stats.add(query.relation, rank, len(query.candidate_ids))
    return stats.as_dict(), skipped


@torch.no_grad()
def score_clinical_jepa_queries(
    model,
    cfg,
    data_list: list,
    graphs: list,
    queries: list[SharedQuery],
    device: torch.device,
) -> tuple[dict, Counter]:
    """Score Fawkes-defined queries with the ``clinical_jepa`` edge head."""

    model.eval()
    stats = RankStats()
    skipped: Counter = Counter()
    id_maps = [
        {str(node.get("id")): idx for idx, node in enumerate(graph.nodes)}
        for graph in graphs
    ]
    data_by_position = [data.to(device) for data in data_list]
    base_masks = []
    for data in data_by_position:
        _positive_mask, message_mask = _trusted_edge_masks(data, cfg)
        base_masks.append(message_mask)

    for query in queries:
        data = data_by_position[query.split_position]
        id_to_idx = id_maps[query.split_position]
        source = id_to_idx.get(query.source_id)
        target = id_to_idx.get(query.target_id)
        relation = EDGE_TYPE_TO_IDX.get(query.relation)
        candidates = [id_to_idx.get(node_id) for node_id in query.candidate_ids]
        if source is None or target is None:
            skipped["skipped_missing_query_endpoint"] += 1
            continue
        if relation is None or relation >= cfg.model.num_relations:
            skipped["skipped_unsupported_relation"] += 1
            continue
        if any(candidate is None for candidate in candidates):
            skipped["skipped_missing_candidate"] += 1
            continue

        message_mask = base_masks[query.split_position].clone()
        exact_edge = (
            (data.edge_index[0] == source)
            & (data.edge_index[1] == target)
            & (data.edge_type == relation)
        )
        if bool(exact_edge.any()):
            message_mask[exact_edge] = False
        else:
            skipped["context_edge_not_present"] += 1

        z_nodes = model.context_node_encoder(
            data.x,
            data.edge_index[:, message_mask],
            data.edge_type[message_mask],
        )
        candidate_t = torch.tensor(candidates, dtype=torch.long, device=device)
        source_t = torch.full(
            (len(candidates),),
            source,
            dtype=torch.long,
            device=device,
        )
        relation_t = torch.full(
            (len(candidates),),
            relation,
            dtype=torch.long,
            device=device,
        )
        logits = model.edge_head(z_nodes[source_t], z_nodes[candidate_t], relation_t)
        target_position = query.candidate_ids.index(query.target_id)
        rank = _rank_from_scores(logits, target_position)
        stats.add(query.relation, rank, len(query.candidate_ids))
    return stats.as_dict(), skipped


def _load_fawkes_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_checkpoint(checkpoint["config"])
    encoder = Encoder(cfg).to(device)
    scorer = build_scorer(cfg).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])
    return encoder, scorer, cfg


def _clinical_pipeline(raw: list[dict], records: list[int], encoder, cfg, source: str):
    graphs = [
        adapt_mimic_subkg(raw[index], source_path=f"{source}:{index + 1}")
        for index in records
    ]
    dataset = PatientGraphDataset(
        graphs,
        encoder,
        use_note_embeddings=cfg.model.use_note_embeddings,
        note_embedding_dim=cfg.model.note_embedding_dim,
        note_ground_by=cfg.model.note_ground_by,
    )
    return graphs, [dataset[idx] for idx in range(len(dataset))]


def _fatal_skips(counters: Counter) -> int:
    return sum(value for key, value in counters.items() if key.startswith("skipped_"))


def _print_summary(label: str, metrics: dict, counters: Counter) -> None:
    print(
        f"[{label}] MRR={metrics['mrr']:.6f} H@1={metrics['hits1']:.6f} "
        f"H@3={metrics['hits3']:.6f} H@10={metrics['hits10']:.6f} "
        f"n={metrics['n']} skipped={_fatal_skips(counters)}"
    )
    if counters:
        print(f"[{label}] diagnostics={dict(counters)}")


def _write_manifest(path: str | None, queries: list[SharedQuery]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for query in queries:
            stream.write(json.dumps(asdict(query), sort_keys=True) + "\n")


def run(args) -> dict:
    device = torch.device(args.device)
    raw, demographics = _load_graphs(Path(args.data), None)
    fawkes_encoder, fawkes_scorer, fawkes_cfg = _load_fawkes_checkpoint(
        args.fawkes_checkpoint,
        device,
    )
    fawkes_graphs, records = paper_test_split(raw, demographics, fawkes_cfg)
    queries = build_query_manifest(raw, records, fawkes_graphs, cap=args.cap)
    _write_manifest(args.manifest_output, queries)

    clinical_model, clinical_cfg = load_model_checkpoint(args.checkpoint, device)
    clinical_encoder = build_checkpoint_encoder(clinical_cfg, args.encoder_cache)
    clinical_graphs, clinical_data = _clinical_pipeline(
        raw,
        records,
        clinical_encoder,
        clinical_cfg,
        args.data,
    )

    fawkes_metrics, fawkes_skipped = score_fawkes_queries_with_raw(
        fawkes_encoder,
        fawkes_scorer,
        raw,
        fawkes_graphs,
        queries,
        device,
        fawkes_cfg,
    )
    clinical_metrics, clinical_skipped = score_clinical_jepa_queries(
        clinical_model,
        clinical_cfg,
        clinical_data,
        clinical_graphs,
        queries,
        device,
    )

    print(
        f"[SHARED] records={len(raw)} test_split={len(records)} "
        f"queries={len(queries)} split_seed={fawkes_cfg.seed} "
        f"test_frac={fawkes_cfg.test_frac}"
    )
    _print_summary("fawkes", fawkes_metrics, fawkes_skipped)
    _print_summary("clinical_jepa", clinical_metrics, clinical_skipped)

    payload = {
        "config": {
            "data": args.data,
            "fawkes_checkpoint": args.fawkes_checkpoint,
            "checkpoint": args.checkpoint,
            "cap": args.cap,
            "device": args.device,
        },
        "population": {
            "records": len(raw),
            "test_split_graphs": len(records),
            "split_seed": fawkes_cfg.seed,
            "test_frac": fawkes_cfg.test_frac,
            "shared_queries": len(queries),
        },
        "coverage": {
            "fawkes_scored": fawkes_metrics["n"],
            "fawkes_fatal_skips": _fatal_skips(fawkes_skipped),
            "fawkes_skipped": dict(fawkes_skipped),
            "clinical_jepa_scored": clinical_metrics["n"],
            "clinical_jepa_fatal_skips": _fatal_skips(clinical_skipped),
            "clinical_jepa_skipped": dict(clinical_skipped),
        },
        "fawkes": fawkes_metrics,
        "clinical_jepa": clinical_metrics,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score fawkes and clinical_jepa on the same Fawkes-defined "
            "leave-one-out query manifest."
        )
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--fawkes-checkpoint", default=DEFAULT_FAWKES_CHECKPOINT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder-cache", default=".cache/clinical_jepa/encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cap", type=int, default=40000)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional JSONL file containing the shared query manifest.",
    )
    return parser


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
