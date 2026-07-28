"""Evaluate a saved paper-v16 checkpoint on a local clinical-graph JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import trainer


def _load_graphs(path: Path, limit: int | None) -> tuple[list, dict]:
    raw = []
    demographics = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                graph = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            raw.append(graph)
            subject = str(graph.get("subject_id"))
            demographics[subject] = {
                "gender": graph.get("gender"),
                "age": graph.get("anchor_age"),
            }
            if limit is not None and len(raw) >= limit:
                break
    if not raw:
        raise ValueError(f"No graphs found in {path}")
    return raw, demographics


def run(args) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    expected = {
        "use_note": trainer.USE_NOTE,
        "ground_by": trainer.GROUND_BY,
        "embed_dim": trainer.EMBED_DIM,
        "use_scores": trainer.USE_SCORES,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Environment does not match checkpoint configuration: "
            f"{mismatches}. Set USE_NOTE/GROUND_BY/EMBED_DIM/USE_SCORES before launch."
        )

    raw, demographics = _load_graphs(Path(args.data), args.max_graphs)
    graphs = []
    for graph in raw:
        data = trainer.to_data(graph, demographics)
        if data.num_nodes >= 3 and data.edge_index.size(1) >= 2:
            graphs.append(data)
    if not graphs:
        raise ValueError("No evaluable graphs remain after conversion")

    device = torch.device(args.device)
    encoder = trainer.Encoder().to(device)
    scorer = trainer.DistMult().to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])
    metrics = trainer.loo_evaluate(
        encoder,
        scorer,
        graphs,
        device,
        cap=args.cap,
    )
    print(
        f"[LOO] graphs={len(graphs)} MRR={metrics['mrr']:.3f} "
        f"H@1={metrics['hits1']:.3f} H@10={metrics['hits10']:.3f} "
        f"over {metrics['n']} edges"
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Clinical graph JSONL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cap", type=int, default=40000)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
