#!/usr/bin/env python3
"""Reproduce the v24 three-condition ACI-Bench transfer evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fawkes.config import Config
from fawkes.data import to_data
from fawkes.evaluate import loo_evaluate
from fawkes.model import Encoder, build_scorer


def load_raw(path: Path):
    raw = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    demographics = {str(g.get("subject_id")): {
        "gender": g.get("gender"), "age": g.get("anchor_age")
    } for g in raw}
    return raw, demographics


def run_condition(raw, demographics, checkpoint, label, ground_by, zero_note, device, cap):
    cfg = Config.from_checkpoint(checkpoint["config"])
    # v24's checkpoint config predates the decoder field, but its scorer state
    # dict is the released MLP readout (the v22/v23 protocol).
    cfg.decoder = "mlp"
    cfg.ground_by = ground_by
    # Keep the 774-d checkpoint architecture in all arms. Graph-only is an
    # inference ablation: the 768-d channel is zeroed, not retrained.
    condition_raw = copy.deepcopy(raw)
    if zero_note:
        for graph in condition_raw:
            graph["note_embedding"] = [0.0] * cfg.embed_dim
    graphs = [to_data(g, demographics, cfg) for g in condition_raw]
    graphs = [g for g in graphs if g.num_nodes >= 3 and g.edge_index.size(1) >= 2]
    encoder = Encoder(cfg).to(device)
    scorer = build_scorer(cfg).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])
    return label, loo_evaluate(encoder, scorer, graphs, device, cfg, cap=cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw, demographics = load_raw(args.data)
    device = torch.device(args.device)
    conditions = [
        ("Graph-only (no note)", "prov", True),
        ("Global note", "all", False),
        ("Entity-grounded note", "prov", False),
    ]
    result = {"protocol": "v25-aci-reproduction", "graphs": len(raw), "conditions": {}}
    for label, ground_by, zero_note in conditions:
        name, metrics = run_condition(raw, demographics, checkpoint, label,
                                       ground_by, zero_note, device, args.cap)
        result["conditions"][name] = metrics
        print(f"{name}: n={metrics['n']} MRR={metrics['mrr']:.6f} "
              f"H@1={metrics['hits1']:.6f} H@3={metrics['hits3']:.6f} "
              f"H@10={metrics['hits10']:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
