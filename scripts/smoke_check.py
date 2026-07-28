#!/usr/bin/env python3
"""Load all three model architectures and their packaged checkpoints."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from graph_jepa_v5.training import load_model_checkpoint as load_v5
from graph_jepa_v6.training import load_model_checkpoint as load_v6
from paper_v16 import trainer


ROOT = Path(__file__).resolve().parents[1]


def check_independence() -> None:
    forbidden = {"graph_jepa", "graph_jepa_v2", "graph_jepa_v3", "graph_jepa_v4"}
    for package in (ROOT / "src/graph_jepa_v5", ROOT / "src/graph_jepa_v6"):
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in forbidden:
                            raise RuntimeError(f"Historical import {alias.name} in {path}")
                if module and module.split(".")[0] in forbidden:
                    raise RuntimeError(f"Historical import {module} in {path}")


def main() -> None:
    check_independence()
    device = torch.device("cpu")
    v5, cfg5 = load_v5(
        str(ROOT / "models/v5_without_note/graph_jepa_v5.pt"), device
    )
    v6, cfg6 = load_v6(
        str(ROOT / "models/v6_with_note/graph_jepa_v6.pt"), device
    )

    checkpoint = torch.load(
        ROOT / "models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt",
        map_location=device,
        weights_only=False,
    )
    paper_encoder = trainer.Encoder().to(device)
    paper_scorer = trainer.DistMult().to(device)
    paper_encoder.load_state_dict(checkpoint["encoder"])
    paper_scorer.load_state_dict(checkpoint["scorer"])

    print(
        "OK: "
        f"v5={type(v5).__name__}(in={cfg5.model.in_dim}), "
        f"v6={type(v6).__name__}(in={cfg6.model.in_dim}), "
        f"paper={checkpoint['config']['model']}"
    )


if __name__ == "__main__":
    main()
