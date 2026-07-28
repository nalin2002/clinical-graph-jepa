from __future__ import annotations

import ast
import json
from pathlib import Path

import torch

from fawkes_core.data import JsonlGraphBuilder
from graph_jepa_v5.training import load_model_checkpoint as load_v5
from graph_jepa_v6.training import load_model_checkpoint as load_v6
from paper_v16 import trainer


ROOT = Path(__file__).resolve().parents[1]


def test_v5_v6_have_no_historical_package_imports():
    forbidden = {"graph_jepa", "graph_jepa_v2", "graph_jepa_v3", "graph_jepa_v4"}
    for package in (ROOT / "src/graph_jepa_v5", ROOT / "src/graph_jepa_v6"):
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
                elif isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
            assert not {name.split(".")[0] for name in imported} & forbidden


def test_jsonl_builder_preserves_graph_metadata(tmp_path):
    path = tmp_path / "graphs.jsonl"
    record = {
        "subject_id": 7,
        "note_embedding": [0.1, 0.2],
        "nodes": [
            {"id": "P", "type": "PATIENT", "name": "patient"},
            {"id": "D", "type": "DIAGNOSIS", "name": "pneumonia"},
        ],
        "edges": [
            {
                "source": "P",
                "target": "D",
                "relation": "HAS_DIAGNOSIS",
                "labels": {"prov_in_note": 1},
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    graph = JsonlGraphBuilder(path).build()[0]
    assert graph.extra["subject_id"] == 7
    assert graph.extra["note_embedding"] == [0.1, 0.2]
    assert graph.edges[0]["labels"]["prov_in_note"] == 1


def test_packaged_checkpoints_match_self_contained_architectures():
    device = torch.device("cpu")
    model5, cfg5 = load_v5(
        str(ROOT / "models/v5_without_note/graph_jepa_v5.pt"), device
    )
    model6, cfg6 = load_v6(
        str(ROOT / "models/v6_with_note/graph_jepa_v6.pt"), device
    )
    assert model5.context_node_encoder.input_proj.in_features == 768
    assert cfg5.encoder == "sapbert"
    assert model6.context_node_encoder.input_proj.in_features == 1536
    assert cfg6.model.use_note_embeddings

    checkpoint = torch.load(
        ROOT / "models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt",
        map_location=device,
        weights_only=False,
    )
    encoder = trainer.Encoder()
    scorer = trainer.DistMult()
    encoder.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])
