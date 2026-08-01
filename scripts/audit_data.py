#!/usr/bin/env python3
"""Report the JSONL features each packaged model needs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# The dataset present in this working tree. The previous default named
# data/fawkes_1k_patients/..., which this repository does not contain.
DEFAULT_PATH = Path(
    "data/fawkes-training-graph-embedded-260615/"
    "fawkes_training_graph_full_embedded_260615.jsonl"
)


def audit(path: Path) -> dict:
    result = Counter()
    node_types = Counter()
    relations = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                graph = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            result["graphs"] += 1
            result["note_text"] += int(bool(graph.get("note")))
            result["note_embedding"] += int(
                bool(graph.get("note_embedding") or graph.get("note_embeddings"))
            )
            for node in graph.get("nodes", []):
                node_types[str(node.get("type"))] += 1
                result["nodes"] += 1
            for edge in graph.get("edges", []):
                relations[str(edge.get("relation") or edge.get("type"))] += 1
                result["edges"] += 1
                result["llm_edges"] += int(edge.get("evidence") == "llm")
                labels = edge.get("labels")
                result["labeled_edges"] += int(isinstance(labels, dict))
                result["note_provenance_edges"] += int(
                    isinstance(labels, dict) and bool(labels.get("prov_in_note"))
                )
    return {
        **dict(result),
        "node_types": dict(sorted(node_types.items())),
        "relations": dict(sorted(relations.items())),
        "compatible": {
            "clinical-jepa-no-note": result["graphs"] > 0,
            "clinical-jepa-localized-note": result["note_embedding"] > 0
            and result["note_provenance_edges"] > 0,
            "fawkes-entity-note": result["note_embedding"] > 0
            and result["note_provenance_edges"] > 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    print(json.dumps(audit(args.path), indent=2))


if __name__ == "__main__":
    main()
