"""Shared fixtures and differential helpers for the restructure gates.

Both ``src`` and ``old_src`` are on ``pythonpath`` (see ``pyproject.toml``), so a
gate can import the new and old implementations in one process and assert they
agree. That is a stronger guarantee than comparing against a pinned JSON file,
and it is available only because every package is being renamed — there are no
module-name collisions between the two trees.

``old_src`` is removed in Phase 8, at which point the remaining gates assert
against ``baseline/*.json`` instead.
"""

import pytest
import torch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models"
BASELINE = ROOT / "baseline"

# The 4,000-record embedded dataset. Note this is NOT the file that
# fawkes_core/training.py:306 defaults --jsonl-path to; that one
# (data/fawkes_1k_patients/...) is absent from the working tree. Only the
# fawkes/paper lineage reads this; the clinical_jepa gates use seeded synthetic
# graphs and need no data file at all.
DATA = ROOT / "data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl"

# Phase 7 renames these to clinical-jepa-no-note / clinical-jepa-localized-note.
requires_checkpoints = pytest.mark.skipif(
    not any((CKPT / "v5_without_note").glob("*.pt")),
    reason="released checkpoints not present in the working tree",
)

requires_data = pytest.mark.skipif(
    not DATA.exists(),
    reason="embedded dataset not present in the working tree",
)


def assert_pyg_equal(a, b, path=""):
    """Assert two PyG Data objects are elementwise identical."""
    ka, kb = set(a.keys()), set(b.keys())
    assert ka == kb, f"{path}: key mismatch {ka ^ kb}"
    for key in sorted(ka):
        va, vb = a[key], b[key]
        if torch.is_tensor(va):
            assert torch.equal(va, vb), f"{path}.{key}: tensor mismatch"
        else:
            assert va == vb, f"{path}.{key}: {va!r} != {vb!r}"


def assert_state_dict_equal(actual, expected_path, section=None):
    """Assert a state_dict's key set matches a baseline/*_keys.json dump.

    Key-set equality catches a renamed attribute path, which is the failure mode
    a bad class flatten produces. It does NOT catch a wrong tensor shape: the v5
    and v6 key sets are identical and differ only in two input_proj shapes, so a
    merged model that ignored use_note_embeddings would still pass this. Pair it
    with load_state_dict(..., strict=True), which checks shapes.
    """
    import json

    expected = json.loads(Path(expected_path).read_text())
    if section is not None:
        expected = expected[section]
    assert sorted(actual.keys()) == expected, (
        f"state_dict key set differs from {Path(expected_path).name}"
        f"{f'[{section}]' if section else ''}"
    )
