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
# clinical_jepa/train/loop.py defaults --jsonl-path to; that one
# (data/fawkes_1k_patients/...) is absent from the working tree. Only the
# fawkes/paper lineage reads this; the clinical_jepa gates use seeded synthetic
# graphs and need no data file at all.
DATA = ROOT / "data/fawkes-training-graph-embedded-260615/fawkes_training_graph_full_embedded_260615.jsonl"

requires_checkpoints = pytest.mark.skipif(
    not any((CKPT / "clinical-jepa-no-note").glob("*.pt")),
    reason="released checkpoints not present in the working tree",
)

requires_data = pytest.mark.skipif(
    not DATA.exists(),
    reason="embedded dataset not present in the working tree",
)


def assert_pyg_equal(a, b, path=""):
    """Assert two PyG Data objects are elementwise identical.

    NaN-aware, and it has to be: torch.equal returns False for NaN == NaN, and
    edge_llm_confidence is all-NaN whenever edges carry no confidence — which is
    every synthetic graph, on both the old and new side. A plain torch.equal here
    cannot compare any clinical_jepa Data object at all.

    This is stricter than torch.equal, not looser: the NaN masks must match
    positionally AND every non-NaN value must be bit-identical. No tolerance.
    """
    ka, kb = set(a.keys()), set(b.keys())
    assert ka == kb, f"{path}: key mismatch {ka ^ kb}"
    for key in sorted(ka):
        va, vb = a[key], b[key]
        if torch.is_tensor(va):
            if va.is_floating_point():
                na, nb = torch.isnan(va), torch.isnan(vb)
                assert torch.equal(na, nb), f"{path}.{key}: NaN mask mismatch"
                assert torch.equal(va[~na], vb[~nb]), f"{path}.{key}: tensor mismatch"
            else:
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

    `section` is required for EVERY baseline dump, v5/v6 included — they are all
    written as {"state_dict": [...]}, and the paper dump as
    {"encoder": [...], "scorer": [...]}. Use section="state_dict" for v5/v6.
    """
    import json

    expected = json.loads(Path(expected_path).read_text())
    if section is not None:
        expected = expected[section]
    assert sorted(actual.keys()) == expected, (
        f"state_dict key set differs from {Path(expected_path).name}"
        f"{f'[{section}]' if section else ''}"
    )
