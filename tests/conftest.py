"""Shared fixtures and pinned-oracle helpers for the restructure gates.

Phases 1–7 ran the old implementation and the new one in the same process and
asserted they agreed exactly. That was available only because every package was
renamed, so both trees could sit on ``pythonpath`` with no module-name
collisions — and it is the strongest form of gate there is.

Phase 8 deleted the old tree, so those gates now assert against artifacts
recorded *from* it: ``baseline/*.json``, written by
``baseline/record_old_pins.py`` while both trees were still importable. Nothing
about what is asserted changed; only where the expected value is read from.
``baseline/COVERAGE.md`` records every gate that lost coverage in the move.
"""

import hashlib
import json

import pytest
import torch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models"
BASELINE = ROOT / "baseline"

# The 4,000-record embedded dataset. Both lineages now read it: the fawkes gates
# use it directly, and it is what clinical_jepa/train/loop.py defaults
# --jsonl-path to. The clinical_jepa gates themselves use seeded synthetic graphs
# and need no data file at all.
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
    expected = json.loads(Path(expected_path).read_text())
    if section is not None:
        expected = expected[section]
    assert sorted(actual.keys()) == expected, (
        f"state_dict key set differs from {Path(expected_path).name}"
        f"{f'[{section}]' if section else ''}"
    )


def load_pin(name):
    """Read one of the artifacts ``baseline/record_old_pins.py`` recorded.

    Everything in these files was produced by the pre-restructure tree while it
    still existed, so a gate reading one is asserting against the original
    implementation rather than against a number someone chose. Never edit one to
    make a test pass -- ``baseline/README.md`` says why.
    """
    return json.loads((BASELINE / name).read_text(encoding="utf-8"))


def digest_fields(fields):
    """sha256 per field, over dtype, shape and raw bytes.

    The pinned form of "these tensors are identical", used where the tensors
    themselves are far too large to track (256 synthetic graphs carry a
    17x768 ``x`` each).

    Bytes rather than ``torch.equal`` because a byte comparison is exact for NaN
    as well: torch writes one canonical quiet NaN, so an all-NaN
    ``edge_llm_confidence`` digests stably, where ``torch.equal`` reports two
    identical tensors as different. That is why ``assert_pyg_equal`` needs its
    NaN mask and this does not.

    One digest per field rather than one per object, so a mismatch names the
    tensor that moved and not merely the object holding it.
    """
    out = {}
    for key in sorted(fields):
        value = fields[key]
        if torch.is_tensor(value):
            payload = (
                f"{value.dtype}:{tuple(value.shape)}:".encode()
                + value.detach().contiguous().cpu().numpy().tobytes()
            )
        else:
            payload = f"{type(value).__name__}:{value!r}".encode()
        out[key] = hashlib.sha256(payload).hexdigest()[:16]
    return out


def digest_data(data):
    """``digest_fields`` over a PyG ``Data`` object."""
    return digest_fields({key: data[key] for key in data.keys()})


def fold_digests(digests):
    """One digest over a collection of them, for pins with thousands of items.

    ``baseline/record_old_pins.py`` uses this same function, so a folded pin and
    the value a gate computes are comparable by construction.
    """
    return hashlib.sha256(
        json.dumps(digests, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def assert_digests_match(actual, expected, path=""):
    """Compare two ``digest_fields`` maps, naming every field that differs."""
    assert set(actual) == set(expected), (
        f"{path}: field set differs from the pin: {set(actual) ^ set(expected)}"
    )
    differing = sorted(key for key in actual if actual[key] != expected[key])
    assert not differing, (
        f"{path}: {differing} differ from the recorded digest "
        f"(pinned {[expected[key] for key in differing]}, "
        f"got {[actual[key] for key in differing]})"
    )
