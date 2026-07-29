"""Phase 7 gates — the ``models/`` directory rename and its manifest.

Phase 7 renamed the three model directories and the four config sidecars while
deliberately leaving every checkpoint *filename* alone, so that the rename is
provably content-neutral: if the directories moved but the bytes did not, every
``sha256`` in ``MANIFEST.json`` still matches. These tests are what keeps that
true, and they are the only place the manifest is checked against reality at
all — nothing in ``src/`` reads it.

They also pin the one checksum Phase 7 corrected. See
``models/fawkes-entity-note/README.md`` for the evidence that the file on disk,
not the recorded hash, was the published artifact.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from conftest import CKPT, requires_checkpoints

MANIFEST = json.loads((CKPT / "MANIFEST.json").read_text())

# Every entry that names a checkpoint, flattened to (label, entry).
CHECKPOINT_ENTRIES = [
    (f"{group}/{stage}", entry)
    for group in ("clinical-jepa-no-note", "clinical-jepa-localized-note",
                  "fawkes-entity-note")
    for stage, entry in MANIFEST[group].items()
]

# (directory, sidecar, checkpoint beside it). The sidecar names lost their
# ``_v5``/``_v6`` suffixes in the rename because the directory already scopes
# them -- which is exactly why something has to check they did not get swapped.
SIDECARS = [
    ("clinical-jepa-no-note", "config.json", "graph_jepa_v5.pt"),
    ("clinical-jepa-no-note", "config_pretrain.json", "graph_jepa_v5_pretrain.pt"),
    ("clinical-jepa-localized-note", "config.json", "graph_jepa_v6.pt"),
    ("clinical-jepa-localized-note", "config_pretrain.json", "graph_jepa_v6_pretrain.pt"),
]


@requires_checkpoints
@pytest.mark.parametrize(("label", "entry"), CHECKPOINT_ENTRIES,
                         ids=[label for label, _ in CHECKPOINT_ENTRIES])
def test_manifest_checksums_match_the_files_on_disk(label, entry):
    """The rename is content-neutral, asserted rather than assumed.

    Byte count *and* digest, because the two failed independently once: the
    paper entry matched on bytes and not on sha256, which is what made it
    identifiable as a re-serialization rather than a different model.
    """
    path = CKPT / entry["path"]
    assert path.is_file(), f"{label}: manifest names a missing file: {entry['path']}"
    payload = path.read_bytes()
    assert len(payload) == entry["bytes"], f"{label}: byte count"
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"], f"{label}: sha256"


def test_manifest_keeps_the_corrected_paper_checksum_auditable():
    """Phase 7 corrected one hash; the superseded value stays recorded.

    A checksum that is quietly rewritten is indistinguishable from one that was
    never wrong. Keeping the old value next to the new one is what makes the
    correction reviewable, so this pins the evidence trail rather than the
    number alone.
    """
    entry = MANIFEST["fawkes-entity-note"]["final"]
    assert entry["sha256"].startswith("6c21abb2")
    assert entry["sha256_superseded"].startswith("fc8c494a")
    assert entry["sha256"] != entry["sha256_superseded"]
    assert "note" in entry


@requires_checkpoints
@pytest.mark.parametrize(("directory", "sidecar", "checkpoint"), SIDECARS,
                         ids=[f"{d}/{s}" for d, s, _ in SIDECARS])
def test_config_sidecars_match_their_checkpoints(directory, sidecar, checkpoint):
    """Each ``config.json`` is the config its neighbouring checkpoint carries.

    The sidecars were ``config_v5.json`` / ``config_v6.json`` before Phase 7 and
    were self-identifying; now two files four directories apart share the name
    ``config.json``, and only their location distinguishes them. A rename that
    put a sidecar in the wrong directory would leave both files present, both
    parseable, and silently describe the wrong model -- this is what notices.
    """
    base = CKPT / directory
    declared = json.loads((base / sidecar).read_text())
    embedded = torch.load(base / checkpoint, map_location="cpu",
                          weights_only=False)["config"]
    assert declared == embedded
