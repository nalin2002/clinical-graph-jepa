#!/usr/bin/env python3
"""Load every packaged checkpoint and validate its input dimensions.

The import-boundary guard that used to live here AST-walked
``src/graph_jepa_v5`` and ``src/graph_jepa_v6`` for imports of
``graph_jepa_v2/v3/v4`` -- packages this repository has never contained. After
Phase 0 moved those directories out of ``src/`` it scanned zero files and passed
green regardless. ``tests/test_import_boundaries.py`` replaces it with a real
boundary test that asserts on its own walked-file count.
"""

from __future__ import annotations

from pathlib import Path

import torch

from clinical_jepa.train.loop import load_model_checkpoint
from fawkes.config import Config as FawkesConfig
from fawkes.model import DistMult, Encoder


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    device = torch.device("cpu")
    no_note, cfg_no_note = load_model_checkpoint(
        str(ROOT / "models/clinical-jepa-no-note/graph_jepa_v5.pt"), device
    )
    localized_note, cfg_localized_note = load_model_checkpoint(
        str(ROOT / "models/clinical-jepa-localized-note/graph_jepa_v6.pt"), device
    )

    checkpoint = torch.load(
        ROOT / "models/fawkes-entity-note/fawkes_trainer_jepa_entity_note_v16_260615.pt",
        map_location=device,
        weights_only=False,
    )
    fawkes_cfg = FawkesConfig()
    encoder = Encoder(fawkes_cfg).to(device)
    scorer = DistMult(fawkes_cfg).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])

    print(
        "OK: "
        f"clinical-jepa-no-note={type(no_note).__name__}"
        f"(in={cfg_no_note.model.in_dim}), "
        f"clinical-jepa-localized-note={type(localized_note).__name__}"
        f"(in={cfg_localized_note.model.in_dim}), "
        f"fawkes-entity-note={checkpoint['config']['model']}"
    )


if __name__ == "__main__":
    main()
