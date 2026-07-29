"""The Clinical Graph-JEPA paper implementation. Compatible with the released
paper checkpoint.

Despite the repository name, **this** package — not ``clinical_jepa`` — is the
implementation behind ``paper/clinical_jepa.pdf``. The name is the original
author's. See ``docs/LINEAGE.md``.

Was ``paper_v16``. Entity-grounded-note Graph-JEPA (v16, 260615): a JEPA world
model over per-admission clinical knowledge graphs, with the Clinical-ModernBERT
note vector localized onto the entity nodes the note actually grounds.

Importing this package has no side effects and reads no environment variables.
``Config.from_env()`` reads them on demand, so the documented
``USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 PRUNE_NO_EVIDENCE=1``
invocations still work, and two configurations can coexist in one process.

    from fawkes.config import Config
    from fawkes.model import Encoder, DistMult

    cfg = Config.from_env()          # or Config(use_note=False) directly
"""
