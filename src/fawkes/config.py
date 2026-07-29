"""Experiment configuration for the paper implementation.

``old_src/paper_v16/trainer.py`` read roughly thirty ``os.environ`` values at
module import time and computed ``NUMERIC_DIM`` — a tensor shape — from them.
That had three consequences: environment variables had to be set before Python
imported the module, two configurations could not coexist in one process, and
the trainer was effectively untestable.

Every one of those globals is a field below. ``Config.from_env()`` reads the
same variable names, with the same defaults and the same parsing rules, so the
documented ``USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 ...`` invocations keep
working unchanged. Nothing in this module reads the environment at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

# Width of the demographic block of ``numfeat`` (age, M, F, and three unused
# slots — see ``data.numeric``). Fixed by the feature layout, not configurable;
# it lives here because ``Config.numeric_dim`` derives from it.
BASE_NUMERIC = 6


def _env(key, default=None, required=False):
    """``trainer.py::env`` — verbatim, including the failure message."""
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(f"[FAILURE] env {key} unset; no fallback.")
    return value


def _flag(key, default):
    """Truthiness rule used by every boolean env var in the original trainer."""
    return str(os.environ.get(key, default)).lower() in ("1", "true", "yes")


def _default_cascade_order():
    return ["MANAGED_FOR", "CONFIRMS", "COMPLICATED_BY", "INDICATES"]


@dataclass
class Config:
    """One experiment. Defaults are the v16 (260615) entity-grounded-note run.

    The defaults here are the defaults that produced the released checkpoint —
    ``Config()`` and an unset environment describe the same experiment.
    """

    # ---- data source ----
    data_repo: str = "wmatbooth/fawkes-training-graph-embedded-260615"
    data_file: str = "fawkes_training_graph_full_embedded_260615.jsonl"
    data_path: str | None = None          # optional local JSONL; bypasses Hugging Face download

    # ---- output ----
    push: bool = True
    output_repo: str = "wmatbooth/fawkes-graph-jepa-v16-260615"

    # ---- schedule ----
    jepa_epochs: int = 60
    readout_epochs: int = 40
    batch: int = 16
    lr: float = 1e-3
    seed: int = 42
    deterministic: bool = True

    # ---- encoder ----
    hid: int = 128
    layers: int = 2
    heads: int = 4
    edge_emb: int = 32
    entity_vocab: int = 8192
    use_entity_emb: bool = True
    query_entity: bool = False
    decoder: str = "distmult"
    semantic_ent: bool = False            # dead in trainer.py too: parsed, never read. Kept so the env surface is unchanged.

    # ---- graph construction ----
    use_scores: bool = False              # (v15) DEFAULT OFF — scores-gate was a wash (v14); structure-only encoder, the NOTE node is the new signal
    prune_no_evidence: bool = True        # (v14) drop LLM edges with zero KB/provenance evidence (keep backbone + borderline)
    use_note: bool = True                 # (v16) inject the Clinical-ModernBERT note vector onto grounded ENTITY nodes (v15 NOTE node retired)
    embed_dim: int = 768                  # (v15) note_embedding dim (Clinical-ModernBERT hidden)
    ground_by: str = "prov"               # (v16) where the note goes: prov=entities with a prov_in_note edge | name=entities named in the note | all=every entity

    # ---- JEPA pretraining ----
    node_mask: float = 0.4
    edge_mask: float = 0.3
    ema_base: float = 0.996
    ema_final: float = 0.9999

    # ---- readout ----
    freeze_encoder: bool = True
    neg_k: int = 8
    temp: float = 1.0
    loss: str = "infonce"
    mask_schedule: bool = True            # (B) sample the held-edge fraction per step
    mask_lo: float = 0.1                  # (B) range for the mask schedule
    mask_hi: float = 0.6
    target_weight: float = 3.0            # (B) upweight the 4 inferred relations in the InfoNCE loss

    # ---- splits and evaluation ----
    val_frac: float = 0.1
    test_frac: float = 0.1
    mrr_cap: int = 3000
    freeze_eval: bool = True
    loo_cap: int = 40000                  # (v12) max leave-one-out queries (each edge re-encodes the graph)
    run_eir: bool = False                 # (v12) v11 batch-holdout EIR eval OFF by default (full-gold F1 is hub-dominated)
    eir_holdout: float = 0.30             # frac of gold LLM edges held out for the ADD test
    eir_fuzzy: float = 0.80               # EIR fuzzy-node match threshold (kg_similarity_scorer)
    run_cascade: bool = True              # (A) run the cascade eval
    cascade_order: list[str] = field(default_factory=_default_cascade_order)

    def __post_init__(self):
        if self.hid % self.heads:
            raise ValueError(f"[FAILURE] HID {self.hid} not divisible by HEADS {self.heads}")

    @property
    def numeric_dim(self) -> int:
        """(v16) demographics + note vector (rides in numfeat on the grounded entities).

        A tensor shape derived from the configuration — ``NUMERIC_DIM`` in the
        original, where it was computed at import.
        """
        return BASE_NUMERIC + (self.embed_dim if self.use_note else 0)

    @classmethod
    def from_env(cls) -> "Config":
        """Build from the same environment variables the original trainer read.

        Same names, same defaults, same parsing. ``OUTPUT_REPO`` keeps its
        ``required=PUSH`` behaviour: pushing with it explicitly blanked fails
        loud rather than silently writing nowhere.
        """
        push = _flag("PUSH", "1")
        return cls(
            data_repo=_env("DATA_REPO", "wmatbooth/fawkes-training-graph-embedded-260615"),
            data_file=_env("DATA_FILE", "fawkes_training_graph_full_embedded_260615.jsonl"),
            data_path=_env("DATA_PATH", None),
            push=push,
            output_repo=_env("OUTPUT_REPO", "wmatbooth/fawkes-graph-jepa-v16-260615", required=push),
            jepa_epochs=int(os.environ.get("JEPA_EPOCHS", 60)),
            readout_epochs=int(os.environ.get("READOUT_EPOCHS", 40)),
            batch=int(os.environ.get("BATCH", 16)),
            lr=float(os.environ.get("LR", 1e-3)),
            seed=int(os.environ.get("SEED", 42)),
            deterministic=_flag("DETERMINISTIC", "1"),
            hid=int(os.environ.get("HID", 128)),
            layers=int(os.environ.get("LAYERS", 2)),
            heads=int(os.environ.get("HEADS", 4)),
            edge_emb=int(os.environ.get("EDGE_EMB", 32)),
            entity_vocab=int(os.environ.get("ENTITY_VOCAB", 8192)),
            use_entity_emb=_flag("ENTITY_EMB", "1"),
            query_entity=_flag("QUERY_ENTITY", "0"),
            decoder=os.environ.get("DECODER", "distmult").lower(),
            semantic_ent=_flag("SEMANTIC_ENT", "0"),
            use_scores=_flag("USE_SCORES", "0"),
            prune_no_evidence=_flag("PRUNE_NO_EVIDENCE", "1"),
            use_note=_flag("USE_NOTE", "1"),
            embed_dim=int(os.environ.get("EMBED_DIM", 768)),
            ground_by=os.environ.get("GROUND_BY", "prov").lower(),
            node_mask=float(os.environ.get("NODE_MASK", 0.4)),
            edge_mask=float(os.environ.get("EDGE_MASK", 0.3)),
            ema_base=float(os.environ.get("EMA_BASE", 0.996)),
            ema_final=float(os.environ.get("EMA_FINAL", 0.9999)),
            freeze_encoder=_flag("FREEZE_ENCODER", "1"),
            neg_k=int(os.environ.get("NEG_K", 8)),
            temp=float(os.environ.get("TEMP", 1.0)),
            loss=os.environ.get("LOSS", "infonce").lower(),
            mask_schedule=_flag("MASK_SCHEDULE", "1"),
            mask_lo=float(os.environ.get("MASK_LO", 0.1)),
            mask_hi=float(os.environ.get("MASK_HI", 0.6)),
            target_weight=float(os.environ.get("TARGET_WEIGHT", 3.0)),
            val_frac=float(os.environ.get("VAL_FRAC", 0.1)),
            test_frac=float(os.environ.get("TEST_FRAC", 0.1)),
            mrr_cap=int(os.environ.get("MRR_CAP", 3000)),
            freeze_eval=_flag("FREEZE_EVAL", "1"),
            loo_cap=int(os.environ.get("LOO_CAP", 40000)),
            run_eir=_flag("RUN_EIR", "0"),
            eir_holdout=float(os.environ.get("EIR_HOLDOUT", 0.30)),
            eir_fuzzy=float(os.environ.get("EIR_FUZZY", 0.80)),
            run_cascade=_flag("RUN_CASCADE", "1"),
            cascade_order=[r.strip() for r in os.environ.get(
                "CASCADE_ORDER", "MANAGED_FOR,CONFIRMS,COMPLICATED_BY,INDICATES").split(",") if r.strip()],
        )

    @classmethod
    def from_checkpoint(cls, saved: dict) -> "Config":
        """Rebuild the architecture-relevant fields from a checkpoint's ``config``.

        Only the keys ``checkpoint_dict`` writes are recoverable; everything else
        falls back to the dataclass default. This is what lets a released
        checkpoint be loaded without setting any environment variable.
        """
        known = {
            "hid": "hid", "layers": "layers", "heads": "heads",
            "use_note": "use_note", "ground_by": "ground_by", "embed_dim": "embed_dim",
            "use_scores": "use_scores", "prune_no_evidence": "prune_no_evidence",
            "node_mask": "node_mask", "edge_mask": "edge_mask",
            "mask_schedule": "mask_schedule", "mask_lo": "mask_lo", "mask_hi": "mask_hi",
            "target_weight": "target_weight", "cascade_order": "cascade_order",
            "loo_cap": "loo_cap", "data_repo": "data_repo", "seed": "seed",
        }
        overrides = {attr: saved[key] for key, attr in known.items() if key in saved}
        return replace(cls(), **overrides)

    def checkpoint_dict(self) -> dict:
        """The ``config`` block written into the checkpoint — trainer.py:706-710.

        Deliberately a hand-picked subset rather than ``asdict(self)``: the
        released checkpoint carries exactly these keys and
        ``evaluate.py`` compares four of them against the running config.
        """
        return {
            "model": "graph_jepa_entity_note_v16",
            "hid": self.hid, "layers": self.layers, "heads": self.heads,
            "use_note": self.use_note, "ground_by": self.ground_by,
            "embed_dim": self.embed_dim, "numeric_dim": self.numeric_dim,
            "use_scores": self.use_scores,
            "prune_no_evidence": self.prune_no_evidence,
            "node_mask": self.node_mask, "edge_mask": self.edge_mask,
            "mask_schedule": self.mask_schedule, "mask_lo": self.mask_lo,
            "mask_hi": self.mask_hi, "target_weight": self.target_weight,
            "cascade_order": self.cascade_order, "loo_cap": self.loo_cap,
            "data_repo": self.data_repo, "seed": self.seed,
        }

    @property
    def checkpoint_name(self) -> str:
        """Plan §5.3. Replaces the hardcoded ``..._v16_260615.pt`` at trainer.py:704."""
        return "fawkes_entity_note.pt" if self.use_note else "fawkes_no_note.pt"
