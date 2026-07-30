"""Phase 3 gates for the merged ``clinical_jepa`` training loop and evaluator.

The evaluator gate was asserted twice, deliberately:

* **differentially** — old and new ran in one process on one input and their
  metric dictionaries had to be equal. That needed both trees importable, so
  Phase 8 retired it (``baseline/COVERAGE.md``).
* **against the pinned payload** — the JSON written by the new evaluator must be
  byte-identical to ``baseline/v{5,6}_loo.json``. That covers strictly more than
  the return value did: the graph count, the recorded invocation fields, and
  float formatting. Phase 0 recorded those files *from the old evaluator on this
  exact invocation* and verified they are byte-reproducible, so the surviving
  half is the same comparison against the same oracle — exact, not toleranced.

``train/loop.py`` had no pinned oracle in Phase 0 — no metric file was recorded
for a training run — so Phase 8 recorded one: one epoch of each stage from a
seeded initialisation, run through ``graph_jepa_v5.training.train_epochs``, in
``baseline/old_clinical_jepa_train.json``.
"""

from __future__ import annotations

import argparse
import json

import pytest
import torch

from conftest import (BASELINE, ROOT, assert_digests_match, digest_fields, load_pin,
                      requires_checkpoints)

from clinical_jepa import evaluate as new_evaluate
from clinical_jepa.config import Config
from clinical_jepa.encoders import MockEncoder
from clinical_jepa.model import GraphJEPA
from clinical_jepa.train import finetune, loop, pretrain

PIN = "old_clinical_jepa_train.json"

# baseline/README.md records this invocation verbatim; the gate is reproducing
# the files it produced, so the flags are copied rather than paraphrased.
LOO_ARGV = [
    "--data", "synthetic",
    "--synthetic-graphs", "256",
    "--synthetic-min-nodes", "8",
    "--synthetic-max-nodes", "28",
    "--device", "cpu",
    "--cap", "40000",
    "--candidate-mode", "schema",
    "--start-graph", "0",
]

VARIANTS = {
    # name: (checkpoint, encoder cache, baseline payload)
    "v5": (
        "models/clinical-jepa-no-note/graph_jepa_v5.pt",
        ".cache/graph_jepa_v5/encoder",
        "v5_loo.json",
    ),
    "v6": (
        "models/clinical-jepa-localized-note/graph_jepa_v6.pt",
        ".cache/graph_jepa_v6/encoder",
        "v6_loo.json",
    ),
}

# The payload records ``args.checkpoint`` verbatim, and Phase 7 renamed the model
# directories, so that one string differs from what Phase 0 recorded. The
# baselines are the regression oracle and are never regenerated; instead the
# expected text is rewritten by exactly this substitution before comparison, so
# the byte comparison stays exact for every metric, count and per-relation row.
# Anything beyond the directory rename still fails.
RENAMED_CHECKPOINT_DIRS = {
    "models/v5_without_note/": "models/clinical-jepa-no-note/",
    "models/v6_with_note/": "models/clinical-jepa-localized-note/",
}

# Both released configs set ``encoder: sapbert``, so a cold cache would download
# SapBERT rather than fail. The two caches hold the same 61 content-hashed
# vectors -- the synthetic vocabulary -- and both are gitignored, like the
# checkpoints.
requires_encoder_cache = pytest.mark.skipif(
    not (ROOT / ".cache/graph_jepa_v5/encoder").is_dir(),
    reason="SapBERT encoder cache is cold; a run would download the model",
)


def _loo_argv(variant: str, output=None) -> list[str]:
    checkpoint, cache, _baseline = VARIANTS[variant]
    argv = LOO_ARGV + ["--checkpoint", checkpoint, "--encoder-cache", cache]
    if output is not None:
        argv += ["--output", str(output)]
    return argv


# --------------------------------------------------------------------------- #
# Phase 3 gate: byte-identical payload, not "within tolerance".
# --------------------------------------------------------------------------- #
@requires_checkpoints
@requires_encoder_cache
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_evaluate_reproduces_baseline_payload(variant, tmp_path, monkeypatch):
    """Every metric, every per-relation row, every count -- as written bytes.

    The expected file is what the old evaluator wrote on this exact invocation
    (``baseline/README.md`` records the command), so this is the same
    old-vs-new comparison the retired differential half made, over strictly more
    of the output.

    ``chdir`` because the payload records ``args.checkpoint`` verbatim and the
    baselines were recorded with repo-relative paths. Those paths carry the
    pre-Phase-7 directory names, which is the only permitted difference -- see
    ``RENAMED_CHECKPOINT_DIRS``.
    """
    monkeypatch.chdir(ROOT)
    expected = BASELINE / VARIANTS[variant][2]
    produced = tmp_path / f"{variant}_loo.json"

    new_evaluate.main(_loo_argv(variant, output=produced))

    expected_text = expected.read_text(encoding="utf-8")
    for old_dir, new_dir in RENAMED_CHECKPOINT_DIRS.items():
        expected_text = expected_text.replace(old_dir, new_dir)
    # The substitution must actually have applied, or this silently degrades
    # into comparing the baseline against itself under a stale assumption.
    assert VARIANTS[variant][0] in expected_text
    assert produced.read_text(encoding="utf-8") == expected_text

    # Pin the headline numbers here too, so a failure reads as a number rather
    # than as a diff of 1,700 lines of JSON.
    metrics = json.loads(produced.read_text(encoding="utf-8"))["metrics"]
    assert metrics["n"] == 1755
    assert round(metrics["mrr"], 3) == {"v5": 0.605, "v6": 0.621}[variant]
    assert round(metrics["hits1"], 3) == {"v5": 0.343, "v6": 0.366}[variant]


@requires_checkpoints
def test_checkpoint_defaults_point_at_files_that_exist():
    """Plan section 5.2. The old defaults pointed into ``checkpoints/``, a
    directory this repository has never contained. Phase 7 renames the model
    directories, and this is what will notice if the defaults are not renamed
    with them."""
    for parser in (new_evaluate.build_arg_parser(), finetune.build_arg_parser()):
        default = parser.get_default("checkpoint")
        assert (ROOT / default).is_file(), f"missing default checkpoint: {default}"


# --------------------------------------------------------------------------- #
# train/loop.py: differential, both stages of the epoch loop.
# --------------------------------------------------------------------------- #
IN_DIM = 32


def _small_config(config_cls):
    """``baseline/record_old_pins.py`` builds this same config for the old side."""
    cfg = config_cls()
    cfg.model.in_dim = IN_DIM
    cfg.model.num_patches = 4
    cfg.train.synthetic_graphs = 8
    cfg.train.batch_size = 4
    cfg.model.use_note_embeddings = False
    cfg.model.base_in_dim = IN_DIM
    return cfg


@pytest.fixture
def single_threaded():
    """Make a training step bit-reproducible so the gate can assert equality.

    Backward on CPU reduces across intra-op threads in an order that is not
    fixed, so at the default thread count a run is not bit-identical even
    against *itself*: two fresh copies of the same model, same seeds, same
    batches, drift ~5e-7 on the parameters after one epoch. That is float noise,
    not behaviour -- but it makes exact comparison impossible. One thread
    removes it entirely (measured: 0.0 both new-vs-new and new-vs-old), so this
    tightens the assertion to exact rather than relaxing it to a tolerance.

    The evaluator gates above need no such thing: they are forward-only under
    ``no_grad`` and reproduce byte-identically at the default thread count.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("use_revision", "pin_key"), [(False, "no_revision"), (True, "revision")]
)
def test_train_epochs_matches_old_loop(use_revision, pin_key, single_threaded):
    """One epoch, identical seeds and starting weights, every parameter compared.

    ``use_revision=False`` is the masked-pretraining branch (which routes through
    ``pretrain_sanitized_graph_data``); ``True`` is the joint branch that adds the
    revision and candidate-ranking objectives. Both draw negatives from the
    global RNG and patches from the passed generator, so both are re-seeded
    immediately before each call.

    The starting weights come from ``model_init_seed`` rather than from an
    unseeded ``GraphJEPA``: the old loop's parameters after one epoch are pinned,
    which is only meaningful from a reproducible initialisation. The recorder
    copied exactly these weights into ``GraphJEPAv5`` before training it, so the
    comparison is still one epoch of the old loop against one epoch of the new
    one from the same point.
    """
    pinned = load_pin(PIN)
    device = torch.device("cpu")
    encoder = MockEncoder(dim=IN_DIM)

    parser = argparse.ArgumentParser()
    loop.add_data_args(parser)
    args = parser.parse_args(["--data", "synthetic"])

    cfg = _small_config(Config)
    torch.manual_seed(pinned["model_init_seed"])
    model = GraphJEPA(cfg.model)

    _dataset, train_loader = loop.build_train_loader(args, cfg, encoder)
    torch.manual_seed(1234)
    steps = loop.train_epochs(
        model,
        loop.build_optimizer(model, cfg),
        train_loader,
        cfg,
        stage_name="gate",
        epochs=1,
        use_revision=use_revision,
        device=device,
        generator=torch.Generator().manual_seed(0),
        wandb_run=None,
    )

    expected = pinned["train_epochs"][pin_key]
    assert steps == expected["steps"] == 2  # 8 graphs, batch_size 4
    assert_digests_match(
        digest_fields(model.state_dict()), expected["state_dict"], path=pin_key
    )


# --------------------------------------------------------------------------- #
# Plan section 5.3: variant-derived checkpoint names.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("use_note", "is_pretrain", "expected"),
    [
        (False, True, "clinical_jepa_no_note_pretrain.pt"),
        (False, False, "clinical_jepa_no_note.pt"),
        (True, True, "clinical_jepa_note_pretrain.pt"),
        (True, False, "clinical_jepa_note.pt"),
    ],
)
def test_checkpoint_filename_follows_plan_5_3(use_note, is_pretrain, expected):
    """The table has no oracle in the pre-restructure tree -- it replaces the
    hardcoded ``graph_jepa_v{5,6}*.pt`` constants -- so it is asserted directly."""
    cfg = Config()
    cfg.model.use_note_embeddings = use_note
    assert loop.checkpoint_filename(cfg, pretrain=is_pretrain) == expected


@pytest.mark.parametrize(
    ("variant_argv", "expected"),
    [
        (["--no-note-embeddings"], "clinical_jepa_no_note_pretrain.pt"),
        (["--note-embedding-dim", "16"], "clinical_jepa_note_pretrain.pt"),
    ],
)
def test_pretrain_names_its_output_by_variant(variant_argv, expected, tmp_path):
    """End-to-end through the real script: the name and the ``config_pretrain.json``
    sidecar come off the variant, and fine-tuning can pick the result back up."""
    argv = [
        "--data", "synthetic",
        "--synthetic-graphs", "4",
        "--epochs", "1",
        "--encoder", "mock",
        "--mock-dim", str(IN_DIM),
        "--num-patches", "4",
        "--batch_size", "2",
        "--out", str(tmp_path),
    ] + variant_argv

    path = pretrain.pretrain(pretrain.build_arg_parser().parse_args(argv))

    assert path.name == expected
    assert (tmp_path / "config_pretrain.json").is_file()

    _model, cfg = loop.load_model_checkpoint(str(path), torch.device("cpu"))
    assert cfg.model.use_note_embeddings == ("--no-note-embeddings" not in variant_argv)
    assert loop.checkpoint_filename(cfg, pretrain=False) == expected.replace(
        "_pretrain.pt", ".pt"
    )
