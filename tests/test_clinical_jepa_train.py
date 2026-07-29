"""Phase 3 gates for the merged ``clinical_jepa`` training loop and evaluator.

The evaluator gate is asserted twice, deliberately:

* **differentially** — old and new run in one process on one input and their
  metric dictionaries must be equal. This is the stronger form, and it exists
  only because every package was renamed, so both trees import side by side.
* **against the pinned payload** — the JSON written by the new evaluator must be
  byte-identical to ``baseline/v{5,6}_loo.json``. That covers the parts the
  return value does not: the graph count, the recorded invocation fields, and
  float formatting. Phase 0 verified those files are byte-reproducible, so this
  is exact, not toleranced. It is also what survives Phase 8, when ``old_src``
  goes away and the differential half of the gate becomes unrunnable.

``train/loop.py`` has no pinned oracle — no metric file was recorded for a
training run — so it is gated purely differentially, one epoch of each stage
against ``graph_jepa_v5.training.train_epochs`` with identical seeds and
weights.
"""

from __future__ import annotations

import argparse
import json

import pytest
import torch

from conftest import ROOT, requires_checkpoints

from clinical_jepa import evaluate as new_evaluate
from clinical_jepa.config import Config
from clinical_jepa.encoders import MockEncoder
from clinical_jepa.model import GraphJEPA
from clinical_jepa.train import finetune, loop, pretrain

from graph_jepa_v5 import evaluate as old_v5_evaluate
from graph_jepa_v5 import training as old_v5_training
from graph_jepa_v5.config import Config as OldV5Config
from graph_jepa_v5.model import GraphJEPAv5 as OldGraphJEPAv5
from graph_jepa_v6 import evaluate as old_v6_evaluate

BASELINE = ROOT / "baseline"

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
    # name: (old module, checkpoint, encoder cache, baseline payload)
    "v5": (
        old_v5_evaluate,
        "models/v5_without_note/graph_jepa_v5.pt",
        ".cache/graph_jepa_v5/encoder",
        "v5_loo.json",
    ),
    "v6": (
        old_v6_evaluate,
        "models/v6_with_note/graph_jepa_v6.pt",
        ".cache/graph_jepa_v6/encoder",
        "v6_loo.json",
    ),
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
    _old, checkpoint, cache, _baseline = VARIANTS[variant]
    argv = LOO_ARGV + ["--checkpoint", checkpoint, "--encoder-cache", cache]
    if output is not None:
        argv += ["--output", str(output)]
    return argv


# --------------------------------------------------------------------------- #
# Phase 3 gate, differential half.
# --------------------------------------------------------------------------- #
@requires_checkpoints
@requires_encoder_cache
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_evaluate_matches_old_evaluator(variant, monkeypatch):
    """One merged evaluator reproduces both of the two it replaces.

    Each side parses the same argv with its own parser, so the merged parser has
    to accept the recorded invocation as well as agree on every metric.
    """
    monkeypatch.chdir(ROOT)
    old_module = VARIANTS[variant][0]
    argv = _loo_argv(variant)

    new = new_evaluate.run(new_evaluate.build_arg_parser().parse_args(argv))
    old = old_module.run(old_module.build_arg_parser().parse_args(argv))

    assert new == old


# --------------------------------------------------------------------------- #
# Phase 3 gate, pinned half: byte-identical payload, not "within tolerance".
# --------------------------------------------------------------------------- #
@requires_checkpoints
@requires_encoder_cache
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_evaluate_reproduces_baseline_payload(variant, tmp_path, monkeypatch):
    """Every metric, every per-relation row, every count -- as written bytes.

    ``chdir`` because the payload records ``args.checkpoint`` verbatim and the
    baselines were recorded with repo-relative paths.
    """
    monkeypatch.chdir(ROOT)
    expected = BASELINE / VARIANTS[variant][3]
    produced = tmp_path / f"{variant}_loo.json"

    new_evaluate.main(_loo_argv(variant, output=produced))

    assert produced.read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")

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
    cfg = config_cls()
    cfg.model.in_dim = IN_DIM
    cfg.model.num_patches = 4
    cfg.train.synthetic_graphs = 8
    cfg.train.batch_size = 4
    if hasattr(cfg.model, "use_note_embeddings"):
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


@pytest.mark.parametrize("use_revision", [False, True])
def test_train_epochs_matches_old_loop(use_revision, single_threaded):
    """One epoch, identical seeds and starting weights, every parameter compared.

    ``use_revision=False`` is the masked-pretraining branch (which routes through
    ``pretrain_sanitized_graph_data``); ``True`` is the joint branch that adds the
    revision and candidate-ranking objectives. Both draw negatives from the
    global RNG and patches from the passed generator, so both are re-seeded
    immediately before each call.
    """
    device = torch.device("cpu")
    encoder = MockEncoder(dim=IN_DIM)

    parser = argparse.ArgumentParser()
    loop.add_data_args(parser)
    args = parser.parse_args(["--data", "synthetic"])

    cfg, old_cfg = _small_config(Config), _small_config(OldV5Config)
    new_model = GraphJEPA(cfg.model)
    old_model = OldGraphJEPAv5(old_cfg.model)
    old_model.load_state_dict(new_model.state_dict(), strict=True)

    def run(module, model, config):
        _dataset, train_loader = module.build_train_loader(args, config, encoder)
        torch.manual_seed(1234)
        return module.train_epochs(
            model,
            module.build_optimizer(model, config),
            train_loader,
            config,
            stage_name="gate",
            epochs=1,
            use_revision=use_revision,
            device=device,
            generator=torch.Generator().manual_seed(0),
            wandb_run=None,
        )

    new_steps = run(loop, new_model, cfg)
    old_steps = run(old_v5_training, old_model, old_cfg)

    assert new_steps == old_steps == 2  # 8 graphs, batch_size 4
    new_state, old_state = new_model.state_dict(), old_model.state_dict()
    assert sorted(new_state) == sorted(old_state)
    for key, value in new_state.items():
        assert torch.equal(value, old_state[key]), f"{key} diverged after training"


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
    """The table has no oracle in ``old_src`` -- it replaces the hardcoded
    ``graph_jepa_v{5,6}*.pt`` constants -- so it is pinned directly."""
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
