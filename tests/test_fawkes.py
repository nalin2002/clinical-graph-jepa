"""Phase 5 gates — `fawkes`, the paper implementation.

`docs/RESTRUCTURE_PLAN.md` calls this the highest-risk phase: `paper_v16/trainer.py`
is the one artifact that backs the paper, and it was split five ways. Every test
here is an equality check against either `old_src/paper_v16` running in the same
process or a `baseline/*.json` recorded from it before the split.

The three gates the plan names are `test_loo_reproduces_baseline`,
`test_state_dict_keys_match_baseline`, and `test_no_module_scope_environment_reads`.
The rest localize a failure: if `to_data` drifts, the LOO gate fails too, but the
`to_data` differential says which of the five modules moved.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import BASELINE, DATA, ROOT, assert_state_dict_equal, requires_data

from fawkes.config import Config
from fawkes.data import resolve_rel, to_data
from fawkes.evaluate import (_load_graphs, cascade_evaluate, eir_uplift_eval, evaluate,
                             loo_evaluate)
from fawkes.model import JEPA, DistMult, Encoder
from fawkes.train import jepa_step, readout_step

PAPER_CKPT = ROOT / "models/fawkes-entity-note/fawkes_trainer_jepa_entity_note_v16_260615.pt"
FAWKES_SRC = ROOT / "src/fawkes"

requires_paper_checkpoint = pytest.mark.skipif(
    not PAPER_CKPT.exists(),
    reason="released paper checkpoint not present in the working tree",
)

TOL = 1e-6


def _load_paper_model(cfg):
    checkpoint = torch.load(PAPER_CKPT, map_location="cpu", weights_only=False)
    encoder, scorer = Encoder(cfg), DistMult(cfg)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    scorer.load_state_dict(checkpoint["scorer"], strict=True)
    return encoder, scorer, checkpoint


def _assert_metrics_equal(actual, expected, tol=TOL):
    """Compare a loo_evaluate result against a baseline dump, per-relation included."""
    assert actual["n"] == expected["n"], f"query count {actual['n']} != {expected['n']}"
    for key in ("mrr", "hits1", "hits3", "hits10"):
        assert abs(actual[key] - expected[key]) <= tol, (
            f"{key}: {actual[key]!r} vs baseline {expected[key]!r}")
    assert [r["rel"] for r in actual["per_rel"]] == [r["rel"] for r in expected["per_rel"]]
    for got, want in zip(actual["per_rel"], expected["per_rel"]):
        assert got["n"] == want["n"], f"{got['rel']}: n {got['n']} != {want['n']}"
        for key in ("mrr", "h1", "h3", "h10", "C", "chance_mrr", "chance_h1"):
            assert abs(got[key] - want[key]) <= tol, (
                f"{got['rel']}.{key}: {got[key]!r} vs baseline {want[key]!r}")


# ---- Gate 1: the released checkpoint still produces the recorded metrics ----

@requires_paper_checkpoint
@requires_data
def test_loo_reproduces_baseline():
    """Gate 1. Same invocation as baseline/README.md's `paper_loo.json` command.

    The whole 4,000-record file, the evaluator's own >=2-edge filter, cap 40000.
    This is old evaluate.py against new evaluate.py on identical input.
    """
    cfg = Config()
    encoder, scorer, _ = _load_paper_model(cfg)
    raw, demographics = _load_graphs(DATA, None)
    graphs = [d for g in raw
              if (d := to_data(g, demographics, cfg)).num_nodes >= 3 and d.edge_index.size(1) >= 2]
    metrics = loo_evaluate(encoder, scorer, graphs, torch.device("cpu"), cfg, cap=40000)
    _assert_metrics_equal(metrics, json.loads((BASELINE / "paper_loo.json").read_text()))


@requires_paper_checkpoint
@requires_data
def test_loo_reproduces_published_test_split():
    """The published number, not just the refactor's self-consistency.

    baseline/README.md asks Phase 5 to gate on both files: paper_loo.json proves
    the evaluator refactor is behaviour-preserving, this one proves the model and
    data path still reproduce what the checkpoint itself reports. Mirrors the
    trainer's split — RandomState(seed).permutation, first test_frac — including
    the >=4 edge filter, which is what makes this population differ from gate 1's.
    """
    cfg = Config()
    encoder, scorer, checkpoint = _load_paper_model(cfg)
    raw, demographics = _load_graphs(DATA, None)
    items = [d for g in raw
             if (d := to_data(g, demographics, cfg)).num_nodes >= 3 and d.edge_index.size(1) >= 4]
    idx = np.random.RandomState(cfg.seed).permutation(len(items))
    test = [items[i] for i in idx[:int(cfg.test_frac * len(items))]]
    metrics = loo_evaluate(encoder, scorer, test, torch.device("cpu"), cfg)

    _assert_metrics_equal(metrics, json.loads((BASELINE / "paper_loo_testsplit.json").read_text()))
    # ...and against the metrics the checkpoint carries internally.
    embedded = checkpoint["recovery_test_loo"]
    assert metrics["n"] == embedded["n"]
    for key in ("mrr", "hits1", "hits3", "hits10"):
        assert abs(metrics[key] - embedded[key]) <= TOL


# ---- Gate 2: attribute paths did not move ----

def test_state_dict_keys_match_baseline():
    """Gate 2. state_dict keys derive from attribute paths, so a renamed
    submodule shows up here and nowhere else until inference silently degrades.
    """
    cfg = Config()
    assert_state_dict_equal(Encoder(cfg).state_dict(), BASELINE / "paper_keys.json", "encoder")
    assert_state_dict_equal(DistMult(cfg).state_dict(), BASELINE / "paper_keys.json", "scorer")


@requires_paper_checkpoint
def test_released_checkpoint_loads_strict():
    """Key-set equality is blind to tensor shapes; strict=True is what checks them."""
    _load_paper_model(Config())


@requires_paper_checkpoint
def test_checkpoint_dict_matches_released_config():
    """Config.checkpoint_dict() must keep writing the block the released file carries."""
    saved = torch.load(PAPER_CKPT, map_location="cpu", weights_only=False)["config"]
    assert Config().checkpoint_dict() == saved


# ---- Gate 3: no import-time environment reads ----

def _import_time_nodes(tree):
    """Yield AST nodes evaluated at import: everything but function bodies.

    Decorators, argument defaults and annotations of a function DO run at import,
    so only the body is skipped.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            children = [c for c in ast.iter_child_nodes(node) if c not in node.body]
        elif isinstance(node, ast.Lambda):
            children = [c for c in ast.iter_child_nodes(node) if c is not node.body]
        else:
            children = list(ast.iter_child_nodes(node))
        stack.extend(children)


def _reads_environment(node):
    if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
        return isinstance(node.value, ast.Name) and node.value.id == "os"
    return isinstance(node, ast.Name) and node.id in ("environ", "getenv")


def test_no_module_scope_environment_reads():
    """Gate 3a. §2.5: the trainer read ~30 os.environ values at import and computed
    NUMERIC_DIM — a tensor shape — from them, so two configurations could not
    coexist in one process. Config.from_env() reads them on demand instead.
    """
    offenders = []
    for path in sorted(FAWKES_SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _import_time_nodes(tree):
            if _reads_environment(node):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"environment read at module scope: {offenders}"


def test_imports_cleanly_with_no_environment():
    """Gate 3b. The AST check proves no env read; this proves import works without one."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c",
         "import fawkes, fawkes.config, fawkes.data, fawkes.model, fawkes.train, fawkes.evaluate"],
        env=env, capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_two_configurations_coexist():
    """The point of gate 3: build both variants in one process, which the
    import-time globals made impossible.
    """
    with_note, without_note = Config(use_note=True), Config(use_note=False)
    assert (with_note.numeric_dim, without_note.numeric_dim) == (774, 6)
    assert Encoder(with_note).num_proj.in_features == 774
    assert Encoder(without_note).num_proj.in_features == 6
    assert with_note.checkpoint_name == "fawkes_entity_note.pt"
    assert without_note.checkpoint_name == "fawkes_no_note.pt"


# ---- Differential against old_src: localize a failure to one module ----

def test_config_from_env_matches_trainer_globals():
    """Every env var the trainer read is a Config field with the same value.

    Reads the ambient environment on both sides, so this holds whatever
    USE_NOTE/GROUND_BY/... the suite is run under, not just the defaults.
    """
    from paper_v16 import trainer

    pairs = [
        ("data_repo", "DATA_REPO"), ("data_file", "DATA_FILE"), ("data_path", "DATA_PATH"),
        ("push", "PUSH"), ("output_repo", "OUTPUT_REPO"),
        ("jepa_epochs", "JEPA_EPOCHS"), ("readout_epochs", "READOUT_EPOCHS"),
        ("batch", "BATCH"), ("lr", "LR"), ("seed", "SEED"), ("deterministic", "DETERMINISTIC"),
        ("hid", "HID"), ("layers", "LAYERS"), ("heads", "HEADS"), ("edge_emb", "EDGE_EMB"),
        ("entity_vocab", "ENTITY_VOCAB"), ("use_entity_emb", "USE_ENTITY_EMB"),
        ("query_entity", "QUERY_ENTITY"), ("decoder", "DECODER"), ("semantic_ent", "SEMANTIC_ENT"),
        ("use_scores", "USE_SCORES"), ("prune_no_evidence", "PRUNE_NO_EVIDENCE"),
        ("use_note", "USE_NOTE"), ("embed_dim", "EMBED_DIM"), ("ground_by", "GROUND_BY"),
        ("node_mask", "NODE_MASK"), ("edge_mask", "EDGE_MASK"),
        ("ema_base", "EMA_BASE"), ("ema_final", "EMA_FINAL"),
        ("freeze_encoder", "FREEZE"), ("neg_k", "NEG_K"), ("temp", "TEMP"), ("loss", "LOSS"),
        ("mask_schedule", "MASK_SCHEDULE"), ("mask_lo", "MASK_LO"), ("mask_hi", "MASK_HI"),
        ("target_weight", "TARGET_WEIGHT"),
        ("val_frac", "VAL_FRAC"), ("test_frac", "TEST_FRAC"), ("mrr_cap", "MRR_CAP"),
        ("freeze_eval", "FREEZE_EVAL"), ("loo_cap", "LOO_CAP"), ("run_eir", "RUN_EIR"),
        ("eir_holdout", "EIR_HOLDOUT"), ("eir_fuzzy", "EIR_FUZZY"),
        ("run_cascade", "RUN_CASCADE"), ("cascade_order", "CASCADE_ORDER"),
    ]
    assert {f.name for f in fields(Config)} == {field for field, _ in pairs}, (
        "a Config field has no counterpart in trainer.py — map it or justify it here")

    cfg = Config.from_env()
    for field, global_name in pairs:
        assert getattr(cfg, field) == getattr(trainer, global_name), f"{field} / {global_name}"
    assert cfg.numeric_dim == trainer.NUMERIC_DIM


def _sample_graphs(cfg, n):
    raw, demographics = _load_graphs(DATA, n)
    pairs = [(d, g) for g in raw
             if (d := to_data(g, demographics, cfg)).num_nodes >= 3 and d.edge_index.size(1) >= 4]
    return pairs


@requires_paper_checkpoint
@requires_data
def test_training_steps_match_trainer():
    """jepa_step and readout_step, bit-for-bit, against old_src.

    Neither is on the inference path, so the LOO gates say nothing about them —
    but they are ~90 lines of hand-transcribed tensor code and a drift here would
    only surface as a bad checkpoint after a full retrain. Both draw randomness,
    so each side is seeded identically before the call.
    """
    from torch_geometric.loader import DataLoader
    from paper_v16 import trainer as old

    cfg = Config()
    device = torch.device("cpu")
    graphs = [d for d, _ in _sample_graphs(cfg, 64)]
    batch = next(iter(DataLoader(graphs, batch_size=16, shuffle=False)))

    new_model, old_model = JEPA(cfg), old.JEPA()
    old_model.load_state_dict(new_model.state_dict())      # same weights, different class
    torch.manual_seed(0)
    new_loss, new_std = jepa_step(new_model, batch.clone(), device, cfg)
    torch.manual_seed(0)
    old_loss, old_std = old.jepa_step(old_model, batch.clone(), device)
    assert torch.equal(new_loss, old_loss) and torch.equal(new_std, old_std)

    encoder, scorer, _ = _load_paper_model(cfg)
    old_enc, old_sc = old.Encoder(), old.DistMult()
    old_enc.load_state_dict(encoder.state_dict())
    old_sc.load_state_dict(scorer.state_dict())
    gen_args = dict(gen=torch.Generator(device=device).manual_seed(7), mask_ratio=0.3)
    new_out = readout_step(encoder, scorer, batch.clone(), device, False, cfg, **gen_args)
    gen_args["gen"] = torch.Generator(device=device).manual_seed(7)
    old_out = old.readout_step(old_enc, old_sc, batch.clone(), device, False, **gen_args)
    for i in range(4):
        assert torch.equal(new_out[i], old_out[i]), f"readout_step return[{i}]"
    assert new_out[4][-1] == old_out[4][-1], "qsig"


@requires_paper_checkpoint
@requires_data
def test_batchmask_cascade_and_eir_match_trainer():
    """The three evaluators the LOO gates do not exercise."""
    from torch_geometric.loader import DataLoader
    from paper_v16 import trainer as old

    cfg = Config()
    device = torch.device("cpu")
    pairs = _sample_graphs(cfg, 64)
    graphs = [d for d, _ in pairs]
    encoder, scorer, _ = _load_paper_model(cfg)
    old_enc, old_sc = old.Encoder(), old.DistMult()
    old_enc.load_state_dict(encoder.state_dict())
    old_sc.load_state_dict(scorer.state_dict())

    loader = lambda: DataLoader(graphs, batch_size=1, shuffle=False)
    assert evaluate(encoder, scorer, loader(), device, cfg) == \
        old.evaluate(old_enc, old_sc, loader(), device)

    order = [resolve_rel(r) for r in cfg.cascade_order]
    assert cascade_evaluate(encoder, scorer, graphs, order, device, cfg) == \
        old.cascade_evaluate(old_enc, old_sc, graphs, order, device)

    assert eir_uplift_eval(encoder, scorer, pairs, 0.5, device, cfg) == \
        old.eir_uplift_eval(old_enc, old_sc, pairs, 0.5, device)


@requires_data
def test_to_data_matches_trainer():
    """The data path, elementwise, over every record in the shipped dataset.

    Uses the ambient environment for both sides for the same reason as above.
    """
    from paper_v16 import trainer

    cfg = Config.from_env()
    raw, demographics = _load_graphs(DATA, None)
    for i, graph in enumerate(raw):
        new, old = to_data(graph, demographics, cfg), trainer.to_data(graph, demographics)
        assert set(new.keys()) == set(old.keys()), i
        for key in sorted(new.keys()):
            a, b = new[key], old[key]
            if torch.is_tensor(a):
                assert a.dtype == b.dtype and torch.equal(a, b), f"record {i}, {key}"
            else:
                assert a == b, f"record {i}, {key}: {a!r} != {b!r}"
