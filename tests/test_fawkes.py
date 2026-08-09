"""Phase 5 gates — `fawkes`, the paper implementation.

`docs/RESTRUCTURE_PLAN.md` calls this the highest-risk phase: `paper_v16/trainer.py`
is the one artifact that backs the paper, and it was split five ways. Every test
here is an equality check against a `baseline/*.json` recorded from that trainer
— the LOO metrics and `state_dict` keys in Phase 0, everything else in Phase 8
(`baseline/old_fawkes.json`) as the old tree was retired.

The three gates the plan names are `test_loo_reproduces_baseline`,
`test_state_dict_keys_match_baseline`, and `test_no_module_scope_environment_reads`.
The rest localize a failure: if `to_data` drifts, the LOO gate fails too, but the
`to_data` comparison says which of the five modules moved.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import (BASELINE, DATA, ROOT, assert_digests_match, assert_state_dict_equal,
                      digest_data, digest_fields, fold_digests, load_pin,
                      requires_data)

from fawkes.config import Config
from fawkes.data import resolve_rel, to_data
from fawkes.evaluate import (_classification_metrics, _load_graphs, cascade_evaluate,
                             eir_uplift_eval, evaluate, loo_evaluate)
from fawkes.model import JEPA, DistMult, Encoder
from fawkes.patches import balanced_bfs_partition, patch_mask
from fawkes.train import _split_items, jepa_step, readout_step, train_readout

PIN = "old_fawkes.json"
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
    idx = np.random.RandomState(cfg.data_split_seed).permutation(len(items))
    test = [items[i] for i in idx[:int(cfg.test_frac * len(items))]]
    metrics = loo_evaluate(encoder, scorer, test, torch.device("cpu"), cfg)

    _assert_metrics_equal(metrics, json.loads((BASELINE / "paper_loo_testsplit.json").read_text()))
    # ...and against the metrics the checkpoint carries internally.
    embedded = checkpoint["recovery_test_loo"]
    assert metrics["n"] == embedded["n"]
    for key in ("mrr", "hits1", "hits3", "hits10"):
        assert abs(metrics[key] - embedded[key]) <= TOL


def test_split_follows_data_split_seed_not_seed():
    """(v20) The split must move with ``data_split_seed`` and hold under ``seed``.

    The Table 1 variance study varies the two independently. If ``_split_items``
    still read ``cfg.seed``, every "same patients, new initialisation" run would
    silently resample the test set, which would make the per-split comparison a
    comparison of nothing and would put unpaired runs in a paired column.
    """
    items = list(range(500))
    _, _, test = _split_items(items, Config())

    assert _split_items(items, Config(seed=7))[2] == test, "changing seed moved the split"
    assert _split_items(items, Config(data_split_seed=43))[2] != test, "data_split_seed did not move the split"
    # The defaults still describe the published split, whichever field carries it.
    assert _split_items(items, Config(data_split_seed=42))[2] == test


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


# ---- Against the recorded trainer: localize a failure to one module ----

@pytest.mark.parametrize("environment", ["defaults", "non_default"])
def test_config_from_env_matches_trainer_globals(environment):
    """Every env var the trainer read is a Config field with the same value.

    The trainer read ~30 of them at import, so one process could only ever
    observe one configuration -- which is why the recorded globals come from a
    subprocess per environment and why ``from_env`` is read the same way here.
    Two environments rather than one: the retired differential read the ambient
    environment on both sides, so it held for any configuration, and a pin can
    only hold for the configurations recorded. ``non_default`` is the one that
    keeps the int/float/bool/lowercase/comma-list parsing rules under test.
    """
    pinned = load_pin(PIN)["trainer_globals"]
    field_to_global = pinned["field_to_global"]
    assert {f.name for f in fields(Config)} == set(field_to_global), (
        "a Config field has no counterpart in trainer.py — map it or justify it here")

    entry = pinned["environments"][environment]
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src"),
           **entry["env"]}
    result = subprocess.run(
        [sys.executable, "-c",
         "import json, sys\n"
         "from dataclasses import asdict\n"
         "from fawkes.config import Config\n"
         "cfg = Config.from_env()\n"
         "print(json.dumps({'globals': asdict(cfg), 'numeric_dim': cfg.numeric_dim}))\n"],
        env=env, capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    produced = json.loads(result.stdout)

    for field in sorted(field_to_global):
        assert produced["globals"][field] == entry["globals"][field], (
            f"{field} / {field_to_global[field]}")
    assert produced["numeric_dim"] == entry["numeric_dim"]

    # Config() and an unset environment describe the same experiment -- the
    # class docstring's claim, which nothing else checks.
    if environment == "defaults":
        assert asdict(Config()) == entry["globals"]


def _sample_graphs(cfg, n):
    raw, demographics = _load_graphs(DATA, n)
    pairs = [(d, g) for g in raw
             if (d := to_data(g, demographics, cfg)).num_nodes >= 3 and d.edge_index.size(1) >= 4]
    return pairs


@requires_paper_checkpoint
@requires_data
def test_training_steps_match_trainer():
    """jepa_step and readout_step, bit-for-bit, against the recorded trainer.

    Neither is on the inference path, so the LOO gates say nothing about them —
    but they are ~90 lines of hand-transcribed tensor code and a drift here would
    only surface as a bad checkpoint after a full retrain. Both draw randomness,
    so both sides were seeded identically before the call, and `JEPA`'s
    initialisation comes off `model_init_seed` rather than being unseeded — the
    recorder copied exactly these weights into the old class.
    """
    from torch_geometric.loader import DataLoader

    pinned = load_pin(PIN)["training_steps"]
    cfg = Config()
    device = torch.device("cpu")
    graphs = [d for d, _ in _sample_graphs(cfg, 64)]
    batch = next(iter(DataLoader(graphs, batch_size=16, shuffle=False)))

    torch.manual_seed(pinned["model_init_seed"])
    model = JEPA(cfg)
    torch.manual_seed(0)
    loss, std = jepa_step(model, batch.clone(), device, cfg)

    encoder, scorer, _ = _load_paper_model(cfg)
    out = readout_step(
        encoder, scorer, batch.clone(), device, False, cfg,
        gen=torch.Generator(device=device).manual_seed(7), mask_ratio=0.3,
    )
    assert_digests_match(
        digest_fields({
            "jepa_step.loss": loss,
            "jepa_step.emb_std": std,
            **{f"readout_step[{i}]": out[i] for i in range(4)},
        }),
        pinned["tensors"],
    )
    assert out[4][-1] == pinned["readout_step.qsig"], "qsig"


@requires_paper_checkpoint
@requires_data
def test_batchmask_cascade_and_eir_match_trainer():
    """The three evaluators the LOO gates do not exercise."""
    from torch_geometric.loader import DataLoader

    pinned = load_pin(PIN)["evaluators"]
    cfg = Config()
    device = torch.device("cpu")
    pairs = _sample_graphs(cfg, 64)
    graphs = [d for d, _ in pairs]
    encoder, scorer, _ = _load_paper_model(cfg)

    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    # `auc_nonobvious` is the one value this repo deliberately moved after the pin was
    # recorded: the trainer scored non-PATIENT positives against the *whole* negative
    # pool, ~68% of which belongs to patient-anchored edges. The pin keeps the inflated
    # number because its job is to record what the old code did, so the corrected metric
    # is asserted separately — and every other key must still match the trainer exactly,
    # which is what shows the fix is confined to the negative-pool selection.
    metrics = evaluate(encoder, scorer, loader, device, cfg)
    corrected = metrics.pop("auc_nonobvious")
    assert metrics == {k: v for k, v in pinned["evaluate"].items() if k != "auc_nonobvious"}
    assert corrected < pinned["evaluate"]["auc_nonobvious"], (
        "the matched pool should be harder than the pool padded with patient-anchored "
        "negatives; if this ever fails the inflation argument needs re-checking")

    order = [resolve_rel(r) for r in cfg.cascade_order]
    assert cascade_evaluate(encoder, scorer, graphs, order, device, cfg) == \
        pinned["cascade_evaluate"]

    assert eir_uplift_eval(encoder, scorer, pairs, 0.5, device, cfg) == \
        pinned["eir_uplift_eval"]


@requires_data
def test_to_data_matches_trainer():
    """The data path, over every record in the shipped dataset.

    The pin is a per-field digest folded per record, plus one digest per field
    over all 4,000, so a failure names the record *and* the field rather than
    only reporting that something in a 234 MB conversion moved. Tracking the
    tensors themselves is not an option: ``numfeat`` alone is (nodes x 774) per
    record.

    ``Config()`` rather than ``Config.from_env()``: the pin is the released
    configuration, and ``test_config_from_env_matches_trainer_globals`` is what
    covers ``from_env``'s parsing. Reading the ambient environment here would
    make this fail for a stray ``USE_NOTE=0`` rather than for a real drift.
    """
    pinned = load_pin(PIN)["to_data"]
    cfg = Config()
    raw, demographics = _load_graphs(DATA, None)
    assert len(raw) == pinned["records"]

    per_key = {key: [] for key in pinned["keys"]}
    for index, graph in enumerate(raw):
        fields_ = digest_data(to_data(graph, demographics, cfg))
        assert sorted(fields_) == pinned["keys"], index
        for key, value in fields_.items():
            per_key[key].append(value)
        assert fold_digests(fields_) == pinned["per_record"][index], f"record {index}"

    # Named per field too, so a systematic drift reads as "numfeat moved"
    # instead of "record 0 moved, and so did the other 3,999".
    assert {key: fold_digests(values) for key, values in per_key.items()} == pinned["per_key"]


# ---- (v17) BFS-balanced patch masking ----

def _path_graph_edges(n, offset=0):
    src = torch.arange(n - 1) + offset
    return torch.stack([src, src + 1])


def test_mask_strategy_validated():
    """Unknown strategies and a context-less patch count fail loud at config time."""
    with pytest.raises(ValueError):
        Config(mask_strategy="bogus")
    with pytest.raises(ValueError):
        Config(mask_strategy="patch", jepa_patches=1)


def test_balanced_bfs_partition_covers_and_balances():
    """Every node lands in exactly one patch; on a path graph the patches are
    balanced and contiguous (contiguity on a path == connectivity)."""
    torch.manual_seed(0)
    patches = balanced_bfs_partition(_path_graph_edges(12), 12, 4)
    assert sorted(node for patch in patches for node in patch) == list(range(12))
    assert sorted(len(patch) for patch in patches) == [3, 3, 3, 3]
    for patch in patches:
        assert max(patch) - min(patch) + 1 == len(patch), f"patch {patch} not contiguous"


def test_patch_mask_masks_whole_local_patches():
    """Per graph in the batch: >= 1 target, >= 1 context, the node_mask fraction
    reached, and masked nodes coherent (every masked node has a masked neighbor)
    rather than the scattered singletons a Bernoulli mask draws. Both node
    counts divide evenly into the 4 patches — a remainder (e.g. 9 nodes) can
    legitimately produce a size-1 patch, for which the neighbor check is false."""
    torch.manual_seed(0)
    n1, n2 = 12, 8
    edges = torch.cat([_path_graph_edges(n1), _path_graph_edges(n2, offset=n1)], dim=1)
    ptr = torch.tensor([0, n1, n1 + n2])
    mask = patch_mask(ptr, edges, 0.4, 4, torch.device("cpu"))
    for start, end in ((0, n1), (n1, n1 + n2)):
        masked = int(mask[start:end].sum())
        assert 0.4 * (end - start) <= masked < end - start
    src, dst = edges[0], edges[1]
    both_masked = mask[src] & mask[dst]
    has_masked_neighbor = torch.zeros_like(mask)
    has_masked_neighbor[src[both_masked]] = True
    has_masked_neighbor[dst[both_masked]] = True
    assert bool(has_masked_neighbor[mask].all()), "a masked node has no masked neighbor"


@requires_data
def test_jepa_step_runs_with_patch_masking():
    """The patch strategy composes with the unchanged jepa_step on real graphs."""
    from torch_geometric.loader import DataLoader

    cfg = Config(mask_strategy="patch", jepa_patches=4)
    graphs = [d for d, _ in _sample_graphs(cfg, 16)]
    batch = next(iter(DataLoader(graphs, batch_size=8, shuffle=False)))
    torch.manual_seed(0)
    model = JEPA(cfg)
    loss, emb_std = jepa_step(model, batch, torch.device("cpu"), cfg)
    assert torch.isfinite(loss) and float(emb_std) > 0


# ---- run_config: a checkpoint records the knobs its experiment turned ----

def test_run_config_carries_every_knob_checkpoint_dict_omits():
    """The two blocks together must cover the fields an experiment varies.

    `checkpoint_dict` is pinned against the released file and cannot grow, so
    every knob turned since v18 went unrecorded — v19's arms C and Cp serialised
    identically, and a v21 MLP checkpoint could not be told from a v20 DistMult
    one. If a future experiment adds a field to `RUN_FIELDS` this stays true; if
    it turns a knob that is in neither block, this is the test that should have
    caught it.
    """
    varied = {"jepa_epochs", "readout_epochs", "freeze_encoder", "mask_strategy",
              "jepa_patches", "decoder", "data_split_seed"}
    recorded = set(Config().checkpoint_dict()) | set(Config().run_dict())
    assert varied <= recorded


def test_run_config_round_trips_through_from_checkpoint():
    """A non-default arm must rebuild from its own checkpoint, without the environment.

    The `decoder` case is the one that bit: loading a v21 MLP checkpoint without
    DECODER=mlp fails a strict state_dict load on the `mlp.0.*` keys.
    """
    arm = Config(decoder="mlp", mask_strategy="patch", jepa_patches=8,
                 freeze_encoder=False, jepa_epochs=0, data_split_seed=43)
    rebuilt = Config.from_checkpoint(arm.checkpoint_dict(), arm.run_dict())

    for field in ("decoder", "mask_strategy", "jepa_patches", "freeze_encoder",
                  "jepa_epochs", "data_split_seed"):
        assert getattr(rebuilt, field) == getattr(arm, field), field


def test_from_checkpoint_without_run_config_is_unchanged():
    """Released checkpoints carry no run_config; they must load exactly as before."""
    released = Config().checkpoint_dict()
    assert Config.from_checkpoint(released) == Config.from_checkpoint(released, None)
    assert Config.from_checkpoint(released).decoder == Config().decoder


# ---- auc_nonobvious scores against a matched negative pool ----

def test_nonobvious_auc_uses_its_own_negatives():
    """auc_nonobvious must score non-PATIENT positives against the negatives drawn
    FOR those positives, not against the whole pool.

    `readout_step` returns `neg_scores[:, 0]` — one negative per positive, in the
    same order — so `nonpat_mask` indexes the negatives exactly as it indexes the
    positives. Filtering only the positives leaves the negatives of the
    patient-anchored edges in the comparison, and those are the easy ones the
    metric exists to exclude, so it reports a discrimination the model never made.

    Here the six non-patient edges are near chance while the four patient-anchored
    negatives sit far below every positive. A metric that leaves them in cannot
    help but look good.
    """
    positives = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0])
    negatives = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, -10.0, -11.0, -12.0, -13.0])
    nonpatient = torch.tensor([True] * 6 + [False] * 4)

    _auc, _ap, auc_nonobvious = _classification_metrics([positives], [negatives], [nonpatient])

    from sklearn.metrics import roc_auc_score
    matched = roc_auc_score(np.concatenate([np.ones(6), np.zeros(6)]),
                            np.concatenate([positives[:6].numpy(), negatives[:6].numpy()]))
    unfiltered = roc_auc_score(np.concatenate([np.ones(6), np.zeros(10)]),
                               np.concatenate([positives[:6].numpy(), negatives.numpy()]))

    assert auc_nonobvious == pytest.approx(matched, abs=1e-12)
    assert unfiltered > matched, (
        "this fixture no longer separates the two computations, so it cannot gate "
        "the bug — widen the gap between the patient-anchored negatives and the rest")


# ---- best-VAL selection when the encoder is not frozen ----

@requires_data
def test_readout_selection_rolls_back_the_encoder_too(monkeypatch):
    """train_readout must return the pair that actually scored best on VAL.

    Under `freeze_encoder` the encoder never moves, so snapshotting the scorer
    alone captures the whole model — which is why the paper runs are unaffected.
    Without it the encoder keeps training past the selected epoch, and returning
    the best-VAL scorer beside the final-epoch encoder is a combination that
    existed at no epoch. That under-reports exactly the ablation arms that train
    the encoder from scratch, so the baselines the JEPA claim is measured against
    would be the ones handicapped.
    """
    from torch_geometric.loader import DataLoader
    import fawkes.train as train_module

    cfg = Config(jepa_epochs=0, readout_epochs=20, freeze_encoder=False, push=False)
    graphs = [d for d, _ in _sample_graphs(cfg, 64)]
    train_graphs, val_graphs = graphs[:48], graphs[48:56]
    device = torch.device("cpu")

    val_aucs = []
    inner_evaluate = train_module.evaluate

    def recording_evaluate(*args, **kwargs):
        metrics = inner_evaluate(*args, **kwargs)
        val_aucs.append(metrics["auc"])
        return metrics

    monkeypatch.setattr(train_module, "evaluate", recording_evaluate)

    torch.manual_seed(0)
    encoder, scorer = train_readout(JEPA(cfg), train_graphs, val_graphs, device, cfg)

    assert max(val_aucs) > val_aucs[-1], (
        "VAL AUC peaked at the final epoch, so this run cannot tell a rolled-back "
        "encoder from a final-epoch one — re-pin the seed or the epoch count")
    selected = evaluate(encoder, scorer, DataLoader(val_graphs, batch_size=1, shuffle=False), device, cfg)
    assert selected["auc"] == pytest.approx(max(val_aucs), abs=1e-9)
