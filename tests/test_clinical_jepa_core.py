"""Phase 1 and 2 gates for the merged ``clinical_jepa`` core.

Every assertion here is differential: the new implementation and the old one run
in the same process on the same input and must agree exactly. ``old_src`` is
removed in Phase 8, at which point the checkpoint-key gates fall back to
``baseline/*_keys.json`` (which they already compare against today).

Test data is the seeded synthetic graph set — the same population the Phase 0
baselines were recorded over (``baseline/README.md``). It needs no data file:
``--jsonl-path``'s default does not exist on disk, and synthetic generation is
deterministic in ``TrainConfig.seed``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.loader import DataLoader

from conftest import CKPT, assert_pyg_equal, assert_state_dict_equal, requires_checkpoints

from clinical_jepa.config import Config, TrainConfig
from clinical_jepa.encoders import MockEncoder
from clinical_jepa.graph.builders import JsonlGraphBuilder, SyntheticGraphGenerator
from clinical_jepa.graph.patches import build_patch_data, sample_patch_task
from clinical_jepa.graph.tensors import to_graph_data
from clinical_jepa.model import GraphJEPA
from clinical_jepa.schema import PatientGraph

from fawkes_core.data import JsonlGraphBuilder as OldJsonlGraphBuilder
from graph_jepa_v5 import data as old_v5_data
from graph_jepa_v5.config import Config as OldV5Config
from graph_jepa_v5.model import GraphJEPAv5 as OldGraphJEPAv5
from graph_jepa_v6 import data as old_v6_data
from graph_jepa_v6.config import Config as OldV6Config
from graph_jepa_v6.model import GraphJEPAv6 as OldGraphJEPAv6

BASE_IN_DIM = 768  # models/clinical-jepa-no-note/config.json::model.in_dim
NOTE_DIM = 768  # models/clinical-jepa-localized-note/config.json::model.note_embedding_dim

V5_CONFIG = CKPT / "clinical-jepa-no-note/config.json"
V6_CONFIG = CKPT / "clinical-jepa-localized-note/config.json"
V5_CHECKPOINT = CKPT / "clinical-jepa-no-note/graph_jepa_v5.pt"
V6_CHECKPOINT = CKPT / "clinical-jepa-localized-note/graph_jepa_v6.pt"
BASELINE = Path(__file__).resolve().parents[1] / "baseline"

# The four ModelConfig fields the v5 lineage never had. Plan section 2.1: these
# plus the note-append branch are the entire substantive v5/v6 difference.
V6_ONLY_MODEL_FIELDS = (
    "base_in_dim",
    "use_note_embeddings",
    "note_embedding_dim",
    "note_ground_by",
)


def assert_graph_data_equal(new, old, path=""):
    """``assert_pyg_equal``, made NaN-aware for ``edge_llm_confidence``.

    ``torch.equal`` is False for any tensor holding NaN, and
    ``edge_llm_confidence`` is NaN for every edge that carries no LLM
    confidence -- which is every edge of a synthetic graph. So conftest's helper
    cannot compare these objects at all. Asserting that the NaN mask matches and
    that every non-NaN element is bit-identical is strictly stronger than
    ``torch.equal`` would be, not weaker. Every other field still goes through
    the shared helper unchanged.
    """
    a, b = new.edge_llm_confidence, old.edge_llm_confidence
    nan_a, nan_b = a.isnan(), b.isnan()
    assert torch.equal(nan_a, nan_b), f"{path}.edge_llm_confidence: NaN mask differs"
    assert torch.equal(a[~nan_a], b[~nan_b]), f"{path}.edge_llm_confidence: value mismatch"

    stripped_new, stripped_old = copy.copy(new), copy.copy(old)
    del stripped_new.edge_llm_confidence
    del stripped_old.edge_llm_confidence
    assert_pyg_equal(stripped_new, stripped_old, path)


def synthetic_graphs() -> list[PatientGraph]:
    """The graph population the Phase 0 baselines were recorded over."""
    train = TrainConfig()
    generator = SyntheticGraphGenerator(
        seed=train.seed,
        min_nodes=train.synthetic_min_nodes,
        max_nodes=train.synthetic_max_nodes,
    )
    return generator.generate_many(train.synthetic_graphs)


# --------------------------------------------------------------------------- #
# Phase 1 gate: one to_graph_data reproduces both released conversions.
# --------------------------------------------------------------------------- #
def test_tensors_match_old_v5_and_v6():
    """Not marked ``requires_checkpoints``: seeded synthetic graphs and the mock
    encoder need no artifact on disk, so this gate runs on a clean clone."""
    graphs = synthetic_graphs()
    assert len(graphs) == 256

    encoder = MockEncoder(dim=BASE_IN_DIM)
    for old_module, use_note in ((old_v5_data, False), (old_v6_data, True)):
        for index, graph in enumerate(graphs):
            new = to_graph_data(
                graph,
                encoder,
                use_note_embeddings=use_note,
                note_embedding_dim=NOTE_DIM,
            )
            old = old_module.to_graph_data(graph, encoder)
            assert_graph_data_equal(new, old, path=f"{old_module.__name__}[{index}]")


@pytest.mark.parametrize("config_path", [V5_CONFIG, V6_CONFIG])
def test_config_from_dict_matches_old_v6(config_path):
    """The merged Config is v6's, which already loaded both released sidecars.

    Plan section 2.1: v6's ``from_dict`` defaults ``use_note_embeddings`` to
    False and derives ``base_in_dim`` when absent, so one class reads both
    checkpoints. Those defaults are kept verbatim, and the round-trip is exact
    for both files.
    """
    raw = json.loads(config_path.read_text())
    new = Config.from_dict(raw).to_dict()

    assert new == OldV6Config.from_dict(raw).to_dict()

    # to_dict is the checkpoint payload, so it has to survive a reload.
    assert Config.from_dict(new).to_dict() == new


def test_config_matches_old_v5_where_old_v5_could_read_it():
    """The v5/v6 config asymmetry, pinned rather than assumed.

    v6's Config reads both sidecars; v5's reads only its own -- its ModelConfig
    has no ``base_in_dim``. So the merged class can only be compared against old
    v5 on the no-note ``config.json``, and there it agrees on every field v5
    had. The
    difference is exactly the four ModelConfig note fields, which is what plan
    section 2.1 predicts and nothing more.
    """
    raw = json.loads(V5_CONFIG.read_text())
    new = Config.from_dict(raw).to_dict()
    old_v5 = OldV5Config.from_dict(raw).to_dict()

    assert set(new["model"]) - set(old_v5["model"]) == set(V6_ONLY_MODEL_FIELDS)
    shared = dict(new)
    shared["model"] = {
        key: value
        for key, value in new["model"].items()
        if key not in V6_ONLY_MODEL_FIELDS
    }
    assert shared == old_v5

    with pytest.raises(TypeError):
        OldV5Config.from_dict(json.loads(V6_CONFIG.read_text()))


# --------------------------------------------------------------------------- #
# The JSONL builder. Carried over from the pre-restructure ``test_suite.py``
# (Phase 7 split it into per-module files) and strengthened into a differential:
# the original only asserted the three metadata fields, which cannot notice the
# builder dropping or reshaping anything else on the record.
# --------------------------------------------------------------------------- #
def test_jsonl_builder_preserves_graph_metadata(tmp_path):
    """Admission metadata, the note vector and provenance labels survive the load.

    These three are what the localized-note variant depends on: without
    ``note_embedding`` in ``extra`` and ``prov_in_note`` on the edge, the note
    vector has nothing to ground onto and the variant silently degrades to
    all-zero note features.
    """
    path = tmp_path / "graphs.jsonl"
    record = {
        "subject_id": 7,
        "note_embedding": [0.1, 0.2],
        "nodes": [
            {"id": "P", "type": "PATIENT", "name": "patient"},
            {"id": "D", "type": "DIAGNOSIS", "name": "pneumonia"},
        ],
        "edges": [
            {
                "source": "P",
                "target": "D",
                "relation": "HAS_DIAGNOSIS",
                "labels": {"prov_in_note": 1},
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    graph = JsonlGraphBuilder(path).build()[0]
    assert graph.extra["subject_id"] == 7
    assert graph.extra["note_embedding"] == [0.1, 0.2]
    assert graph.edges[0]["labels"]["prov_in_note"] == 1

    # ...and the whole record, against the builder this one replaces.
    old = OldJsonlGraphBuilder(path).build()[0]
    assert graph.nodes == old.nodes
    assert graph.edges == old.edges
    assert graph.extra == old.extra


# --------------------------------------------------------------------------- #
# Plan section 2.6: endpoint-alias normalization. This is a deliberate behaviour
# change for the v5 variant, not a move -- see clinical_jepa/graph/builders.py.
# --------------------------------------------------------------------------- #
def _alias_fixture(source_key: str, target_key: str) -> PatientGraph:
    """One graph keyed by the requested endpoint spelling.

    Deliberately targeted rather than read from ``data/``: it exercises the
    mixed case no shipped file contains, and it keeps these tests runnable on a
    clean clone.
    """
    return PatientGraph(
        nodes=[
            {"id": "P", "type": "PATIENT", "text": "patient"},
            {"id": "D", "type": "DIAGNOSIS", "name": "pneumonia"},  # text absent
            {"id": "M", "type": "MEDICATION", "text": "amoxicillin"},
        ],
        edges=[
            {
                source_key: "P",
                target_key: "D",
                "type": "HAS_DIAGNOSIS",
                "labels": {"prov_in_note": 1},
            },
            {source_key: "D", target_key: "M", "type": "TREATED_BY"},
        ],
        extra={"subject_id": 7, "note_embedding": [0.25, -0.5, 0.75, 1.0]},
    )


ALIAS_KEY_STYLES = {
    "canonical": ("source_id", "target_id"),
    "aliased": ("source", "target"),
    "mixed": ("source", "target_id"),
}


@pytest.mark.parametrize("style", sorted(ALIAS_KEY_STYLES))
def test_alias_keyed_edges_are_normalized(style):
    """All three endpoint spellings produce the same tensors, and match old v6.

    v6 already normalized (``graph_jepa_v6/data.py:105``); v5 did not. Both
    variants now behave like v6 -- see ``test_v5_used_to_reject_alias_keys``
    for the behaviour this replaces.
    """
    encoder = MockEncoder(dim=8)
    source_key, target_key = ALIAS_KEY_STYLES[style]
    graph = _alias_fixture(source_key, target_key)
    canonical = _alias_fixture(*ALIAS_KEY_STYLES["canonical"])

    for use_note in (False, True):
        kwargs = {"use_note_embeddings": use_note, "note_embedding_dim": 4}
        new = to_graph_data(graph, encoder, **kwargs)
        assert_graph_data_equal(
            new, to_graph_data(canonical, encoder, **kwargs), path=style
        )
        assert_graph_data_equal(
            new,
            old_v6_data.to_graph_data(graph, encoder, **kwargs),
            path=f"{style}/old_v6",
        )

    # The note vector reaches both endpoints of the prov_in_note edge whichever
    # way that edge spells them.
    noted = to_graph_data(graph, encoder, use_note_embeddings=True, note_embedding_dim=4)
    assert noted.note_grounded_mask.tolist() == [True, True, False]
    assert bool(noted.note_embedding_present)

    # Node text falls back to `name` when absent, so D is not encoded as "".
    assert not torch.equal(noted.x[1, :8], torch.zeros(8))


def test_v5_used_to_reject_alias_keys():
    """Pins the divergence this phase removes rather than assuming it.

    ``graph_jepa_v5/data.py:75`` indexes ``edge["source_id"]`` unguarded, so the
    no-note variant raised ``KeyError`` on the endpoint spelling used by the
    only dataset shipped in this repository. The merged converter does not.
    """
    encoder = MockEncoder(dim=8)
    graph = _alias_fixture("source", "target")

    with pytest.raises(KeyError):
        old_v5_data.to_graph_data(graph, encoder)

    to_graph_data(graph, encoder, use_note_embeddings=False)


# --------------------------------------------------------------------------- #
# Phase 2 gate 1: state_dict key sets. Catches a renamed attribute path.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("config_path", "keys_name"),
    [(V5_CONFIG, "v5_keys.json"), (V6_CONFIG, "v6_keys.json")],
)
def test_state_dict_keys_match_baseline(config_path, keys_name):
    """Key-set equality only. The two baselines are identical -- 119 keys each --
    so this is blind to the variant difference; ``test_released_checkpoints_load``
    is what checks shapes."""
    cfg = Config.from_dict(json.loads(config_path.read_text()))
    model = GraphJEPA(cfg.model)
    assert_state_dict_equal(model.state_dict(), BASELINE / keys_name, section="state_dict")


# --------------------------------------------------------------------------- #
# Phase 2 gate 2: strict load. This is what catches a wrong variant.
# --------------------------------------------------------------------------- #
@requires_checkpoints
@pytest.mark.parametrize(
    ("checkpoint_path", "expected_in_dim"),
    [(V5_CHECKPOINT, 768), (V6_CHECKPOINT, 1536)],
)
def test_released_checkpoints_load_strict(checkpoint_path, expected_in_dim):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])
    model = GraphJEPA(cfg.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert model.context_node_encoder.input_proj.in_features == expected_in_dim
    assert model.target_node_encoder.input_proj.in_features == expected_in_dim


# --------------------------------------------------------------------------- #
# Phase 2 gate 3: bit-identical forward pass against the old classes.
# --------------------------------------------------------------------------- #
def _batch(cfg, encoder):
    graphs = synthetic_graphs()[:8]
    data = [
        to_graph_data(
            graph,
            encoder,
            use_note_embeddings=cfg.model.use_note_embeddings,
            note_embedding_dim=cfg.model.note_embedding_dim,
            note_ground_by=cfg.model.note_ground_by,
        )
        for graph in graphs
    ]
    return next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))


def _patch_inputs(cfg, batch):
    generator = torch.Generator().manual_seed(0)
    patch_data = build_patch_data(
        batch,
        num_patches=cfg.model.num_patches,
        patch_pe_dim=cfg.model.patch_pe_dim,
        generator=generator,
    )
    task = sample_patch_task(
        patch_data,
        context_patches=cfg.train.context_patches,
        target_patches=cfg.train.target_patches,
        generator=generator,
    )
    return patch_data, task


@requires_checkpoints
@pytest.mark.parametrize(
    ("checkpoint_path", "old_config_cls", "old_model_cls"),
    [
        (V5_CHECKPOINT, OldV5Config, OldGraphJEPAv5),
        (V6_CHECKPOINT, OldV6Config, OldGraphJEPAv6),
    ],
)
def test_forward_matches_old_model(checkpoint_path, old_config_cls, old_model_cls):
    """Released weights, one seeded batch, every output tensor compared exactly.

    Each loss is re-seeded immediately before the call because the revision and
    ranking objectives draw their masks and negatives from the global RNG.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])
    old_cfg = old_config_cls.from_dict(checkpoint["config"])

    new_model = GraphJEPA(cfg.model)
    new_model.load_state_dict(checkpoint["state_dict"], strict=True)
    new_model.eval()
    old_model = old_model_cls(old_cfg.model)
    old_model.load_state_dict(checkpoint["state_dict"], strict=True)
    old_model.eval()

    batch = _batch(cfg, MockEncoder(dim=cfg.model.base_in_dim))
    assert batch.x.size(1) == cfg.model.in_dim
    patch_data, task = _patch_inputs(cfg, batch)

    assert torch.equal(new_model.encode_nodes(batch), old_model.encode_nodes(batch))
    assert torch.equal(
        new_model.encode_target_nodes(batch), old_model.encode_target_nodes(batch)
    )

    target_patch_idx = task.target_idx[:1]
    assert torch.equal(
        new_model.patch_prediction_energy(batch, patch_data, target_patch_idx),
        old_model.patch_prediction_energy(batch, patch_data, target_patch_idx),
    )

    def paired(call):
        torch.manual_seed(1234)
        new_loss, new_log = call(new_model)
        torch.manual_seed(1234)
        old_loss, old_log = call(old_model)
        assert torch.equal(new_loss, old_loss), f"{new_loss} != {old_loss}"
        assert new_log == old_log

    paired(
        lambda model: model.jepa_loss(
            batch,
            patch_data,
            task,
            var_weight=cfg.train.vicreg_var_weight,
            cov_weight=cfg.train.vicreg_cov_weight,
        )
    )
    paired(
        lambda model: model.revision_loss(
            batch,
            mask_ratio=cfg.train.revision_mask_ratio,
            neg_per_pos=cfg.train.revision_neg_per_pos,
            llm_confidence_negatives=cfg.train.llm_confidence_negatives,
            llm_negative_threshold=cfg.train.llm_negative_threshold,
            llm_positive_threshold=cfg.train.llm_positive_threshold,
            llm_negative_threshold_by_relation=cfg.train.llm_negative_threshold_by_relation,
            llm_positive_threshold_by_relation=cfg.train.llm_positive_threshold_by_relation,
            llm_negative_weight=cfg.train.llm_negative_weight,
            clinical_artifact_filters=cfg.train.clinical_artifact_filters,
        )
    )
    paired(
        lambda model: model.candidate_ranking_loss(
            batch,
            mask_ratio=cfg.train.ranking_mask_ratio,
            neg_per_pos=cfg.train.ranking_neg_per_pos,
            max_pos=cfg.train.ranking_max_pos,
            temperature=cfg.train.ranking_temperature,
        )
    )
