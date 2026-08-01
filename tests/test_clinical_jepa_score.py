"""Phase 4 gate for the merged ``clinical_jepa.score``.

The gate: scoring one fixed input graph with each released checkpoint produces
**byte-identical output JSON** before and after the merge, for the
KEEP/REVIEW/PRUNE path and for the candidate-addition path. The four files the
old code produced are tracked under
``baseline/old_clinical_jepa_score_output/``, so the comparison is still a byte
comparison; before Phase 8 the old side wrote them in the same process.

v5 and v6 genuinely diverge here, so the old side was recorded for *both* -- not
just the one the new code was modelled on. ``test_alias_keyed_input_...`` pins
one place they disagreed. The other, ``_install_v6_data_conversion``'s global
rebinding, is recorded in ``baseline/COVERAGE.md``: it could only be
demonstrated by running the old module, and what replaces it is
``test_variants_coexist_in_one_process``.

Two things about the recorded old side:

* **The old scorer drew its patch partition from the global RNG.**
  ``score_base.py:262`` passed ``generator=None`` to ``build_patch_data``, so
  the structural half of every score depended on how much randomness the process
  had already consumed. That is preserved, not fixed -- it is behaviour, not a
  move. It does mean a byte comparison is only well defined from a known seed,
  so every scoring test calls ``torch.manual_seed(SEED)`` immediately before
  scoring, and the recorder used the same seed. Same reason
  ``test_clinical_jepa_core.py`` re-seeds around each loss.

* **The old v6 path mutated ``fawkes_core`` globally.** The recorder restored
  the two module attributes after each localized-note run, or the no-note pins
  taken afterwards would have been scored with 1536-wide features.

Scoring uses ``MockEncoder`` rather than the SapBERT encoder both checkpoints
name in their config: it was the same encoder on both sides of every
comparison, it is deterministic, and it keeps these tests runnable offline.
``test_load_builds_the_checkpoint_encoder`` covers the real ``_load``.
"""

from __future__ import annotations

import json
import random

import pytest
import torch

from conftest import BASELINE, CKPT, DATA, load_pin, requires_checkpoints, requires_data

from clinical_jepa import score as new_score
from clinical_jepa.config import Config
from clinical_jepa.encoders import MockEncoder, SapBertNodeEncoder
from clinical_jepa.model import GraphJEPA
from clinical_jepa.schema import PatientGraph as NewPatientGraph

PIN = "old_clinical_jepa_score.json"
PINNED_OUTPUT = BASELINE / "old_clinical_jepa_score_output"

# Phase 7 renamed these directories; the checkpoint filenames did not change.
V5_CHECKPOINT = CKPT / "clinical-jepa-no-note/graph_jepa_v5.pt"
V6_CHECKPOINT = CKPT / "clinical-jepa-localized-note/graph_jepa_v6.pt"

SEED = 20260729

# Chosen against the released weights so the fixture actually reaches every
# revision action on both variants -- see ``_assert_not_vacuous``. Only the
# thresholds are chosen; nothing downstream of them is.
PRUNE_THRESHOLD = 0.07
CANDIDATE_THRESHOLD = 0.5
WEAK_OVERRIDES = ["INDICATES=0.10", "LOCATED_AT=0.05", "CONFIRMS=0.08"]

VARIANTS = [
    ("no_note", V5_CHECKPOINT),
    ("localized_note", V6_CHECKPOINT),
]
VARIANT_IDS = [name for name, _ in VARIANTS]


# --------------------------------------------------------------------------- #
# Fixture graph
# --------------------------------------------------------------------------- #
_NODES = [
    {"id": "N1", "text": "fever", "type": "SYMPTOM", "evidence": "reports fever", "turn_id": "t1"},
    {"id": "N2", "text": "productive cough", "type": "SYMPTOM", "evidence": "cough for 3 days", "turn_id": "t1"},
    {"id": "N3", "text": "pneumonia", "type": "DIAGNOSIS", "evidence": "assessment", "turn_id": "t3"},
    {"id": "N4", "text": "asthma", "type": "DIAGNOSIS", "evidence": "history", "turn_id": "t2"},
    {"id": "N5", "text": "amoxicillin", "type": "TREATMENT", "evidence": "prescribed", "turn_id": "t4"},
    {"id": "N6", "text": "chest x-ray", "type": "PROCEDURE", "evidence": "ordered", "turn_id": "t3"},
    {"id": "N7", "text": "lungs", "type": "LOCATION", "evidence": "chest exam", "turn_id": "t2"},
    {"id": "N8", "text": "smoking history", "type": "MEDICAL_HISTORY", "evidence": "20 pack years", "turn_id": "t1"},
    {"id": "N9", "text": "elevated wbc", "type": "LAB_RESULT", "evidence": "cbc", "turn_id": "t3"},
]

# (source, relation, target, prov_in_note). The last edge is schema-invalid on
# purpose -- there is no RELATION_SCHEMA rule for LOCATION --INDICATES--> -- so
# the run exercises the schema guard and produces a prunable zero score.
_EDGES = [
    ("N1", "INDICATES", "N3", True),
    ("N2", "INDICATES", "N3", False),
    ("N2", "LOCATED_AT", "N7", False),
    ("N5", "TAKEN_FOR", "N3", True),
    ("N6", "CONFIRMS", "N3", False),
    ("N6", "RULES_OUT", "N4", False),
    ("N8", "CAUSES", "N4", False),
    ("N9", "INDICATES", "N3", False),
    ("N3", "CAUSES", "N1", False),
    ("N7", "INDICATES", "N3", False),
]

# No node type here is MIMIC-only and there is no subject_id/hadm_ids, so this
# graph reaches _looks_like_mimic_subkg's alias clause and nothing else -- which
# is what makes the alias test below isolate exactly that clause.
_MIMIC_ONLY_TYPES = {"PATIENT", "MEDICATION", "LAB_TEST", "MICROBIOLOGY", "SERVICE"}
assert not {node["type"] for node in _NODES} & _MIMIC_ONLY_TYPES


def kg_dict(source_key: str = "source_id", target_key: str = "target_id") -> dict:
    """The fixture KG, with edge endpoints under the requested spelling."""
    rng = random.Random(11)
    edges = []
    for source, relation, target, prov in _EDGES:
        edge = {
            source_key: source,
            target_key: target,
            "type": relation,
            "evidence": f"{source}->{target}",
            "turn_id": "t1",
        }
        if prov:
            edge["labels"] = {"prov_in_note": 1}
        edges.append(edge)
    return {
        "nodes": [dict(node) for node in _NODES],
        "edges": edges,
        "note": "patient with fever and productive cough",
        # 768 = ModelConfig.note_embedding_dim; without it the localized-note
        # variant scores against an all-zero note and the two variants differ
        # for an uninteresting reason.
        "note_embedding": [round(rng.uniform(-1.0, 1.0), 6) for _ in range(768)],
    }


def write_kg(tmp_path, name: str = "kg.json", **kwargs):
    path = tmp_path / name
    path.write_text(json.dumps(kg_dict(**kwargs), indent=2))
    return path


# --------------------------------------------------------------------------- #
# Driving one scoring run. These are ``run()``'s CLI override block and per-file
# loop body: the merged module keeps every name the old CLI used, which is why
# the same lines drove the old side when it was still here.
# --------------------------------------------------------------------------- #
def configure(cfg):
    """The slice of ``run()``'s CLI override block this gate exercises."""
    cfg.score.prune_threshold = PRUNE_THRESHOLD
    cfg.score.candidate_threshold = CANDIDATE_THRESHOLD
    cfg.score.candidate_threshold_by_relation = {
        relation: CANDIDATE_THRESHOLD for relation in new_score.RELATIONS
    }
    new_score._update_relation_thresholds(
        cfg.score.weak_threshold_by_relation, WEAK_OVERRIDES
    )
    return cfg


def revise(graph, model, encoder, cfg, device, *, add_candidates):
    """``run()``'s per-file loop body, verbatim."""
    scores, flags = new_score.score_graph(graph, model, encoder, cfg, device)
    graph.annotate_edges(scores, flags)
    new_score._annotate_revision_actions(graph, cfg)

    added = 0
    if add_candidates:
        added = new_score.add_candidate_edges(graph, model, encoder, cfg, device)
        new_score._annotate_revision_actions(graph, cfg)

    pruned = 0
    if cfg.score.prune_threshold is not None:
        pruned = new_score._prune(graph, cfg.score.prune_threshold)
    return added, pruned


def score_with_new(checkpoint_path, kg_path, out_path, *, add_candidates):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = configure(Config.from_dict(checkpoint["config"]))
    model = GraphJEPA(cfg.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    torch.manual_seed(SEED)
    graph = new_score._load_graph_for_scoring(kg_path)
    new_score._validate_checkpoint_relation_capacity(graph, cfg)
    counts = revise(
        graph,
        model,
        MockEncoder(dim=cfg.model.base_in_dim),
        cfg,
        torch.device("cpu"),
        add_candidates=add_candidates,
    )
    graph.save(out_path)
    return graph, counts


def _assert_not_vacuous(graph, added, pruned, *, add_candidates):
    """Guard against a gate that passes because nothing happened."""
    actions = {edge["jepa_revision_action"] for edge in graph.edges}
    assert {"keep", "review"} <= actions, actions
    assert pruned > 0, "no edge reached the prune path"
    # The schema guard ran over every edge; the one schema-invalid edge scored
    # 0.0 and is exactly what the prune path removed.
    assert all("jepa_schema_valid" in edge for edge in graph.edges)
    if add_candidates:
        assert added > 0, "no candidate edge was added"
        assert "add_candidate" in actions


# --------------------------------------------------------------------------- #
# The gate.
# --------------------------------------------------------------------------- #
@requires_checkpoints
@pytest.mark.parametrize("path_name", ["revise", "add_candidates"])
@pytest.mark.parametrize(("name", "checkpoint_path"), VARIANTS, ids=VARIANT_IDS)
def test_scoring_output_is_byte_identical(tmp_path, name, checkpoint_path, path_name):
    """Phase 4 gate, run for both released variants and both output paths.

    Each of the four pinned files was written by the old scoring path this
    variant actually used: the no-note variant scored through ``fawkes_core``'s
    converter, the localized-note variant through the one v6 installed over it.
    Recording only the second would have left any change to the first invisible.
    """
    kg_path = write_kg(tmp_path)
    produced = tmp_path / f"{name}_{path_name}.json"
    add_candidates = path_name == "add_candidates"

    graph, counts = score_with_new(
        checkpoint_path, kg_path, produced, add_candidates=add_candidates
    )

    pinned = load_pin(PIN)["revision_counts"][name][path_name]
    assert list(counts) == pinned
    assert produced.read_bytes() == (PINNED_OUTPUT / f"{name}_{path_name}.json").read_bytes()
    _assert_not_vacuous(graph, *counts, add_candidates=add_candidates)


@requires_checkpoints
def test_the_two_variants_do_not_produce_the_same_output(tmp_path):
    """The gate above would also pass if the variant flag did nothing.

    The two released checkpoints are the two configurations of one GraphJEPA,
    and they are supposed to score differently. Pin that so a merge which
    ignored ``use_note_embeddings`` could not slip through.
    """
    kg_path = write_kg(tmp_path)
    outputs = []
    for name, checkpoint_path in VARIANTS:
        out = tmp_path / f"{name}.json"
        score_with_new(checkpoint_path, kg_path, out, add_candidates=True)
        outputs.append(out.read_bytes())
    assert outputs[0] != outputs[1]


# --------------------------------------------------------------------------- #
# Divergence 1 (plan section 2.4): the global rebinding, and its removal.
# ``test_old_v6_monkeypatch_made_the_variants_mutually_exclusive`` demonstrated
# the defect by triggering it -- installing ``_install_v6_data_conversion`` and
# watching the no-note scorer raise ``RuntimeError: mat1 and mat2 shapes cannot
# be multiplied``. That needed the old module to exist, so Phase 8 retired it
# (``baseline/COVERAGE.md``). The live property is below and needs nothing old.
# --------------------------------------------------------------------------- #
@requires_checkpoints
def test_variants_coexist_in_one_process(tmp_path):
    """What replaces the rebinding: the converter is chosen from ``cfg.model``.

    Score no-note, then localized-note, then no-note again. The first and third
    runs are byte-identical, so the localized-note run left nothing behind --
    which is exactly what the old rebinding could not promise.
    """
    kg_path = write_kg(tmp_path)
    first = tmp_path / "first.json"
    middle = tmp_path / "middle.json"
    third = tmp_path / "third.json"

    score_with_new(V5_CHECKPOINT, kg_path, first, add_candidates=True)
    score_with_new(V6_CHECKPOINT, kg_path, middle, add_candidates=True)
    score_with_new(V5_CHECKPOINT, kg_path, third, add_candidates=True)

    assert first.read_bytes() == third.read_bytes()
    assert middle.read_bytes() != first.read_bytes()


# --------------------------------------------------------------------------- #
# Divergence 2 (plan section 2.6): endpoint aliases in the scoring path.
# --------------------------------------------------------------------------- #
_MIMIC_STAMPS = ("_method", "_source_path", "_mimic_adapter")


def test_canonical_input_loads_identically(tmp_path):
    """Canonical-keyed input that also avoids the dropped clause is unchanged.

    This fixture keys endpoints ``source_id``/``target_id`` and the relation as
    ``type``, so none of the old third clause's three terms matched and both
    loaders skip the adapter. It does NOT show the change is scoped to
    alias-keyed input -- see
    :func:`test_relation_keyed_canonical_input_also_stops_being_adapted`.
    """
    kg_path = write_kg(tmp_path)
    old = load_pin(PIN)["loader"]["canonical_fixture"]
    new = new_score._load_graph_for_scoring(kg_path)

    assert new.nodes == old["nodes"]
    assert new.edges == old["edges"]
    assert new.extra == old["extra"]


def test_relation_keyed_canonical_input_also_stops_being_adapted(tmp_path):
    """The dropped clause was wider than the alias rule, and this is the proof.

    Endpoints here are canonical ``source_id``/``target_id`` -- no alias
    anywhere -- but the edges carry a ``relation`` key, which is what the old
    third clause's final term matched. So the old loader adapted this graph and
    the new one does not, on input that plan §2.6 never spoke to.

    The shipped dataset keys its edges ``relation``, so this term was live on
    real data, not a theoretical branch.

    Scores are unaffected: everything the encoder and scorer read is identical.
    Only the provenance stamps differ, exactly as for the alias-keyed case.
    """
    kg_path = tmp_path / "relation_keyed.json"
    kg_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "N1", "type": "SYMPTOM", "name": "cough"},
                    {"id": "N2", "type": "DIAGNOSIS", "name": "pneumonia"},
                ],
                "edges": [
                    {
                        "source_id": "N1",
                        "target_id": "N2",
                        "relation": "INDICATES",
                        "confidence": 0.9,
                    }
                ],
            }
        )
    )

    old = load_pin(PIN)["loader"]["relation_keyed_fixture"]
    assert old["looks_like_mimic_subkg"] is True
    assert not new_score._looks_like_mimic_subkg(json.loads(kg_path.read_text()))

    new = new_score._load_graph_for_scoring(kg_path)

    assert old["method"] == "mimic_subkg_adapter"
    assert all(stamp not in new.extra for stamp in _MIMIC_STAMPS)

    # What the encoder embeds and what the scorer indexes -- identical, which is
    # why routing changed and the numbers did not.
    encoder_visible = [[n.get("type"), n.get("text")] for n in new.nodes]
    endpoints = [
        [e.get("source_id"), e.get("target_id"), e.get("relation"), e.get("type")]
        for e in new.edges
    ]
    assert encoder_visible == old["encoder_visible"]
    assert endpoints == old["endpoints"]


def test_alias_keyed_input_no_longer_routes_through_the_mimic_adapter(tmp_path):
    """The intentional difference, labelled rather than smuggled in.

    Old: ``source``/``target`` edges failed ``_schema_compatibility_errors``, and
    ``_looks_like_mimic_subkg``'s third clause -- the scoring path's private copy
    of the alias rule -- treated the alias spelling as evidence the file was a
    MIMIC sub-KG, so it went through ``adapt_mimic_subkg``.

    New: ``_load_graph_for_scoring`` normalizes aliases with THE normalizer in
    ``graph/builders.py`` (the Phase 1 decision), the schema check passes, and a
    non-MIMIC graph is no longer stamped as a MIMIC one.

    The scored content is unchanged -- same nodes, same edges, same order, so
    the same scores. What changes is the provenance metadata in the output JSON.
    """
    kg_path = write_kg(tmp_path, source_key="source", target_key="target")
    old = load_pin(PIN)["loader"]["alias_fixture"]
    new = new_score._load_graph_for_scoring(kg_path)

    # Old only got there via the adapter; new never touches it.
    assert old["method"] == "mimic_subkg_adapter"
    assert all(stamp not in new.extra for stamp in _MIMIC_STAMPS)
    assert old["every_node_stamped"] is True
    assert not any("mimic_type" in node for node in new.nodes)
    assert old["every_edge_stamped"] is True
    assert not any("mimic_relation" in edge for edge in new.edges)

    # Everything the scorer actually reads is identical, which is why this
    # changes the output file and not the numbers in it.
    assert new.nodes == old["nodes_without_stamp"]
    assert new.edges == old["edges_without_stamp"]
    assert new.extra == old["extra_without_stamps"]


@requires_data
def test_alias_fold_on_the_real_dataset(tmp_path):
    """The same difference, measured on shipped data rather than a fixture.

    The one real dataset here keys its edges ``source``/``target``, so it is
    exactly the input the change above affects, and it is MIMIC-shaped: it has
    ``subject_id`` and PATIENT/MEDICATION/SERVICE nodes, so the two surviving
    ``_looks_like_mimic_subkg`` clauses still fire. The adapter is therefore
    still reachable for anything that genuinely needs it -- what stopped
    happening is being sent there *because of the endpoint spelling alone*.

    Nodes, edges and endpoints come out the same, so scores are unaffected; the
    difference is entirely the adapter's provenance stamping.
    """
    with DATA.open(encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))

    assert all("source_id" not in edge for edge in record["edges"])
    assert "subject_id" in record

    old = load_pin(PIN)["loader"]["real_dataset_first_record"]
    new = new_score._load_graph_for_scoring(path)

    assert old["method"] == "mimic_subkg_adapter"
    assert all(stamp not in new.extra for stamp in _MIMIC_STAMPS)
    assert old["dropped_nodes"] == 0
    assert old["dropped_edges"] == 0

    endpoints = [[e["source_id"], e["type"], e["target_id"]] for e in new.edges]
    assert endpoints == old["endpoints"]
    assert [n["id"] for n in new.nodes] == old["node_ids"]
    # node_encoder_keys() is (type, text); identical keys mean identical scores.
    assert [list(key) for key in new.node_encoder_keys()] == old["node_encoder_keys"]


# --------------------------------------------------------------------------- #
# The other two demolitions this phase names.
# --------------------------------------------------------------------------- #
def test_unscoreable_graph_still_gets_the_schema_guard():
    """The one control-flow join the merge had to make by hand.

    v4's ``score_graph`` wrapped v3's and ran ``_apply_schema_guard`` on whatever
    came back -- including v3's two early returns for a graph with no nodes or no
    patches. Collapsing the wrapper into one function makes it easy to return
    early and skip the guard instead, so this pins the old behaviour. No
    checkpoint needed: nothing reaches the model.
    """
    edge = {"source_id": "A", "target_id": "B", "type": "INDICATES",
            "evidence": "", "turn_id": ""}
    new_graph = NewPatientGraph(nodes=[], edges=[dict(edge)])
    device = torch.device("cpu")

    old = load_pin(PIN)["schema_guard"]
    scores, flags = new_score.score_graph(new_graph, None, None, Config(), device)

    assert [scores, flags] == old["result"] == [[0.0], ["inconsistent"]]
    assert new_graph.edges == old["edges"]
    assert new_graph.edges[0]["jepa_schema_valid"] is False
    assert new_graph.edges[0]["jepa_schema_error"] == "missing source node 'A'"


def test_install_v6_data_conversion_is_gone():
    """``graph_jepa_v6/score.py`` really did define ``_install_v6_data_conversion``
    -- that was the other half of this assertion until Phase 8, and
    ``baseline/COVERAGE.md`` records it. The half that still runs is the one that
    would notice it coming back."""
    assert not hasattr(new_score, "_install_v6_data_conversion")


# The 15 module-level assignments score_revision.py used to re-export from
# score_base, so the v4 CLI could reuse v3's private helpers.
_OLD_REEXPORTS = [
    "RELATIONS",
    "NEGATED_OR_ABSENT_MARKERS",
    "_annotate_revision_actions",
    "_candidate_threshold",
    "_flag",
    "_iter_candidate_specs",
    "_iter_inputs",
    "_load_graph_for_scoring",
    "_normalise_relation",
    "_prune",
    "_revision_action",
    "_schema_compatibility_errors",
    "_update_disabled_relations",
    "_update_relation_thresholds",
    "_validate_checkpoint_relation_capacity",
]


def test_reexport_block_is_gone():
    """Every name score_revision re-exported is defined in score.py itself.

    That each of the 15 really was a re-export -- ``getattr(score_revision, n) is
    getattr(score_base, n)`` -- needed both old modules to assert, so Phase 8
    retired that half (``baseline/COVERAGE.md``). What survives is the half that
    matters going forward: all 15 names exist here, are defined here, and the two
    vocabularies still hold what the old module held.
    """
    old = load_pin(PIN)["reexports"]
    assert len(_OLD_REEXPORTS) == 15
    for name in _OLD_REEXPORTS:
        value = getattr(new_score, name)
        if callable(value):
            assert value.__module__ == "clinical_jepa.score", name
    assert sorted(new_score.RELATIONS) == old["RELATIONS"]
    assert list(new_score.NEGATED_OR_ABSENT_MARKERS) == old["NEGATED_OR_ABSENT_MARKERS"]


def test_parser_keeps_every_released_option():
    """Four parsers became one. Nothing the released CLIs accepted was dropped.

    The one deliberate default change: ``--encoder-cache`` was per package
    (``.cache/graph_jepa_v5/encoder``, ``.cache/graph_jepa_v6/encoder``,
    ``.cache/fawkes_core/encoder``). One module cannot keep three, and the cache
    is keyed by a hash of ``(type, text)``, so a shared directory is a superset
    of any of them.

    The three old parsers had identical option sets, which
    ``baseline/record_old_pins.py`` asserted before writing one list.
    """
    new = new_score.build_arg_parser()
    options = sorted(list(a.option_strings) for a in new._actions)
    assert options == load_pin(PIN)["parser_options"]

    args = new.parse_args([
        "--input", "in.json", "--checkpoint", "c.pt", "--output", "out.json",
        "--add-candidates", "--prune-threshold", "0.1",
        "--candidate-threshold", "0.9",
        "--candidate-threshold-relation", "INDICATES=0.5",
        "--review-weak-threshold", "CONFIRMS=0.3",
        "--review-inconsistent-threshold", "CONFIRMS=0.1",
        "--enable-candidate-relation", "CAUSES",
        "--disable-candidate-relation", "RULES_OUT",
        "--max-candidates", "5",
        "--allow-cross-res-candidates", "--allow-negated-candidates",
    ])
    assert args.add_candidates and args.max_candidates == 5
    assert args.candidate_threshold_relation == ["INDICATES=0.5"]
    assert args.encoder_cache == ".cache/clinical_jepa/encoder"


@requires_checkpoints
@pytest.mark.parametrize("checkpoint_path", [V5_CHECKPOINT, V6_CHECKPOINT], ids=VARIANT_IDS)
def test_load_builds_the_checkpoint_encoder(tmp_path, checkpoint_path):
    """``_load`` is the one part of ``run()`` the byte gate cannot cover.

    Both released checkpoints name the SapBERT encoder, which the scoring gate
    replaces with MockEncoder to stay offline. SapBertNodeEncoder loads its
    weights lazily, on the first cache miss, so constructing it here downloads
    nothing.
    """
    model, encoder, cfg = new_score._load(
        str(checkpoint_path), torch.device("cpu"), str(tmp_path / "encoder")
    )
    assert isinstance(model, GraphJEPA)
    assert not model.training
    assert cfg.encoder == "sapbert"
    assert isinstance(encoder, SapBertNodeEncoder)
    assert encoder.dim == cfg.model.base_in_dim
    assert model.context_node_encoder.input_proj.in_features == cfg.model.in_dim
