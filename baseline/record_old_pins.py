"""Record what ``old_src`` produced, so the Phase 8 gates survive its deletion.

A Phase 8 one-off, and the only reason the differential gates written in Phases
1-6 still assert anything. ``baseline/`` held LOO metrics, ``state_dict`` key
lists and the paper test-split numbers; the gates *also* compared per-graph
tensors, scored output JSON, LLM prompts, training-step values and the released
CLIs' option sets against ``old_src`` running in the same process. None of that
was written down anywhere, so deleting the old tree would have retired 24 gates
rather than converting them.

    PYTHONPATH=src:old_src python baseline/record_old_pins.py [outdir]

``outdir`` defaults to ``baseline/``. To check that a tracked pin still agrees
with live ``old_src``, write to a scratch directory and diff:

    PYTHONPATH=src:old_src python baseline/record_old_pins.py /tmp/pins
    diff -r baseline /tmp/pins

Two properties this script has to have, and why neither is optional:

* **Deterministic.** Every value is re-derived from a seed, so a second run is
  byte-identical and the diff above is a real check rather than a formality.
  Randomness is seeded immediately before the call that consumes it, exactly as
  the gates do.
* **The old side is the oracle.** The new tree is imported only to build inputs
  and starting points the old tree cannot produce on its own: the seeded
  synthetic graph population, the two randomly-initialised models whose weights
  the gates copied from new to old before training, and ``benchmarks``'
  test-split record selection. Every recorded *value* is computed by ``old_src``.

The paper trainer's environment variables are scrubbed before it is imported,
because ``paper_v16/trainer.py`` reads ~30 of them at module scope: an ambient
``USE_NOTE=0`` would silently record a different configuration under the same
filename.

This needs ``old_src``, so it cannot be re-run once Phase 8 deletes it. That is
inherent -- it is a record of provenance, like
``baseline/reproduce_paper_testsplit.py``, not a maintained tool.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ``paper_v16/trainer.py`` lines 119-197. Cleared before the import below, so
# everything recorded from the trainer is the released configuration.
TRAINER_ENV_VARS = (
    "DATA_REPO", "DATA_FILE", "DATA_PATH", "PUSH", "OUTPUT_REPO",
    "JEPA_EPOCHS", "READOUT_EPOCHS", "BATCH", "HID", "LAYERS", "HEADS",
    "EDGE_EMB", "ENTITY_VOCAB", "LR", "NODE_MASK", "EDGE_MASK", "EMA_BASE",
    "EMA_FINAL", "VAL_FRAC", "TEST_FRAC", "FREEZE_ENCODER", "MRR_CAP", "SEED",
    "DETERMINISTIC", "QUERY_ENTITY", "ENTITY_EMB", "NEG_K", "TEMP",
    "USE_SCORES", "PRUNE_NO_EVIDENCE", "USE_NOTE", "EMBED_DIM", "GROUND_BY",
    "DECODER", "FREEZE_EVAL", "LOSS", "SEMANTIC_ENT", "EIR_HOLDOUT",
    "EIR_FUZZY", "LOO_CAP", "RUN_EIR", "MASK_SCHEDULE", "MASK_LO", "MASK_HI",
    "TARGET_WEIGHT", "RUN_CASCADE", "CASCADE_ORDER",
)
for _name in TRAINER_ENV_VARS:
    os.environ.pop(_name, None)

sys.path.insert(0, str(ROOT / "tests"))
from conftest import DATA, digest_data, digest_fields, fold_digests  # noqa: E402

import torch  # noqa: E402

CKPT = ROOT / "models"
V5_CHECKPOINT = CKPT / "clinical-jepa-no-note/graph_jepa_v5.pt"
V6_CHECKPOINT = CKPT / "clinical-jepa-localized-note/graph_jepa_v6.pt"
PAPER_CHECKPOINT = CKPT / "fawkes-entity-note/fawkes_trainer_jepa_entity_note_v16_260615.pt"
V5_CONFIG = CKPT / "clinical-jepa-no-note/config.json"
V6_CONFIG = CKPT / "clinical-jepa-localized-note/config.json"

DEVICE = torch.device("cpu")

# Every seed and threshold below is the one the gate that reads the pin uses.
# They are duplicated rather than shared because this script must keep working
# from the recorded values alone; see baseline/README.md.
SCORE_SEED = 20260729
MODEL_INIT_SEED = 20260808
LOSS_SEED = 1234
TRAIN_SEED = 1234
PATCH_SEED = 0
IN_DIM = 32
DIFFERENTIAL_GRAPHS = 120
SAMPLE_GRAPHS = 64

# Config field -> the module-scope global in paper_v16/trainer.py it replaces.
FIELD_TO_TRAINER_GLOBAL = {
    "data_repo": "DATA_REPO", "data_file": "DATA_FILE", "data_path": "DATA_PATH",
    "push": "PUSH", "output_repo": "OUTPUT_REPO",
    "jepa_epochs": "JEPA_EPOCHS", "readout_epochs": "READOUT_EPOCHS",
    "batch": "BATCH", "lr": "LR", "seed": "SEED", "deterministic": "DETERMINISTIC",
    "hid": "HID", "layers": "LAYERS", "heads": "HEADS", "edge_emb": "EDGE_EMB",
    "entity_vocab": "ENTITY_VOCAB", "use_entity_emb": "USE_ENTITY_EMB",
    "query_entity": "QUERY_ENTITY", "decoder": "DECODER", "semantic_ent": "SEMANTIC_ENT",
    "use_scores": "USE_SCORES", "prune_no_evidence": "PRUNE_NO_EVIDENCE",
    "use_note": "USE_NOTE", "embed_dim": "EMBED_DIM", "ground_by": "GROUND_BY",
    "node_mask": "NODE_MASK", "edge_mask": "EDGE_MASK",
    "ema_base": "EMA_BASE", "ema_final": "EMA_FINAL",
    "freeze_encoder": "FREEZE", "neg_k": "NEG_K", "temp": "TEMP", "loss": "LOSS",
    "mask_schedule": "MASK_SCHEDULE", "mask_lo": "MASK_LO", "mask_hi": "MASK_HI",
    "target_weight": "TARGET_WEIGHT",
    "val_frac": "VAL_FRAC", "test_frac": "TEST_FRAC", "mrr_cap": "MRR_CAP",
    "freeze_eval": "FREEZE_EVAL", "loo_cap": "LOO_CAP", "run_eir": "RUN_EIR",
    "eir_holdout": "EIR_HOLDOUT", "eir_fuzzy": "EIR_FUZZY",
    "run_cascade": "RUN_CASCADE", "cascade_order": "CASCADE_ORDER",
}

# A second, deliberately non-default environment. The gate this feeds used to
# read the ambient environment on both sides, so it held for any configuration;
# a pin can only hold for the configurations recorded. Two of them keep the
# int/float/bool/lowercase/comma-list parsing rules under test rather than only
# the defaults. Every name here is read by both implementations (trainer.py
# 119-197, fawkes/config.py::from_env).
NON_DEFAULT_TRAINER_ENV = {
    "USE_NOTE": "0", "EMBED_DIM": "512", "GROUND_BY": "NAME", "USE_SCORES": "true",
    "PRUNE_NO_EVIDENCE": "0", "HID": "64", "HEADS": "8", "LAYERS": "3",
    "SEED": "7", "BATCH": "4", "LR": "5e-4", "DECODER": "MLP",
    "ENTITY_EMB": "no", "FREEZE_ENCODER": "0", "PUSH": "0",
    "MRR_CAP": "100", "RUN_EIR": "yes", "DATA_PATH": "/tmp/local.jsonl",
    "CASCADE_ORDER": "INDICATES, CONFIRMS ,",
}


def as_json(value):
    """Round-trip through JSON, and refuse anything that does not survive it.

    Guards the whole file: a value that changes under the round-trip (a tuple, a
    NaN, an int dict key) would be pinned as something the gate can never
    compare equal to, and the gate would fail for a serialization reason rather
    than a real one.
    """
    encoded = json.dumps(value, allow_nan=False)
    decoded = json.loads(encoded)
    if decoded != json.loads(json.dumps(decoded, allow_nan=False)):
        raise ValueError(f"value does not round-trip through JSON: {value!r}")
    return decoded


def write(outdir, name, payload):
    path = outdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")


# --------------------------------------------------------------------------- #
# tests/test_clinical_jepa_core.py
# --------------------------------------------------------------------------- #
def record_core():
    from clinical_jepa.config import Config, TrainConfig
    from clinical_jepa.encoders import MockEncoder
    from clinical_jepa.graph.builders import SyntheticGraphGenerator
    from clinical_jepa.graph.patches import build_patch_data, sample_patch_task
    from clinical_jepa.graph.tensors import to_graph_data
    from clinical_jepa.schema import PatientGraph
    from torch_geometric.loader import DataLoader

    from fawkes_core.data import JsonlGraphBuilder as OldJsonlGraphBuilder
    from graph_jepa_v5 import data as old_v5_data
    from graph_jepa_v5.config import Config as OldV5Config
    from graph_jepa_v5.model import GraphJEPAv5 as OldGraphJEPAv5
    from graph_jepa_v6 import data as old_v6_data
    from graph_jepa_v6.config import Config as OldV6Config
    from graph_jepa_v6.model import GraphJEPAv6 as OldGraphJEPAv6

    train = TrainConfig()
    graphs = SyntheticGraphGenerator(
        seed=train.seed,
        min_nodes=train.synthetic_min_nodes,
        max_nodes=train.synthetic_max_nodes,
    ).generate_many(train.synthetic_graphs)
    assert len(graphs) == 256

    # -- the 512 per-graph tensor comparisons --------------------------------
    encoder = MockEncoder(dim=768)
    tensors = {}
    for variant, module in (("graph_jepa_v5", old_v5_data), ("graph_jepa_v6", old_v6_data)):
        tensors[variant] = [
            digest_data(module.to_graph_data(graph, encoder)) for graph in graphs
        ]

    # -- the endpoint-alias fixture, all three spellings ---------------------
    def alias_fixture(source_key, target_key):
        return PatientGraph(
            nodes=[
                {"id": "P", "type": "PATIENT", "text": "patient"},
                {"id": "D", "type": "DIAGNOSIS", "name": "pneumonia"},
                {"id": "M", "type": "MEDICATION", "text": "amoxicillin"},
            ],
            edges=[
                {source_key: "P", target_key: "D", "type": "HAS_DIAGNOSIS",
                 "labels": {"prov_in_note": 1}},
                {source_key: "D", target_key: "M", "type": "TREATED_BY"},
            ],
            extra={"subject_id": 7, "note_embedding": [0.25, -0.5, 0.75, 1.0]},
        )

    small = MockEncoder(dim=8)
    alias = {}
    for style, (source_key, target_key) in (
        ("canonical", ("source_id", "target_id")),
        ("aliased", ("source", "target")),
        ("mixed", ("source", "target_id")),
    ):
        graph = alias_fixture(source_key, target_key)
        alias[style] = {
            variant: digest_data(old_v6_data.to_graph_data(
                graph, small, use_note_embeddings=use_note, note_embedding_dim=4))
            for variant, use_note in (("no_note", False), ("note", True))
        }

    # -- the two old Config classes' to_dict, per released sidecar -----------
    raw_v5 = json.loads(V5_CONFIG.read_text())
    raw_v6 = json.loads(V6_CONFIG.read_text())
    configs = {
        "graph_jepa_v6": {
            "clinical-jepa-no-note": as_json(OldV6Config.from_dict(raw_v5).to_dict()),
            "clinical-jepa-localized-note": as_json(OldV6Config.from_dict(raw_v6).to_dict()),
        },
        # v5's ModelConfig has no base_in_dim, so it can only read its own file.
        "graph_jepa_v5": {
            "clinical-jepa-no-note": as_json(OldV5Config.from_dict(raw_v5).to_dict()),
        },
    }

    # -- the JSONL builder, on the fixture record the gate writes ------------
    record = {
        "subject_id": 7,
        "note_embedding": [0.1, 0.2],
        "nodes": [
            {"id": "P", "type": "PATIENT", "name": "patient"},
            {"id": "D", "type": "DIAGNOSIS", "name": "pneumonia"},
        ],
        "edges": [
            {"source": "P", "target": "D", "relation": "HAS_DIAGNOSIS",
             "labels": {"prov_in_note": 1}},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graphs.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        built = OldJsonlGraphBuilder(path).build()[0]
        extra = dict(built.extra)
        # The builder stamps the file it read, which here is a temporary
        # directory -- the one value in this file that cannot be pinned
        # literally. The gate substitutes its own path the same way and asserts
        # the real one against its own input separately, so nothing is dropped.
        assert extra["_source_path"] == f"{path}:1"
        extra["_source_path"] = "<input>:1"
        jsonl_builder = {
            "nodes": as_json(built.nodes),
            "edges": as_json(built.edges),
            "extra": as_json(extra),
        }

    # -- the forward pass and the three losses, per released checkpoint ------
    forward = {}
    for name, checkpoint_path, old_config_cls, old_model_cls in (
        ("no_note", V5_CHECKPOINT, OldV5Config, OldGraphJEPAv5),
        ("localized_note", V6_CHECKPOINT, OldV6Config, OldGraphJEPAv6),
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(checkpoint["config"])
        old_cfg = old_config_cls.from_dict(checkpoint["config"])
        model = old_model_cls(old_cfg.model)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()

        batch_encoder = MockEncoder(dim=cfg.model.base_in_dim)
        data = [
            to_graph_data(
                graph, batch_encoder,
                use_note_embeddings=cfg.model.use_note_embeddings,
                note_embedding_dim=cfg.model.note_embedding_dim,
                note_ground_by=cfg.model.note_ground_by,
            )
            for graph in graphs[:8]
        ]
        batch = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))

        generator = torch.Generator().manual_seed(PATCH_SEED)
        patch_data = build_patch_data(
            batch, num_patches=cfg.model.num_patches,
            patch_pe_dim=cfg.model.patch_pe_dim, generator=generator,
        )
        task = sample_patch_task(
            patch_data,
            context_patches=cfg.train.context_patches,
            target_patches=cfg.train.target_patches,
            generator=generator,
        )

        outputs = {
            "encode_nodes": model.encode_nodes(batch),
            "encode_target_nodes": model.encode_target_nodes(batch),
            "patch_prediction_energy": model.patch_prediction_energy(
                batch, patch_data, task.target_idx[:1]),
        }
        logs = {}
        for label, call in (
            ("jepa_loss", lambda: model.jepa_loss(
                batch, patch_data, task,
                var_weight=cfg.train.vicreg_var_weight,
                cov_weight=cfg.train.vicreg_cov_weight)),
            ("revision_loss", lambda: model.revision_loss(
                batch,
                mask_ratio=cfg.train.revision_mask_ratio,
                neg_per_pos=cfg.train.revision_neg_per_pos,
                llm_confidence_negatives=cfg.train.llm_confidence_negatives,
                llm_negative_threshold=cfg.train.llm_negative_threshold,
                llm_positive_threshold=cfg.train.llm_positive_threshold,
                llm_negative_threshold_by_relation=cfg.train.llm_negative_threshold_by_relation,
                llm_positive_threshold_by_relation=cfg.train.llm_positive_threshold_by_relation,
                llm_negative_weight=cfg.train.llm_negative_weight,
                clinical_artifact_filters=cfg.train.clinical_artifact_filters)),
            ("candidate_ranking_loss", lambda: model.candidate_ranking_loss(
                batch,
                mask_ratio=cfg.train.ranking_mask_ratio,
                neg_per_pos=cfg.train.ranking_neg_per_pos,
                max_pos=cfg.train.ranking_max_pos,
                temperature=cfg.train.ranking_temperature)),
        ):
            torch.manual_seed(LOSS_SEED)
            loss, log = call()
            outputs[label] = loss
            logs[label] = as_json(log)
        forward[name] = {"tensors": digest_fields(outputs), "logs": logs}

    return {
        "tensors": tensors,
        "alias_tensors": alias,
        "config_to_dict": configs,
        "jsonl_builder": jsonl_builder,
        "forward": forward,
    }


# --------------------------------------------------------------------------- #
# tests/test_clinical_jepa_score.py
# --------------------------------------------------------------------------- #
_SCORE_NODES = [
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

_SCORE_EDGES = [
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

PRUNE_THRESHOLD = 0.07
CANDIDATE_THRESHOLD = 0.5
WEAK_OVERRIDES = ["INDICATES=0.10", "LOCATED_AT=0.05", "CONFIRMS=0.08"]


def score_kg_dict(source_key="source_id", target_key="target_id"):
    rng = random.Random(11)
    edges = []
    for source, relation, target, prov in _SCORE_EDGES:
        edge = {source_key: source, target_key: target, "type": relation,
                "evidence": f"{source}->{target}", "turn_id": "t1"}
        if prov:
            edge["labels"] = {"prov_in_note": 1}
        edges.append(edge)
    return {
        "nodes": [dict(node) for node in _SCORE_NODES],
        "edges": edges,
        "note": "patient with fever and productive cough",
        "note_embedding": [round(rng.uniform(-1.0, 1.0), 6) for _ in range(768)],
    }


def record_score(outdir):
    from clinical_jepa.config import Config
    from clinical_jepa.encoders import MockEncoder

    from fawkes_core import data_graph as old_data_graph
    from fawkes_core import score_base as old_v3
    from fawkes_core import score_revision as old_v4
    from fawkes_core.config import Config as OldFawkesConfig
    from fawkes_core.schema import PatientGraph as OldPatientGraph
    from graph_jepa_v5 import score as old_v5_score
    from graph_jepa_v5.config import Config as OldV5Config
    from graph_jepa_v5.model import GraphJEPAv5
    from graph_jepa_v6 import score as old_v6_score
    from graph_jepa_v6.config import Config as OldV6Config
    from graph_jepa_v6.model import GraphJEPAv6

    def configure(cfg):
        cfg.score.prune_threshold = PRUNE_THRESHOLD
        cfg.score.candidate_threshold = CANDIDATE_THRESHOLD
        cfg.score.candidate_threshold_by_relation = {
            relation: CANDIDATE_THRESHOLD for relation in old_v4.RELATIONS
        }
        old_v4._update_relation_thresholds(cfg.score.weak_threshold_by_relation, WEAK_OVERRIDES)
        return cfg

    tmp = Path(tempfile.mkdtemp())
    kg_path = tmp / "kg.json"
    kg_path.write_text(json.dumps(score_kg_dict(), indent=2))

    # -- the Phase 4 gate: four scored output files, byte for byte -----------
    counts = {}
    for name, checkpoint_path, old_config_cls, old_model_cls, patched in (
        ("no_note", V5_CHECKPOINT, OldV5Config, GraphJEPAv5, False),
        ("localized_note", V6_CHECKPOINT, OldV6Config, GraphJEPAv6, True),
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = configure(old_config_cls.from_dict(checkpoint["config"]))
        model = old_model_cls(cfg.model)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        mock_dim = Config.from_dict(checkpoint["config"]).model.base_in_dim

        for path_name, add_candidates in (("revise", False), ("add_candidates", True)):
            saved = (old_v3.to_graph_data, old_v4.to_graph_data)
            if patched:
                old_v6_score._install_v6_data_conversion()
            try:
                torch.manual_seed(SCORE_SEED)
                graph = old_v4._load_graph_for_scoring(kg_path)
                old_v4._validate_checkpoint_relation_capacity(graph, cfg)
                scores, flags = old_v4.score_graph(
                    graph, model, MockEncoder(dim=mock_dim), cfg, DEVICE)
                graph.annotate_edges(scores, flags)
                old_v4._annotate_revision_actions(graph, cfg)
                added = 0
                if add_candidates:
                    added = old_v4.add_candidate_edges(
                        graph, model, MockEncoder(dim=mock_dim), cfg, DEVICE)
                    old_v4._annotate_revision_actions(graph, cfg)
                pruned = old_v4._prune(graph, cfg.score.prune_threshold)
            finally:
                old_v3.to_graph_data, old_v4.to_graph_data = saved
            out = outdir / "old_clinical_jepa_score_output" / f"{name}_{path_name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            graph.save(out)
            print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
            counts.setdefault(name, {})[path_name] = [added, pruned]

    assert old_v3.to_graph_data is old_data_graph.to_graph_data
    assert old_v4.to_graph_data is old_data_graph.to_graph_data

    # -- _load_graph_for_scoring, on the four inputs the gates use -----------
    canonical = old_v4._load_graph_for_scoring(kg_path)

    relation_keyed_raw = {
        "nodes": [
            {"id": "N1", "type": "SYMPTOM", "name": "cough"},
            {"id": "N2", "type": "DIAGNOSIS", "name": "pneumonia"},
        ],
        "edges": [
            {"source_id": "N1", "target_id": "N2", "relation": "INDICATES", "confidence": 0.9},
        ],
    }
    relation_keyed_path = tmp / "relation_keyed.json"
    relation_keyed_path.write_text(json.dumps(relation_keyed_raw))
    relation_keyed = old_v4._load_graph_for_scoring(relation_keyed_path)

    alias_path = tmp / "alias.json"
    alias_path.write_text(json.dumps(score_kg_dict("source", "target"), indent=2))
    aliased = old_v4._load_graph_for_scoring(alias_path)

    with DATA.open(encoding="utf-8") as stream:
        first_record = json.loads(stream.readline())
    real_path = tmp / "record.json"
    real_path.write_text(json.dumps(first_record))
    real = old_v4._load_graph_for_scoring(real_path)

    loader = {
        "canonical_fixture": {
            "nodes": as_json(canonical.nodes),
            "edges": as_json(canonical.edges),
            "extra": as_json(canonical.extra),
        },
        "relation_keyed_fixture": {
            "looks_like_mimic_subkg": old_v3._looks_like_mimic_subkg(relation_keyed_raw),
            "method": relation_keyed.extra["_method"],
            "encoder_visible": as_json(
                [[n.get("type"), n.get("text")] for n in relation_keyed.nodes]),
            "endpoints": as_json([
                [e.get("source_id"), e.get("target_id"), e.get("relation"), e.get("type")]
                for e in relation_keyed.edges
            ]),
        },
        "alias_fixture": {
            "method": aliased.extra["_method"],
            "every_node_stamped": all("mimic_type" in n for n in aliased.nodes),
            "every_edge_stamped": all("mimic_relation" in e for e in aliased.edges),
            "nodes_without_stamp": as_json(
                [{k: v for k, v in n.items() if k != "mimic_type"} for n in aliased.nodes]),
            "edges_without_stamp": as_json(
                [{k: v for k, v in e.items() if k != "mimic_relation"} for e in aliased.edges]),
            "extra_without_stamps": as_json({
                k: v for k, v in aliased.extra.items()
                if k not in ("_method", "_source_path", "_mimic_adapter")
            }),
        },
        "real_dataset_first_record": {
            "method": real.extra["_method"],
            "dropped_nodes": real.extra["_mimic_adapter"]["dropped_nodes"],
            "dropped_edges": real.extra["_mimic_adapter"]["dropped_edges"],
            "node_ids": as_json([n["id"] for n in real.nodes]),
            "endpoints": as_json(
                [[e["source_id"], e["type"], e["target_id"]] for e in real.edges]),
            "node_encoder_keys": as_json([list(k) for k in real.node_encoder_keys()]),
        },
    }

    # -- the schema guard on a graph nothing can score ----------------------
    guard_edge = {"source_id": "A", "target_id": "B", "type": "INDICATES",
                  "evidence": "", "turn_id": ""}
    guard_graph = OldPatientGraph(nodes=[], edges=[dict(guard_edge)])
    guard_scores, guard_flags = old_v4.score_graph(
        guard_graph, None, None, OldFawkesConfig(), DEVICE)

    # -- what score_revision re-exported, and the three parsers' options ----
    options = sorted(list(action.option_strings)
                     for action in old_v3.build_arg_parser()._actions)
    for parser in (old_v5_score.build_arg_parser(), old_v6_score.build_arg_parser()):
        assert sorted(list(a.option_strings) for a in parser._actions) == options

    shutil.rmtree(tmp)
    return {
        "revision_counts": counts,
        "loader": loader,
        "schema_guard": {
            "result": [as_json(guard_scores), as_json(guard_flags)],
            "edges": as_json(guard_graph.edges),
        },
        "reexports": {
            "RELATIONS": sorted(old_v3.RELATIONS),
            "NEGATED_OR_ABSENT_MARKERS": list(old_v3.NEGATED_OR_ABSENT_MARKERS),
        },
        "parser_options": options,
    }


# --------------------------------------------------------------------------- #
# tests/test_clinical_jepa_train.py
# --------------------------------------------------------------------------- #
def record_train():
    import argparse as _argparse

    from clinical_jepa.config import Config
    from clinical_jepa.encoders import MockEncoder
    from clinical_jepa.model import GraphJEPA
    from clinical_jepa.train import loop

    from graph_jepa_v5 import training as old_v5_training
    from graph_jepa_v5.config import Config as OldV5Config
    from graph_jepa_v5.model import GraphJEPAv5 as OldGraphJEPAv5

    def small(config_cls):
        cfg = config_cls()
        cfg.model.in_dim = IN_DIM
        cfg.model.num_patches = 4
        cfg.train.synthetic_graphs = 8
        cfg.train.batch_size = 4
        if hasattr(cfg.model, "use_note_embeddings"):
            cfg.model.use_note_embeddings = False
            cfg.model.base_in_dim = IN_DIM
        return cfg

    parser = _argparse.ArgumentParser()
    loop.add_data_args(parser)
    args = parser.parse_args(["--data", "synthetic"])
    encoder = MockEncoder(dim=IN_DIM)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out = {}
        for label, use_revision in (("no_revision", False), ("revision", True)):
            cfg, old_cfg = small(Config), small(OldV5Config)
            # The starting weights are the new model's, exactly as the gate had
            # them: it built GraphJEPA and copied its state into GraphJEPAv5.
            # GraphJEPA is constructed first, and immediately after the seed, so
            # the gate reproduces these weights from the seed alone.
            torch.manual_seed(MODEL_INIT_SEED)
            new_model = GraphJEPA(cfg.model)
            model = OldGraphJEPAv5(old_cfg.model)
            model.load_state_dict(new_model.state_dict(), strict=True)

            _dataset, train_loader = old_v5_training.build_train_loader(args, old_cfg, encoder)
            torch.manual_seed(TRAIN_SEED)
            steps = old_v5_training.train_epochs(
                model,
                old_v5_training.build_optimizer(model, old_cfg),
                train_loader,
                old_cfg,
                stage_name="gate",
                epochs=1,
                use_revision=use_revision,
                device=DEVICE,
                generator=torch.Generator().manual_seed(PATCH_SEED),
                wandb_run=None,
            )
            out[label] = {"steps": steps, "state_dict": digest_fields(model.state_dict())}
    finally:
        torch.set_num_threads(previous_threads)
    return {"train_epochs": out, "model_init_seed": MODEL_INIT_SEED}


# --------------------------------------------------------------------------- #
# tests/test_fawkes.py
# --------------------------------------------------------------------------- #
def _trainer_globals(env):
    """Read the trainer's module-scope globals in a subprocess under ``env``.

    A subprocess because ``paper_v16/trainer.py`` reads the environment at
    import: one process can only ever observe one configuration, which is the
    whole defect ``Config.from_env`` removes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "globals.json"
        script = (
            "import json, sys\n"
            "from paper_v16 import trainer\n"
            "names = json.loads(sys.argv[1])\n"
            "payload = {'globals': {f: getattr(trainer, g) for f, g in names.items()},\n"
            "           'numeric_dim': trainer.NUMERIC_DIM}\n"
            "open(sys.argv[2], 'w').write(json.dumps(payload))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script,
             json.dumps(FIELD_TO_TRAINER_GLOBAL), str(out)],
            env={"PATH": os.environ.get("PATH", ""),
                 "PYTHONPATH": str(ROOT / "old_src"), **env},
            capture_output=True, text=True, cwd=ROOT,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(out.read_text())


def record_fawkes(raw, demographics):
    from fawkes.config import Config
    from fawkes.data import resolve_rel, to_data
    from fawkes.evaluate import _load_graphs
    from fawkes.model import JEPA
    from torch_geometric.loader import DataLoader

    from paper_v16 import trainer as old

    cfg = Config()

    # -- Config.from_env against the globals it replaced, in two environments
    environments = {
        "defaults": {"env": {}, **_trainer_globals({})},
        "non_default": {"env": NON_DEFAULT_TRAINER_ENV,
                        **_trainer_globals(NON_DEFAULT_TRAINER_ENV)},
    }
    for name, entry in environments.items():
        entry["globals"] = as_json(entry["globals"])

    # -- jepa_step and readout_step, on the first 64 records ----------------
    sample_raw, sample_demographics = _load_graphs(DATA, SAMPLE_GRAPHS)
    pairs = [
        (d, g) for g in sample_raw
        if (d := to_data(g, sample_demographics, cfg)).num_nodes >= 3
        and d.edge_index.size(1) >= 4
    ]
    graphs = [d for d, _ in pairs]
    batch = next(iter(DataLoader(graphs, batch_size=16, shuffle=False)))

    # Same starting weights as the gate: it built the new JEPA first, right
    # after the seed, and copied its state into the old one.
    torch.manual_seed(MODEL_INIT_SEED)
    new_jepa = JEPA(cfg)
    old_jepa = old.JEPA()
    old_jepa.load_state_dict(new_jepa.state_dict())
    torch.manual_seed(PATCH_SEED)
    jepa_loss, jepa_std = old.jepa_step(old_jepa, batch.clone(), DEVICE)

    checkpoint = torch.load(PAPER_CHECKPOINT, map_location="cpu", weights_only=False)
    old_encoder, old_scorer = old.Encoder(), old.DistMult()
    old_encoder.load_state_dict(checkpoint["encoder"])
    old_scorer.load_state_dict(checkpoint["scorer"])
    readout = old.readout_step(
        old_encoder, old_scorer, batch.clone(), DEVICE, False,
        gen=torch.Generator(device=DEVICE).manual_seed(7), mask_ratio=0.3,
    )
    steps = {
        "model_init_seed": MODEL_INIT_SEED,
        "tensors": digest_fields({
            "jepa_step.loss": jepa_loss,
            "jepa_step.emb_std": jepa_std,
            **{f"readout_step[{i}]": readout[i] for i in range(4)},
        }),
        "readout_step.qsig": as_json(readout[4][-1]),
    }

    # -- the three evaluators the LOO gates do not reach --------------------
    evaluators = {
        "evaluate": as_json(old.evaluate(
            old_encoder, old_scorer,
            DataLoader(graphs, batch_size=1, shuffle=False), DEVICE)),
        "cascade_evaluate": as_json(old.cascade_evaluate(
            old_encoder, old_scorer, graphs,
            [resolve_rel(r) for r in cfg.cascade_order], DEVICE)),
        "eir_uplift_eval": as_json(old.eir_uplift_eval(
            old_encoder, old_scorer, pairs, 0.5, DEVICE)),
    }

    # -- to_data over every record in the shipped dataset -------------------
    keys = None
    per_record = []
    per_key = {}
    for index, graph in enumerate(raw):
        fields = digest_data(old.to_data(graph, demographics))
        if keys is None:
            keys = sorted(fields)
            per_key = {key: [] for key in keys}
        assert sorted(fields) == keys, f"record {index}: field set differs from record 0"
        per_record.append(fold_digests(fields))
        for key, value in fields.items():
            per_key[key].append(value)

    return {
        "trainer_globals": {
            "field_to_global": FIELD_TO_TRAINER_GLOBAL,
            "environments": environments,
        },
        "training_steps": steps,
        "evaluators": evaluators,
        "to_data": {
            "records": len(per_record),
            "keys": keys,
            "per_key": {key: fold_digests(values) for key, values in per_key.items()},
            "per_record": per_record,
        },
    }


# --------------------------------------------------------------------------- #
# tests/test_benchmarks.py
# --------------------------------------------------------------------------- #
def record_benchmarks(raw, demographics):
    import argparse as _argparse
    from dataclasses import astuple

    from benchmarks import vs_fawkes
    from clinical_jepa.encoders import MockEncoder

    from fawkes_core.data import adapt_mimic_subkg as old_adapt_mimic_subkg
    from graph_jepa_v6 import evaluate as old_evaluate
    from graph_jepa_v6 import evaluate_llm as old_llm
    from graph_jepa_v6 import evaluate_loo_v12_jepa_llm as old_three_way
    from graph_jepa_v6 import training as old_training
    from graph_jepa_v6.data import PatientGraphDataset as OldPatientGraphDataset

    # -- the clinical_jepa arm, over the fawkes arm's test-split records -----
    # The record selection is the new driver's, because it is the population
    # both arms are compared on; the pipeline and the metrics are the old ones.
    _metrics, records, _cfg = vs_fawkes.fawkes_arm(
        str(PAPER_CHECKPOINT), raw, demographics, DEVICE, cap=40000)
    records = records[:DIFFERENTIAL_GRAPHS]

    old_model, old_cfg = old_training.load_model_checkpoint(str(V6_CHECKPOINT), DEVICE)
    old_model.eval()
    encoder = MockEncoder(dim=old_cfg.model.base_in_dim)
    old_dataset = OldPatientGraphDataset(
        [
            old_adapt_mimic_subkg(raw[index], source_path=f"{DATA}:{index + 1}")
            for index in records
        ],
        encoder,
        use_note_embeddings=old_cfg.model.use_note_embeddings,
        note_embedding_dim=old_cfg.model.note_embedding_dim,
        note_ground_by=old_cfg.model.note_ground_by,
    )
    arm = old_evaluate.leave_one_out_recovery(
        old_model,
        [old_dataset[index] for index in range(len(old_dataset))],
        old_cfg, DEVICE, cap=40000, candidate_mode="same-type",
    )

    # -- query sampling, prompt text and ranks ------------------------------
    parser = _argparse.ArgumentParser()
    old_training.add_data_args(parser)
    args = parser.parse_args(["--data", "synthetic", "--synthetic-graphs", "32"])
    llm_model, llm_cfg = old_training.load_model_checkpoint(str(V6_CHECKPOINT), DEVICE)
    llm_model.eval()
    llm_cfg.train.synthetic_graphs = args.synthetic_graphs
    llm_graphs = old_training.build_graphs(args, llm_cfg)
    llm_dataset = OldPatientGraphDataset(
        llm_graphs,
        MockEncoder(dim=llm_cfg.model.base_in_dim),
        use_note_embeddings=llm_cfg.model.use_note_embeddings,
        note_embedding_dim=llm_cfg.model.note_embedding_dim,
        note_ground_by=llm_cfg.model.note_ground_by,
    )
    llm_data = [llm_dataset[index] for index in range(len(llm_dataset))]
    queries = old_llm.collect_recovery_queries(
        llm_data, llm_cfg,
        samples_per_relation=5, candidate_mode="same-type", max_candidates=8, seed=0,
    )
    assert queries
    prompts = [
        old_llm.build_prompt(
            llm_graphs[q.graph_index], llm_data[q.graph_index], llm_cfg, q,
            context_mode="full", max_context_edges=120)
        for q in queries
    ]
    ranks = [
        old_llm.jepa_rank_query(llm_model, llm_data[q.graph_index], q, llm_cfg, DEVICE)
        for q in queries
    ]

    # -- the reply parser, and the two _summarize copies --------------------
    replies = [
        '{"ranking":[3,1,2]}',
        'sure, here you go:\n{"ranked_candidates": ["2", "1"]}\nhope that helps',
        "I think 2 then 1 then 3",
        "no numbers at all",
        '{"ranking":[99,2]}',
        "",
    ]
    ranking_pairs = [(1, 4), (2, 1), (3, 3), (1, 2), (7, 7)]
    relations = ["MANAGED_FOR", "MANAGED_FOR", "INDICATES", "INDICATES", "CONFIRMS"]
    shared = dict(
        graph_index=0, edge_index=0, source="a", target="b",
        candidates=["a", "b", "c", "d"],
        llm_parse_ok=True, llm_parse_complete=True,
        prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, total_tokens=0,
        finish_reason="stop", llm_response="",
    )
    pairwise = [
        old_llm.QueryResult(relation=rel, jepa_rank=j, llm_rank=m, **shared)
        for rel, (j, m) in zip(relations, ranking_pairs)
    ]
    three_way = [
        old_three_way.ComparisonResult(
            relation=rel, loo_jepa_rank=j, v6_jepa_rank=j, llm_rank=m, **shared)
        for rel, (j, m) in zip(relations, ranking_pairs)
    ]

    return {
        "clinical_jepa_arm": {"graphs": DIFFERENTIAL_GRAPHS, "metrics": as_json(arm)},
        "llm": {
            "queries": as_json([astuple(q) for q in queries]),
            "prompts": prompts,
            "jepa_ranks": as_json(ranks),
        },
        "parse_ranking": {
            reply: as_json(old_llm.parse_ranking(reply, 3)) for reply in replies
        },
        "summaries": {
            "vs_llm.jepa_rank": as_json(old_llm._summarize(pairwise, "jepa_rank")),
            "three_way.llm_rank": as_json(old_three_way._summarize(three_way, "llm_rank")),
        },
        "loo_baseline_symbols": sorted(
            name for name in ("LooEncoder", "LooDistMult", "LooMLPScorer", "LOO_NUMERIC_DIM")
            if hasattr(old_three_way, name)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", default=str(ROOT / "baseline"))
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    from fawkes.evaluate import _load_graphs

    print("recording tests/test_clinical_jepa_core.py")
    write(outdir, "old_clinical_jepa_core.json", record_core())

    print("recording tests/test_clinical_jepa_score.py")
    write(outdir, "old_clinical_jepa_score.json", record_score(outdir))

    print("recording tests/test_clinical_jepa_train.py")
    write(outdir, "old_clinical_jepa_train.json", record_train())

    print(f"loading {DATA.name}")
    raw, demographics = _load_graphs(DATA, None)
    print(f"  {len(raw):,} records")

    print("recording tests/test_fawkes.py")
    write(outdir, "old_fawkes.json", record_fawkes(raw, demographics))

    print("recording tests/test_benchmarks.py")
    write(outdir, "old_benchmarks.json", record_benchmarks(raw, demographics))


if __name__ == "__main__":
    main()
