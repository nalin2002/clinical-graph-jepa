"""Vocabulary, feature construction, and PyG tensor conversion.

Split out of ``paper_v16/trainer.py`` lines 92-254. The vocabularies are
the paper's own — note the ``NOTE`` and ``PROCUREMENT`` node types, which
``clinical_jepa`` does not have. The two schemas genuinely differ and are
deliberately not unified (plan §9).

``NOTE`` and ``HAS_NOTE`` are retained but unused: v15 put the note on a
per-admission NOTE node, v16 retired it and localized the same vector onto the
entity nodes the note actually grounds. The definitions stay so v15 checkpoints
remain describable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os

import torch
from torch_geometric.data import Data

from .config import BASE_NUMERIC

logger = logging.getLogger("fawkes_jepa")

NODE_TYPES = {"PATIENT": 0, "DIAGNOSIS": 1, "MEDICATION": 2, "MICROBIOLOGY": 3, "PROCEDURE": 4,
              "SERVICE": 5, "LAB_TEST": 6, "PROCUREMENT": 7, "NOTE": 8}
RELATION_CANONICAL = {"HAS_DIAGNOSIS": 0, "TAKES_MEDICATION": 1, "TREATED_BY": 2, "HAD_MICROBIOLOGY": 3,
                      "CO_OCCURS_WITH": 4, "UNDERWENT_PROCEDURE": 5, "MANAGED_BY_SERVICE": 6,
                      "MANAGED_FOR": 7, "PERFORMED_FOR": 8, "CONFIRMS": 9, "HAD_LAB_TEST": 10,
                      "DIAGNOSED_BY": 11, "TARGETS_ORGANISM": 12, "MONITORED_BY": 13,
                      "ADMINISTERED_DURING": 14, "INDICATES": 15, "INVESTIGATED_BY": 16,
                      "ASSOCIATED_WITH": 17, "COMPLICATED_BY": 18, "PART_OF_REGIMEN": 19, "HAS_NOTE": 20}
RELATION_ALIASES = {"DIAGNORED_BY": "DIAGNOSED_BY", "TARGET_ORGANISM": "TARGETS_ORGANISM",
                    "HAS_MICROBIOLOGY": "HAD_MICROBIOLOGY", "HAD_PROCEDURE": "UNDERWENT_PROCEDURE",
                    "HAS_MEDICATION": "TAKES_MEDICATION", "MANAGES_FOR": "MANAGED_FOR"}

NUM_NODE_TYPES = len(NODE_TYPES)
NUM_BASE = len(RELATION_CANONICAL)
NUM_RELATIONS = 2 * NUM_BASE

# (v9) the v8 edge-score vector fed to the encoder. Numeric signals only; None -> 0.0. lca_dist scaled.
SCORE_FEATS = ["model", "drug_link_cos", "dx_disease_cos", "het_treats_ctd", "het_treats_cpd",
               "het_drug_cos", "het_dx_cos", "het_resembles_drd", "het_presents_dps",
               "omop_src_cos", "omop_dst_cos", "omop_lca_dist", "prov_in_note", "prov_ratio"]
SCORE_DIM = len(SCORE_FEATS)

TARGET_RELS = {"MANAGED_FOR", "CONFIRMS", "COMPLICATED_BY", "INDICATES"}    # (v11) the 4 inferred LLM edges
TARGET_REL_IDS = set(RELATION_CANONICAL[x] for x in TARGET_RELS)            # (v13) forward ids of the 4 inferred edges

SUPPORT_FEATS = ["dx_disease_cos", "het_treats_ctd", "het_treats_cpd", "het_resembles_drd",
                 "het_presents_dps", "prov_ratio"]
EVIDENCE_FEATS = ["drug_link_cos", "dx_disease_cos", "het_treats_ctd", "het_treats_cpd",
                  "het_drug_cos", "het_dx_cos", "het_resembles_drd", "het_presents_dps",
                  "omop_src_cos", "omop_dst_cos"]


def score_vec(labels):
    labels = labels or {}
    values = []
    for feat in SCORE_FEATS:
        value = labels.get(feat)
        if not isinstance(value, (int, float)):
            value = 0.0
        if feat == "omop_lca_dist":
            value = math.exp(-float(value) / 5.0) if value else 0.0   # distance -> closeness in (0,1]
        values.append(float(value))
    return values


def support_graded(labels):
    """(v11) graded evidence-support in [0,1], non-saturating."""
    labels = labels or {}
    return max([float(labels.get(feat) or 0.0) for feat in SUPPORT_FEATS] + [0.0])   # max KB endorsement / provenance ratio (no binary prov->1.0)


def has_evidence(labels):
    """(v14) any external KB hit or note provenance -> keep; else no-evidence junk."""
    labels = labels or {}
    if labels.get("prov_in_note"):
        return True

    for feat in EVIDENCE_FEATS:
        value = labels.get(feat)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def note_grounded_edge(labels):
    """(v16) edge grounds its endpoints in the note iff prov_in_note."""
    return bool((labels or {}).get("prov_in_note"))


def normalize_text(text):
    """(v16) for GROUND_BY=name, and for EIR triple matching."""
    return " ".join(str(text or "").lower().split())


def ehash(name, entity_vocab):
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % entity_vocab


def resolve_rel(relation):
    if relation in RELATION_CANONICAL:
        return RELATION_CANONICAL[relation]
    if relation in RELATION_ALIASES:
        return RELATION_CANONICAL[RELATION_ALIASES[relation]]

    raise KeyError(f"[FAILURE] unknown relation '{relation}'; no fallback.")


def add_inverses(edge_index, edge_type, edge_feat=None):
    """(v9) ``edge_feat`` is the (E, SCORE_DIM) feature matrix (or None)."""
    if edge_index.size(1) == 0:
        return (edge_index, edge_type) if edge_feat is None else (edge_index, edge_type, edge_feat)

    both_index = torch.cat([edge_index, edge_index.flip(0)], 1)
    both_type = torch.cat([edge_type, edge_type + NUM_BASE])

    if edge_feat is None:
        return both_index, both_type

    return both_index, both_type, torch.cat([edge_feat, edge_feat], 0)  # inverse edge inherits its forward edge's v8 scores


def load_full_dataset(cfg):
    """(v14) ONE dataset: note + MIMIC + nodes + edges(scores) per admission."""
    if cfg.data_path:
        path = cfg.data_path
        if not os.path.isfile(path):
            raise RuntimeError(f"[FAILURE] DATA_PATH not found: {path}")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(cfg.data_repo, cfg.data_file, repo_type="dataset")

    graphs = []
    demographics = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        graphs.append(record)
        demographics[str(record.get("subject_id"))] = {"gender": record.get("gender"), "age": record.get("anchor_age")}

    if not graphs:
        raise RuntimeError(f"[FAILURE] 0 records in {cfg.data_repo}/{cfg.data_file}; no fallback.")
    logger.info(f"[DATA] {len(graphs)} admission graphs + {len(demographics)} subjects from {path}")

    return graphs, demographics


def numeric(demo_rec):
    if demo_rec is None:
        return [0.0] * BASE_NUMERIC
    age = demo_rec.get("age")
    age = float(age) / 100.0 if str(age).strip() not in ("", "None") else 0.0
    gender = demo_rec.get("gender")
    return [age, 1.0 if gender == "M" else 0.0, 1.0 if gender == "F" else 0.0, 0.0, 0.0, 0.0]   # (v9) age/sex; structure carries the rest (type-only >= full)


def to_data(graph, demographics, cfg):
    subject_id = str(graph.get("subject_id"))
    demo_record = demographics.get(subject_id)
    nodes, edges = graph.get("nodes", []), graph.get("edges", [])
    id_to_index, type_ids, entity_hashes, names = {}, [], [], []
    demo_feats = numeric(demo_record)
    for index, node in enumerate(nodes):
        id_to_index[node["id"]] = index
        node_type = node.get("type")
        if node_type not in NODE_TYPES:
            raise KeyError(f"[FAILURE] unknown node type '{node_type}' subj {subject_id}")
        name = node.get("normalized_name") or node.get("name") or ""
        type_ids.append(NODE_TYPES[node_type])
        entity_hashes.append(ehash(name, cfg.entity_vocab))
        names.append(name)

    src_indices, dst_indices, edge_type_ids, edge_scores = [], [], [], []
    grounded = set()                                             # (v16) ENTITY indices the note grounds (carry the note vector); rest stay zero
    for edge in edges:
        if cfg.prune_no_evidence and edge.get("evidence") == "llm" and not has_evidence(edge.get("labels")):
            continue                                             # (v14) drop no-evidence junk

        source_id, target_id = edge["source"], edge["target"]

        if source_id not in id_to_index or target_id not in id_to_index:
            raise ValueError(f"[FAILURE] dangling edge subj {subject_id}")

        source_index, target_index = id_to_index[source_id], id_to_index[target_id]
        src_indices.append(source_index)
        dst_indices.append(target_index)
        edge_type_ids.append(resolve_rel(edge.get("relation")))
        edge_scores.append(score_vec(edge.get("labels")))

        if cfg.use_note and cfg.ground_by == "prov" and note_grounded_edge(edge.get("labels")):
            grounded.update((source_index, target_index))         # (v16) note-grounded edge -> both endpoints

    if cfg.use_note and cfg.ground_by == "name":                 # (v16) entity named in the note text
        note_text = normalize_text(graph.get("note"))
        for index, name in enumerate(names):
            normalized_name = normalize_text(name)
            if len(normalized_name) >= 3 and normalized_name in note_text:
                grounded.add(index)

    if cfg.use_note and cfg.ground_by == "all":                  # (v16) every non-PATIENT entity (ablation: localized-but-everywhere)
        grounded = set(i for i in range(len(type_ids)) if type_ids[i] != NODE_TYPES["PATIENT"])

    if cfg.use_note:                                             # (v16) note vector LOCALIZED onto grounded entities (NO NOTE node)
        note_embedding = graph.get("note_embedding")
        if note_embedding is not None and len(note_embedding) != cfg.embed_dim:
            raise ValueError(f"[FAILURE] note_embedding dim {len(note_embedding)} != EMBED_DIM {cfg.embed_dim} subj {subject_id}")
        note_embedding = note_embedding or [0.0] * cfg.embed_dim
        zero_note = [0.0] * cfg.embed_dim
        node_feats = [demo_feats + (note_embedding if i in grounded else zero_note) for i in range(len(type_ids))]
    else:
        node_feats = [list(demo_feats) for _ in range(len(type_ids))]

    data = Data()
    data.num_nodes = len(type_ids)
    data.n_edges_real = len(src_indices)
    data.node_type = torch.tensor(type_ids, dtype=torch.long)
    data.entity_id = torch.tensor(entity_hashes, dtype=torch.long)
    data.numfeat = torch.tensor(node_feats, dtype=torch.float)
    data.edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long) if src_indices else torch.zeros((2, 0), dtype=torch.long)   # FORWARD only
    data.edge_type = torch.tensor(edge_type_ids, dtype=torch.long) if edge_type_ids else torch.zeros((0,), dtype=torch.long)
    data.edge_feat = torch.tensor(edge_scores, dtype=torch.float) if edge_scores else torch.zeros((0, SCORE_DIM), dtype=torch.float)   # (v9) per-edge v8 score vector
    data.sem_id = torch.zeros(len(type_ids), dtype=torch.long)
    data.n_grounded = torch.tensor([len(grounded)], dtype=torch.long)   # (v16) entities carrying the note vector (for the [NOTE] log)
    data.gid = torch.tensor(
        [int(subject_id) if str(subject_id).isdigit() else int(hashlib.md5(str(subject_id).encode()).hexdigest()[:12], 16)],
        dtype=torch.long)

    return data
