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
    out = []
    for k in SCORE_FEATS:
        v = labels.get(k)
        if not isinstance(v, (int, float)):
            v = 0.0
        if k == "omop_lca_dist":
            v = math.exp(-float(v) / 5.0) if v else 0.0   # distance -> closeness in (0,1]
        out.append(float(v))
    return out


def support_graded(labels):
    """(v11) graded evidence-support in [0,1], non-saturating."""
    labels = labels or {}
    return max([float(labels.get(k) or 0.0) for k in SUPPORT_FEATS] + [0.0])   # max KB endorsement / provenance ratio (no binary prov->1.0)


def has_evidence(labels):
    """(v14) any external KB hit or note provenance -> keep; else no-evidence junk."""
    labels = labels or {}
    if labels.get("prov_in_note"):
        return True
    for k in EVIDENCE_FEATS:
        v = labels.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return True
    return False


def note_grounded_edge(labels):
    """(v16) edge grounds its endpoints in the note iff prov_in_note."""
    return bool((labels or {}).get("prov_in_note"))


def normalize_text(t):
    """(v16) for GROUND_BY=name, and for EIR triple matching."""
    return " ".join(str(t or "").lower().split())


def ehash(name, entity_vocab):
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % entity_vocab


def resolve_rel(r):
    if r in RELATION_CANONICAL:
        return RELATION_CANONICAL[r]
    if r in RELATION_ALIASES:
        return RELATION_CANONICAL[RELATION_ALIASES[r]]
    raise KeyError(f"[FAILURE] unknown relation '{r}'; no fallback.")


def add_inverses(ei, et, ef=None):
    """(v9) ``ef`` is the (E, SCORE_DIM) feature matrix (or None)."""
    if ei.size(1) == 0:
        return (ei, et) if ef is None else (ei, et, ef)
    ei2 = torch.cat([ei, ei.flip(0)], 1)
    et2 = torch.cat([et, et + NUM_BASE])
    if ef is None:
        return ei2, et2
    return ei2, et2, torch.cat([ef, ef], 0)                 # inverse edge inherits its forward edge's v8 scores


def load_full_dataset(cfg):
    """(v14) ONE dataset: note + MIMIC + nodes + edges(scores) per admission."""
    if cfg.data_path:
        p = cfg.data_path
        if not os.path.isfile(p):
            raise RuntimeError(f"[FAILURE] DATA_PATH not found: {p}")
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(cfg.data_repo, cfg.data_file, repo_type="dataset")
    graphs = []
    demo = {}
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        graphs.append(r)
        demo[str(r.get("subject_id"))] = {"gender": r.get("gender"), "age": r.get("anchor_age")}
    if not graphs:
        raise RuntimeError(f"[FAILURE] 0 records in {cfg.data_repo}/{cfg.data_file}; no fallback.")
    logger.info(f"[DATA] {len(graphs)} admission graphs + {len(demo)} subjects from {p}")
    return graphs, demo


def numeric(demo_rec):
    if demo_rec is None:
        return [0.0] * BASE_NUMERIC
    age = demo_rec.get("age")
    age = float(age) / 100.0 if str(age).strip() not in ("", "None") else 0.0
    g = demo_rec.get("gender")
    return [age, 1.0 if g == "M" else 0.0, 1.0 if g == "F" else 0.0, 0.0, 0.0, 0.0]   # (v9) age/sex; structure carries the rest (type-only >= full)


def to_data(g, demo, cfg):
    sid = str(g.get("subject_id"))
    drec = demo.get(sid)
    nodes, edges = g.get("nodes", []), g.get("edges", [])
    id2i, types, ents, names = {}, [], [], []
    nf = numeric(drec)
    for i, n in enumerate(nodes):
        id2i[n["id"]] = i
        t = n.get("type")
        if t not in NODE_TYPES:
            raise KeyError(f"[FAILURE] unknown node type '{t}' subj {sid}")
        name = n.get("normalized_name") or n.get("name") or ""
        types.append(NODE_TYPES[t])
        ents.append(ehash(name, cfg.entity_vocab))
        names.append(name)
    src, dst, et, feats = [], [], [], []
    grounded = set()                                             # (v16) ENTITY indices the note grounds (carry the note vector); rest stay zero
    for e in edges:
        if cfg.prune_no_evidence and e.get("evidence") == "llm" and not has_evidence(e.get("labels")):
            continue                                             # (v14) drop no-evidence junk
        s, d = e["source"], e["target"]
        if s not in id2i or d not in id2i:
            raise ValueError(f"[FAILURE] dangling edge subj {sid}")
        si, di = id2i[s], id2i[d]
        src.append(si)
        dst.append(di)
        et.append(resolve_rel(e.get("relation")))
        feats.append(score_vec(e.get("labels")))
        if cfg.use_note and cfg.ground_by == "prov" and note_grounded_edge(e.get("labels")):
            grounded.update((si, di))                            # (v16) note-grounded edge -> both endpoints
    if cfg.use_note and cfg.ground_by == "name":                 # (v16) entity named in the note text
        ntxt = normalize_text(g.get("note"))
        for i, nm in enumerate(names):
            nm2 = normalize_text(nm)
            if len(nm2) >= 3 and nm2 in ntxt:
                grounded.add(i)
    if cfg.use_note and cfg.ground_by == "all":                  # (v16) every non-PATIENT entity (ablation: localized-but-everywhere)
        grounded = set(i for i in range(len(types)) if types[i] != NODE_TYPES["PATIENT"])
    if cfg.use_note:                                             # (v16) note vector LOCALIZED onto grounded entities (NO NOTE node)
        ne = g.get("note_embedding")
        if ne is not None and len(ne) != cfg.embed_dim:
            raise ValueError(f"[FAILURE] note_embedding dim {len(ne)} != EMBED_DIM {cfg.embed_dim} subj {sid}")
        ne = ne or [0.0] * cfg.embed_dim
        zero_note = [0.0] * cfg.embed_dim
        numf = [nf + (ne if i in grounded else zero_note) for i in range(len(types))]
    else:
        numf = [list(nf) for _ in range(len(types))]
    data = Data()
    data.num_nodes = len(types)
    data.n_edges_real = len(src)
    data.node_type = torch.tensor(types, dtype=torch.long)
    data.entity_id = torch.tensor(ents, dtype=torch.long)
    data.numfeat = torch.tensor(numf, dtype=torch.float)
    data.edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)   # FORWARD only
    data.edge_type = torch.tensor(et, dtype=torch.long) if et else torch.zeros((0,), dtype=torch.long)
    data.edge_feat = torch.tensor(feats, dtype=torch.float) if feats else torch.zeros((0, SCORE_DIM), dtype=torch.float)   # (v9) per-edge v8 score vector
    data.sem_id = torch.zeros(len(types), dtype=torch.long)
    data.n_grounded = torch.tensor([len(grounded)], dtype=torch.long)   # (v16) entities carrying the note vector (for the [NOTE] log)
    data.gid = torch.tensor(
        [int(sid) if str(sid).isdigit() else int(hashlib.md5(str(sid).encode()).hexdigest()[:12], 16)],
        dtype=torch.long)
    return data
