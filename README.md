# Clinical Graph-JEPA

### Predictive patient-state knowledge graphs for cognitive decision support

[![Paper](https://img.shields.io/badge/paper-OpenReview-8c1b13.svg)](https://openreview.net/forum?id=HXsMPubPqE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C.svg)](https://pytorch.org/)

Clinical Graph-JEPA learns predictive patient-state representations from typed
clinical knowledge graphs. This repository provides a self-contained,
documented implementation suite for the paper, including note-free and
note-augmented Graph-JEPA models, released checkpoints, evaluation code, and a
400-admission graph dataset.

**Paper:** [OpenReview](https://openreview.net/forum?id=HXsMPubPqE) ·
[repository PDF](paper/clinical_jepa.pdf) ·
[paper-to-code map](docs/PAPER_CODE_MAP.md)

> [!IMPORTANT]
> This is a research system for improving structured patient representations.
> It does not make autonomous clinical decisions and is not intended for direct
> clinical use.

## Abstract

Clinical narratives contain relationships that are absent from structured
tables, but extraction errors, ontology mismatch, missing edges, and temporal
ambiguity can make automatically constructed knowledge graphs unreliable.
Clinical Graph-JEPA treats an extracted graph as a draft patient-state
representation rather than a finished artifact. It combines a deterministic
clinical backbone with inferred relations, then learns to recover masked or
missing graph structure from observed context using joint-embedding predictive
learning.

The paper studies two deployment settings: **Option A**, a note-free model for
admissions without available notes, and **Option B**, a note-augmented model
that places a Clinical-ModernBERT embedding only on the entities grounded by
the note. The reported entity-localized design improves inferred-edge recovery
from **0.274 to 0.419 MRR** over the note-free option while keeping the learned
world-model encoder frozen during relation readout.

## Method at a glance

![Clinical Graph-JEPA architecture from Figure 1 of the paper](docs/assets/clinical_graph_jepa_overview.png)

*Figure 1 from the paper. See the [full PDF](paper/clinical_jepa.pdf) for its
caption and surrounding discussion.*

The figure has three stages:

1. **Admission record to patient-state KG.** Structured MIMIC-IV tables provide
   high-confidence backbone edges. The clinical narrative provides cross-links
   such as a medication being managed for a diagnosis or a procedure confirming
   a diagnosis. Nodes and relations are normalized into a typed graph.
2. **Graph-JEPA world-model learning.** A GNN converts node features and typed
   edges into patient-state latents. The modular models partition the graph into
   balanced local patches; an online branch predicts masked target-patch latents
   produced by a slowly updated EMA target branch. The standalone paper-v16
   implementation follows the same predictive principle at node level.
3. **Graph revision and recovery.** Learned latents score observed and candidate
   relations. Schema rules reject clinically invalid type combinations, while
   leave-one-out evaluation measures whether a hidden true target is recovered
   above type-compatible alternatives.

The important JEPA idea is that the model predicts a **latent representation**
of missing graph state, not raw node text. The target encoder is not optimized
directly by backpropagation; its weights track the online encoder by an
exponential moving average. This supplies a stable learning target without
requiring a manually labeled target for every masked region.

## Included implementations

| Model | Package | Entity representation | Note input | Checkpoint input |
| --- | --- | --- | --- | --- |
| **Raw Graph-JEPA v5** | `graph_jepa_v5` | 768-d SapBERT | None | 768 |
| **MIMIC Graph-JEPA v6** | `graph_jepa_v6` | 768-d SapBERT | 768-d entity-localized Clinical-ModernBERT | 1536 |
| **Paper entity-note v16** | `paper_v16` | Learned type + hashed-entity + demographics | 768-d entity-localized Clinical-ModernBERT | 774-dimensional numeric branch projected to hidden space |

The modular `v5`/`v6` models and standalone paper `v16` model come from two
related development lineages. `v16` is not a direct architectural successor of
modular `v6`, and their checkpoints are not interchangeable. See the
[paper-to-code map](docs/PAPER_CODE_MAP.md) for the exact correspondence.

### Architecture comparison

| Stage | Raw v5 | Modular v6 | Paper-v16 |
| --- | --- | --- | --- |
| Raw node signal | `TYPE: normalized_name` | `TYPE: normalized_name` plus admission note embedding | Node type, hashed normalized name, demographics, optional note embedding |
| Initial representation | Frozen SapBERT CLS embedding, 768-d | SapBERT 768-d + localized note 768-d = 1536-d | Three 128-d terms are added: learned type, learned hash bucket, projected numeric/note branch |
| GNN | 2-layer relation-aware GINE | Same as v5 | 2-layer, 4-head `TransformerConv` |
| Graph unit predicted by JEPA | Balanced BFS patch | Balanced BFS patch | Masked node latent |
| Latent width | 160 | 160 | 128 |
| Target branch | EMA node encoder + EMA patch transformer | Same as v5 | EMA copy of the graph encoder |
| Downstream objective | Schema-aware revision BCE + hard candidate ranking | Same, with confidence/artifact metadata | Frozen-encoder DistMult with InfoNCE and 8 type-matched negatives |
| Intended role | Note-free baseline and deployment option | Modular note-localization experiment | Direct standalone paper experiment/checkpoint |

For a full tensor-by-tensor walkthrough, use the model guides:

- [Raw Graph-JEPA v5 explained](models/v5_without_note/README.md)
- [MIMIC Graph-JEPA v6 explained](models/v6_with_note/README.md)
- [Paper entity-note v16 explained](models/paper_v16/README.md)

## Project evaluation snapshot

The latest supplied leave-one-out edge-recovery results are:

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: |
| **Modular v6** | **0.865** | **0.779** | **0.950** | **1.000** |
| Paper-v16 | 0.571 | 0.429 | 0.626 | 0.872 |

MRR rewards placing the true target near the top of the candidate ranking.
Hits@K is the fraction of queries for which the true target appears among the
top K candidates. In this evaluation snapshot, modular v6 ranks the true target
first for 77.9% of queries and within its top ten for every evaluated query.

> [!NOTE]
> These are project-supplied evaluation results, separate from the paper's
> reported Option A/Option B experiment above. A model comparison is strictly
> valid only when both rows use the same graph records, held-out queries,
> candidate construction, note availability, filtering rules, and query cap.
> Until the corresponding result JSON/configuration is committed, treat this
> table as a reported evaluation snapshot rather than an independently
> reproducible paper result.

## Repository contents

```text
clinical-graph-jepa/
├── README.md                           Project overview and runnable examples
├── pyproject.toml                     Python package and dependency definition
├── uv.lock                            Reproducible dependency lockfile
│
├── data/
│   ├── README.md                       Dataset contract, audit, and privacy notes
│   └── fawkes_1k_patients/
│       └── fawkes_1k_patients_graphs_260615.jsonl
│                                      400 admission-level patient graphs
│
├── docs/
│   ├── ARCHITECTURE.md                  Cross-model architecture walkthrough
│   ├── DATA.md                          JSONL fields and model compatibility
│   ├── EVALUATION.md                    LOO protocol, metrics, and result snapshot
│   ├── PAPER_CODE_MAP.md                Paper concepts mapped to code symbols
│   └── assets/
│       └── clinical_graph_jepa_overview.png
│                                      Figure 1 extracted from the paper
│
├── models/
│   ├── MANIFEST.json                    Paths, sizes, and SHA-256 checksums
│   ├── v5_without_note/
│   │   ├── README.md                   Complete v5 model guide
│   │   ├── config_v5_pretrain.json    Pretraining configuration
│   │   ├── config_v5.json             Final configuration
│   │   ├── graph_jepa_v5_pretrain.pt  Pretrained checkpoint
│   │   └── graph_jepa_v5.pt           Final checkpoint
│   ├── v6_with_note/
│   │   ├── README.md                   Complete v6 model guide
│   │   ├── config_v6_pretrain.json    Pretraining configuration
│   │   ├── config_v6.json             Final configuration
│   │   ├── graph_jepa_v6_pretrain.pt  Pretrained checkpoint
│   │   └── graph_jepa_v6.pt           Final checkpoint
│   └── paper_v16/
│       ├── README.md                   Complete paper-v16 model guide
│       └── fawkes_trainer_jepa_entity_note_v16_260615.pt
│                                      Encoder + DistMult checkpoint
│
├── paper/
│   ├── clinical_jepa.pdf                Repository copy of the manuscript
│   └── README.md                       Paper links and checksum
│
├── scripts/
│   ├── audit_data.py                   Counts graphs/features and checks compatibility
│   └── smoke_check.py                  Loads every checkpoint and validates dimensions
│
├── src/
│   ├── fawkes_core/                    Shared, version-neutral implementation
│   │   ├── schema.py                   Node/relation vocabularies and graph object
│   │   ├── data.py                     JSONL adapters and schema normalization
│   │   ├── encoders.py                 SapBERT, BGE, and deterministic mock encoders
│   │   ├── patches.py                  Balanced BFS graph partitioning
│   │   ├── model_base.py               GNN, patch transformer, EMA, and edge head
│   │   ├── revision.py                 Schema-aware revision losses
│   │   ├── training.py                 Shared training utilities
│   │   └── score_revision.py           KEEP/REVIEW/PRUNE/ADD scoring workflow
│   │
│   ├── graph_jepa_v5/                  Note-free modular implementation
│   │   ├── config.py, data.py, model.py Model definition and input pipeline
│   │   ├── pretrain.py, finetune.py    Two-stage training CLIs
│   │   ├── evaluate.py                 Native leave-one-out evaluator
│   │   └── score.py                    Revision/candidate-scoring CLI
│   │
│   ├── graph_jepa_v6/                  Entity-localized-note implementation
│   │   ├── config.py, data.py, model.py Note-aware definition and input pipeline
│   │   ├── pretrain.py, finetune.py    Two-stage training CLIs
│   │   ├── evaluate.py                 Native leave-one-out evaluator
│   │   └── score.py                    Revision/candidate-scoring CLI
│   │
│   └── paper_v16/                      Standalone paper experiment
│       ├── trainer.py                   Data conversion, JEPA, DistMult, and training
│       └── evaluate.py                  Checkpoint-only local LOO evaluator
│
└── tests/
    └── test_suite.py                   Package independence, data, and checkpoint tests
```

The modular packages are self-contained: neither imports historical
`graph_jepa_v2`, `graph_jepa_v3`, or `graph_jepa_v4` modules.

### Where should a new reader start?

| Goal | Start here | Then inspect |
| --- | --- | --- |
| Understand the research idea | [Paper](paper/clinical_jepa.pdf) | [Paper-to-code map](docs/PAPER_CODE_MAP.md) |
| Compare all three architectures | [Architecture guide](docs/ARCHITECTURE.md) | The three model READMEs under `models/` |
| Understand the JSONL | [Data contract](docs/DATA.md) | `fawkes_core/data.py` and the model's `data.py` |
| Run the packaged dataset | [v5 guide](models/v5_without_note/README.md) | `graph_jepa_v5.evaluate` |
| Understand localized notes | [v6 guide](models/v6_with_note/README.md) | `graph_jepa_v6/data.py` |
| Reproduce the standalone paper model | [paper-v16 guide](models/paper_v16/README.md) | `paper_v16/trainer.py` |
| Understand metrics/results | [Evaluation guide](docs/EVALUATION.md) | Each package's `evaluate.py` |
| Verify downloaded/copied artifacts | `models/MANIFEST.json` | `scripts/smoke_check.py` |

## Quick start

### Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- CPU, Apple Silicon, or CUDA-supported PyTorch device

### Install and verify

```bash
git clone https://github.com/nalin2002/clinical-graph-jepa.git
cd clinical-graph-jepa
uv sync --extra test

uv run python scripts/audit_data.py
uv run python scripts/smoke_check.py
uv run pytest
```

SapBERT weights are downloaded from Hugging Face on the first modular scoring
or evaluation run and cached under `.cache/`.

## Evaluation

### Option A: raw v5 without notes

This model can be evaluated directly on the included graph dataset:

```bash
uv run python -m graph_jepa_v5.evaluate \
  --checkpoint models/v5_without_note/graph_jepa_v5.pt \
  --data jsonl \
  --jsonl-path data/fawkes_1k_patients/fawkes_1k_patients_graphs_260615.jsonl \
  --max-graphs 10 \
  --cap 500 \
  --device cpu \
  --output outputs/v5_loo.json
```

Remove `--max-graphs` and increase `--cap` for a complete run.

### Option B: modular v6 with localized notes

The included raw JSONL has no note embeddings. Supply the embedded dataset used
by the checkpoint:

```bash
uv run python -m graph_jepa_v6.evaluate \
  --checkpoint models/v6_with_note/graph_jepa_v6.pt \
  --data jsonl \
  --jsonl-path /path/to/fawkes_training_graph_full_embedded_260615.jsonl \
  --device cpu \
  --output outputs/v6_loo.json
```

### Standalone paper-v16

The standalone evaluator reads experiment switches from the environment. Match
the released checkpoint explicitly:

```bash
USE_NOTE=1 GROUND_BY=prov EMBED_DIM=768 USE_SCORES=0 PRUNE_NO_EVIDENCE=1 \
uv run python -m paper_v16.evaluate \
  --checkpoint models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt \
  --data /path/to/fawkes_training_graph_full_embedded_260615.jsonl \
  --device cpu \
  --output outputs/paper_v16_loo.json
```

## Data and reproducibility

The included JSONL contains **400 admission graphs**, **17,665 nodes**, and
**22,795 edges**. It contains the structured and LLM-inferred graph topology,
but not the private note text, note embeddings, or evidence-score vectors.

| Reproduction path | Included data sufficient? |
| --- | --- |
| Raw v5 / Option A | Yes |
| Modular v6 / Option B | No - requires the original embedded JSONL |
| Paper-v16 with notes | No - requires note embeddings and provenance fields |

Running a note model with zero-filled missing vectors is structurally possible,
but it is not a faithful reproduction of the note-augmented checkpoint.

## Training

Training is model-specific. Start with the corresponding guide:

- [Raw v5 training and evaluation](models/v5_without_note/README.md)
- [v6 localized-note training and evaluation](models/v6_with_note/README.md)
- [paper-v16 training and evaluation](models/paper_v16/README.md)

For implementation details, read:

- [Architecture](docs/ARCHITECTURE.md)
- [Data contract](docs/DATA.md)
- [Evaluation protocol](docs/EVALUATION.md)
- [Paper-to-code implementation map](docs/PAPER_CODE_MAP.md)

## Citation

If you use this repository, cite the
[Clinical Graph-JEPA OpenReview paper](https://openreview.net/forum?id=HXsMPubPqE).
The repository includes a [local PDF](paper/clinical_jepa.pdf) for offline
reference. Author metadata in the supplied manuscript is anonymized, so this
README does not invent a BibTeX author list.

## Acknowledgements

This implementation builds on PyTorch, PyTorch Geometric, Hugging Face
Transformers, SapBERT, Clinical-ModernBERT, MIMIC-IV, and ACI-Bench.
