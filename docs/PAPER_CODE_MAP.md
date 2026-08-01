# Paper-to-code implementation map

## References

- [Repository copy of the paper](../paper/clinical_jepa.pdf)
- [OpenReview record](https://openreview.net/forum?id=HXsMPubPqE)

## Which model is the paper implementation?

**`src/fawkes/` is the paper implementation. `src/clinical_jepa/` is not.**

This inverts what the repository name (`clinical-graph-jepa`) and the manuscript
filename (`paper/clinical_jepa.pdf`) suggest. `fawkes` keeps its original
author's name. If you are looking for the code behind the paper, read
`src/fawkes/`.

`src/fawkes/` is the direct packaged form of
`fawkes_trainer_jepa_entity_note_v16_wmatbooth_260723.py` and is the closest
match to the experiments and entity-localized note method described in the
paper. Its compatible artifact is
`models/fawkes-entity-note/fawkes_trainer_jepa_entity_note_v16_260615.pt`. The
original single `trainer.py` is now split across
`fawkes/{config,data,model,train,evaluate}.py`; `docs/LINEAGE.md` records the
mapping.

The two `clinical_jepa` checkpoints are useful comparison models:

- `models/clinical-jepa-no-note/` is the note-free SapBERT Graph-JEPA model.
- `models/clinical-jepa-localized-note/` adds a localized 768-dimensional note
  embedding to the SapBERT node representation.

Both are the same `clinical_jepa.model.GraphJEPA` class under two settings of
`cfg.model.use_note_embeddings`. They implement the patch-based Graph-JEPA
schematic shown in the paper, but they are not checkpoint-compatible rewrites of
the paper model, and the paper model is not a successor of either: the numbers
come from different development lineages.

## Method mapping

| Paper concept | `fawkes` (the paper implementation) | `clinical_jepa` counterpart | Important detail |
| --- | --- | --- | --- |
| Admission-level patient-state KG | `fawkes.data.to_data` consumes one graph record and converts typed nodes/edges to PyG tensors | `clinical_jepa.graph.builders` provides schema normalization and builders; `clinical_jepa.graph.tensors` does PyG conversion | This suite consumes the supplied prebuilt JSONL. It does not reproduce upstream MIMIC extraction or LLM graph generation. |
| Typed node input | `fawkes.model.Encoder`: learned node-type embedding, stable hashed-entity bucket embedding, demographics/numeric projection | `clinical_jepa`: SapBERT entity vectors plus type and numeric features | These are different entity-representation choices, not interchangeable checkpoint inputs. |
| Option A: no note | Run `fawkes` with `USE_NOTE=0` | `models/clinical-jepa-no-note/` is the packaged no-note comparison checkpoint | Option A removes the 768 note features; it does not replace them with SapBERT in `fawkes`. |
| Option B: localized note | `fawkes.data.to_data` identifies grounded entities and appends the admission note vector only to those nodes | `clinical_jepa.graph.tensors.to_graph_data` with `use_note_embeddings=True` supplies localized-note node inputs | The expected vector is a 768-dimensional Clinical-ModernBERT embedding. Ungrounded entities receive a 768-dimensional zero vector. |
| Graph encoder | `fawkes.model.Encoder`: two `TransformerConv` layers with four attention heads | `clinical_jepa.model.GraphJEPA`'s node encoder: relation-aware GINE message passing | Both are GNN encoders, but their parameters and state dictionaries differ. |
| Self-supervised JEPA phase | `fawkes.model.JEPA` and `fawkes.train.jepa_step`: mask node representations and predict the target latent | `clinical_jepa.model.GraphJEPA`, patch transformer, EMA target encoder, and VICReg terms | `fawkes` uses node masking; `clinical_jepa` uses balanced BFS patches and an EMA target branch. |
| Frozen relation readout | `fawkes.model.DistMult` and `fawkes.train.readout_step`: freeze the encoder and train type-matched InfoNCE relation recovery | `clinical_jepa.model`'s edge plausibility head and the `clinical_jepa.losses` revision loss | The released paper checkpoint uses the DistMult route. |
| Leave-one-out evaluation | `fawkes.evaluate.loo_evaluate` and the `fawkes-eval` CLI | `clinical_jepa.evaluate` and the `clinical-jepa-eval` CLI | Use the evaluator belonging to the checkpoint family. |
| Cascading update test | `fawkes.evaluate.cascade_evaluate` | scoring/revision paths live in `clinical_jepa.score` | The paper model evaluates cascades; the modular path additionally exposes revision actions and schema guards. |
| KEEP/REVIEW/PRUNE and candidate addition | Not exposed as a standalone `fawkes` command | `clinical_jepa.score` and the `clinical-jepa-score` CLI | These operational actions are part of the modular scoring workflow, not a claim of paper-checkpoint compatibility. |

## Input dimensions: correction to 778 versus 1550

In this exact `fawkes` code, the numeric branch is **774, not 778**:
`BASE_NUMERIC=6` plus `EMBED_DIM=768`. The six values are normalized age, male,
female, and three reserved zeros. The code projects this 774-dimensional branch
to the 128-dimensional hidden space, then **adds** the separate learned type and
hashed-entity embeddings. It does not concatenate those learned embeddings to
the raw note vector, so neither 778 nor 1550 is the `fawkes` input dimension.

By contrast, the localized-note `clinical_jepa` variant intentionally combines a
768-dimensional SapBERT vector with a separate 768-dimensional localized note
vector before its learned projection. Its configured node input is therefore
**1536** dimensions, but it belongs to a different architecture.

## Reproduction boundaries

The repository includes the source, configuration, checkpoints, evaluation entry
points, and the 4,000-record embedded JSONL. That file carries note text,
768-dimensional `note_embedding` vectors, and per-edge provenance labels, so it
faithfully exercises all three released checkpoints — `fawkes-eval` on it
reproduces the metrics stored inside the released checkpoint exactly.

The 400-record raw JSONL whose checksum `models/MANIFEST.json` records
(`data/fawkes_1k_patients/`) is not present in this working tree and is not
recoverable from its source. It had no note embeddings, so it could faithfully
exercise only the no-note variant and `fawkes` Option A. Running a note model on
zero-filled vectors is structurally possible but is not a faithful reproduction
of a note-augmented checkpoint — which is why `--jsonl-path` now defaults to the
embedded dataset, whose note embeddings make both variants faithful.
