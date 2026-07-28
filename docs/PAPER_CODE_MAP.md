# Paper-to-code implementation map

## References

- [Repository copy of the paper](../paper/clinical_jepa.pdf)
- [OpenReview record](https://openreview.net/forum?id=HXsMPubPqE)

## Which model is the paper implementation?

`src/paper_v16/trainer.py` is the direct packaged form of
`fawkes_trainer_jepa_entity_note_v16_wmatbooth_260723.py` and is the closest
match to the experiments and entity-localized note method described in the
paper. Its compatible artifact is
`models/paper_v16/fawkes_trainer_jepa_entity_note_v16_260615.pt`.

The two modular checkpoints are useful comparison models:

- `graph_jepa_v5` is the note-free SapBERT Graph-JEPA model.
- `graph_jepa_v6` adds a localized 768-dimensional note embedding to the
  SapBERT node representation.

Those modular models implement the patch-based Graph-JEPA schematic shown in
the paper, but they are not checkpoint-compatible rewrites of paper-v16.
Likewise, `v16` is not a direct successor of modular `v6`; the numbers come
from different development lineages.

## Method mapping

| Paper concept | Direct paper-v16 implementation | Modular v5/v6 counterpart | Important detail |
| --- | --- | --- | --- |
| Admission-level patient-state KG | `paper_v16.trainer.to_data` consumes one graph record and converts typed nodes/edges to PyG tensors | `fawkes_core.data` provides schema normalization, builders, and PyG conversion | This suite consumes the supplied prebuilt JSONL. It does not reproduce upstream MIMIC extraction or LLM graph generation. |
| Typed node input | `paper_v16.trainer.Encoder`: learned node-type embedding, stable hashed-entity bucket embedding, demographics/numeric projection | `graph_jepa_v5` and `graph_jepa_v6`: SapBERT entity vectors plus type and numeric features | These are different entity-representation choices, not interchangeable checkpoint inputs. |
| Option A: no note | Run paper-v16 with `USE_NOTE=0` | `graph_jepa_v5` is the packaged no-note comparison checkpoint | Option A removes the 768 note features; it does not replace them with SapBERT in paper-v16. |
| Option B: localized note | `paper_v16.trainer.to_data` identifies grounded entities and appends the admission note vector only to those nodes | `graph_jepa_v6.data` supplies localized-note node inputs | The expected vector is a 768-dimensional Clinical-ModernBERT embedding. Ungrounded entities receive a 768-dimensional zero vector. |
| Graph encoder | `paper_v16.trainer.Encoder`: two `TransformerConv` layers with four attention heads | `fawkes_core.model_base.GraphNodeEncoder`: typed message passing used by modular v5/v6 | Both are GNN encoders, but their parameters and state dictionaries differ. |
| Self-supervised JEPA phase | `paper_v16.trainer.JEPA` and `jepa_step`: mask node representations and predict the target latent | `fawkes_core.model_base.GraphJEPAv3`, patch transformer, EMA target encoder, and VICReg terms | Paper-v16 uses node masking; modular v5/v6 use balanced BFS patches and an EMA target branch. |
| Frozen relation readout | `paper_v16.trainer.DistMult` and `readout_step`: freeze the encoder and train type-matched InfoNCE relation recovery | `fawkes_core.model_base.EdgePlausibilityHead` and revision loss | The paper-v16 checkpoint uses the DistMult route. |
| Leave-one-out evaluation | `paper_v16.trainer.loo_evaluate` and `src/paper_v16/evaluate.py` | each modular package has `evaluate.py`; v6 also includes the released LOO/LLM comparison evaluator | Use the evaluator belonging to the checkpoint family. |
| Cascading update test | `paper_v16.trainer.cascade_evaluate` | scoring/revision paths live in `fawkes_core.score_revision` | The standalone trainer evaluates cascades; the modular path additionally exposes revision actions and schema guards. |
| KEEP/REVIEW/PRUNE and candidate addition | Not exposed as a standalone paper-v16 command | `fawkes_core.score_revision` | These operational actions are part of the modular scoring workflow, not a claim of paper-v16 checkpoint compatibility. |

## Input dimensions: correction to 778 versus 1550

In this exact paper-v16 code, the numeric branch is **774, not 778**:
`BASE_NUMERIC=6` plus `EMBED_DIM=768`. The six values are normalized age, male,
female, and three reserved zeros. The code projects this 774-dimensional branch
to the 128-dimensional hidden space, then **adds** the separate learned type and
hashed-entity embeddings. It does not concatenate those learned embeddings to
the raw note vector, so neither 778 nor 1550 is the paper-v16 input dimension.

By contrast, modular v6 intentionally combines a 768-dimensional SapBERT vector
with a separate 768-dimensional localized note vector before its learned
projection. Its configured node input is therefore **1536** dimensions, but it
belongs to a different architecture.

## Reproduction boundaries

The repository includes the source, configuration, checkpoints, evaluation
entry points, and the supplied 1K-patient graph JSONL. The packaged JSONL has no
note embeddings, so it can faithfully exercise v5 and paper-v16 Option A only.
Faithful v6 and paper-v16 Option B reproduction requires the original embedded
JSONL containing each admission's 768-dimensional `note_embedding` and the
evidence/provenance fields expected by the checkpoint.
