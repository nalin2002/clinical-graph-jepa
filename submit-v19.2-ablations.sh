cd /Users/kushagrayadav/Code/clinical-graph-jepa

# v19.2: v19's JEPA-pretraining ablation, re-run on the v22 configuration.
#
# v19 asked how much of the result comes from phase 1 rather than from having any
# graph encoder at all, and answered it on the v16/v20 configuration (random masking,
# DistMult head). The configuration has moved since: v22 is patch masking + MLP head.
# The ablation has to move with it, or the paper's headline number and the ablation
# that qualifies it describe two different models.
#
# Same six arms as v19, same knobs, same seeds — only the base configuration differs:
#   A   JEPA 60 ep, frozen encoder      the v22 reference
#   B   no phase 1, frozen              random-init frozen encoder
#   C   no phase 1, trained jointly     from scratch, end-to-end
#   Cp  as C, READOUT_EPOCHS=100        compute-matched, so budget is not the excuse
#   D   as C, LAYERS=0                  no message passing — entity memorisation alone
#   E   JEPA 60 ep, fine-tuned          pretrain then unfreeze
#
# ARM A IS NOT SUBMITTED. With JEPA_EPOCHS, FREEZE_ENCODER, READOUT_EPOCHS and LAYERS
# all left at their defaults, arm A's environment is v22's exactly, and DETERMINISTIC=1
# — so kushagrayadv/fawkes-v22-patch-mlp-sp42-s{42,43,44} ARE arm A. print_v19.2_ablation_table.py
# reads them from there. Re-running would spend three 4h T4 jobs to reproduce runs that
# already exist.
#
# JEPA_EPOCHS=0 makes pretrain_jepa's loop body never execute (train.py:144), so
# MASK_STRATEGY and JEPA_PATCHES are inert in B, C, Cp and D — the patch pretext task
# is exercised only by A and E. Those four arms therefore differ from their v19
# counterparts by DECODER=mlp alone. That is not redundancy: the ablation is measured
# against the v22 reference, and the reference now has an MLP head.

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check — fail here rather than eleven jobs later

submit () {                        # submit <arm> <seed> <timeout> [EXTRA=VAL ...]
  local arm=$1 seed=$2 timeout=$3; shift 3
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting v19.2 $arm-s$seed ==="
  $HF jobs uv run --detach --flavor t4-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-v19-2-abl-$arm-s$seed" \
    -v ./src:/workspace/src \
    -e MASK_STRATEGY=patch -e JEPA_PATCHES=8 -e NODE_MASK=0.4 \
    -e USE_NOTE=1 -e GROUND_BY=prov -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e DECODER=mlp \
    -e PUSH=1 -e DATA_SPLIT_SEED=42 -e SEED="$seed" \
    -e OUTPUT_REPO="kushagrayadv/fawkes-v19-2-ablation-$arm-sp42-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

# The experiment is v19.2 but the names say v19-2: a job --name is stored as a tag, and
# tags reject '.'. huggingface_hub sanitizes auto-generated names but passes an explicit
# --name through untouched, so the dot fails server-side at submit. Repo names would
# accept it; they drop it too so the job you find in `hf jobs ps` and the repo it writes
# are the same string.
#
# DATA_SPLIT_SEED=42 holds the split fixed across every arm and matches v22's, so each
# arm is paired against A seed by seed on the same held-out admissions. None of
# MASK_STRATEGY, JEPA_PATCHES, DECODER or DATA_SPLIT_SEED reaches checkpoint_dict(),
# so the repo name is the only provenance record — do not rename.
for s in 42 43 44; do
  submit B  $s 4h JEPA_EPOCHS=0
  submit C  $s 4h JEPA_EPOCHS=0 FREEZE_ENCODER=0
  submit Cp $s 4h JEPA_EPOCHS=0 FREEZE_ENCODER=0 READOUT_EPOCHS=100
done
submit D 42 4h JEPA_EPOCHS=0 FREEZE_ENCODER=0 LAYERS=0
submit E 42 4h FREEZE_ENCODER=0
