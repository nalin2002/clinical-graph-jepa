cd /Users/kushagrayadav/Code/clinical-graph-jepa

# v22: BFS-balanced patch masking (v18's pretext task) on top of v21's MLP readout.
#
# This is the second cell of a 2x2 over {random, patch} x {distmult, mlp}:
#   (random, distmult) = v20    (random, mlp) = v21    (patch, mlp) = v22
# The fourth cell, (patch, distmult), is not run here. Without it the mask-strategy
# effect cannot be separated from an interaction with the readout head — see the note
# at the bottom of print_v22_patch_mlp_table.py.
#
# Paired against v21 seed by seed: same DATA_SPLIT_SEED=42, same seeds 42-51, only
# MASK_STRATEGY differs. Unlike v21 vs v20, this changes the PRETEXT TASK, so phase 1
# differs between the arms and the encoders are genuinely different models — the
# pairing controls the split and the seed, not the encoder.

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check — fail here rather than ten jobs later

submit () {                        # submit <split_seed> <seed> <timeout> [EXTRA=VAL ...]
  local split=$1 seed=$2 timeout=$3; shift 3
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting v22 patch+mlp sp$split-s$seed ==="
  $HF jobs uv run --detach --flavor t4-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-v22-patch-mlp-sp$split-s$seed" \
    -v ./src:/workspace/src \
    -e MASK_STRATEGY=patch -e JEPA_PATCHES=8 -e NODE_MASK=0.4 \
    -e USE_NOTE=1 -e GROUND_BY=prov -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e DECODER=mlp \
    -e PUSH=1 -e DATA_SPLIT_SEED="$split" -e SEED="$seed" \
    -e OUTPUT_REPO="kushagrayadv/fawkes-v22-patch-mlp-sp$split-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

# MASK_STRATEGY and JEPA_PATCHES are passed explicitly even though the job script
# already defaults to them: the job log should record the experiment, not inherit it.
# None of MASK_STRATEGY, JEPA_PATCHES, DECODER or DATA_SPLIT_SEED reaches
# checkpoint_dict(), so the repo name is the only provenance record — do not rename.
for seed in 42 43 44 45 46 47 48 49 50 51; do
  submit 42 $seed 4h
done
