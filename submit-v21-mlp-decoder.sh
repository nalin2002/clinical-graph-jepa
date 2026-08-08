cd /Users/kushagrayadav/Code/clinical-graph-jepa

# v21: swap the DistMult readout for the MLP head — the same architecture
# clinical_jepa scores edges with (relation embedding + MLP over [src, tgt, rel]),
# differing only in ReLU vs GELU. No new code: build_scorer already returns it for
# any DECODER != distmult.
#
# This arm is paired against v20. Same DATA_SPLIT_SEED=42, same seeds 42-51, same
# everything else, so the comparison is seed-by-seed against
# kushagrayadv/fawkes-v20-variance-sp42-s{seed}. Phase 1 is identical and identically
# seeded in both arms (the scorer is only built in phase 2, and FREEZE_ENCODER=1),
# so this isolates the readout head against the same encoder.

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check — fail here rather than ten jobs later

submit () {                        # submit <split_seed> <seed> <timeout> [EXTRA=VAL ...]
  local split=$1 seed=$2 timeout=$3; shift 3
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting v21 mlp sp$split-s$seed ==="
  $HF jobs uv run --detach --flavor t4-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-v21-mlp-sp$split-s$seed" \
    -v ./src:/workspace/src \
    -e MASK_STRATEGY=random \
    -e USE_NOTE=1 -e GROUND_BY=prov -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e DECODER=mlp \
    -e PUSH=1 -e DATA_SPLIT_SEED="$split" -e SEED="$seed" \
    -e OUTPUT_REPO="kushagrayadv/fawkes-v21-mlp-sp$split-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

# DECODER is NOT written into checkpoint_dict(), so a v21 checkpoint is
# indistinguishable from a v20 one by its config block. The repo name is the only
# provenance record — do not rename these.
for seed in 42 43 44 45 46 47 48 49 50 51; do
  submit 42 $seed 4h
done
