cd /Users/kushagrayadav/Code/clinical-graph-jepa

# v20: the Table 1 variance study. One arm — the paper config — over 10 training
# seeds at the published split. DATA_SPLIT_SEED stays 42 throughout, so every run
# holds out the same 400 admissions and the spread is initialisation, batch order
# and JEPA masking only. sp42-s42 is the published run and should reproduce its
# published MRR.
#
# This measures initialisation variance, NOT patient-sample variance. To answer
# the review's "one particular random sample of patients" as well, add:
#   for split in 43 44 45 46; do submit $split 42 4h; done
# which gives 5 splits at seed 42, sharing the sp42-s42 cell.

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check — fail here rather than ten jobs later

submit () {                        # submit <split_seed> <seed> <timeout> [EXTRA=VAL ...]
  local split=$1 seed=$2 timeout=$3; shift 3
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting sp$split-s$seed ==="
  $HF jobs uv run --detach --flavor t4-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-v20-var-sp$split-s$seed" \
    -v ./src:/workspace/src \
    -e MASK_STRATEGY=random \
    -e USE_NOTE=1 -e GROUND_BY=prov -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e PUSH=1 -e DATA_SPLIT_SEED="$split" -e SEED="$seed" \
    -e OUTPUT_REPO="kushagrayadv/fawkes-v20-variance-sp$split-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

# RUN_CASCADE=0 above: Table 1's published row is the LOO metric block. Flip it
# to 1 in submit() if Table 1 also carries cascade columns.
for seed in 42 43 44 45 46 47 48 49 50 51; do
  submit 42 $seed 4h
done
