#!/usr/bin/env bash
set -euo pipefail

cd /Users/kushagrayadav/Code/clinical-graph-jepa

# Clinical-JEPA seed variance over the fixed Fawkes paper split.
#
# This mirrors the Fawkes HF sweep shape:
#   - DATA_SPLIT_SEED stays 42, so the held-out 400 admissions are fixed.
#   - SEED varies, so the spread is model initialisation, batch order, and JEPA
#     masking/task sampling.
#   - Both Clinical-JEPA variants run for each seed and are evaluated on the
#     same Fawkes-defined 8,283-query shared manifest.
#
# The job mounts ./src and ./models, downloads the 4,000-record dataset inside
# HF Jobs, recreates outputs/splits/fawkes_train_plus_val.jsonl there, trains,
# evaluates, and uploads checkpoints + shared_queries_eval.json to OUTPUT_REPO.

HF=.venv/bin/hf
HF_AUTH_USER=$($HF auth whoami | head -1)
HF_OWNER=${CLINICAL_JEPA_HF_USER:-wmatbooth}
echo "HF auth: $HF_AUTH_USER"
echo "HF output namespace: $HF_OWNER"

submit () {              # submit <variant> <split_seed> <seed> <timeout> [EXTRA=VAL ...]
  local variant=$1 split=$2 seed=$3 timeout=$4; shift 4
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  local label=${variant//_/-}
  echo "=== submitting clinical-jepa-$label sp$split-s$seed ==="
  $HF jobs uv run --detach --flavor a10g-small --timeout "$timeout" \
    --namespace "$HF_OWNER" \
    --secrets HF_TOKEN --name "clinical-jepa-$label-sp$split-s$seed" \
    -v ./src:/workspace/src \
    -v ./models:/workspace/models \
    -e VARIANT="$label" \
    -e DATA_SPLIT_SEED="$split" -e SEED="$seed" \
    -e PRETRAIN_EPOCHS=60 \
    -e USE_TORCH=1 -e USE_TF=0 \
    -e PUSH=1 \
    -e OUTPUT_REPO="$HF_OWNER/clinical-jepa-$label-fawkes-sp$split-s$seed" \
    "${envs[@]}" \
    scripts/clinical_jepa_hf_job.py
}

for seed in 42 43 44 45 46 47 48 49 50 51; do
  submit note 42 "$seed" 8h FINETUNE_EPOCHS=50
  submit no-note 42 "$seed" 8h FINETUNE_EPOCHS=50
done
