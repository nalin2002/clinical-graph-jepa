#!/usr/bin/env bash
set -euo pipefail

# Submit after pushing the exact source commit named by SOURCE_REF.
: "${SOURCE_REF:?Set SOURCE_REF to the pushed 40-character git commit}"

HF_USER=$(hf auth whoami --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["user"])')
NOTE_MEMORY_REPO=${NOTE_MEMORY_REPO:-nalin9/fawkes-training-note-memory-v23-260808}
FLAVOR=${FLAVOR:-a10g-small}

submit() {
  local mode=$1 seed=$2 timeout=$3
  local output_repo="${HF_USER}/fawkes-v23-${mode}-note-sp42-s${seed}"
  hf jobs uv run --detach --flavor "$FLAVOR" --timeout "$timeout" \
    --secrets HF_TOKEN --label experiment=fawkes-v23-note-injection \
    --env SOURCE_REF="$SOURCE_REF" --env NOTE_INJECTION="$mode" \
    --env NOTE_MEMORY_REPO="$NOTE_MEMORY_REPO" \
    --env DATA_SPLIT_SEED=42 --env SEED="$seed" \
    --env OUTPUT_REPO="$output_repo" \
    scripts/fawkes_v23_hf_job.py
}

SEEDS=(42 44 45 47 49 50 51)

for mode in mean uniform attention; do
  for seed in "${SEEDS[@]}"; do
    submit "$mode" "$seed" 5h
  done
done
