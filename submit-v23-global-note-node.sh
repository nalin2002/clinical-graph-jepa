#!/usr/bin/env bash
set -euo pipefail

# Submit after pushing the exact source commit named by SOURCE_REF.
: "${SOURCE_REF:?Set SOURCE_REF to the pushed 40-character git commit}"

HF_USER=$(hf auth whoami --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["user"])')
NOTE_MEMORY_REPO=${NOTE_MEMORY_REPO:-wmatbooth/fawkes-training-note-memory-v23-260808}
SOURCE_REPO=${SOURCE_REPO:-wmatbooth/clinical-graph-jepa-v23-source}
FLAVOR=${FLAVOR:-a10g-small}

submit() {
  local seed=$1 timeout=$2
  local output_repo="${HF_USER}/fawkes-v23-global-note-node-sp42-s${seed}"
  hf jobs uv run --detach --flavor "$FLAVOR" --timeout "$timeout" \
    --secrets HF_TOKEN --label experiment=fawkes-v23-global-note-node \
    --env SOURCE_REF="$SOURCE_REF" --env SOURCE_REPO="$SOURCE_REPO" \
    --env NOTE_INJECTION=mean \
    --env GLOBAL_NOTE_NODE=1 \
    --env NOTE_MEMORY_REPO="$NOTE_MEMORY_REPO" \
    --env DATA_SPLIT_SEED=42 --env SEED="$seed" \
    --env OUTPUT_REPO="$output_repo" \
    scripts/fawkes_v23_hf_job.py
}

# Fixed ten-seed confirmatory set used for the global NOTE-node aggregate.
SEEDS=(42 43 44 45 46 47 48 49 50 51)
for seed in "${SEEDS[@]}"; do
  submit "$seed" 5h
done
