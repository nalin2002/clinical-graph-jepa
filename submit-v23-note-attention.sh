cd /Users/kushagrayadav/Code/clinical-graph-jepa

# v23: the note-injection ablation -- how the discharge note is POOLED before it is
# written onto the provenance-grounded entities.
#
#   mean       one global 768-d Clinical ModernBERT mean per admission = the v22 path
#   uniform    token-count-weighted mean over 32-token spans (preprocessing control)
#   attention  entity-conditioned four-head attention over those spans
#
# Paired against v22 seed by seed: same DATA_SPLIT_SEED=42, same seeds 42-51, same
# GROUND_BY=prov / patch masking / MLP readout. Only the pooling changes. The `mean`
# arm is the v22 path retrained here rather than v22's own checkpoints, so all three
# arms sit on one GPU type and the comparison carries no cross-hardware confound.
#
# WORKFLOW -- this is now the v22 workflow, and that is the point:
# the job MOUNTS ./src, so the code that runs is your working tree. Edit, submit, read
# the log. There is no source tarball, no pushed commit, no SOURCE_REF.
#
# scripts/fawkes_v23_hf_job.py is the OTHER path. It fetches <SOURCE_REF>.tar.gz from
# a dataset repo and pins the code to a commit. That exists to reproduce nalin9's
# published runs bit for bit; use it only when you need those exact numbers, not for
# your own experiments. scripts/fawkes_patch_masking_hf_job.py already carries the
# safetensors dependency and logs every NOTE_* knob, so it runs all three arms.

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check -- fail here rather than thirty jobs later

NOTE_MEMORY_REPO=${NOTE_MEMORY_REPO:-wmatbooth/fawkes-training-note-memory-v23-260808}

submit () {              # submit <mode> <split_seed> <seed> <timeout> [EXTRA=VAL ...]
  local mode=$1 split=$2 seed=$3 timeout=$4; shift 4
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting v23-$mode sp$split-s$seed ==="
  $HF jobs uv run --detach --flavor a10g-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-v23-$mode-note-sp$split-s$seed" \
    -v ./src:/workspace/src \
    -e USE_NOTE=1 -e GROUND_BY=prov -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e NOTE_INJECTION="$mode" \
    -e NOTE_MEMORY_REPO="$NOTE_MEMORY_REPO" \
    -e NOTE_MEMORY_FILE=fawkes_note_memory_v23.safetensors \
    -e NOTE_SPAN_TOKENS=32 -e NOTE_MAX_SPANS=64 -e NOTE_ATTN_HEADS=4 \
    -e PUSH=1 -e DATA_SPLIT_SEED="$split" -e SEED="$seed" \
    -e OUTPUT_REPO="$HF_USER/fawkes-v23-$mode-note-sp$split-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

# MASK_STRATEGY, JEPA_PATCHES, NODE_MASK and DECODER are passed explicitly even though
# the job script defaults to them: the job log should record the experiment, not
# inherit it. a10g-small rather than v22's t4-small because the attention arm holds
# the span memory -- and all three arms take it so the comparison is same-hardware.
for seed in 42 43 44 45 46 47 48 49 50 51; do
  for mode in mean uniform attention; do
    submit "$mode" 42 "$seed" 5h MASK_STRATEGY=patch JEPA_PATCHES=8 NODE_MASK=0.4 DECODER=mlp
  done
done
