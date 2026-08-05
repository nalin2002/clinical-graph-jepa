cd /Users/kushagrayadav/Code/clinical-graph-jepa

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check — every repo name is built from this


submit () {                        # submit <arm> <seed> <timeout> [EXTRA=VAL ...]
  local arm=$1 seed=$2 timeout=$3; shift 3
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting $arm-s$seed ==="
  $HF jobs uv run --detach --flavor t4-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-v19-abl-$arm-s$seed" \
    -v ./src:/workspace/src \
    -e MASK_STRATEGY=random \
    -e USE_NOTE=1 -e GROUND_BY=prov -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e PUSH=1 -e SEED="$seed" \
    -e OUTPUT_REPO="kushagrayadv/fawkes-v19-ablation-$arm-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

for s in 42 43 44; do
  submit A  $s 4h
  submit B  $s 4h JEPA_EPOCHS=0
  submit C  $s 4h JEPA_EPOCHS=0 FREEZE_ENCODER=0
  submit Cp $s 4h JEPA_EPOCHS=0 FREEZE_ENCODER=0 READOUT_EPOCHS=100
done
submit D 42 4h JEPA_EPOCHS=0 FREEZE_ENCODER=0 LAYERS=0
submit E 42 4h FREEZE_ENCODER=0