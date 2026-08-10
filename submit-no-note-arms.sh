cd /Users/kushagrayadav/Code/clinical-graph-jepa

# The no-note cells of v20, v21 and v22 — USE_NOTE=0 across all three arms.
#
# Every v20/v21/v22 run so far set USE_NOTE=1, so the whole 2x2 sits on one side of
# the note axis. v18 ran both variants but at ONE seed per cell, and v20 later
# measured the noise floor at +-0.019 — which is most of v18's no-note effect. So:
#
#   * v18's only POSITIVE result (no-note patch masking, +0.0237 MRR) has never been
#     tested at power. v22 gave the with-note arm ten paired seeds and found nothing;
#     the no-note arm, which carried v18's entire mechanism argument, never got them.
#   * v21's +0.0530 readout-head win — the change this project has adopted — has only
#     been measured with the note present. Whether it is a general property of the
#     head or entangled with the note channel is untested.
#   * The note lift itself (v18 put it at +0.144, narrowing to +0.110) has no error
#     bar, because both numbers came from single runs.
#
# Paired against the existing note arms seed by seed: same DATA_SPLIT_SEED=42, same
# seeds 42-51, same everything but USE_NOTE. Note that USE_NOTE changes the encoder's
# input width (numeric_dim 774 -> 6), so these are different models, not a readout
# swap — the pairing controls the split and the seed, not the encoder.
#
# Pre-flight: scripts/preflight_no_note.py runs all three arms end to end locally
# (both phases, both evaluators) before any of this is submitted. Run it first.
#
# These are the FIRST runs to carry the run_config provenance block, so unlike the
# existing thirty they identify their own arm. The repo names below still follow the
# old convention; keep them anyway until every checkpoint in the series is
# self-describing.

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check — fail here rather than thirty jobs later

submit () {              # submit <name> <split_seed> <seed> <timeout> [EXTRA=VAL ...]
  local name=$1 split=$2 seed=$3 timeout=$4; shift 4
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting $name sp$split-s$seed ==="
  $HF jobs uv run --detach --flavor t4-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-$name-sp$split-s$seed" \
    -v ./src:/workspace/src \
    -e USE_NOTE=0 -e GROUND_BY=prov -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e PUSH=1 -e DATA_SPLIT_SEED="$split" -e SEED="$seed" \
    -e OUTPUT_REPO="kushagrayadv/fawkes-$name-sp$split-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

# MASK_STRATEGY, JEPA_PATCHES and DECODER are passed explicitly even where the job
# script already defaults to them: the job log should record the experiment, not
# inherit it. NODE_MASK is pinned at the value every arm in the series used.
for seed in 42 43 44 45 46 47 48 49 50 51; do
  submit v20-nonote          42 $seed 4h MASK_STRATEGY=random DECODER=distmult NODE_MASK=0.4
  submit v21-nonote-mlp      42 $seed 4h MASK_STRATEGY=random DECODER=mlp      NODE_MASK=0.4
  submit v22-nonote-patchmlp 42 $seed 4h MASK_STRATEGY=patch  DECODER=mlp      NODE_MASK=0.4 JEPA_PATCHES=8
done
