cd /Users/kushagrayadav/Code/clinical-graph-jepa

# The GLOBAL NOTE-NODE arm: the v15 placement, at ten paired seeds.
#
# v15 hung the discharge note on a per-admission NOTE node (PATIENT->NOTE, HAS_NOTE)
# and lost to the no-note control on every inferred relation (overall LOO MRR
# 0.274 -> 0.265). v16 moved the same vector onto the entities the note grounds and
# nearly doubled recovery (-> 0.419). That three-way is the paper's argument that
# LOCALIZATION, not the note, is the effect -- and it rests on three single runs.
#
# v20 later put this pipeline's seed noise at +-0.019. The v15 effect is -0.009, so
# the negative that retired the NOTE node sits inside the noise floor of the harness
# that produced it. This arm is the missing cell:
#
#   USE_NOTE=0                no-note control     v22-nonote-patchmlp   (10 seeds)
#   GLOBAL_NOTE_NODE=1        the v15 placement   THIS FILE             (10 seeds)
#   GROUND_BY=prov            the v16 placement   v22-patch-mlp         (10 seeds)
#
# Same DATA_SPLIT_SEED=42, same seeds 42-51, same v22 cell (patch masking + MLP
# readout), so the three arms are paired seed by seed and differ only in where the
# note sits. The pairing controls the split and the seed, not the encoder: a NOTE node
# changes what phase 1 masks, so these are different models -- as is equally true of
# the two arms they are compared against.
#
# WORKFLOW -- this is v22's, and that is the point: the job MOUNTS ./src, so the code
# that runs is your working tree. Edit, submit, read the log. No source tarball, no
# pushed commit, no SOURCE_REF. scripts/fawkes_v23_hf_job.py is the other path; it
# pins the code to a commit and exists to reproduce nalin9's published v23 runs, not
# to run your own.

HF=.venv/bin/hf
HF_USER=$($HF auth whoami | head -1)
echo "$HF_USER"          # sanity check -- fail here rather than ten jobs later

submit () {              # submit <split_seed> <seed> <timeout> [EXTRA=VAL ...]
  local split=$1 seed=$2 timeout=$3; shift 3
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "=== submitting global-note-node sp$split-s$seed ==="
  $HF jobs uv run --detach --flavor t4-small --timeout "$timeout" \
    --secrets HF_TOKEN --name "fawkes-v23-global-note-node-sp$split-s$seed" \
    -v ./src:/workspace/src \
    -e USE_NOTE=1 -e GLOBAL_NOTE_NODE=1 -e NOTE_INJECTION=mean -e EMBED_DIM=768 \
    -e USE_SCORES=0 -e PRUNE_NO_EVIDENCE=1 \
    -e RUN_CASCADE=0 -e RUN_EIR=0 \
    -e PUSH=1 -e DATA_SPLIT_SEED="$split" -e SEED="$seed" \
    -e OUTPUT_REPO="wmatbooth/fawkes-v23-global-note-node-sp$split-s$seed" \
    "${envs[@]}" \
    scripts/fawkes_patch_masking_hf_job.py
}

# NOTE_INJECTION=mean because the NOTE node carries the stored global mean in numfeat;
# the span-memory paths are a different experiment and would need NOTE_MEMORY_REPO.
# GROUND_BY is deliberately NOT passed: with GLOBAL_NOTE_NODE=1 the vector goes to the
# NOTE node and no entity is grounded, so a GROUND_BY here would only mislead the log.
# t4-small, matching the v22 arms this is paired against, rather than v23's a10g.
for seed in 42 43 44 45 46 47 48 49 50 51; do
  submit 42 $seed 4h MASK_STRATEGY=patch JEPA_PATCHES=8 NODE_MASK=0.4 DECODER=mlp
done
