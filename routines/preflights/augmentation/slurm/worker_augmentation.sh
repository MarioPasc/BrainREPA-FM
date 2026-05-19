#!/usr/bin/env bash
# Worker for the BraTS-2026 augmentation pre-flight on Picasso.
# Singularity-only (no Docker). A100 inside the NGC image.
#
# Expects env vars from the launcher: REPO_DIR, CONFIG_PATH.
# A100-40GB sizing: full 1,251 training subjects + 219 challenge-val, ~6-8 h.
set -euo pipefail

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}

# Singularity image — adjust to whichever NGC PyTorch image is current on Picasso.
IMG="${SINGULARITY_IMG:-/mnt/home/users/tic_163_uma/mpascual/fscratch/singularity/nvidia_pytorch_24.10-py3.sif}"

# Project conda env (system Python is not used).
PY="$REPO_DIR/.conda-bin/python"
if [[ ! -x "$PY" ]]; then
    PY="$HOME/.conda/envs/brainrepa/bin/python"
fi

cd "$REPO_DIR"
echo "[worker] starting on $(hostname) at $(date -u +%FT%TZ)"
echo "[worker] REPO_DIR=$REPO_DIR"
echo "[worker] CONFIG_PATH=$CONFIG_PATH"
echo "[worker] PY=$PY"
echo "[worker] IMG=$IMG"

# Print GPU info for the log.
nvidia-smi || true

singularity exec --nv \
    -B "$REPO_DIR:$REPO_DIR" \
    -B /mnt/home/users/tic_163_uma/mpascual/fscratch:/mnt/home/users/tic_163_uma/mpascual/fscratch \
    "$IMG" \
    bash -c "cd '$REPO_DIR' && '$PY' -m pip install --quiet -e . && '$PY' -m routines.preflights.augmentation.cli '$CONFIG_PATH'"

echo "[worker] done at $(date -u +%FT%TZ)"
