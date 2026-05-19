#!/usr/bin/env bash
# Launcher for the BraTS-2026 augmentation pre-flight on Picasso (UMA HPC).
#
# Usage (from a Picasso login node):
#   bash routines/preflights/augmentation/slurm/launcher_augmentation.sh path/to/config.yaml
#
# Conventions inherited from MenGrowth / IsalSR launcher scripts:
# - Singularity-only (no Docker).
# - A100 selection via `--constraint=dgx`.
# - Repos live at /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/<repo>.
# - Logs under ~/execs/<project>/logs.
#
# This script wraps an `sbatch` invocation; the actual worker is
# `worker_augmentation.sh`.
set -euo pipefail

CONFIG_PATH=${1:-routines/preflights/augmentation/configs/default.yaml}
PROJECT=brainrepa_fm
REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/BrainREPA-FM"
LOG_DIR="$HOME/execs/${PROJECT}/logs"
mkdir -p "$LOG_DIR"

JOB_NAME="brainrepa-preflight-aug-$(date +%Y%m%d-%H%M%S)"

sbatch \
    --job-name="${JOB_NAME}" \
    --partition=dgx2q \
    --constraint=dgx \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=08:00:00 \
    --output="${LOG_DIR}/${JOB_NAME}.out" \
    --error="${LOG_DIR}/${JOB_NAME}.err" \
    --export=ALL,REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}" \
    "${REPO_DIR}/routines/preflights/augmentation/slurm/worker_augmentation.sh"
