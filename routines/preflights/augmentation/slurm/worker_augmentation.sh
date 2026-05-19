#!/usr/bin/env bash
#SBATCH -J brainrepa-preflight-aug
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
# --output / --error paths come from the launcher's sbatch flags so the LOGS_DIR
# resolves on the login node.

# Worker for the BraTS-2026 augmentation pre-flight on Picasso.
# Singularity-only (no Docker). A100-40GB inside the NGC PyTorch image.
#
# Driven by env vars exported via the launcher's `sbatch --export=ALL,...`:
#   REPO_DIR        absolute path to the BrainREPA-FM clone on /mnt/.../fscratch/repos
#   CONFIG_PATH     absolute path to the routine YAML (picasso.yaml by default)
#   SINGULARITY_IMG absolute path to the NGC .sif image
#   DATASET_ROOT    absolute path to the dataset dir (read-only mount)
#   CHECKPOINTS_ROOT absolute path to the checkpoints dir on fscratch
#   LEGACY_CKPT_ROOT host path the MAISI wrapper hard-codes (default /media/mpascual/Sandisk2TB/checkpoints).
#                    The worker bind-mounts CHECKPOINTS_ROOT here so the wrapper resolves without code changes.
#   PIP_USER_BASE   where pip --user installs our non-torch deps (per-repo, persistent across re-runs).

set -euo pipefail
START_TIME=$(date +%s)

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}
SINGULARITY_IMG=${SINGULARITY_IMG:?missing SINGULARITY_IMG}
DATASET_ROOT=${DATASET_ROOT:?missing DATASET_ROOT}
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:?missing CHECKPOINTS_ROOT}
LEGACY_CKPT_ROOT=${LEGACY_CKPT_ROOT:-/media/mpascual/Sandisk2TB/checkpoints}
PIP_USER_BASE=${PIP_USER_BASE:-${REPO_DIR}/.python-user}

# ============================================================================
# JOB HEADER
# ============================================================================
echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date -u +%FT%TZ)"
echo "REPO_DIR:     ${REPO_DIR}"
echo "CONFIG_PATH:  ${CONFIG_PATH}"
echo "IMG:          ${SINGULARITY_IMG}"
echo "DATASET_ROOT: ${DATASET_ROOT}"
echo "CKPT_ROOT:    ${CHECKPOINTS_ROOT}"
echo "LEGACY_CKPT:  ${LEGACY_CKPT_ROOT}"
echo "PIP_USER:     ${PIP_USER_BASE}"
echo "Git commit:   $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

mkdir -p "${PIP_USER_BASE}"

# GPU info for the log.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "[warn] nvidia-smi unavailable"
echo ""

# ============================================================================
# RUN INSIDE SINGULARITY
# ============================================================================
# Bind-mounts:
#   - REPO_DIR (repo code visible at the same path inside the container)
#   - DATASET_ROOT (read-only)
#   - CHECKPOINTS_ROOT mounted at LEGACY_CKPT_ROOT so src/brainrepa_fm/common/maisi.py
#     resolves its hard-coded DEFAULT_MAISI_CHECKPOINT / DEFAULT_MAISI_CONFIG_PATH paths
#     without code changes (the wrapper currently encodes /media/.../Sandisk2TB/checkpoints).
#   - PIP_USER_BASE (writable; pip --user lands here).
#
# The container ships torch preinstalled; we user-install only our project + non-torch
# deps so torch is never reinstalled. PYTHONUSERBASE makes pip --user land in
# the repo-scoped directory.
singularity exec \
    --nv \
    --cleanenv \
    -B "${REPO_DIR}:${REPO_DIR}" \
    -B "${DATASET_ROOT}:${DATASET_ROOT}:ro" \
    -B "${CHECKPOINTS_ROOT}:${LEGACY_CKPT_ROOT}:ro" \
    -B "${PIP_USER_BASE}:${PIP_USER_BASE}" \
    --env "PYTHONUSERBASE=${PIP_USER_BASE}" \
    --env "PYTHONUNBUFFERED=1" \
    --env "PATH=${PIP_USER_BASE}/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    "${SINGULARITY_IMG}" \
    bash -lc "
        set -euo pipefail
        cd '${REPO_DIR}'
        echo '[container] python: '\$(python --version)
        echo '[container] torch: '\$(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')
        python -m pip install --user --quiet --no-warn-script-location \
            'monai[einops]>=1.4,<2.0' 'h5py>=3.10' 'nibabel>=5.2' 'SimpleITK>=2.3' \
            'scipy>=1.11' 'scikit-image>=0.22' 'einops>=0.7' 'omegaconf>=2.3' \
            'pydantic>=2.6' 'rich>=13.7' 'matplotlib>=3.8' 'pandas>=2.1' 'PyYAML>=6.0'
        python -m pip install --user --quiet --no-warn-script-location --no-deps -e .
        python -m routines.preflights.augmentation.cli '${CONFIG_PATH}'
    "

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date -u +%FT%TZ)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
