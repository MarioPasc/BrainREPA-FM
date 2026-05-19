#!/usr/bin/env bash
#SBATCH -J brainrepa-preflight-aug
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
# --output / --error paths come from the launcher's sbatch flags.

# Worker for the BraTS-2026 augmentation pre-flight on Picasso.
# Conda-based (no Singularity). Expects the `brainrepa` env to already exist on Picasso
# with torch+cu121 working against the A100 driver. The MAISI paths come from the
# routine YAML (parameterized in AugmentationRoutineConfig), so no bind-mount tricks.
#
# Driven by env vars exported via the launcher's `sbatch --export=ALL,...`:
#   REPO_DIR         absolute path to the BrainREPA-FM clone on /mnt/.../fscratch/repos
#   CONFIG_PATH      absolute path to the routine YAML (configs/picasso.yaml by default)
#   CONDA_ENV_NAME   conda env name on Picasso (default: brainrepa)

set -euo pipefail
START_TIME=$(date +%s)

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-brainrepa}

# ============================================================================
# JOB HEADER
# ============================================================================
echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date -u +%FT%TZ)"
echo "REPO_DIR:     ${REPO_DIR}"
echo "CONFIG_PATH:  ${CONFIG_PATH}"
echo "CONDA_ENV:    ${CONDA_ENV_NAME}"
echo "Git commit:   $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ============================================================================
# ENVIRONMENT
# ============================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module on this node; assuming conda in PATH."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[env] python: $(which python)"
python --version

# BrainREPA-FM uses Python 3.10+ syntax (X | None, match, etc.). Fail fast with a
# clear message if the activated env is older.
python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' || {
    echo "[FATAL] conda env '${CONDA_ENV_NAME}' has $(python --version 2>&1)." >&2
    echo "[FATAL] BrainREPA-FM requires Python >= 3.10." >&2
    echo "[hint]  recreate the env on Picasso:" >&2
    echo "        conda create -n brainrepa python=3.11 -y" >&2
    echo "        conda activate brainrepa" >&2
    echo "        pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision" >&2
    echo "        pip install -e '${REPO_DIR}'" >&2
    exit 1
}

python -c "import torch; print(f'[env] torch={torch.__version__}, cuda={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')"

# GPU info for the log.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "[warn] nvidia-smi unavailable"
echo ""

# ============================================================================
# COMMAND
# ============================================================================
python -m routines.preflights.augmentation.cli "${CONFIG_PATH}"

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date -u +%FT%TZ)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
