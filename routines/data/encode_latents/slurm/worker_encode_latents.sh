#!/usr/bin/env bash
#SBATCH -J brainrepa-encode-latents
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# Worker for the Schema-A -> Schema-B latent encoder on Picasso.
# Conda-based. The MAISI paths come from the routine YAML.
#
# Env vars (from launcher):
#   REPO_DIR        absolute path to the BrainREPA-FM clone
#   CONFIG_PATH     absolute path to the routine YAML
#   CONDA_ENV_NAME  conda env name (default: brainrepa)

set -euo pipefail
START_TIME=$(date +%s)

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-brainrepa}

echo "=========================================="
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date -u +%FT%TZ)"
echo "REPO_DIR:    ${REPO_DIR}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "CONDA_ENV:   ${CONDA_ENV_NAME}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; assuming conda in PATH."

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
python -c "import torch; print(f'[env] torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo ""

python -m routines.data.encode_latents.cli "${CONFIG_PATH}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date -u +%FT%TZ)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
