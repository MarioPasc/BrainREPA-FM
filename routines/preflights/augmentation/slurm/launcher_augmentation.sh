#!/usr/bin/env bash
# Submit the BraTS-2026 augmentation pre-flight to Picasso.
#
# Usage (login node):
#   bash routines/preflights/augmentation/slurm/launcher_augmentation.sh
#   bash routines/preflights/augmentation/slurm/launcher_augmentation.sh path/to/other_config.yaml
#   bash routines/preflights/augmentation/slurm/launcher_augmentation.sh --dry-run
#
# Notes:
#   - The default config (configs/picasso.yaml) targets the A100 envelope (256×256×192).
#   - The launcher does NOT touch the dataset or checkpoints; both are expected to
#     already live under /mnt/.../fscratch on Picasso (see paths below).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable paths -----------------------------------------------------
REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/BrainREPA-FM}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/inpainting}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints}"
SINGULARITY_IMG="${SINGULARITY_IMG:-/mnt/home/users/tic_163_uma/mpascual/fscratch/singularity/nvidia_pytorch_24.10-py3.sif}"
LEGACY_CKPT_ROOT="${LEGACY_CKPT_ROOT:-/media/mpascual/Sandisk2TB/checkpoints}"

LOGS_DIR="${LOGS_DIR:-${HOME}/execs/brainrepa_fm/logs}"
mkdir -p "${LOGS_DIR}"

# ---- Args -------------------------------------------------------------------
DRY_RUN=false
CONFIG_PATH=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# //; s/^#$//'
            exit 0
            ;;
        *) CONFIG_PATH="${arg}" ;;
    esac
done
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/routines/preflights/augmentation/configs/picasso.yaml}"

# ---- Pre-submit sanity checks ----------------------------------------------
echo "[launcher] REPO_DIR        = ${REPO_DIR}"
echo "[launcher] CONFIG_PATH     = ${CONFIG_PATH}"
echo "[launcher] DATASET_ROOT    = ${DATASET_ROOT}"
echo "[launcher] CHECKPOINTS_ROOT= ${CHECKPOINTS_ROOT}"
echo "[launcher] LEGACY_CKPT     = ${LEGACY_CKPT_ROOT}"
echo "[launcher] SINGULARITY_IMG = ${SINGULARITY_IMG}"
echo "[launcher] LOGS_DIR        = ${LOGS_DIR}"

for path in \
    "${REPO_DIR}" \
    "${CONFIG_PATH}" \
    "${SINGULARITY_IMG}" \
    "${DATASET_ROOT}/brats_inpainting_2026.h5" \
    "${CHECKPOINTS_ROOT}/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt" \
    "${CHECKPOINTS_ROOT}/MAISI_V2_RM/code/NV-Generate-CTMR/configs/config_network_ddpm.json" ;
do
    if [[ ! -e "${path}" ]]; then
        echo "[FATAL] required path missing: ${path}" >&2
        echo "[hint]  fix the paths (env vars on the launcher) or rsync the missing artifact in." >&2
        exit 1
    fi
done

# ---- Job name with timestamp suffix -----------------------------------------
JOB_NAME="brainrepa-preflight-aug-$(date -u +%Y%m%dT%H%M%SZ)"

SBATCH_CMD=(
    sbatch
    --parsable
    --job-name="${JOB_NAME}"
    --output="${LOGS_DIR}/${JOB_NAME}_%j.out"
    --error="${LOGS_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}",SINGULARITY_IMG="${SINGULARITY_IMG}",DATASET_ROOT="${DATASET_ROOT}",CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT}",LEGACY_CKPT_ROOT="${LEGACY_CKPT_ROOT}"
    "${SCRIPT_DIR}/worker_augmentation.sh"
)

if ${DRY_RUN}; then
    echo
    echo "[DRY-RUN] ${SBATCH_CMD[*]}"
    exit 0
fi

JOB_ID=$("${SBATCH_CMD[@]}")
echo
echo "Submitted job ${JOB_ID} (name: ${JOB_NAME})"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
