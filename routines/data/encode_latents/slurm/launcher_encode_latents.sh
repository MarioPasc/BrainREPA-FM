#!/usr/bin/env bash
# Submit the Schema-A -> Schema-B latent encoder to Picasso.
#
# Usage (login node):
#   bash routines/data/encode_latents/slurm/launcher_encode_latents.sh
#   bash routines/data/encode_latents/slurm/launcher_encode_latents.sh path/to/other_config.yaml
#   bash routines/data/encode_latents/slurm/launcher_encode_latents.sh --dry-run
#
# Default config (configs/picasso.yaml): A100 envelope, full 2,721 passes (~3 h).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/BrainREPA-FM}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-brainrepa}"
LOGS_DIR="${LOGS_DIR:-${HOME}/execs/brainrepa_fm/logs}"
mkdir -p "${LOGS_DIR}"

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
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/routines/data/encode_latents/configs/picasso.yaml}"

echo "[launcher] REPO_DIR    = ${REPO_DIR}"
echo "[launcher] CONFIG_PATH = ${CONFIG_PATH}"
echo "[launcher] CONDA_ENV   = ${CONDA_ENV_NAME}"
echo "[launcher] LOGS_DIR    = ${LOGS_DIR}"

for path in "${REPO_DIR}" "${CONFIG_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[FATAL] required path missing: ${path}" >&2
        exit 1
    fi
done

# Pure-bash top-level scalar extraction from a flat YAML file (no python dependency).
_yaml_get() {
    local file="$1" key="$2"
    grep -E "^[[:space:]]*${key}[[:space:]]*:" "${file}" 2>/dev/null \
        | head -1 \
        | sed -E "s/^[[:space:]]*${key}[[:space:]]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]+$//; s/^[\"']//; s/[\"']$//"
}

_validate_yaml_paths() {
    local cfg="$1" key val fatal=0
    for key in source_h5 maisi_checkpoint_path maisi_config_path; do
        val="$(_yaml_get "${cfg}" "${key}")"
        if [[ -z "${val}" || "${val}" == "null" ]]; then
            if [[ "${key}" == "source_h5" ]]; then
                echo "[FATAL] '${key}' is not set in ${cfg}" >&2
                fatal=1
            fi
            continue
        fi
        if [[ ! -e "${val}" ]]; then
            echo "[FATAL] '${key}' points at a missing path: ${val}" >&2
            fatal=1
        fi
    done
    if [[ ${fatal} -ne 0 ]]; then
        echo "[hint] fix the path in ${cfg} or rsync the missing artifact in." >&2
        return 1
    fi
    echo "[launcher] YAML paths validated."
    return 0
}
_validate_yaml_paths "${CONFIG_PATH}" || exit 1

JOB_NAME="brainrepa-encode-latents-$(date -u +%Y%m%dT%H%M%SZ)"

SBATCH_CMD=(
    sbatch
    --parsable
    --job-name="${JOB_NAME}"
    --output="${LOGS_DIR}/${JOB_NAME}_%j.out"
    --error="${LOGS_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}",CONDA_ENV_NAME="${CONDA_ENV_NAME}"
    "${SCRIPT_DIR}/worker_encode_latents.sh"
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
