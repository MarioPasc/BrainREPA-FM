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

_validate_yaml_paths() {
    local cfg="$1"
    python - "$cfg" <<'PYEOF'
import sys, yaml
from pathlib import Path
cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = yaml.safe_load(f) or {}
missing = []
for key in ("source_h5", "maisi_checkpoint_path", "maisi_config_path"):
    val = cfg.get(key)
    if val is None:
        if key == "source_h5":
            missing.append(f"{key}: not set in {cfg_path}")
        continue
    if not Path(val).exists():
        missing.append(f"{key}: missing on filesystem: {val}")
if missing:
    print("[FATAL] required artifact paths missing on Picasso:", file=sys.stderr)
    for m in missing:
        print(f"  - {m}", file=sys.stderr)
    sys.exit(1)
print("[launcher] YAML paths validated.")
PYEOF
}
if command -v python >/dev/null 2>&1; then
    _validate_yaml_paths "${CONFIG_PATH}" || exit 1
fi

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
