#!/usr/bin/env bash
# Submit the BraTS-2026 augmentation pre-flight to Picasso.
#
# Usage (login node):
#   bash routines/preflights/augmentation/slurm/launcher_augmentation.sh
#   bash routines/preflights/augmentation/slurm/launcher_augmentation.sh path/to/other_config.yaml
#   bash routines/preflights/augmentation/slurm/launcher_augmentation.sh --dry-run
#
# The default config (configs/picasso.yaml) targets the A100 envelope (256x256x192)
# and points at the dataset + MAISI paths under /mnt/.../fscratch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable paths -----------------------------------------------------
REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/BrainREPA-FM}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-brainrepa}"
LOGS_DIR="${LOGS_DIR:-${HOME}/execs/brainrepa_fm/logs}"
mkdir -p "${LOGS_DIR}"

# ---- Activate the project conda env on the login node -----------------------
# The launcher itself runs python (for YAML path validation); the login node's
# (base) env lacks PyYAML, so activate brainrepa first.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null \
        || echo "[launcher] WARNING: could not activate conda env '${CONDA_ENV_NAME}'." >&2
else
    echo "[launcher] WARNING: conda not on PATH; YAML validation may fail." >&2
fi

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

# Extract the source_h5, maisi_checkpoint_path, maisi_config_path from the YAML and verify each.
# Use python (any conda env with PyYAML works on the login node).
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
        # maisi_* are optional; only flag source_h5 if absent
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
else
    echo "[launcher] WARNING: no python on PATH at launcher time; skipping YAML path validation." >&2
fi

# ---- Job name with timestamp suffix -----------------------------------------
JOB_NAME="brainrepa-preflight-aug-$(date -u +%Y%m%dT%H%M%SZ)"

SBATCH_CMD=(
    sbatch
    --parsable
    --job-name="${JOB_NAME}"
    --output="${LOGS_DIR}/${JOB_NAME}_%j.out"
    --error="${LOGS_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}",CONDA_ENV_NAME="${CONDA_ENV_NAME}"
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
