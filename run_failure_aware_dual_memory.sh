#!/usr/bin/env bash

set -euo pipefail

# Failure-Aware Dual-Memory baseline runner
#
# Default behavior:
# 1. activate the `failure_aware_dual_memory` conda environment
# 2. enter the project root
# 3. run one baseline pipeline job through the single supported CLI
#
# You can override the defaults below by prefixing environment variables:
#
#   CONDA_ENV_NAME=failure_aware_dual_memory \
#   DATA_PATH=/path/to/failure_aware_dual_memory/data/mp_20/train.csv \
#   INITIAL_GUESS=from_file \
#   INITIAL_GUESS_FILE=/tmp/my_init.txt \
#   LLM_MODEL=/path/to/local-llm \
#   TARGET_VALUE=-3.8 \
#   N_INIT=1 \
#   N_ITERATIONS=1 \
#   ADDITIONAL_PROMPT="Prefer oxides." \
#   bash /path/to/failure_aware_dual_memory/run_failure_aware_dual_memory.sh
#
# Common optional parameters and meanings:
# - CONDA_ENV_NAME:
#   Conda environment name. Default: failure_aware_dual_memory
# - PROJECT_ROOT:
#   failure_aware_dual_memory project root. Default: directory containing this script
# - DATA_PATH:
#   CSV used for random initialization sampling. Default: data/mp_20/train.csv
# - INITIAL_GUESS:
#   Initial composition mode. Allowed by current baseline code:
#   random | llm | from_file
# - INITIAL_GUESS_FILE:
#   Required when INITIAL_GUESS=from_file. One composition per line.
# - LLM_MODEL:
#   LLM path or model id used by the proposer.
# - TARGET_VALUE:
#   Target formation energy per atom in eV/atom.
# - N_INIT:
#   Number of independent runs.
# - N_ITERATIONS:
#   Number of proposal-feedback iterations per run.
# - MAX_NEW_TOKENS:
#   Max generation tokens for transformer proposer.
# - ADDITIONAL_PROMPT:
#   Extra instruction appended to the proposer prompt.
# - NO_CUDA:
#   Set to 1 to force CPU. Default: 0
# - SEED:
#   Random seed passed to the CLI. Default: 42
#
# Notes:
# - The single supported public interface is `scripts/run_pipeline.py`.
# - This wrapper keeps the same environment-variable UX, but now launches that
#   script directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-failure_aware_dual_memory}"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/mp_20/train.csv}"
INITIAL_GUESS="${INITIAL_GUESS:-from_file}"
INITIAL_GUESS_FILE="${INITIAL_GUESS_FILE:-/tmp/feedback_init.txt}"
LLM_MODEL="${LLM_MODEL:-${FADM_LLM_MODEL:-}}"
TARGET_VALUE="${TARGET_VALUE:--3.8}"
N_INIT="${N_INIT:-1}"
N_ITERATIONS="${N_ITERATIONS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
ADDITIONAL_PROMPT="${ADDITIONAL_PROMPT:-}"
LLM_DEVICE_MAP="${LLM_DEVICE_MAP:-${FADM_LLM_DEVICE_MAP:-auto}}"
LLM_TORCH_DTYPE="${LLM_TORCH_DTYPE:-${FADM_LLM_TORCH_DTYPE:-auto}}"
LLM_TRUST_REMOTE_CODE="${LLM_TRUST_REMOTE_CODE:-0}"
NO_CUDA="${NO_CUDA:-0}"
SEED="${SEED:-42}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "Project root does not exist: $PROJECT_ROOT" >&2
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Data path does not exist: $DATA_PATH" >&2
    exit 1
fi

if [[ "$INITIAL_GUESS" == "from_file" && ! -f "$INITIAL_GUESS_FILE" ]]; then
    echo "INITIAL_GUESS=from_file but file does not exist: $INITIAL_GUESS_FILE" >&2
    exit 1
fi

if [[ -z "$LLM_MODEL" ]]; then
    echo "LLM model is not configured. Set LLM_MODEL or FADM_LLM_MODEL to a local model path or Hugging Face model id." >&2
    exit 1
fi

CONDA_SH_PATH="${CONDA_SH_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [[ -f "$CONDA_SH_PATH" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_SH_PATH"
else
    echo "Cannot find conda.sh: $CONDA_SH_PATH" >&2
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
cd "$PROJECT_ROOT"

CMD=(
    python "$PROJECT_ROOT/scripts/run_pipeline.py"
    --initial_guess "$INITIAL_GUESS"
    --data_path "$DATA_PATH"
    --memory_storage_dir "${MEMORY_STORAGE_DIR:-$PROJECT_ROOT/memory_storage}"
    --output_dir "${OUTPUT_DIR:-$PROJECT_ROOT/test_outputs/run_$(date +%Y%m%d_%H%M%S)}"
    --llm_model "$LLM_MODEL"
    --llm_device_map "$LLM_DEVICE_MAP"
    --llm_torch_dtype "$LLM_TORCH_DTYPE"
    --target_value "$TARGET_VALUE"
    --n_init "$N_INIT"
    --n_iterations "$N_ITERATIONS"
    --max_new_tokens "$MAX_NEW_TOKENS"
)

if [[ "$INITIAL_GUESS" == "from_file" ]]; then
    CMD+=(--initial_guess_file "$INITIAL_GUESS_FILE")
fi

if [[ -n "$ADDITIONAL_PROMPT" ]]; then
    CMD+=(--additional_prompt "$ADDITIONAL_PROMPT")
fi

if [[ "$LLM_TRUST_REMOTE_CODE" == "1" ]]; then
    CMD+=(--llm_trust_remote_code)
fi

if [[ "$NO_CUDA" == "1" ]]; then
    CMD+=(--no_cuda)
fi

CMD+=(--seed "$SEED")

echo "Running failure_aware_dual_memory with command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
