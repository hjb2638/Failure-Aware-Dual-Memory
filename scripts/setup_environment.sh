#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MANAGER="conda"
ENV_NAME="failure_aware_dual_memory_verify"
ENV_PATH=""
PYTHON_VERSION="3.12"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-2.7.0+cu128.html"
SETUPTOOLS_SPEC="setuptools<81"
SKIP_EDITABLE=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup_environment.sh [options]

This script creates a brand-new environment for Failure-Aware Dual-Memory and installs the
verified dependency stack without touching any existing project environment.

Options:
  --manager conda|uv        Environment manager to use. Default: conda
  --env-name NAME           Conda environment name. Default: failure_aware_dual_memory_verify
  --env-path PATH           Environment path. For conda this uses `conda create -p`.
                            For uv this is the venv directory path.
  --python VERSION          Python version. Default: 3.12
  --skip-editable           Skip `pip install -e .`
  --help                    Show this message

Examples:
  bash scripts/setup_environment.sh
  bash scripts/setup_environment.sh --manager conda --env-name failure_aware_dual_memory_clean
  bash scripts/setup_environment.sh --manager conda --env-path /tmp/failure_aware_dual_memory_env
  bash scripts/setup_environment.sh --manager uv --env-path .venv
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manager)
            MANAGER="${2:?missing value for --manager}"
            shift 2
            ;;
        --env-name)
            ENV_NAME="${2:?missing value for --env-name}"
            shift 2
            ;;
        --env-path)
            ENV_PATH="${2:?missing value for --env-path}"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="${2:?missing value for --python}"
            shift 2
            ;;
        --skip-editable)
            SKIP_EDITABLE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

run_pip_install() {
    "$@" pip install "$SETUPTOOLS_SPEC"
    "$@" pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"
    "$@" pip install \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "$TORCH_INDEX_URL" \
        --no-deps
    "$@" pip install \
        pyg_lib \
        torch_scatter \
        torch_sparse \
        torch_cluster \
        torch_spline_conv \
        -f "$PYG_WHEEL_URL"
    "$@" pip install torch_geometric==2.7.0
    "$@" pip install -r "$PROJECT_ROOT/requirements.txt"

    if [[ "$SKIP_EDITABLE" != "1" ]]; then
        "$@" pip install -e "$PROJECT_ROOT"
    fi
}

echo "Project root: $PROJECT_ROOT"
echo "Environment manager: $MANAGER"

case "$MANAGER" in
    conda)
        CONDA_SH_PATH="${CONDA_SH_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
        if [[ ! -f "$CONDA_SH_PATH" ]]; then
            echo "Cannot find conda.sh: $CONDA_SH_PATH" >&2
            exit 1
        fi

        # shellcheck disable=SC1090
        source "$CONDA_SH_PATH"

        if [[ -n "$ENV_PATH" ]]; then
            echo "Creating new conda env at: $ENV_PATH"
            conda create -y -p "$ENV_PATH" "python=${PYTHON_VERSION}" pip
            run_pip_install conda run -p "$ENV_PATH"
            cat <<EOF

Environment setup complete.
Activate with:
  source "$CONDA_SH_PATH"
  conda activate "$ENV_PATH"
EOF
        else
            echo "Creating new conda env named: $ENV_NAME"
            conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
            run_pip_install conda run -n "$ENV_NAME"
            cat <<EOF

Environment setup complete.
Activate with:
  source "$CONDA_SH_PATH"
  conda activate "$ENV_NAME"
EOF
        fi
        ;;
    uv)
        if ! command -v uv >/dev/null 2>&1; then
            echo "uv is not installed or not on PATH." >&2
            exit 1
        fi

        UV_ENV_PATH="${ENV_PATH:-$PROJECT_ROOT/.venv}"
        echo "Creating new uv environment at: $UV_ENV_PATH"
        uv venv "$UV_ENV_PATH" --python "$PYTHON_VERSION"
        run_pip_install "$UV_ENV_PATH/bin/python" -m
        cat <<EOF

Environment setup complete.
Activate with:
  source "$UV_ENV_PATH/bin/activate"
EOF
        ;;
    *)
        echo "Unsupported manager: $MANAGER" >&2
        usage >&2
        exit 1
        ;;
esac
