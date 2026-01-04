#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Killarney ONLINE setup (run on login or compute node - has internet)
# =============================================================================
# Purpose:
# - Create venv with UV
# - Install all Python dependencies
# - Pre-download gated Llama model into HF cache
#
# Usage:
#   cd /project/6105522/harmful-agents-meta-dataset
#   export HF_TOKEN=hf_...   # required for gated Llama models (generate new one if expired)
#   bash scripts/killarney_online_setup.sh
#
# To generate a new HF token:
#   1. Go to https://huggingface.co/settings/tokens
#   2. Create a new token with "Read" access
#   3. Accept the Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#
# Optional overrides:
#   PROJECT_DIR=/project/XXXX \
#   HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
#   bash scripts/killarney_online_setup.sh
# =============================================================================

PROJECT_DIR="${PROJECT_DIR:-/project/6105522}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/harmful-agents-meta-dataset}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venvs/cb_env}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_DIR/cb_cache}"
HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

# PyTorch wheel source
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

uv_install() {
  uv pip install "$@"
}

if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: REPO_DIR not found: $REPO_DIR"
  echo "Set REPO_DIR or PROJECT_DIR, then rerun."
  exit 1
fi

cd "$REPO_DIR"
mkdir -p logs "$PROJECT_DIR/.venvs" "$CACHE_ROOT"/{hf,wandb,torch,xdg}

# Hugging Face / Transformers caches (shared across jobs)
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# --- Modules (Alliance) ---
if command -v module >/dev/null 2>&1; then
  module --force purge || true
  module load StdEnv/2023
  module load cuda/12.6
  module load python/3.11.5
fi

# --- Install UV if not present ---
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- Create + activate venv ---
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv: $VENV_DIR"
  python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo ""
echo "Python: $(python -V)"
echo "Venv:   $VENV_DIR"
echo "Cache:  $CACHE_ROOT"
echo ""

echo "Upgrading packaging tools..."
uv_install --upgrade pip setuptools wheel

echo "Installing Python deps (requirements.txt)..."
uv_install -r requirements.txt

echo "Installing CUDA PyTorch wheels..."
uv_install torch torchvision --index-url "$TORCH_INDEX_URL"

echo ""
echo "Checking torch installation..."
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("✅ PyTorch is installed")
except ImportError as e:
    print("ERROR: torch not found.")
    import sys
    sys.exit(1)
PY

# --- Download model into HF cache ---
# IMPORTANT: Llama models are gated. You must:
# 1) have access on your HF account
# 2) accept the model terms on Hugging Face
# 3) provide a valid (non-expired) HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo ""
  echo "ERROR: HF_TOKEN is not set."
  echo "This model is gated; export HF_TOKEN and rerun:"
  echo "  export HF_TOKEN=hf_..."
  echo ""
  echo "Generate a new token at: https://huggingface.co/settings/tokens"
  exit 1
fi

echo ""
echo "Downloading model snapshot into HF cache (this can take a while)..."
echo "Model: $HF_MODEL_ID"

python - <<PY
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("HF_MODEL_ID", "$HF_MODEL_ID")
token = os.environ.get("HF_TOKEN")

try:
    path = snapshot_download(
        repo_id=model_id,
        token=token,
        resume_download=True,
    )
    print("✅ Downloaded/cached at:", path)
except Exception as e:
    print(f"ERROR downloading model: {e}")
    print("")
    print("If you see 401 Unauthorized or 'token expired':")
    print("  1. Go to https://huggingface.co/settings/tokens")
    print("  2. Create a new token with Read access")
    print("  3. Accept the Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("  4. Re-export HF_TOKEN and rerun this script")
    import sys
    sys.exit(1)
PY

echo ""
echo "=============================================="
echo "✅ Killarney setup complete"
echo "Repo:   $REPO_DIR"
echo "Venv:   $VENV_DIR"
echo "Cache:  $CACHE_ROOT"
echo "Model:  $HF_MODEL_ID"
echo ""
echo "Next: submit a training job"
echo "  sbatch slurm/killarney_cb_llama31_8b_4xh100.sbatch"
echo "=============================================="
