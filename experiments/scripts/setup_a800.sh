#!/usr/bin/env bash
# One-shot environment bootstrap for running CL-LLMOps experiments on an A800.
#
# Usage: bash scripts/setup_a800.sh
#
# What this does:
#   1. Creates a Python venv at .venv (skipped if it already exists)
#   2. Installs experiments/requirements.txt
#   3. Verifies CUDA + GPU visibility
#   4. Prints the HuggingFace / dataset / vLLM environment variables you need

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== CL-LLMOps A800 Environment Bootstrap ==="
echo "Repo : ${REPO_ROOT}"
echo ""

# 1. Python venv
if [[ ! -d .venv ]]; then
    echo "[1/4] Creating .venv ..."
    python3 -m venv .venv
else
    echo "[1/4] .venv already exists, skipping creation"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 2. Install dependencies
echo ""
echo "[2/4] Installing experiments/requirements.txt ..."
python -m pip install -r experiments/requirements.txt

# 3. GPU check
echo ""
echo "[3/4] Checking CUDA + GPU ..."
python - <<'PY'
import torch
print(f"  torch             : {torch.__version__}")
print(f"  cuda available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU[{i}]           : {props.name} ({props.total_memory / 1e9:.1f} GB)")
PY

# 4. Environment variable hints
echo ""
echo "[4/4] Environment variables you should set before running experiments:"
cat <<'ENV'

  # HuggingFace (needed for Llama-3 weights)
  export HF_TOKEN="hf_xxx_your_token_here"
  export HF_HOME="${HOME}/.cache/huggingface"

  # GPU selection
  export ASOC_GPUS="0"            # comma-separated device ids
  export ASOC_DEVICE="cuda"

  # vLLM endpoint (start server via scripts/start_vllm_server.sh)
  export VLLM_ENDPOINT="http://127.0.0.1:8000/v1"
  export VLLM_API_KEY="EMPTY"

  # Optional: Kaggle credentials for PQ dataset
  export KAGGLE_USERNAME="your_username"
  export KAGGLE_KEY="your_kaggle_api_key"

ENV

echo ""
echo "Next steps:"
echo "  1) Configure env vars above"
echo "  2) Download datasets  : see experiments/data/README.md"
echo "  3) Start vLLM server  : bash experiments/scripts/start_vllm_server.sh llama3"
echo "  4) Run experiments    : bash experiments/run_all.sh"
echo ""
echo "Bootstrap complete."
