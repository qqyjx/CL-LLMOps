#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for CL-LLMOps experiments.
#
# Usage:
#   bash scripts/start_vllm_server.sh [llama3|mistral7b] [gpu_id]
#
# Environment:
#   HF_TOKEN      Hugging Face token (required for gated Llama-3 weights)
#   HF_HOME       Hugging Face cache dir (defaults to $HOME/.cache/huggingface)
#   VLLM_PORT     Server port (default 8000)
#   ASOC_LLAMA_MODEL_PATH
#                 Optional local Llama checkpoint directory. The API keeps the
#                 canonical model name expected by experiments/config.py.
#
# The server exposes an OpenAI-compatible API at:
#   http://127.0.0.1:${VLLM_PORT}/v1
#
# Set VLLM_ENDPOINT in your shell to point at this when running experiments:
#   export VLLM_ENDPOINT="http://127.0.0.1:8000/v1"

set -euo pipefail

MODEL_KEY="${1:-llama3}"
GPU_ID="${2:-0}"
PORT="${VLLM_PORT:-8000}"

case "${MODEL_KEY}" in
    llama3)
        CANONICAL_MODEL_ID="NousResearch/Meta-Llama-3.1-8B-Instruct"
        MODEL_ID="${ASOC_LLAMA_MODEL_PATH:-${CANONICAL_MODEL_ID}}"
        MAX_MODEL_LEN=8192
        ;;
    mistral7b)
        CANONICAL_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
        MODEL_ID="${ASOC_MISTRAL_MODEL_PATH:-${CANONICAL_MODEL_ID}}"
        MAX_MODEL_LEN=8192
        ;;
    *)
        echo "Unknown model key: ${MODEL_KEY}" >&2
        echo "Use: llama3 | mistral7b" >&2
        exit 1
        ;;
esac

echo "============================================"
echo "Starting vLLM OpenAI-compatible server"
echo "  Load from : ${MODEL_ID}"
echo "  Serve as  : ${CANONICAL_MODEL_ID}"
echo "  GPU       : ${GPU_ID}"
echo "  Port      : ${PORT}"
echo "  MaxLen    : ${MAX_MODEL_LEN}"
echo "============================================"

if [[ -z "${HF_TOKEN:-}" ]] && [[ "${MODEL_KEY}" == "llama3" ]] \
        && [[ ! -d "${MODEL_ID}" ]]; then
    echo "WARNING: HF_TOKEN is not set. Llama-3-8B-Instruct requires HuggingFace"
    echo "  access approval; run 'huggingface-cli login' or export HF_TOKEN."
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --served-model-name "${CANONICAL_MODEL_ID}" \
    --dtype float16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --host 127.0.0.1 \
    --port "${PORT}"
