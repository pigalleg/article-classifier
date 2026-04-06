#!/usr/bin/env bash
set -euo pipefail

# Enable GPU runtime in Ollama and restart the server.
# Usage: scripts/enable_ollama_gpu.sh [runtime] [model]
# Examples:
#   scripts/enable_ollama_gpu.sh
#   scripts/enable_ollama_gpu.sh cuda_v12
#   scripts/enable_ollama_gpu.sh cuda_v12 llama3.1:latest

RUNTIME=""
MODEL="llama3.1:latest"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama not found in PATH." >&2
  exit 1
fi

# Parse arguments
case "${1:-}" in
  cuda_v12|rocm_v6|metal|hip|vulkan)
    RUNTIME="${1}"
    MODEL="${2:-$MODEL}"
    ;;
  "")
    ;;
  *)
    MODEL="${1}"
    ;;
esac

# Auto-detect GPU runtime if not specified
if [ -z "$RUNTIME" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    RUNTIME="cuda_v12"
  elif command -v rocm-smi >/dev/null 2>&1; then
    RUNTIME="rocm_v6"
  else
    echo "Error: No GPU runtime detected (nvidia-smi or rocm-smi not found)." >&2
    exit 1
  fi
fi

echo "Killing Ollama..."
pkill -x ollama >/dev/null 2>&1 || true
sleep 1

# Wait for port 11434 to be free
for i in {1..5}; do
  if ! lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Restarting Ollama with GPU runtime: $RUNTIME"
nohup env OLLAMA_LLM_LIBRARY="$RUNTIME" ollama serve >/tmp/ollama-serve.log 2>&1 &
sleep 2

if ! pgrep -x ollama >/dev/null; then
  echo "Error: Ollama failed to restart. Check /tmp/ollama-serve.log" >&2
  cat /tmp/ollama-serve.log >&2
  exit 1
fi

echo "Warming up model: $MODEL"
ollama run "$MODEL" "Reply with the single word: ready" >/dev/null 2>&1 || true

echo ""
echo "GPU runtime enabled. Status:"
ollama ps || true
