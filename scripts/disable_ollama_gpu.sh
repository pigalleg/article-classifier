#!/usr/bin/env bash
set -euo pipefail

# Disable GPU runtime in Ollama and force CPU-only mode.
# Usage: scripts/disable_ollama_gpu.sh [model]
# Examples:
#   scripts/disable_ollama_gpu.sh
#   scripts/disable_ollama_gpu.sh llama3.1:latest

MODEL="llama3.1:latest"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama not found in PATH." >&2
  exit 1
fi

if [ -n "${1:-}" ]; then
  MODEL="${1}"
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

echo "Restarting Ollama in CPU-only mode..."
nohup env OLLAMA_NUM_GPU=0 ollama serve >/tmp/ollama-serve.log 2>&1 &
sleep 2

if ! pgrep -x ollama >/dev/null; then
  echo "Error: Ollama failed to restart. Check /tmp/ollama-serve.log" >&2
  cat /tmp/ollama-serve.log >&2
  exit 1
fi

echo "Warming up model: $MODEL"
ollama run "$MODEL" "Reply with the single word: ready" >/dev/null 2>&1 || true

echo ""
echo "GPU runtime disabled (CPU-only). Status:"
ollama ps || true
