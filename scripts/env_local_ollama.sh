#!/usr/bin/env bash
# set -euo pipefail
# Usage: source scripts/env_local_ollama.sh
# Force-set local (Ollama) config
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
export OPENAI_MODEL="llama3.1:latest" #"gemma3:27b"
export OPENAI_REQUESTS_PER_MINUTE="9999"

echo "Local LLM env set: base=$OPENAI_BASE_URL, model=$OPENAI_MODEL, rpm=$OPENAI_REQUESTS_PER_MINUTE"