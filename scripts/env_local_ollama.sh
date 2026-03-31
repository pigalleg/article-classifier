#!/usr/bin/env bash
# set -euo pipefail
# Usage: source scripts/env_local_ollama.sh
# Force local backend.
export LLM_BACKEND="local"

# Clear cloud-only overrides so settings.yaml local defaults are used.
unset OPENAI_BASE_URL 2>/dev/null || true
unset OPENAI_API_KEY 2>/dev/null || true
unset OPENAI_MODEL 2>/dev/null || true
unset OPENAI_REQUESTS_PER_MINUTE 2>/dev/null || true

echo "Local LLM env set via settings.yaml defaults (backend=$LLM_BACKEND)"