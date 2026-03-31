#!/usr/bin/env bash
# Usage: source scripts/env_cloud.sh
# Force cloud backend
export LLM_BACKEND="cloud"
# Ensure local override is cleared
unset OPENAI_BASE_URL 2>/dev/null || true
# Force-set cloud config
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_REQUESTS_PER_MINUTE="20"
echo "Cloud LLM env set: backend=$LLM_BACKEND, model=$OPENAI_MODEL, rpm=$OPENAI_REQUESTS_PER_MINUTE"