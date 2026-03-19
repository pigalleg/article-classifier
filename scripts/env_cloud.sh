#!/usr/bin/env bash
# Usage: source scripts/env_cloud.sh
# Ensure local override is cleared
unset OPENAI_BASE_URL
# Force-set cloud config
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_REQUESTS_PER_MINUTE="20"
echo "Cloud LLM env set: model=$OPENAI_MODEL, rpm=$OPENAI_REQUESTS_PER_MINUTE"