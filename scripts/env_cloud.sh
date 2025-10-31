#!/usr/bin/env bash
# Usage: source scripts/env_cloud.sh

# Ensure local override is cleared
unset OPENAI_BASE_URL

# Force-set cloud config
export OPENAI_API_KEY="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"   # set your key
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_REQUESTS_PER_MINUTE="20"

echo "Cloud LLM env set: model=$OPENAI_MODEL, rpm=$OPENAI_REQUESTS_PER_MINUTE"