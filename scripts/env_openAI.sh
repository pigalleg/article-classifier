#!/usr/bin/env bash
# Usage: source scripts/env_openAI.sh
# Rate limits: https://platform.openai.com/home
# Force cloud backend.
export LLM_BACKEND="cloud"

# Ensure local/OpenAI-compatible override is cleared.
unset OPENAI_BASE_URL 2>/dev/null || true

# Cloud key source (separate from OPENAI_API_KEY so this script can restore it).
# Set OPENAI_CLOUD_API_KEY once, then re-source this script whenever needed.
export OPENAI_CLOUD_API_KEY="${OPENAI_CLOUD_API_KEY:-${OPENAI_API_KEY:-}}"

# Cloud defaults (can still be overridden by caller before running Python).
export OPENAI_MODEL="gpt-5.4-nano"
export OPENAI_REQUESTS_PER_MINUTE="20" 

# Validate key presence early to avoid confusing runtime failures.
if [[ -z "${OPENAI_CLOUD_API_KEY:-}" ]]; then
  echo "Missing OPENAI_CLOUD_API_KEY for cloud backend."
  echo "Set it first, for example: export OPENAI_CLOUD_API_KEY='sk-...'"
  return 1 2>/dev/null || exit 1
fi

# Always set the runtime key from the dedicated cloud key source.
export OPENAI_API_KEY="$OPENAI_CLOUD_API_KEY"

# Optional project/org routing info can help debug quota issues.
project_info="${OPENAI_PROJECT:-<unset>}"
org_info="${OPENAI_ORG:-<unset>}"

echo "Cloud LLM env set: backend=$LLM_BACKEND, model=$OPENAI_MODEL, rpm=$OPENAI_REQUESTS_PER_MINUTE, project=$project_info, org=$org_info"
