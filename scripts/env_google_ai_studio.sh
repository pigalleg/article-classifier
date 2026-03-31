#!/usr/bin/env bash
# Usage: source scripts/env_google_ai_studio.sh
# Configures OpenAI-compatible endpoint for Google AI Studio (Gemini).

# Force cloud backend.
export LLM_BACKEND="cloud"

# Clear any previous local backend override (e.g., Ollama)
unset OPENAI_BASE_URL

# Prefer GOOGLE_API_KEY; fall back to GEMINI_API_KEY if present.
if [[ -z "${GOOGLE_API_KEY:-}" && -n "${GEMINI_API_KEY:-}" ]]; then
  export GOOGLE_API_KEY="$GEMINI_API_KEY"
fi

if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "Missing GOOGLE_API_KEY (or GEMINI_API_KEY)."
  echo "Set it first, for example: export GOOGLE_API_KEY='your-key'"
  return 1 2>/dev/null || exit 1
fi

# OpenAI-compatible endpoint for Google AI Studio.
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
export OPENAI_API_KEY="$GOOGLE_API_KEY"

# A commonly available Gemini model in AI Studio.
export OPENAI_MODEL="gemini-2.5-flash"

# Conservative default rate limit; tune as needed. Monitor request rates at: https://aistudio.google.com/rate-limit?project=gen-lang-client-0684680159
export OPENAI_REQUESTS_PER_MINUTE="5"

echo "Google AI Studio env set: backend=$LLM_BACKEND, base=$OPENAI_BASE_URL, model=$OPENAI_MODEL, rpm=$OPENAI_REQUESTS_PER_MINUTE"
