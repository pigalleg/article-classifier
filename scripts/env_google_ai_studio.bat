@echo off
REM Usage: env_google_ai_studio.bat
REM Configures OpenAI-compatible endpoint for Google AI Studio (Gemini).

REM Clear any previous local backend override (e.g., Ollama)
set OPENAI_BASE_URL=

REM Prefer GOOGLE_API_KEY; fall back to GEMINI_API_KEY if present.
if defined GEMINI_API_KEY if not defined GOOGLE_API_KEY (
  set GOOGLE_API_KEY=%GEMINI_API_KEY%
)

if not defined GOOGLE_API_KEY (
  echo Missing GOOGLE_API_KEY ^(or GEMINI_API_KEY^).
  echo Set it first, for example: set GOOGLE_API_KEY=your-key
  exit /b 1
)

REM OpenAI-compatible endpoint for Google AI Studio.
set OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
echo Google AI Studio LLM env set: base=%OPENAI_BASE_URL%
