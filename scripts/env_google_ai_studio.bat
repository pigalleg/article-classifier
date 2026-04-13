@echo off
REM Usage: scripts\env_google_ai_studio.bat
REM Configures OpenAI-compatible endpoint for Google AI Studio (Gemini).

REM Force cloud backend.
set LLM_BACKEND=cloud

REM Clear any previous local backend override (e.g., Ollama).
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
set OPENAI_API_KEY=%GOOGLE_API_KEY%

REM A commonly available Gemini model in AI Studio.
set OPENAI_MODEL=gemini-3-flash-preview

REM Conservative default rate limit; tune as needed.
set OPENAI_REQUESTS_PER_MINUTE=500

echo Google AI Studio env set: backend=%LLM_BACKEND%, base=%OPENAI_BASE_URL%, model=%OPENAI_MODEL%, rpm=%OPENAI_REQUESTS_PER_MINUTE%
