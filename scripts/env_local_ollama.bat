@echo off
REM Usage: scripts\env_local_ollama.bat
REM Force local backend.
set LLM_BACKEND=local

REM Clear cloud-only overrides so settings.yaml local defaults are used.
set OPENAI_BASE_URL=
set OPENAI_API_KEY=
set OPENAI_MODEL=
set OPENAI_REQUESTS_PER_MINUTE=

echo Local LLM env set via settings.yaml defaults (backend=%LLM_BACKEND%)
