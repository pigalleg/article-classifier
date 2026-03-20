@echo off
REM Usage: env_cloud.bat
REM Ensure local override is cleared
set OPENAI_BASE_URL=
REM Force-set cloud config
set OPENAI_MODEL=gpt-4o-mini
set OPENAI_REQUESTS_PER_MINUTE=20
echo Cloud LLM env set: model=%OPENAI_MODEL%, rpm=%OPENAI_REQUESTS_PER_MINUTE%
