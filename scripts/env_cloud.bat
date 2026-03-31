@echo off
REM Usage: env_cloud.bat
REM Force cloud backend
set LLM_BACKEND=cloud
REM Ensure local override is cleared
set OPENAI_BASE_URL=
REM Force-set cloud config
set OPENAI_MODEL=gpt-4o-mini
set OPENAI_REQUESTS_PER_MINUTE=20
echo Cloud LLM env set: backend=%LLM_BACKEND%, model=%OPENAI_MODEL%, rpm=%OPENAI_REQUESTS_PER_MINUTE%
