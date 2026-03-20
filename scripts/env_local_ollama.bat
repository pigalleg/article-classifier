@echo off
REM Usage: env_local_ollama.bat
REM Force-set local (Ollama) config
set OPENAI_BASE_URL=http://localhost:11434/v1
set OPENAI_API_KEY=ollama
set OPENAI_MODEL=llama3.1:latest
set OPENAI_REQUESTS_PER_MINUTE=9999
echo Local LLM env set: base=%OPENAI_BASE_URL%, model=%OPENAI_MODEL%, rpm=%OPENAI_REQUESTS_PER_MINUTE%
