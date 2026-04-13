@echo off
REM Usage: scripts\env_openAI.bat
REM Rate limits: https://platform.openai.com/home
REM Force cloud backend.
set LLM_BACKEND=cloud

REM Ensure local/OpenAI-compatible override is cleared.
set OPENAI_BASE_URL=

REM Cloud key source (separate from OPENAI_API_KEY so this script can restore it).
REM Set OPENAI_CLOUD_API_KEY once, then re-run this script whenever needed.
if not defined OPENAI_CLOUD_API_KEY if defined OPENAI_API_KEY (
	set OPENAI_CLOUD_API_KEY=%OPENAI_API_KEY%
)

REM Cloud defaults (can still be overridden by caller before running Python).
set OPENAI_MODEL=gpt-5.4-nano
set OPENAI_REQUESTS_PER_MINUTE=20

REM Validate key presence early to avoid confusing runtime failures.
if not defined OPENAI_CLOUD_API_KEY (
	echo Missing OPENAI_CLOUD_API_KEY for cloud backend.
	echo Set it first, for example: set OPENAI_CLOUD_API_KEY=sk-...
	exit /b 1
)

REM Always set the runtime key from the dedicated cloud key source.
set OPENAI_API_KEY=%OPENAI_CLOUD_API_KEY%

REM Optional project/org routing info can help debug quota issues.
if defined OPENAI_PROJECT (
	set project_info=%OPENAI_PROJECT%
) else (
	set project_info=^<unset^>
)
if defined OPENAI_ORG (
	set org_info=%OPENAI_ORG%
) else (
	set org_info=^<unset^>
)

echo Cloud LLM env set: backend=%LLM_BACKEND%, model=%OPENAI_MODEL%, rpm=%OPENAI_REQUESTS_PER_MINUTE%, project=%project_info%, org=%org_info%
