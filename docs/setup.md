# Setup

Run setup commands from the repository root. This guide covers the environment
and credentials for the benchmark models currently enabled in
`src/config/settings.yaml`.

## Python Environment

Use Python 3.10 or newer. Create and activate a virtual environment, then
install the project dependencies from `requirements.txt`.

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows Command Prompt

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The `openai` package in the requirements supports the configured OpenAI-
compatible APIs; separate Python SDKs for Google and Anthropic are not required
by these profiles.

## API Credentials

Benchmark profiles and credential variable names are configured in
`src/config/settings.yaml`. Obtain keys from the provider and make them
available in the environment of the terminal that launches Python.

| Enabled benchmark profile | Provider | Environment variable | Key portal |
|---|---|---|---|
| `gpt-5.4-mini` | OpenAI | `OPENAI_CLOUD_API_KEY` | [OpenAI API keys](https://platform.openai.com/api-keys) |
| `gemini-3.1-flash-lite` | Google AI Studio | `GOOGLE_API_KEY` | [Google AI Studio API keys](https://aistudio.google.com/apikey) |
| `claude-haiku-4.5` | Anthropic | `ANTHROPIC_API_KEY` | [Anthropic API keys](https://console.anthropic.com/settings/keys) |
| `gemma4-31b-cloud` | Ollama Cloud through local Ollama | Ollama sign-in; no provider key variable | [Ollama sign-in](https://docs.ollama.com/cli) |
| `deepseek-v4-flash-cloud` | Ollama Cloud through local Ollama | Ollama sign-in; no provider key variable | [Ollama sign-in](https://docs.ollama.com/cli) |

The enabled providers you plan to run must have valid API access and any
required provider-side billing or quota configured. The Ollama profiles use a
local Ollama server as a gateway to cloud-hosted models; Ollama account sign-in
is separate from the vendor API keys above.

### Set Keys in Bash or Zsh

Set only the keys for providers you will run. Replace each placeholder locally;
do not commit keys or put them in `settings.yaml`.

```bash
export OPENAI_CLOUD_API_KEY="<your-openai-api-key>"
export GOOGLE_API_KEY="<your-google-api-key>"
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"
```

These variables are available to processes started from this shell. Avoid
typing real secrets directly into commands if your shell saves command history;
use your secret manager or another secure input method where possible.

### Set Keys in Windows Command Prompt

Set only the keys for providers you will run. These values apply to the current
Command Prompt session and child processes.

```cmd
set "OPENAI_CLOUD_API_KEY=<your-openai-api-key>"
set "GOOGLE_API_KEY=<your-google-api-key>"
set "ANTHROPIC_API_KEY=<your-anthropic-api-key>"
```

Do not commit keys or put them in `settings.yaml`. Avoid typing real secrets
directly into commands if they may be retained in command history.

## Ollama

Install [Ollama](https://ollama.com/download/) and sign in to the Ollama account
used for cloud models:

```bash
ollama signin
```

Keep the Ollama application or server running while using these profiles. If a
server is not already running, start one with `ollama serve` and leave that
process running. Cloud model access requires internet connectivity and Ollama
account access.

The enabled profiles use these Ollama model names:

```text
gemma4:31b-cloud
deepseek-v4-flash:cloud
```

Although the profiles say `backend: local`, that means the application connects
to the local Ollama-compatible endpoint at `http://localhost:11434/v1`. The
`:cloud` models run inference in Ollama's cloud, so they do not require
downloading model weights or a local GPU. The configured client key `ollama` is
a required placeholder for the local OpenAI-compatible client and is ignored by
the local server; it is not a secret. `OLLAMA_API_KEY` is not required for
these configured profiles.

For a different, locally hosted Ollama model, use `ollama pull <model-name>`
before running it. Such models use local compute resources.

## GPU Acceleration (Optional)

GPU acceleration is enabled by default when Ollama detects supported hardware.
You do not need to run a GPU setup or enable command for normal use. In
particular, the enabled `:cloud` profiles above do not use your local GPU for
inference.

The following commands are optional diagnostics or overrides when using
locally hosted models. For example, after pulling a local model, check GPU
visibility and model placement with:

```bash
nvidia-smi
ollama pull llama3.1:latest
ollama run llama3.1:latest "Say hello"
ollama ps
```

To explicitly select GPU runtime settings, use the helper scripts (Linux/WSL):

```bash
scripts/enable_ollama_gpu.sh
scripts/enable_ollama_gpu.sh cuda_v12 llama3.1:latest
scripts/enable_ollama_gpu.sh llama3.1:latest
scripts/disable_ollama_gpu.sh
```

The GPU scripts change Ollama runtime behavior; consult their usage before
running them. A restart may be needed after changing runtime settings. Avoid
stopping an Ollama service that is shared with other users or applications.

## Choose a Backend

The default backend in `src/config/settings.yaml` is `cloud`. Backend helper
scripts provide convenient overrides for single-backend runs:

**Linux and macOS:**

```bash
source scripts/env_openAI.sh
# or
source scripts/env_google_ai_studio.sh
# or
source scripts/env_local_ollama.sh
```

**Windows Command Prompt:**

```cmd
call scripts\env_openAI.bat
call scripts\env_google_ai_studio.bat
call scripts\env_local_ollama.bat
```

Run one helper at a time in the same terminal used to launch Python. The
OpenAI and Google helpers set a model and request limit as well as the backend;
set any desired `OPENAI_MODEL` or `OPENAI_REQUESTS_PER_MINUTE` override after
running the helper. The Ollama helper selects the configured local profile and
clears cloud-specific runtime overrides.

The multi-model benchmark uses the enabled per-model profiles in
`src/config/settings.yaml` and reads the corresponding provider credentials
from the environment. Export the credentials for the providers you plan to run;
you do not need to source a one-provider helper to run a mixed benchmark. To
use YAML defaults for a single-model script, clear runtime overrides such as
`LLM_BACKEND`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_REQUESTS_PER_MINUTE` while keeping the required provider key set.