# IEEE Classifier

Project to classify IEEE abstracts against Research Agenda (RA) questions using embeddings and LLM reasoning.

## Running Scripts

Run commands from the repository root. The scripts use paths like `data/processed` and imports from `src`.

### Setup Instructions

Requires Python 3.10+.

#### Linux / macOS

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows

From the repository root:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`pip` will install the `openai` Python package for cloud LLM backends. 

### Installing Ollama
Ollama is needed to run local LLMs. Download and install it from [here](https://ollama.com/download/), and pull any model via command line.

```bash
ollama pull [model_name]
```

### Enable GPU in Ollama (Linux/WSL)

1. Verify your GPU is visible in Linux:

```bash
nvidia-smi
```

2. Source local Ollama env (now GPU-aware):

```bash
source scripts/env_local_ollama.sh
```

3. Restart Ollama so runtime changes take effect:

```bash
pkill ollama || true
ollama serve
```

4. In a second terminal, run a test prompt and check processor:

```bash
ollama run llama3.1:latest "Say hello"
ollama ps
```

`ollama ps` should show the model processor as GPU (or mostly GPU).

For explicit GPU toggle scripts:

```bash
scripts/enable_ollama_gpu.sh
```

Optional runtime override and model selection when enabling:

```bash
scripts/enable_ollama_gpu.sh cuda_v12 llama3.1:latest
```

If you want to use a specific model without specifying a runtime:

```bash
scripts/enable_ollama_gpu.sh llama3.1:latest
```

To disable GPU (force CPU):

```bash
scripts/disable_ollama_gpu.sh
```

### Choose LLM backend

Default behavior: scripts use local Ollama settings from `src/config/settings.yaml`.
Switch between cloud and local by sourcing the cloud/local env scripts.

**Linux / macOS:**
```bash
source scripts/env_cloud.sh
# or
source scripts/env_local_ollama.sh
```

**Windows:**
```cmd
scripts\env_cloud.bat
<!-- or -->
scripts\env_local_ollama.bat
```

Notes:
- `scripts/env_cloud.*` sets `LLM_BACKEND=cloud`.
- `scripts/env_local_ollama.*` sets `LLM_BACKEND=local` and clears runtime overrides so local profile defaults in settings are used.
- You can still override model/base URL/rate limit through env vars (`OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_REQUESTS_PER_MINUTE`, `OPENAI_API_KEY`).

### Model List For Multi-Model Affinity Evaluation

Model profiles for benchmarking are stored in `src/config/settings.yaml` under `models.llm.benchmark.models`.

You can override the configured model list with environment variables:
- `AFFINITY_BENCHMARK_MODELS`: comma-separated model IDs (for example `llama3.1:latest,gemma3:27b`)
- `AFFINITY_BENCHMARK_BACKEND`: force backend for that override list (`local` or `cloud`)

### Run Multi-Model Affinity Benchmark

Run PRP+RA stages for each configured model and isolate outputs per model:

```bash
python scripts/run_affinity_benchmark.py --mode both --ra-retrieval-mode prp_only
```

Useful flags:
- `--continue-on-error` / `--stop-on-error`
- `--resume` / `--no-resume`
- `--output-root data/results/affinity_benchmark`
- `--run-id 20260331_local_benchmark`

Outputs are written under:
- `data/results/affinity_benchmark/<run_id>/<model_slug>/...`
- `data/results/affinity_benchmark/<run_id>/benchmark_manifest.csv`

### Run Affinity Evaluation

Launch affinity evaluation (scored from 0 to 100) for each abstract against each Primary Research Programme (PRP).

```bash
python scripts/run_affinity_evaluation.py --mode prp
```
Launch RA affinity evaluation for a subset of RA questions. Subsets are selected via criteria defined by the `--ra-retrieval-mode` argument (see next instructions for details).

```bash
python scripts/run_affinity_evaluation.py --mode ra --ra-retrieval-mode prp_only
```

#### Arguments for run_affinity_evaluation.py

- `--mode`: controls which stage to run.
  - `prp`: run PRP affinity only.
  - `ra`: run RA affinity only.
  - `both`: run PRP and RA affinity (default behavior).
- `--ra-retrieval-mode`: controls how RA candidates are selected before LLM affinity scoring.
  - `cosine`: pure embedding/cosine top-k retrieval.
  - `prp_filter`: cosine retrieval constrained by PRPs from `--prp-input`.
  - `prp_only`: strict PRP-routed retrieval from `--prp-input` (default behavior).

Example:

```bash
python scripts/run_affinity_evaluation.py --mode ra --ra-retrieval-mode prp_only --prp-input data/results/prp_affinities.csv
```

## Script Outputs

<!-- - `prepare_data.py` -> `data/processed/abstracts_cleaned.csv`, `data/processed/ra_questions_cleaned.csv`, `data/processed/primary_programmes.csv` -->
<!-- - `run_classification.py` -> `data/results/classified_articles_llm.xlsx`, `data/results/llm_mismatches.csv` -->
- `run_affinity_evaluation.py --mode prp` -> `data/results/prp_affinities.csv`
- `run_affinity_evaluation.py --mode ra` -> `data/results/ra_affinities.csv`, `data/results/combined_affinities.xlsx`
<!-- - `run_postprocessing.py` -> analysis artifacts in `outputs/` -->
<!-- - `run_keyword_clustering.py` -> `data/processed/keyword_clusters.csv` -->

## Notes

<!-- - `run_keyword_clustering.py` also needs `scikit-learn`. -->
- If an API backend is not configured, LLM-based scripts will fail before classification/affinity steps.
- Deprecated: `prepare_data.py`, `run_classification.py`, `run_postprocessing.py`, `run_keyword_clustering.py`