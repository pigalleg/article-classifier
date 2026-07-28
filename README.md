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
source scripts/env_openAI.sh
# or
source scripts/env_local_ollama.sh
```

**Windows:**
```cmd
scripts\env_openAI.bat
<!-- or -->
scripts\env_local_ollama.bat
```

Notes:
- `scripts/env_openAI.*` sets `LLM_BACKEND=cloud`.
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

### Run Parallel Benchmark Launcher

Use the launcher when benchmark models are split into independent execution groups.
Models in the same `execution_group` run sequentially inside one benchmark
process; different groups run in parallel.

Usage guidelines:
- Keep models that share one GPU or one constrained backend in the same group.
- Use a logical group name such as `cloud` or `local-gpu`.
- Prefer this launcher when you want parallelism without making the benchmark
script itself more complex.

Examples:

```bash
python scripts/run_affinity_benchmark_parallel.py --dry-run
python scripts/run_affinity_benchmark_parallel.py --spawn-dry-run
python scripts/run_affinity_benchmark_parallel.py --run-id 20260713_parallel
```

The launcher forwards all regular benchmark arguments to
`run_affinity_benchmark.py`.

Final outputs are flattened into `data/results/affinity_benchmark/<run_id>/`.
Group-level metadata files are kept with group-specific prefixes so they do not
overwrite each other.

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
- `--enable-few-shot` / `--no-enable-few-shot`: override few-shot prompting for the current run only.

Example:

```bash
python scripts/run_affinity_evaluation.py --mode ra --ra-retrieval-mode prp_only --prp-input data/results/prp_affinities.csv

# force few-shot on for this run
python scripts/run_affinity_evaluation.py --mode ra --enable-few-shot
```

### Optional: Few-shot Affinity Calibration

You can inject few-shot examples into RA/PRP affinity prompts to reduce model variance.

Configuration is under `runtime.llm_reasoner` in `src/config/settings.yaml`:
- `enable_few_shot`: enable/disable few-shot prompting (default: `false`)
- `enable_affinity_reasons`: request a reason per target ID in the affinity JSON response (default: `false`)
- `few_shot_ra_file`: YAML file for RA examples
- `few_shot_prp_file`: YAML file for PRP examples
- `few_shot_max_examples_per_prompt`: max examples injected per prompt

Default example files:
- `data/prompts/few_shot_ra.yaml`
- `data/prompts/few_shot_prp.yaml`

YAML format:

```yaml
version: 1
examples:
  - id: ra_high_1
    target_type: RA
    abstract: "..."
    targets:
      - target_id: "42"
        score: 88
        rationale: "optional short explanation"
      - target_id: "43"
        score: 61
        rationale: "optional short explanation"
```

Notes:
- Use `target_type: RA` in the RA file and `target_type: PRP` in the PRP file.
- Multiple labels per abstract are supported through `targets`.
- Backward compatibility is kept for legacy single-label fields (`target_id`, `score`, `rationale`).
- Scores are clamped to `[0, 100]`.
- When `enable_affinity_reasons` is on, the model returns one short reason per ID and the output CSV includes `LLM_Affinity_Reason`.
- You can override settings with env vars:
  - `AFFINITY_ENABLE_FEW_SHOT`
  - `AFFINITY_ENABLE_AFFINITY_REASONS`
  - `AFFINITY_FEW_SHOT_RA_FILE`
  - `AFFINITY_FEW_SHOT_PRP_FILE`
  - `AFFINITY_FEW_SHOT_MAX_EXAMPLES`

### Build RA Few-Shot Examples From Excel Evaluations

Use the utility script below to append RA examples from an evaluator Excel file
to `data/prompts/few_shot_ra.yaml`.

Dry run (preview only):

```bash
python scripts/util/populate_few_shot_ra_from_excel_evaluations.py --dry-run
```

Append mode (writes to YAML):

```bash
python scripts/util/populate_few_shot_ra_from_excel_evaluations.py
```

With explicit input/output arguments:

```bash
python scripts/util/populate_few_shot_ra_from_excel_evaluations.py \
  --input-xlsx "<path-to-xlsx>" \
  --sheet-name "Rationale_clean" \
  --output-yaml data/prompts/few_shot_ra.yaml
```

Expected Excel columns in sheet `Rationale_clean`:
- `Abstract_Index`
- `Abstract`
- `RA2025_ID`
- `Evaluator_Score`
- `Why this score was assigned to the abstract?`
- `Why does this abstract fall below the next relevance level?`

Score mapping used:
- `0 -> 10`
- `1 -> 30`
- `2 -> 50`
- `3 -> 70`
- `4 -> 90`
- `5 -> 110`

### Build Calibration File From Raw Excel Evaluation Matrix

Use the utility script below to generate a calibration CSV directly from the
raw evaluation workbook in `data/raw`.

Default workbook:
- `data/raw/Abstracts Evaluation Template - Mark OM.xlsm`

The generator reads `Evaluation Matrix - Mark`, keeps questions that are either
in the row-3 selected program subset or have an affinity score greater than 0,
and writes `calibration_abstract_question_affinities.csv` under a date-stamped
run folder in `data/results/affinity_calibration/`.

Dry run (preview only):

```bash
python scripts/util/generate_affinity_calibration_from_excel.py --dry-run
```

Write mode:

```bash
python scripts/util/generate_affinity_calibration_from_excel.py
```

With explicit input/output arguments:

```bash
python scripts/util/generate_affinity_calibration_from_excel.py \
  --input-xlsx "data/raw/Abstracts Evaluation Template - Mark OM.xlsm" \
  --sheet-name "Evaluation Matrix - Mark" \
  --output-root data/results/affinity_calibration
```

Output columns:
- `Abstract_Index`
- `Document Title`
- `Question ID`
- `RA_Question`
- `Evaluator_Affinity`
- `Evaluator_Affinity_Level`
- `Evaluator_Affinity_Reason`
- `Source_Evaluator`

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