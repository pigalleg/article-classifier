# IEEE Classifier

Project to classify IEEE abstracts against Research Agenda (RA) questions using embeddings and LLM reasoning.

## Run Scripts From scripts/

These scripts use relative paths like `data/processed` and imports from `src`. If your terminal is in `scripts/`, run them through a subshell that changes to the repository root.

### Setup Instructions

#### Linux / macOS

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then navigate to the scripts folder and run steps 3-4 below.

#### Windows (Command Prompt)

From the repository root:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then navigate to the scripts folder and run steps 3-4 below.

#### Windows (PowerShell)

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then navigate to the scripts folder and run steps 3-4 below.

### 3) Enter scripts folder

**Linux / macOS:**
```bash
cd scripts
```

**Windows (Command Prompt):**
```cmd
cd scripts
```

**Windows (PowerShell):**
```powershell
cd scripts
```

### 4) Optional: choose LLM backend

**Linux / macOS:**
```bash
source env_local_ollama.sh
# or
source env_cloud.sh
```

**Windows (Command Prompt or PowerShell):**
```cmd
env_local_ollama.bat
REM or
env_cloud.bat
```

### 5) Run each pipeline step (while staying in scripts/)

**Linux / macOS:**
```bash
<!-- (cd .. && python scripts/prepare_data.py) -->
<!-- (cd .. && python scripts/run_classification.py) -->
(cd .. && python scripts/run_affinity_evaluation.py)
<!-- (cd .. && python scripts/run_postprocessing.py) -->
<!-- (cd .. && python scripts/run_keyword_clustering.py) -->
```

**Windows (Command Prompt):**
```cmd
REM cd .. && python scripts/prepare_data.py && cd scripts
REM cd .. && python scripts/run_classification.py && cd scripts
cd .. && python scripts/run_affinity_evaluation.py && cd scripts
REM cd .. && python scripts/run_postprocessing.py && cd scripts
REM cd .. && python scripts/run_keyword_clustering.py && cd scripts
```

**Windows (PowerShell):**
```powershell
# & python ..\scripts\prepare_data.py
# & python ..\scripts\run_classification.py
& python ..\scripts\run_affinity_evaluation.py
# & python ..\scripts\run_postprocessing.py
# & python ..\scripts\run_keyword_clustering.py
```

### 6) Arguments for run_affinity_evaluation.py

- `--mode`: controls which stage to run.
	- `prp`: run PRP affinity only.
	- `ra`: run RA affinity only.
	- `both`: run PRP and RA affinity (default behavior).
- `--ra-retrieval-mode`: controls how RA candidates are selected before LLM affinity scoring.
	- `cosine`: pure embedding/cosine top-k retrieval.
	- `prp_filter`: cosine retrieval constrained by PRPs from `--prp-input`.
	- `prp_only`: strict PRP-routed retrieval from `--prp-input`.

Example:

```bash
(cd .. && python scripts/run_affinity_evaluation.py --mode ra --ra-retrieval-mode prp_only --prp-input data/results/prp_affinities.csv)
```

## Script Outputs

<!-- - `prepare_data.py` -> `data/processed/abstracts_cleaned.csv`, `data/processed/ra_questions_cleaned.csv`, `data/processed/primary_programmes.csv` -->
<!-- - `run_classification.py` -> `data/results/classified_articles_llm.xlsx`, `data/results/llm_mismatches.csv` -->
- `run_affinity_evaluation.py` -> `data/results/ra_affinities.csv`, `data/results/prp_affinities.csv`, `data/results/combined_affinities.xlsx`
<!-- - `run_postprocessing.py` -> analysis artifacts in `outputs/` -->
<!-- - `run_keyword_clustering.py` -> `data/processed/keyword_clusters.csv` -->

## Notes

<!-- - `run_keyword_clustering.py` also needs `scikit-learn`. -->
- If an API backend is not configured, LLM-based scripts will fail before classification/affinity steps.
