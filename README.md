# IEEE Classifier

Project to classify IEEE abstracts against Research Agenda (RA) questions using embeddings and LLM reasoning.

## Run Scripts From scripts/

These scripts use relative paths like `data/processed` and imports from `src`. If your terminal is in `scripts/`, run them through a subshell that changes to the repository root.

### 1) Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Enter scripts folder

```bash
cd scripts
```

### 3) Optional: choose LLM backend

```bash
source env_cloud.sh
# or
source env_local_ollama.sh
# or
source env_google_ai_studio.sh
```

### 4) Run each pipeline step (while staying in scripts/)

```bash
(cd .. && python scripts/prepare_data.py)
(cd .. && python scripts/run_classification.py)
(cd .. && python scripts/run_affinity_evaluation.py)
(cd .. && python scripts/run_postprocessing.py)
(cd .. && python scripts/run_keyword_clustering.py)
```

## Script Outputs

- `prepare_data.py` -> `data/processed/abstracts_cleaned.csv`, `data/processed/ra_questions_cleaned.csv`, `data/processed/primary_programmes.csv`
- `run_classification.py` -> `data/results/classified_articles_llm.xlsx`, `data/results/llm_mismatches.csv`
- `run_affinity_evaluation.py` -> `data/results/ra_affinities.csv`, `data/results/prp_affinities.csv`, `data/results/combined_affinities.xlsx`
- `run_postprocessing.py` -> analysis artifacts in `outputs/`
- `run_keyword_clustering.py` -> `data/processed/keyword_clusters.csv`

## Notes

- `run_keyword_clustering.py` also needs `scikit-learn`.
- If an API backend is not configured, LLM-based scripts will fail before classification/affinity steps.
