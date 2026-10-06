# Data Management and Lineage

This guide describes the active data flow in the repository. Paths containing
`to delete` are intentionally excluded: they are historical artifacts and are
not inputs to supported workflows.

## Data Classes

| Class | Store in Git? | Purpose |
|---|---|---|
| Source code, configuration, prompts, and small curated reference data | Yes | Makes workflow behavior reviewable and reproducible. |
| Large raw exports and expert workbooks | No; use controlled shared storage or DVC | Required inputs that are too large or restricted for ordinary Git history. |
| Deterministic processed data and embedding caches | Normally no | Rebuild from the corresponding source inputs. |
| High-value benchmark and optimization results | No; archive selected runs in shared storage or DVC | Re-running can be expensive and model responses may not be repeatable. |
| Exploratory results and notebook figures | Normally no | Rebuild or archive only when supporting a report. |

## Raw-to-Processed Data Flow

| Raw source | Processed output | Generator | Active consumers | Notes |
|---|---|---|---|---|
| `data/raw/TEC_2000_2026_ieee.csv`, `TEMPR_2023_2026_ieee.csv`, `TPWRD_2000_2026_ieee.csv`, `TPWRS_2000_2026_ieee.csv`, `TSG_2010_2026_ieee.csv`, `TSTE_2010_2026_ieee.csv` | `data/processed/abstracts_cleaned.csv` | `scripts/util/prepare_ra_and_ieee_data.py` | `run_affinity_evaluation.py`, `populate_few_shot_ra_from_ministral_examples.py`, notebooks 09 and 13 | The six journal exports are merged, identifiers are normalized, and abstracts/keywords are cleaned. |
| `data/raw/Research Agenda 2025.xlsx`, `main` sheet | `data/processed/ra_questions_cleaned.csv` | `scripts/util/prepare_ra_and_ieee_data.py` | `run_affinity_evaluation.py`, `populate_few_shot_ra_from_ministral_examples.py` | RA question identifiers and cleaned question text. |
| `data/raw/Research Agenda 2025.xlsx`, `Primary Research Programme` sheet | `data/processed/primary_programmes.csv` | `scripts/util/prepare_ra_and_ieee_data.py` | `run_affinity_evaluation.py`, `run_affinity_benchmark.py` | PRP names and descriptions for PRP affinity scoring. |
| `data/raw/Abstracts Evaluation Template - Mark OM.xlsm` | `data/processed/affinity_calibration/<run_id>/calibration_abstract_question_affinities.csv` | `scripts/util/generate_affinity_calibration_from_excel.py` | `scripts/util/select_cases_by_levels.py`; notebooks 10, 11, 12, 14, and 15 | Expert annotation export. The default source workbook is Mark OM. |
| `data/raw/Abstracts Evaluation Template - Charlie S.xlsx` | `data/processed/affinity_calibration/<run_id>/calibration_abstract_question_affinities.csv` | `scripts/util/generate_affinity_calibration_from_excel.py` with explicit input/output options | `scripts/util/select_cases_by_levels.py`; calibration notebooks | Expert annotation export. |
| `data/raw/Abstracts Evaluation Template - Janusz.xlsx` | `data/processed/affinity_calibration/<run_id>/calibration_abstract_question_affinities.csv` | `scripts/util/generate_affinity_calibration_from_excel.py` with explicit input/output options | `scripts/util/select_cases_by_levels.py` | Expert annotation export. |
| `data/raw/Abstracts Evaluation Template - Batch 1.xlsx` | No processed CSV | `scripts/util/populate_few_shot_ra_from_excel_evaluations.py` | The same utility | Adds reviewed examples to the RA few-shot prompt YAML. |

## Curated and Derived Processed Data

| File | Origin | Consumers | Tracking recommendation |
|---|---|---|---|
| `data/processed/ra_grouping_ra2025.csv` | Manually curated RA-to-programme mapping | Analysis utilities and notebooks 01, 02, 09, 16, and 17 | Git. It defines the canonical programme ordering. |
| `notebooks/input/ra_extended_grouping_ra2025.csv` | Notebook-specific RA-to-programme mapping | Archived notebook 09 | Git. The file is colocated with its exploratory notebook input rather than the active processed-data pipeline. |
| `data/processed/manifest_tpwrs_2010_2026_by_year.csv` | Manually curated benchmark manifest | Manifest-driven benchmark workflows | Git. It defines reproducible year slices. |
| `data/processed/abstracts_cleaned_benchmark_v1.csv` | Curated corpus subset | Benchmark runs when selected | Git if it is the frozen study population; document its selection rule. |
| `data/processed/abstracts_cleaned_calibration_v1.csv` | Curated corpus subset | Calibration workflow | Git if it is the frozen study population; document its selection rule. |
| `data/processed/cache/ra_embeddings_*.npy` | Runtime embedding cache keyed by RA text | `RAClassifier` | Ignore and regenerate. |

## Results Lineage

| Input | Generator | Output | Keep |
|---|---|---|---|
| Processed abstracts, RA questions, PRPs, prompts, and settings | `scripts/run_affinity_evaluation.py` | Per-model `ra_affinities.csv`, `prp_affinities.csv`, `combined_affinities.xlsx`, lookup and timing data | Archive selectively; generated outputs are normally ignored. |
| Per-model affinity outputs | `scripts/run_postprocessing_affinity_benchmark.py` | `merged_ra_affinities.csv`, `merged_prp_affinities.csv` | Archive selected benchmark runs with their metadata. |
| Merged affinity outputs | `scripts/run_affinity_adjudication.py` | `adjudicated_ra_affinities.csv`, `adjudicated_prp_affinities.csv`, model-row variants | Archive selected benchmark runs with their metadata. |
| Expert calibration CSV plus benchmark output | Optimizer workflow in `optimization/` | Weights, reports, predictions | Archive final/reportable runs with their inputs and run metadata. |

Each selected result directory should retain its `run_metadata.json`, benchmark
manifest, settings snapshot, and all result files together. This preserves the
model configuration, source revision, and run identity required to interpret a
score later.

## Active Notebook Data Selections

This table records the currently active data-folder selectors in notebooks as
of September 2026. Commented alternatives are excluded. For conditional
loaders, the row reflects the branch selected by the current flag value.

| Notebook | Active benchmark folder or pattern | Active calibration folder |
|---|---|---|
| `01_ensemble_affinity.ipynb` | `tpwrs_2026_tpwrs_2026` | - |
| `02_ensemble_affinity_adjudication.ipynb` | `tpwrs_2026_tpwrs_2026` | - |
| `06_prp_adjudication_error_bars.ipynb` | `tpwrs_2026_tpwrs_2026` | - |
| `07_intra_panel_rank_consistency - promp modificatiom.ipynb` | `20260517_175302` | - |
| `07_intra_panel_rank_consistency base case.ipynb` | `20260422_212121_1` | - |
| `07_intra_panel_rank_consistency.ipynb` | `20260726_154151` | - |
| `10_affinity_diagnostics_metrics_per_model.ipynb` | `20260724_155639` | `20260618_Mark_batch1_v4` |
| `11_affinity_diagnostisc_metrics_pipeline.ipynb` | `20260726_134607_5` | `20260728_abstracts_evaluation_template_mark_om_v1` |
| `12_affinity_diagnostics_metrics_binary.ipynb` | `20260726_134607_5` | `20260728_abstracts_evaluation_template_mark_om_v1` |
| `13_pipeline_timing_price.ipynb` | all folders containing `affinity_timing_log.csv` | - |
| `14_calibration_methodology_comparison.ipynb` | `20260726_134607_1` | `20260728_abstracts_evaluation_template_mark_om_v1`; `20260728_abstracts_evaluation_template_charlie_s_v1` |
| `15_adjudication_impact_deep_dive.ipynb` | `20260726_134607_5.1` | `20260728_abstracts_evaluation_template_mark_om_v1` |
| `16_tpwrs_2025_2026_affinity_analysis.ipynb` | `tpwrs_*` | - |
| `17_*`, `18_*`, and `19_*` TPWRS notebooks | `tpwrs_*` | - |

Notebook 10 currently uses `select = "calibration"`. Changing that flag to
`"benchmark"` instead selects benchmark run `20260726_134607_5` and
calibration run `20260728_abstracts_evaluation_template_mark_om_v1`.

## DVC Storage Decision

| Repository path | Current scale | Storage decision |
|---|---:|---|
| Selected folders in `data/results/affinity_benchmark/` | 19 individually tracked runs | DVC-tracked and uploaded to the private `gdrive` remote. The targets are the seven fixed notebook selections plus TPWRS runs 2015-2026. Other local benchmark runs remain outside DVC. |
| `data/results/ensemble_weight_optimizer/` | 6.1 MB; 283 files | DVC-tracked and uploaded to the private `gdrive` remote. |
| `data/processed/affinity_calibration/` | 22 files; 2.4 MB | Staged for ordinary Git tracking. DVC and Git LFS are unnecessary at this scale. |
| `data/processed/affinity_calibration/to delete/` | Historical artifacts | Ignored by Git and not DVC-tracked. Retain locally or archive outside the active dataset. |

DVC is preferred over Git LFS for benchmark results because the outputs are
numerous, generated, and large. A DVC remote keeps ordinary Git history small
while allowing notebooks to retrieve the selected result folders with
`dvc pull`.

The DVC target list, manifest paths, checksums, and storage classes are
recorded in `data/data-registry.yaml`. The registry names the remote as
`gdrive`, but deliberately does not store its Google Drive URL or credentials.
Repository-specific installation, authentication, push/pull, and troubleshooting
steps are documented in `docs/dvc-setup.md`.

## Sharing a Fresh Clone

1. Clone the Git repository to obtain code, prompts, settings, small reference
   mappings, manifests, and curated study subsets.
2. Retrieve raw inputs from the agreed shared storage, or run `dvc pull` after
   DVC is configured.
3. Run `python scripts/util/prepare_ra_and_ieee_data.py` to recreate the main
   processed inputs.
4. Retrieve an archived benchmark run only when analysis requires it. Do not
   download the entire result history by default.
5. Treat expert workbooks and their calibration exports as restricted data;
   grant access through a separate shared-storage location or DVC remote.

## Storage Record

Keep this guide as the human-readable policy. The versioned
`data/data-registry.yaml` records one entry per shared dataset or archived run:
logical name, repository path, source or producing script, storage class,
access tier, checksum or DVC target, and a short description. Never store
personal OneDrive paths, credentials, or shared links with embedded secrets in
the repository.