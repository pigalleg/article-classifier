# One-Shot: Build a Calibration CSV From Excel

This is a manual, optional data-preparation utility. It is not called by the
benchmark or the normal affinity-evaluation and adjudication stages. Use it
when a calibration analysis needs a CSV exported from an evaluator workbook.

## Inputs and Output

By default, the script reads `data/raw/Abstracts Evaluation Template - Mark OM.xlsm`.
It uses the first worksheet whose name contains `Evaluation Matrix`, unless
`--sheet-name` is supplied.

For each abstract, the script keeps a question row when it belongs to the
abstract's selected program subset in row 3 of the worksheet, or when its
recorded affinity score is greater than 0. Evaluator scores must be integers
from 0 through 5 and map to affinity scores as follows:

| Evaluator score | Affinity score |
|---:|---:|
| 0 | 10 |
| 1 | 30 |
| 2 | 50 |
| 3 | 70 |
| 4 | 90 |
| 5 | 110 |

By default, the CSV is written under
`data/processed/affinity_calibration/<generated_run_id>/` as
`calibration_abstract_question_affinities.csv`. The generated run folder also
contains `run_metadata.json`. The CSV uses the calibration schema:

- `Abstract_Index`
- `Document Title`
- `RA2025_ID`
- `RA_Question`
- `Evaluator_Affinity`
- `Evaluator_Affinity_Level`
- `Evaluator_Affinity_Reason`
- `Source_Evaluator`

The default run ID includes the current UTC date/time and workbook name. This
calibration export supports calibration analysis; it is not an input to the
main benchmark/adjudication workflow.

## Run It

Preview the output row count and destination without writing the CSV or metadata:

```bash
python scripts/util/generate_affinity_calibration_from_excel.py --dry-run
```

Generate the CSV with defaults:

```bash
python scripts/util/generate_affinity_calibration_from_excel.py
```

Specify a workbook, worksheet, and output root:

```bash
python scripts/util/generate_affinity_calibration_from_excel.py \
  --input-file "Abstracts Evaluation Template - Mark OM.xlsm" \
  --sheet-name "Evaluation Matrix - Mark" \
  --output-root data/processed/affinity_calibration
```

Relative input filenames are resolved from `data/raw`. Use `--output-run-id`
to choose a stable run-folder name or `--output-file` to change the CSV
filename. Calibration generation and its downstream analysis are described in
[Data Management](data-management.md).
