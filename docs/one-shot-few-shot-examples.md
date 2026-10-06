# One-Shot: Build RA Few-Shot Examples From Excel

This is a manual, optional data-preparation utility. It is not called by the
benchmark, affinity-evaluation, postprocessing, or adjudication workflow. Run it
when you want to import reviewed RA annotations into the few-shot prompt file.
The examples then affect later affinity runs when few-shot prompting is enabled.

## Inputs and Output

By default, the script reads `data/raw/Abstracts Evaluation Template - Batch 1.xlsx`,
worksheet `Rationale_clean`, and appends examples to
`data/prompts/few_shot_ra.yaml`.

The worksheet must contain these columns:

- `Abstract_Index`
- `Abstract`
- `RA2025_ID`
- `Evaluator_Score`
- `Why this score was assigned to the abstract?`
- `Why does this abstract fall below the next relevance level?`

The two rationale columns are combined into the example rationale. Rows without
a rationale are skipped. Evaluator scores map to affinity scores as follows:

| Evaluator score | Affinity score |
|---:|---:|
| 0 | 10 |
| 1 | 30 |
| 2 | 50 |
| 3 | 70 |
| 4 | 90 |
| 5 | 110 |

Generated examples use `target_type: RA`. Existing example IDs are not
appended again, so rerunning the utility does not duplicate examples that it
has already written.

## Run It

Preview the examples without changing the YAML file:

```bash
python scripts/util/populate_few_shot_ra_from_excel_evaluations.py --dry-run
```

Append new examples using the defaults:

```bash
python scripts/util/populate_few_shot_ra_from_excel_evaluations.py
```

Specify a workbook, worksheet, and output file:

```bash
python scripts/util/populate_few_shot_ra_from_excel_evaluations.py \
  --input-xlsx "<path-to-xlsx>" \
  --sheet-name "Rationale_clean" \
  --output-yaml data/prompts/few_shot_ra.yaml
```

The utility appends to the selected YAML file; review the dry-run output and
existing prompt examples before writing. Few-shot defaults and prompt-file
settings are documented in the [Few-Shot Prompting and Affinity Scoring Scale
guide](few-shot-affinity-calibration.md).
