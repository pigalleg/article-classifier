"""Generate a calibration dataset from a raw evaluation workbook.

The default workbook is the Mark OM annotation template under ``data/raw``.
The script reads the evaluation matrix, keeps cells that either:

- belong to the abstract's selected row-3 program subset, or
- have an affinity score greater than 0,

then writes a calibration CSV using the repo's standard schema.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openpyxl import load_workbook


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.run_affinity_adjudication import affinity_level_fixed

DEFAULT_INPUT_FILE = Path("Abstracts Evaluation Template - Mark OM.xlsm")
DEFAULT_SHEET_NAME = None
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "affinity_calibration"
DEFAULT_OUTPUT_FILE = "calibration_abstract_question_affinities.csv"
DEFAULT_RUN_SUFFIX = "v1"

ABSTRACT_ROW = 1
TITLE_ROW = 2
SELECTION_ROW = 3
HEADER_ROW = 4
FIRST_ABSTRACT_COL = 5

SCORE_MAPPING = {
    0: 10,
    1: 30,
    2: 50,
    3: 70,
    4: 90,
    5: 110,
}


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_") or "item"


def _canonical_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _is_missing(value: object) -> bool:
    return bool(pd.isna(value))


def _to_int_score(value: object) -> int:
    if _is_missing(value):
        raise ValueError("Affinity score is missing")

    try:
        parsed = int(float(str(value).strip()))
    except Exception as exc:
        raise ValueError(f"Affinity score '{value}' is not numeric") from exc

    if parsed not in SCORE_MAPPING:
        raise ValueError(
            f"Affinity score '{value}' is outside the expected 0-5 range"
        )
    return parsed


def _map_score(raw_score: object) -> int:
    return SCORE_MAPPING[_to_int_score(raw_score)]


def _normalize_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _find_header_column(ws, row_number: int, expected_header: str) -> int:
    for cell in ws[row_number]:
        if cell.value is None:
            continue
        if _canonical_text(cell.value) == _canonical_text(expected_header):
            return cell.column
    raise ValueError(
        f"Could not find header '{expected_header}' in row {row_number}"
    )


def _find_helper_column(ws) -> int:
    return _find_header_column(ws, HEADER_ROW, "Row Program (helper)")


def _iter_abstract_columns(ws, helper_col: int) -> Iterable[int]:
    for col_idx in range(FIRST_ABSTRACT_COL, helper_col):
        if ws.cell(ABSTRACT_ROW, col_idx).value is None:
            continue
        if ws.cell(TITLE_ROW, col_idx).value is None:
            continue
        yield col_idx


def _selected_by_row_three(selection_value: object, helper_value: object) -> bool:
    if _is_missing(selection_value) or _is_missing(helper_value):
        return False
    return _canonical_text(selection_value) == _canonical_text(helper_value)


def _should_transfer(selection_value: object, helper_value: object, score: object) -> bool:
    if _is_missing(score):
        return False

    score_int = _to_int_score(score)
    if score_int > 0:
        return True

    return _selected_by_row_three(selection_value, helper_value)


def _build_output_frame(input_xlsx: Path, sheet_name: str) -> pd.DataFrame:
    if not input_xlsx.exists() and input_xlsx.suffix.lower() == ".xlsx":
        alt_path = input_xlsx.with_suffix(".xlsm")
        if alt_path.exists():
            input_xlsx = alt_path

    workbook = load_workbook(input_xlsx, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {input_xlsx.name}. Available sheets: {workbook.sheetnames}"
        )

    ws = workbook[sheet_name]
    helper_col = _find_helper_column(ws)
    question_id_col = _find_header_column(ws, HEADER_ROW, "Question ID")
    question_text_col = _find_header_column(ws, HEADER_ROW, "Question")

    rows: list[dict[str, object]] = []
    selected_subset_rows = 0
    positive_score_rows = 0

    for col_idx in _iter_abstract_columns(ws, helper_col):
        abstract_index = _normalize_text(ws.cell(ABSTRACT_ROW, col_idx).value)
        document_title = _normalize_text(ws.cell(TITLE_ROW, col_idx).value)
        selection_value = ws.cell(SELECTION_ROW, col_idx).value

        if not abstract_index or not document_title:
            continue

        for row_idx in range(HEADER_ROW + 1, ws.max_row + 1):
            question_id = ws.cell(row_idx, question_id_col).value
            question_text = ws.cell(row_idx, question_text_col).value
            helper_value = ws.cell(row_idx, helper_col).value
            score = ws.cell(row_idx, col_idx).value

            if question_id is None and question_text is None:
                continue

            if not _should_transfer(selection_value, helper_value, score):
                continue

            score_int = _to_int_score(score)
            mapped_score = _map_score(score_int)
            if score_int > 0:
                positive_score_rows += 1
            else:
                selected_subset_rows += 1

            rows.append(
                {
                    "Abstract_Index": abstract_index,
                    "Document Title": document_title,
                    "RA2025_ID": _normalize_text(question_id),
                    "RA_Question": _normalize_text(question_text),
                    "Evaluator_Affinity": mapped_score,
                    "Evaluator_Affinity_Level": affinity_level_fixed(mapped_score),
                    "Evaluator_Affinity_Reason": "",
                    "Source_Evaluator": input_xlsx.name,
                }
            )

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError(
            f"No calibration rows were selected from {input_xlsx.name} on sheet '{sheet_name}'"
        )

    column_order = [
        "Abstract_Index",
        "Document Title",
        "RA2025_ID",
        "RA_Question",
        "Evaluator_Affinity",
        "Evaluator_Affinity_Level",
        "Evaluator_Affinity_Reason",
        "Source_Evaluator",
    ]
    output = output[column_order]

    if not set(output["Evaluator_Affinity"].unique()).issubset(set(SCORE_MAPPING.values())):
        raise ValueError("Generated calibration contains unexpected affinity values")

    output.attrs["selected_subset_rows"] = selected_subset_rows
    output.attrs["positive_score_rows"] = positive_score_rows
    return output


def _build_run_id(input_xlsx: Path) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d}_{_slugify(input_xlsx.stem)}_{DEFAULT_RUN_SUFFIX}"


def _find_evaluation_matrix_sheet_name(workbook) -> str:
    matching_sheet_names = [name for name in workbook.sheetnames if "Evaluation Matrix" in name]
    if not matching_sheet_names:
        raise ValueError(
            f"No worksheet name containing 'Evaluation Matrix' found. Available sheets: {workbook.sheetnames}"
        )
    return matching_sheet_names[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-run-id", default=None)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        input_file = REPO_ROOT / "data" / "raw" / input_file

    output_root = Path(args.output_root)
    output_run_id = args.output_run_id or _build_run_id(input_file)
    output_dir = output_root / output_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook = load_workbook(input_file, data_only=True)
    sheet_name = args.sheet_name or _find_evaluation_matrix_sheet_name(workbook)

    output_df = _build_output_frame(input_xlsx=input_file, sheet_name=sheet_name)

    output_csv = output_dir / args.output_file
    if not args.dry_run:
        output_df.to_csv(output_csv, index=False)

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_xlsx": str(input_file),
        "input_xlsx_name": input_file.name,
        "sheet_name": args.sheet_name,
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "output_csv": output_csv.name,
        "generated_rows": int(len(output_df)),
        "selected_subset_rows": int(output_df.attrs.get("selected_subset_rows", 0)),
        "positive_score_rows": int(output_df.attrs.get("positive_score_rows", 0)),
    }

    if not args.dry_run:
        with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {output_csv}")
    print(f"Rows: {len(output_df)}")


if __name__ == "__main__":
    main()