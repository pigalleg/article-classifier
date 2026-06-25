#!/usr/bin/env python3
"""Select (Abstract_Index, RA2025_ID) rows by calibration vs adjudicated level counts.

Usage examples:
  python scripts/select_cases_by_levels.py \
    --calib-dir data/results/affinity_calibration/20260618_Mark_batch1_v2 \
  --adjud-file data/results/affinity_benchmark/20260422_212121_1 \
  --adjud-file data/results/affinity_benchmark/20260422_212121_3 \
  --adjud-file data/results/affinity_benchmark/20260422_212121_4 \
  --pair "Calibration Moderate:Adjudicated High=8" \
  --pair "Calibration Low:Adjudicated High=7" \
  --pair "Calibration Low:Adjudicated Moderate=2" \
  --pair "Calibration Moderate:Adjudicated Low=2" \
  --pair "Calibration High:Adjudicated High=1" \
  --out outputs/selected_by_levels.csv
"""


import argparse
from pathlib import Path
import sys

import pandas as pd


LABEL_ORDER = ["Low", "Moderate", "High"]
LABEL_TO_NUM = {"Low": 1, "Moderate": 2, "High": 3}
NUM_TO_LABEL = {v: k for k, v in LABEL_TO_NUM.items()}


def read_csv_robust(path: Path) -> pd.DataFrame:
    # Some exported CSVs contain non-UTF-8 bytes (e.g. 0xA0 from cp1252/latin1).
    # Try a few common encodings so the script works across files generated on different systems.
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def normalize_level_from_score(score: float) -> str:
    # thresholds used elsewhere in the project (40/70)
    try:
        s = float(score)
    except Exception:
        return str(score)
    if s < 40.0:
        return "Low"
    if s < 70.0:
        return "Moderate"
    return "High"


def infer_calibration_level(df: pd.DataFrame) -> pd.Series:
    # prefer an explicit level column if present
    for col in ["Evaluator_Affinity_Level", "evaluator_affinity_level", "calibration_level"]:
        if col in df.columns:
            return df[col].astype(str)
    # otherwise map numeric `Evaluator_Affinity` to Low/Moderate/High
    if "Evaluator_Affinity" in df.columns:
        return df["Evaluator_Affinity"].apply(normalize_level_from_score)
    # fallback: try any affinity-like numeric column
    for col in df.columns:
        if "affinity" in col.lower() and df[col].dtype.kind in "fi":
            return df[col].apply(normalize_level_from_score)
    raise ValueError("Could not infer calibration affinity column in calibration file")


def infer_adjudicated_level(df: pd.DataFrame) -> pd.Series:
    for col in ["Affinity_Level", "Final_Affinity_Level", "benchmark_level"]:
        if col in df.columns:
            return df[col].astype(str)
    if "Final_Affinity" in df.columns:
        return df["Final_Affinity"].apply(normalize_level_from_score)
    for col in df.columns:
        if "final_affinity" in col.lower() or ("affinity" in col.lower() and df[col].dtype.kind in "fi"):
            return df[col].apply(normalize_level_from_score)
    raise ValueError("Could not infer adjudicated affinity column in adjudicated file")


def level_distance(calibration_level: str, adjudicated_level: str) -> int:
    calib = str(calibration_level).strip().capitalize()
    adjud = str(adjudicated_level).strip().capitalize()
    return abs(LABEL_TO_NUM[calib] - LABEL_TO_NUM[adjud])


def safe_numeric_series(df: pd.DataFrame, preferred_columns: list[str]) -> pd.Series:
    for col in preferred_columns:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def parse_pair_arg(s: str):
    # expected format: "Calibration Moderate:Adjudicated High=8"
    if "=" not in s or ":" not in s:
        raise argparse.ArgumentTypeError("pair must be of form 'Calibration X:Adjudicated Y=N'")
    left, right = s.split("=", 1)
    count = int(right)
    calib, adjud = left.split(":", 1)
    calib = calib.strip()
    adjud = adjud.strip()
    # accept forms like 'Calibration Moderate' or just 'Moderate'
    if calib.lower().startswith("calibration"):
        calib_label = calib.split(None, 1)[1].strip()
    else:
        calib_label = calib
    if adjud.lower().startswith("adjudicated"):
        adjud_label = adjud.split(None, 1)[1].strip()
    else:
        adjud_label = adjud
    return (calib_label, adjud_label, count)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--calib-dir",
        required=True,
        help="Directory containing calibration_abstract_question_affinities.csv",
    )
    p.add_argument(
        "--adjud-file",
        action="append",
        required=True,
        help=(
            "Adjudicated CSV file path or benchmark run directory. "
            "Repeat this option to combine multiple adjudication files."
        ),
    )
    p.add_argument(
        "--adjud-filename",
        default="adjudicated_ra_affinities.csv",
        help="Filename to read inside each adjudication run directory",
    )
    p.add_argument(
        "--pair",
        action="append",
        type=parse_pair_arg,
        help="Pair selection as 'Calibration Moderate:Adjudicated High=8'. Can be given multiple times",
    )
    p.add_argument(
        "--out",
        default="selected_samples.csv",
        help="Output CSV filename or path. Relative paths are saved inside --calib-dir.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    calib_dir = Path(args.calib_dir)
    calib_path = calib_dir / "calibration_abstract_question_affinities.csv"
    if not calib_path.exists():
        print(f"Calibration file not found: {calib_path}")
        sys.exit(2)

    calib_df = read_csv_robust(calib_path)

    adjud_paths = []
    for raw_path in args.adjud_file:
        path = Path(raw_path)
        if path.is_dir():
            path = path / args.adjud_filename
        if not path.exists():
            print(f"Adjudicated file not found: {path}")
            sys.exit(2)
        adjud_paths.append(path)

    adjud_frames = [read_csv_robust(path) for path in adjud_paths]
    adjud_df = pd.concat(adjud_frames, ignore_index=True)

    calib_level = infer_calibration_level(calib_df)
    calib_df = calib_df.copy()
    calib_df["calibration_level"] = calib_level

    adjud_level = infer_adjudicated_level(adjud_df)
    adjud_df = adjud_df.copy()
    adjud_df["adjudicated_level"] = adjud_level

    # ensure key columns exist
    key_cols = ["Abstract_Index", "RA2025_ID"]
    for k in key_cols:
        if k not in calib_df.columns or k not in adjud_df.columns:
            raise ValueError(f"Missing key column '{k}' in one of the files")

    merged = pd.merge(calib_df, adjud_df, on=key_cols, suffixes=("_calib", "_adj"))

    # normalize labels to canonical forms
    merged["calibration_level"] = merged["calibration_level"].apply(lambda s: str(s).strip()).replace({"moderate":"Moderate","low":"Low","high":"High"})
    merged["adjudicated_level"] = merged["adjudicated_level"].apply(lambda s: str(s).strip()).replace({"moderate":"Moderate","low":"Low","high":"High"})

    # build selection mapping
    mapping = {}
    if args.pair:
        for calib_label, adjud_label, count in args.pair:
            mapping[(calib_label, adjud_label)] = int(count)

    calibration_score = safe_numeric_series(merged, ["Evaluator_Affinity", "Evaluator_Affinity_calib"])
    adjudicated_score = safe_numeric_series(merged, ["Final_Affinity", "Final_Affinity_adj"])

    merged["level_gap"] = merged.apply(lambda row: level_distance(row["calibration_level"], row["adjudicated_level"]), axis=1)
    merged["score_gap"] = (calibration_score - adjudicated_score).abs()

    selected_frames = []
    for calib_label in LABEL_ORDER:
        for adj_label in LABEL_ORDER:
            n = mapping.get((calib_label, adj_label), 0)
            if n <= 0:
                continue
            candidates = merged[(merged["calibration_level"] == calib_label) & (merged["adjudicated_level"] == adj_label)]
            if candidates.empty:
                print(f"Warning: no candidates for {calib_label} -> {adj_label}")
                continue
            # Prioritize cases where calibration and adjudication diverge most,
            # so the requested sample size is spent on the most informative rows first.
            ranked = candidates.sort_values(
                by=["level_gap", "score_gap", "Abstract_Index", "RA2025_ID"],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            if len(ranked) > n:
                chosen = ranked.head(n)
            else:
                chosen = ranked
            selected_frames.append(chosen)

    if not selected_frames:
        print("No rows selected (empty mapping or no matches). Exiting.")
        sys.exit(0)

    result = pd.concat(selected_frames, ignore_index=True)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = calib_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} selected rows to {out_path}")


if __name__ == "__main__":
    main()
