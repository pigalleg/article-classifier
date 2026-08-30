"""Run independent optimizer fits across a range of cross-validation fold counts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from util.aggregate_k_fold_optimizer_sweep import write_comparison_workbook
except ModuleNotFoundError:
    from scripts.util.aggregate_k_fold_optimizer_sweep import write_comparison_workbook

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "results" / "ensemble_weight_optimizer"
SELECTION_METRIC = "Quadratic_Weighted_Kappa"


def validate_fold_range(k_min: int, k_max: int) -> range:
    if k_min < 2:
        raise ValueError("--k-min must be at least 2.")
    if k_max < k_min:
        raise ValueError("--k-max must be greater than or equal to --k-min.")
    return range(k_min, k_max + 1)


def build_sweep_directory(output_root: Path, benchmark_run: str, k_min: int, k_max: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_kfold_sweep_{benchmark_run}_k{k_min}-{k_max}"


def child_command(args: argparse.Namespace, fold_count: int, output_dir: Path) -> list[str]:
    command = [
        "bash",
        str(REPOSITORY_ROOT / "optimization" / "run_optimizer.sh"),
        "--expert-run",
        args.expert_run,
        "--benchmark-run",
        args.benchmark_run,
        "--objective",
        args.objective,
        "--weight-scope",
        args.weight_scope,
        "--calibration-scope",
        args.calibration_scope,
        "--low-multiplier",
        str(args.low_multiplier),
        "--moderate-multiplier",
        str(args.moderate_multiplier),
        "--high-multiplier",
        str(args.high_multiplier),
        "--folds",
        str(fold_count),
        "--fit-all",
        "--output-dir",
        str(output_dir),
    ]
    return command


def expected_metadata(args: argparse.Namespace, fold_count: int | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "expert_run": args.expert_run,
        "benchmark_run": args.benchmark_run,
        "objective": args.objective,
        "weight_scope": args.weight_scope,
        "calibration_scope": args.calibration_scope,
        "low_multiplier": args.low_multiplier,
        "moderate_multiplier": args.moderate_multiplier,
        "high_multiplier": args.high_multiplier,
    }
    if fold_count is None:
        metadata.update({"k_min": args.k_min, "k_max": args.k_max})
    else:
        metadata.update({"fold_count": fold_count, "fit_all": True})
    return metadata


def read_metadata(metadata_path: Path) -> dict[str, object]:
    with metadata_path.open("rb") as metadata_file:
        return tomllib.load(metadata_file)


def is_compatible_metadata(metadata_path: Path, expected: dict[str, object]) -> bool:
    if not metadata_path.is_file():
        return False
    metadata = read_metadata(metadata_path)
    return all(metadata.get(key) == value for key, value in expected.items())


def write_sweep_metadata(sweep_dir: Path, args: argparse.Namespace) -> None:
    metadata = expected_metadata(args)
    lines = [
        f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}"
        for key, value in metadata.items()
    ]
    lines.append('selection_metric = "Quadratic_Weighted_Kappa"')
    lines.append('selection_scope = "pooled_out_of_fold_optimized"')
    (sweep_dir / "run_metadata.toml").write_text("\n".join(lines) + "\n", encoding="ascii")


def completed_child_is_compatible(child_dir: Path, args: argparse.Namespace, fold_count: int) -> bool:
    required_paths = [
        child_dir / "optimization_report.xlsx",
        child_dir / "final_weights.csv",
        child_dir / "run_metadata.toml",
    ]
    if not child_dir.exists():
        return False
    if not all(path.is_file() for path in required_paths):
        raise FileNotFoundError(f"Cannot resume incomplete child run: {child_dir}.")
    if not is_compatible_metadata(child_dir / "run_metadata.toml", expected_metadata(args, fold_count)):
        raise ValueError(f"Cannot resume incompatible child run: {child_dir}.")
    return True


def read_optimized_oof_qwk(report_path: Path) -> float:
    metrics = pd.read_excel(report_path, sheet_name="Overall Metrics")
    matches = metrics.loc[
        metrics["Metric"].eq(SELECTION_METRIC), "Optimized"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{report_path} must have exactly one optimized {SELECTION_METRIC} value."
        )
    qwk = float(matches.iloc[0])
    if pd.isna(qwk):
        raise ValueError(f"{report_path} has a missing optimized {SELECTION_METRIC} value.")
    return qwk


def select_best_run(sweep_dir: Path, fold_counts: range) -> tuple[int, float]:
    candidates = [
        (fold_count, read_optimized_oof_qwk(sweep_dir / f"k{fold_count}" / "optimization_report.xlsx"))
        for fold_count in fold_counts
    ]
    return max(candidates, key=lambda candidate: (candidate[1], -candidate[0]))


def write_selection(sweep_dir: Path, selected_fold_count: int, selected_qwk: float) -> None:
    source_dir = sweep_dir / f"k{selected_fold_count}"
    final_weights = source_dir / "final_weights.csv"
    if not final_weights.is_file():
        raise FileNotFoundError(
            f"Selected run is missing deployable weights: {final_weights}."
        )
    shutil.copy2(final_weights, sweep_dir / "selected_programme_model_weights.csv")
    final_calibration = source_dir / "final_calibration.csv"
    if final_calibration.is_file():
        shutil.copy2(final_calibration, sweep_dir / "selected_programme_calibration.csv")
    (sweep_dir / "selection.toml").write_text(
        "selection_metric = \"Quadratic_Weighted_Kappa\"\n"
        "selection_scope = \"pooled_out_of_fold_optimized\"\n"
        f"selected_fold_count = {selected_fold_count}\n"
        f"selected_qwk = {selected_qwk}\n"
        "tie_breaker = \"lowest_fold_count\"\n"
        f"source_run = \"k{selected_fold_count}\"\n",
        encoding="ascii",
    )


def run_sweep(args: argparse.Namespace) -> Path:
    fold_counts = validate_fold_range(args.k_min, args.k_max)
    sweep_dir = args.output_dir or build_sweep_directory(
        args.output_root, args.benchmark_run, args.k_min, args.k_max
    )
    if args.resume:
        if args.output_dir is None:
            raise ValueError("--resume requires --output-dir.")
        if not sweep_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing sweep directory: {sweep_dir}.")
        if not is_compatible_metadata(sweep_dir / "run_metadata.toml", expected_metadata(args)):
            raise ValueError(f"Cannot resume incompatible sweep directory: {sweep_dir}.")
    else:
        sweep_dir.mkdir(parents=True, exist_ok=False)
        write_sweep_metadata(sweep_dir, args)
    for fold_count in fold_counts:
        child_dir = sweep_dir / f"k{fold_count}"
        if args.resume and completed_child_is_compatible(child_dir, args, fold_count):
            continue
        subprocess.run(child_command(args, fold_count, child_dir), check=True)
    selected_fold_count, selected_qwk = select_best_run(sweep_dir, fold_counts)
    write_selection(sweep_dir, selected_fold_count, selected_qwk)
    write_comparison_workbook(sweep_dir)
    return sweep_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-run", required=True)
    parser.add_argument("--benchmark-run", required=True)
    parser.add_argument("--k-min", type=int, required=True)
    parser.add_argument("--k-max", type=int, required=True)
    parser.add_argument("--objective", default="weighted_mae", choices=["weighted_mae", "weighted_mse"])
    parser.add_argument("--weight-scope", default="programme", choices=["global", "programme"])
    parser.add_argument("--calibration-scope", default="none", choices=["none", "programme"])
    parser.add_argument("--low-multiplier", type=float, default=1.0)
    parser.add_argument("--moderate-multiplier", type=float, default=2.0)
    parser.add_argument("--high-multiplier", type=float, default=2.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run_sweep(arguments))