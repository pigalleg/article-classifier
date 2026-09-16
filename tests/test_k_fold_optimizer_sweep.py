from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_k_fold_optimizer_sweep import (
    REPOSITORY_ROOT,
    child_command,
    expected_metadata,
    run_sweep,
    select_best_run,
    validate_fold_range,
    validate_solver,
    write_selection,
)
from scripts.util.aggregate_k_fold_optimizer_sweep import write_comparison_workbook


def _write_child_run(path: Path, fold_count: int, qwk: float) -> None:
    path.mkdir(parents=True)
    with pd.ExcelWriter(path / "optimization_report.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(
            {"Metric": ["Quadratic_Weighted_Kappa"], "Optimized": [qwk], "Equal_Weight": [0.2]}
        ).to_excel(writer, sheet_name="Overall Metrics", index=False)
        pd.DataFrame(
            {"Level": ["High"], "Optimized_Precision": [0.7]}
        ).to_excel(writer, sheet_name="Per-Level Metrics", index=False)
        pd.DataFrame(
            {
                "Grouping": ["Planning"],
                "Model_Slug": ["model-a"],
                "Fold_Count": [fold_count],
                "Mean_Weight": [1.0],
            }
        ).to_excel(writer, sheet_name="Fold Weight Ranges", index=False)
    pd.DataFrame({"Grouping": ["Planning"], "Model_Slug": ["model-a"], "Weight": [1.0]}).to_csv(
        path / "final_weights.csv", index=False
    )
    (path / "run_metadata.toml").write_text(
        f'benchmark_run = "benchmark"\nfold_count = {fold_count}\nfit_all = true\n', encoding="ascii"
    )


def test_validate_fold_range_is_inclusive_and_rejects_invalid_bounds() -> None:
    assert list(validate_fold_range(3, 5)) == [3, 4, 5]
    with pytest.raises(ValueError, match="at least 2"):
        validate_fold_range(1, 4)
    with pytest.raises(ValueError, match="greater than or equal"):
        validate_fold_range(5, 4)


def test_child_command_forces_fit_all_and_forwards_configuration(tmp_path: Path) -> None:
    args = Namespace(
        expert_run="expert",
        benchmark_run="benchmark",
        objective="weighted_mae",
        weight_scope="programme",
        calibration_scope="none",
        low_multiplier=1.0,
        moderate_multiplier=2.0,
        high_multiplier=3.0,
        solver="exact",
    )

    command = child_command(args, 4, tmp_path / "k4")

    assert command[0:2] == ["bash", str(REPOSITORY_ROOT / "optimization" / "run_optimizer.sh")]
    assert command[command.index("--folds") + 1] == "4"
    assert "--fit-all" in command
    assert command[command.index("--weight-scope") + 1] == "programme"
    assert command[command.index("--high-multiplier") + 1] == "3.0"
    assert command[command.index("--solver") + 1] == "exact"
    assert "--learning-rate" not in command


def test_child_command_forwards_gradient_hyperparameters(tmp_path: Path) -> None:
    args = Namespace(
        expert_run="expert",
        benchmark_run="benchmark",
        objective="soft_qwk",
        weight_scope="programme",
        calibration_scope="none",
        low_multiplier=1.0,
        moderate_multiplier=1.0,
        high_multiplier=1.0,
        solver="gradient",
        learning_rate=0.02,
        max_iterations=3000,
        patience=100,
        tolerance=1e-9,
        level_temperature=4.0,
        restarts=3,
    )

    command = child_command(args, 4, tmp_path / "k4")

    assert command[command.index("--objective") + 1] == "soft_qwk"
    assert command[command.index("--solver") + 1] == "gradient"
    assert command[command.index("--learning-rate") + 1] == "0.02"
    assert command[command.index("--level-temperature") + 1] == "4.0"
    assert command[command.index("--restarts") + 1] == "3"
    assert expected_metadata(args, 4)["restart_count"] == 3


def test_soft_kappa_objective_requires_the_gradient_solver() -> None:
    with pytest.raises(ValueError, match="requires --solver gradient"):
        validate_solver("soft_qwk", "exact")
    validate_solver("soft_qwk", "gradient")
    validate_solver("weighted_mae", "exact")


def test_selects_highest_oof_qwk_then_lowest_k_and_aggregates_reports_only(tmp_path: Path) -> None:
    _write_child_run(tmp_path / "k3", 3, 0.72)
    _write_child_run(tmp_path / "k4", 4, 0.72)
    _write_child_run(tmp_path / "k5", 5, 0.69)
    (tmp_path / "k3" / "out_of_fold_predictions.csv").write_text("not,a,valid,csv", encoding="ascii")

    selected_k, selected_qwk = select_best_run(tmp_path, range(3, 6))
    write_selection(tmp_path, selected_k, selected_qwk)
    workbook_path = write_comparison_workbook(tmp_path)

    assert (selected_k, selected_qwk) == (3, 0.72)
    assert (tmp_path / "selected_programme_model_weights.csv").read_text(encoding="ascii") == (
        tmp_path / "k3" / "final_weights.csv"
    ).read_text(encoding="ascii")
    assert pd.read_excel(workbook_path, sheet_name="Overall Metrics by K")["Fold_Count"].tolist() == [3, 4, 5]
    assert pd.read_excel(workbook_path, sheet_name="Fold Weight Stability by K")["Fold_Count"].tolist() == [3, 4, 5]
    selection = pd.read_excel(workbook_path, sheet_name="Selection")
    assert selection.loc[0, "selected_fold_count"] == 3
    assert selection.loc[0, "selection_scope"] == "pooled_out_of_fold_optimized"


def test_run_sweep_creates_layout_and_resumes_compatible_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = Namespace(
        expert_run="expert",
        benchmark_run="benchmark",
        k_min=3,
        k_max=4,
        objective="weighted_mae",
        weight_scope="programme",
        calibration_scope="none",
        low_multiplier=1.0,
        moderate_multiplier=2.0,
        high_multiplier=2.0,
        solver="exact",
        output_root=tmp_path,
        output_dir=tmp_path / "sweep",
        resume=False,
    )
    invocation_count = 0

    def fake_run(command: list[str], check: bool) -> None:
        nonlocal invocation_count
        invocation_count += 1
        fold_count = int(command[command.index("--folds") + 1])
        child_dir = Path(command[command.index("--output-dir") + 1])
        _write_child_run(child_dir, fold_count, 0.6 + fold_count / 100)
        metadata = expected_metadata(args, fold_count)
        child_dir.joinpath("run_metadata.toml").write_text(
            "\n".join(
                f'{key} = "{value}"'
                if isinstance(value, str)
                else f"{key} = {str(value).lower()}"
                for key, value in metadata.items()
            )
            + "\n",
            encoding="ascii",
        )

    monkeypatch.setattr("scripts.run_k_fold_optimizer_sweep.subprocess.run", fake_run)

    sweep_dir = run_sweep(args)

    assert invocation_count == 2
    assert (sweep_dir / "run_metadata.toml").is_file()
    assert (sweep_dir / "k3" / "optimization_report.xlsx").is_file()
    assert (sweep_dir / "k4" / "final_weights.csv").is_file()
    assert (sweep_dir / "k_fold_comparison.xlsx").is_file()
    assert (sweep_dir / "selected_programme_model_weights.csv").is_file()

    args.resume = True
    run_sweep(args)

    assert invocation_count == 2