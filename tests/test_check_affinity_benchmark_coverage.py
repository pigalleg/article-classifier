import csv

import pandas as pd

from scripts.util.check_affinity_benchmark_coverage import check_run_coverage


def _write_manifest(run_dir, model_slugs, filename="benchmark_manifest.csv", windows_paths=False):
    with (run_dir / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "mode", "output_dir"])
        writer.writeheader()
        for slug in model_slugs:
            output_dir = str(run_dir / slug)
            if windows_paths:
                output_dir = f"C:\\results\\_staging\\group\\{slug}"
            writer.writerow({"status": "success", "mode": "ra", "output_dir": output_dir})


def _write_ra_rows(run_dir, slug, rows):
    model_dir = run_dir / slug
    model_dir.mkdir()
    pd.DataFrame(rows).to_csv(model_dir / "ra_affinities.csv", index=False)


def _rows(pair_values):
    return [
        {"Abstract_Index": abstract_index, "RA2025_ID": question_id, "LLM_Affinity": 50}
        for abstract_index, question_id in pair_values
    ]


def test_check_run_coverage_accepts_complete_models(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, ["model-a", "model-b"])
    pd.DataFrame({"Abstract_Index": [1, 2]}).to_csv(run_dir / "abstract_lookup_cleaned.csv", index=False)
    pairs = _rows([(1, 10), (2, 20)])
    _write_ra_rows(run_dir, "model-a", pairs)
    _write_ra_rows(run_dir, "model-b", pairs)

    assert check_run_coverage(run_dir) == []


def test_check_run_coverage_reports_missing_abstract_and_pair(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, ["model-a", "model-b"])
    pd.DataFrame({"Abstract_Index": [1, 2]}).to_csv(run_dir / "abstract_lookup_cleaned.csv", index=False)
    _write_ra_rows(run_dir, "model-a", _rows([(1, 10), (2, 20)]))
    _write_ra_rows(run_dir, "model-b", _rows([(1, 10)]))

    errors = check_run_coverage(run_dir)

    assert any("model-b (ra): missing 1 abstracts" in error for error in errors)
    assert any("model-b (ra): missing 1 abstract-target pairs" in error for error in errors)


def test_check_run_coverage_reports_duplicate_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, ["model-a"])
    _write_ra_rows(run_dir, "model-a", _rows([(1, 10), (1, 10)]))

    errors = check_run_coverage(run_dir)

    assert any("has 1 duplicate abstract-target rows" in error for error in errors)


def test_check_run_coverage_reports_missing_affinity_scores(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, ["model-a"])
    rows = _rows([(1, 10), (2, 20)])
    rows[1]["LLM_Affinity"] = "not returned"
    _write_ra_rows(run_dir, "model-a", rows)

    errors = check_run_coverage(run_dir)

    assert any("has 1 missing or nonnumeric affinity scores" in error for error in errors)


def test_check_run_coverage_reads_parallel_manifest_files(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(
        run_dir,
        ["model-a"],
        filename="cloud_benchmark_manifest.csv",
        windows_paths=True,
    )
    _write_manifest(run_dir, ["model-b"], filename="local_benchmark_manifest.csv")
    pairs = _rows([(1, 10)])
    _write_ra_rows(run_dir, "model-a", pairs)
    _write_ra_rows(run_dir, "model-b", pairs)

    assert check_run_coverage(run_dir) == []