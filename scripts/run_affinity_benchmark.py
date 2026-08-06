#!/usr/bin/env python3
"""Run affinity evaluation across multiple models with isolated outputs.

Features:
- Per-model output isolation under a run directory
- Continue-on-error behavior
- Resume/skip-existing behavior
- Model list override via AFFINITY_BENCHMARK_MODELS
- Automatic fallback PRP affinities generation: If PRP affinities are missing,
  the script falls back to: (1) global data/results/prp_affinities.csv if it exists,
  or (2) generates an empty PRP affinities file with all abstract-PRP pairs from
  the data files (useful for comprehensive evaluation without pre-computed scores)

Execution examples (from repo root):
- Run both PRP and RA stages per model:
    python scripts/run_affinity_benchmark.py --mode both --ra-retrieval-mode prp_only
- Run PRP stage only:
    python scripts/run_affinity_benchmark.py --mode prp
- Run RA stage only with automatic fallback (generates all abstract-PRP pairs if needed):
    python scripts/run_affinity_benchmark.py --mode ra --ra-retrieval-mode prp_only
- Resume an interrupted run id:
    python scripts/run_affinity_benchmark.py --run-id 20260331_120000 --resume
- Dry run (print planned models only):
    python scripts/run_affinity_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CFG_PATH = REPO_ROOT / "src" / "config" / "settings.yaml"
POSTPROCESSING_SKIP_ENV = "AFFINITY_SKIP_POSTPROCESSING"

try:
    from scripts.util.ollama_gpu import _describe_ollama_gpu_state
except ModuleNotFoundError:
    from util.ollama_gpu import _describe_ollama_gpu_state

try:
    from scripts.util.metadata import get_run_metadata, write_run_metadata_file, write_run_metadata
    from scripts.util.generate_prp_affinities import generate_empty_prp_affinities
except ModuleNotFoundError:
    from util.metadata import get_run_metadata, write_run_metadata_file, write_run_metadata
    from util.generate_prp_affinities import generate_empty_prp_affinities


def _load_settings(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    text = str(value).strip()
    return text if text else str(default)


def _slugify(value: str) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _resolve_processed_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / "data" / "processed" / path).resolve()


@dataclass
class ModelProfile:
    name: str
    backend: str
    model: str
    base_url: str | None
    api_key: str | None
    api_key_env: str | None
    requests_per_minute: int | None
    enabled: bool


def _resolve_profiles(settings: dict[str, Any]) -> tuple[list[ModelProfile], str | None]:
    llm_cfg = settings.get("models", {}).get("llm", {})
    bench_cfg = llm_cfg.get("benchmark", {})

    models_env_var = str(bench_cfg.get("models_env_var") or "AFFINITY_BENCHMARK_MODELS")
    backend_env_var = str(bench_cfg.get("backend_env_var") or "AFFINITY_BENCHMARK_BACKEND")

    forced_backend = os.getenv(backend_env_var)
    if forced_backend:
        forced_backend = forced_backend.strip().lower()

    local_profile = llm_cfg.get("local", {})
    cloud_profile = llm_cfg.get("cloud", {})

    configured: list[dict[str, Any]] = bench_cfg.get("models", []) or []
    configured_by_key: dict[str, dict[str, Any]] = {}
    for item in configured:
        name_key = str(item.get("name") or "").strip()
        model_key = str(item.get("model") or "").strip()
        if name_key:
            configured_by_key[name_key] = item
        if model_key:
            configured_by_key[model_key] = item

    # Optional override list: comma-separated model IDs.
    models_override_raw = os.getenv(models_env_var)
    if models_override_raw:
        override_models = [m.strip() for m in models_override_raw.split(",") if m.strip()]
        overridden: list[dict[str, Any]] = []
        for model_name in override_models:
            base_item = dict(configured_by_key.get(model_name, {}))
            base_item.setdefault("name", model_name)
            base_item.setdefault("model", model_name)
            base_item["enabled"] = True
            if forced_backend:
                base_item["backend"] = forced_backend
            overridden.append(base_item)
        configured = overridden

    profiles: list[ModelProfile] = []
    for item in configured:
        enabled = _as_bool(item.get("enabled"), True)
        if not enabled:
            continue

        backend = str(item.get("backend") or llm_cfg.get("default_backend") or "local").strip().lower()
        if forced_backend:
            backend = forced_backend
        active = cloud_profile if backend == "cloud" else local_profile

        model = str(item.get("model") or item.get("name") or active.get("model") or "").strip()
        if not model:
            continue

        name = str(item.get("name") or model).strip()

        base_url = item.get("base_url")
        if base_url is None:
            base_url = active.get("base_url")
        if isinstance(base_url, str):
            base_url = base_url.strip() or None

        api_key_env = item.get("api_key_env")
        if api_key_env is None:
            api_key_env = active.get("api_key_env")
        if api_key_env is not None:
            api_key_env = str(api_key_env).strip() or None

        api_key = item.get("api_key")
        if api_key is None and api_key_env:
            api_key = os.getenv(api_key_env)
        if api_key is None:
            api_key = active.get("api_key")
        if isinstance(api_key, str):
            api_key = api_key.strip() or None

        rpm = item.get("requests_per_minute")
        if rpm is None:
            rpm = active.get("requests_per_minute_default")
        rpm_val = _as_int(rpm, 0)
        if rpm_val <= 0:
            rpm_val = None

        profiles.append(
            ModelProfile(
                name=name,
                backend=backend,
                model=model,
                base_url=base_url,
                api_key=api_key,
                api_key_env=api_key_env,
                requests_per_minute=rpm_val,
                enabled=True,
            )
        )

    return profiles, forced_backend


def _expected_outputs(model_dir: Path, mode: str) -> list[Path]:
    outputs: list[Path] = []
    if mode in {"prp", "both"}:
        outputs.append(model_dir / "prp_affinities.csv")
    if mode in {"ra", "both"}:
        outputs.append(model_dir / "ra_affinities.csv")
    if mode in {"ra", "both", "prp"}:
        outputs.append(model_dir / "combined_affinities.xlsx")
    return outputs


def _is_completed(model_dir: Path, mode: str) -> bool:
    expected = _expected_outputs(model_dir, mode)
    return bool(expected) and all(_is_nonempty_file(p) for p in expected)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_utc",
        "model_name",
        "model",
        "backend",
        "ollama_gpu_state",
        "status",
        "mode",
        "ra_retrieval_mode",
        "git_commit",
        "git_branch",
        "output_dir",
        "duration_seconds",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    affinity_cfg = settings.get("runtime", {}).get("affinity", {})
    bench_cfg = affinity_cfg.get("benchmark", {})

    parser = argparse.ArgumentParser(description="Run multi-model affinity benchmark")
    parser.add_argument("--mode", choices=["both", "prp", "ra"], default=str(affinity_cfg.get("mode", "both")))
    parser.add_argument(
        "--ra-retrieval-mode",
        choices=["cosine", "prp_filter", "prp_only"],
        default=str(affinity_cfg.get("ra_retrieval_mode", "prp_only")),
    )
    parser.add_argument("--prp-top-n", type=int, default=_as_int(affinity_cfg.get("prp_top_n"), 6))
    parser.add_argument("--prp-min-affinity", type=float, default=affinity_cfg.get("prp_min_affinity", None))
    parser.add_argument(
        "--prp-membership-confidence-min",
        type=float,
        default=float(affinity_cfg.get("prp_membership_confidence_min", 10.0)),
    )
    parser.add_argument(
        "--prp-scope-fallback-mode",
        choices=["strict", "legacy_scores"],
        default=str(affinity_cfg.get("prp_scope_fallback_mode", "legacy_scores")),
    )
    parser.add_argument(
        "--abstracts-file",
        default=_as_str(
            os.getenv("AFFINITY_ABSTRACTS_FILE") or affinity_cfg.get("abstracts_file"),
            "abstracts_cleaned.csv",
        ),
        help="Cleaned abstracts input file name or path used by evaluation and fallback PRP generation",
    )
    parser.add_argument("--year-start", type=int, default=None, help="Inclusive lower bound for publication year filtering")
    parser.add_argument("--year-end", type=int, default=None, help="Inclusive upper bound for publication year filtering")
    parser.add_argument(
        "--journals",
        default=None,
        help="Comma/semicolon separated list of journal values to keep (e.g., 'TPWRS;TPWRD')",
    )
    parser.add_argument(
        "--enable-affinity-reasons",
        dest="enable_affinity_reasons",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override whether affinity prompts must return a reason per target ID for all stage calls. "
            "Use --enable-affinity-reasons or --no-enable-affinity-reasons."
        ),
    )
    parser.add_argument(
        "--enable-few-shot",
        dest="enable_few_shot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override few-shot prompting for all stage calls. "
            "Use --enable-few-shot or --no-enable-few-shot."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(bench_cfg.get("output_root", "data/results/affinity_benchmark")),
        help="Root directory for benchmark outputs",
    )
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    parser.set_defaults(continue_on_error=_as_bool(bench_cfg.get("continue_on_error"), True))
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=_as_bool(bench_cfg.get("resume"), True))
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit")
    return parser.parse_args()


def _run_stage(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _run_postprocessing(run_dir: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_postprocessing_affinity_benchmark.py"),
        "--input-root",
        str(run_dir.parent),
        "--run-id",
        run_dir.name,
    ]
    _run_stage(command, dict(os.environ))


def _append_reasoner_overrides(cmd: list[str], args: argparse.Namespace) -> None:
    if args.enable_few_shot is not None:
        cmd.append("--enable-few-shot" if args.enable_few_shot else "--no-enable-few-shot")
    if args.enable_affinity_reasons is not None:
        cmd.append(
            "--enable-affinity-reasons"
            if args.enable_affinity_reasons
            else "--no-enable-affinity-reasons"
        )


def _append_subset_filters(cmd: list[str], args: argparse.Namespace) -> None:
    if args.year_start is not None:
        cmd.extend(["--year-start", str(args.year_start)])
    if args.year_end is not None:
        cmd.extend(["--year-end", str(args.year_end)])
    if args.journals:
        cmd.extend(["--journals", str(args.journals)])


def _build_env(profile: ModelProfile, model_output_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONBREAKPOINT"] = "0"
    env["LLM_BACKEND"] = profile.backend
    env["OPENAI_MODEL"] = profile.model
    env["AFFINITY_RESULTS_DIR"] = str(model_output_dir)
    env["AFFINITY_PRP_SCOPE_FAILURE_LOG"] = str(model_output_dir / "prp_scope_failures.jsonl")

    if profile.base_url:
        env["OPENAI_BASE_URL"] = profile.base_url
    else:
        env.pop("OPENAI_BASE_URL", None)

    if profile.api_key:
        # Set model-specific api_key_env variable (e.g., GOOGLE_API_KEY for Gemini)
        if profile.api_key_env:
            env[profile.api_key_env] = profile.api_key
            # Tell child process which env var to use for the API key
            env["OPENAI_API_KEY_ENV_VAR"] = profile.api_key_env
        # Also set generic OPENAI_API_KEY as fallback for compatibility
        env["OPENAI_API_KEY"] = profile.api_key

    if profile.requests_per_minute is not None:
        env["OPENAI_REQUESTS_PER_MINUTE"] = str(profile.requests_per_minute)

    return env


def main() -> None:
    settings = _load_settings(CFG_PATH)
    args = parse_args(settings)
    profiles, _ = _resolve_profiles(settings)

    if not profiles:
        raise RuntimeError("No enabled benchmark models found. Check settings.models.llm.benchmark.models.")

    output_root = (REPO_ROOT / args.output_root).resolve()
    run_dir = output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "benchmark_manifest.csv"

    # Record run metadata (git + settings) for reproducibility
    try:
        run_meta = get_run_metadata(repo_root=REPO_ROOT, settings_path=CFG_PATH)
        run_meta["benchmark"] = {
            "mode": args.mode,
            "ra_retrieval_mode": args.ra_retrieval_mode,
            "abstracts_file": args.abstracts_file,
            "subset_filter": {
                "year_start": args.year_start,
                "year_end": args.year_end,
                "journals": args.journals,
            },
        }
        write_run_metadata_file(run_dir / "run_metadata.json", run_meta)
    except Exception:
        run_meta = {"git": {"commit": None, "commit_short": None, "branch": None}}

    print(f"Benchmark run directory: {run_dir}")
    print(f"Models to run: {len(profiles)}")

    if args.dry_run:
        for p in profiles:
            print(f"- {p.name} ({p.model}) backend={p.backend}")
        return

    manifest_rows: list[dict[str, Any]] = []

    for profile in profiles:
        model_slug = _slugify(profile.name)
        model_dir = run_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        started = datetime.now(timezone.utc)
        row = {
            "timestamp_utc": started.isoformat().replace("+00:00", "Z"),
            "model_name": profile.name,
            "model": profile.model,
            "backend": profile.backend,
            "ollama_gpu_state": _describe_ollama_gpu_state(),
            "status": "success",
            "mode": args.mode,
            "ra_retrieval_mode": args.ra_retrieval_mode,
            "output_dir": str(model_dir),
            "duration_seconds": "",
            "error": "",
        }
        # Include git metadata in manifest rows for traceability
        row["git_commit"] = run_meta.get("git", {}).get("commit_short") if run_meta else None
        row["git_branch"] = run_meta.get("git", {}).get("branch") if run_meta else None

        if args.resume and _is_completed(model_dir, args.mode):
            row["status"] = "skipped"
            row["duration_seconds"] = "0"
            manifest_rows.append(row)
            _write_manifest(manifest_path, manifest_rows)
            print(f"[SKIP] {profile.name} (existing outputs)")
            continue

        env = _build_env(profile, model_dir)
        env["AFFINITY_ABSTRACTS_FILE"] = args.abstracts_file
        env["AFFINITY_ABSTRACT_LOOKUP_DIR"] = str(run_dir)

        try:
            if args.mode in {"prp", "both"}:
                cmd = [
                    sys.executable,
                    "scripts/run_affinity_evaluation.py",
                    "--mode",
                    "prp",
                    "--prp-membership-confidence-min",
                    str(args.prp_membership_confidence_min),
                    "--prp-scope-fallback-mode",
                    args.prp_scope_fallback_mode,
                ]
                _append_reasoner_overrides(cmd, args)
                _append_subset_filters(cmd, args)
                _run_stage(cmd, env)

            if args.mode in {"ra", "both"}:
                prp_input = model_dir / "prp_affinities.csv"
                if args.ra_retrieval_mode in {"prp_filter", "prp_only"} and not _is_nonempty_file(prp_input):
                    # Generate empty PRP affinities file with all abstract-PRP pairs
                    print(f"[INFO] Generating empty PRP affinities with all abstract-PRP pairs...")
                    abstracts_file = _resolve_processed_path(args.abstracts_file)
                    prp_file = REPO_ROOT / "data" / "processed" / "primary_programmes.csv"
                    
                    if not _is_nonempty_file(abstracts_file) or not _is_nonempty_file(prp_file):
                        raise RuntimeError(
                            f"Cannot generate fallback PRP affinities. Missing:\n"
                            f"  - Abstracts: {abstracts_file}\n"
                            f"  - PRP domains: {prp_file}\n"
                            f"Either provide {prp_input} or ensure data files exist."
                        )
                    
                    generate_empty_prp_affinities(prp_input, abstracts_file, prp_file)
                    print(f"[INFO] Generated fallback PRP affinities: {prp_input}")
        
                cmd = [
                    sys.executable,
                    "scripts/run_affinity_evaluation.py",
                    "--mode",
                    "ra",
                    "--ra-retrieval-mode",
                    args.ra_retrieval_mode,
                    "--prp-input",
                    str(prp_input),
                    "--prp-top-n",
                    str(args.prp_top_n),
                ]
                if args.prp_min_affinity is not None:
                    cmd.extend(["--prp-min-affinity", str(args.prp_min_affinity)])
                _append_reasoner_overrides(cmd, args)
                _append_subset_filters(cmd, args)

                _run_stage(cmd, env)

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            if not args.continue_on_error:
                finished = datetime.now(timezone.utc)
                row["duration_seconds"] = str(round((finished - started).total_seconds(), 6))
                manifest_rows.append(row)
                _write_manifest(manifest_path, manifest_rows)
                raise

        finished = datetime.now(timezone.utc)
        row["duration_seconds"] = str(round((finished - started).total_seconds(), 6))
        manifest_rows.append(row)
        _write_manifest(manifest_path, manifest_rows)
        print(f"[{row['status'].upper()}] {profile.name} -> {model_dir}")

    print(f"Saved benchmark manifest -> {manifest_path}")
    if not _as_bool(os.getenv(POSTPROCESSING_SKIP_ENV)):
        _run_postprocessing(run_dir)


if __name__ == "__main__":
    main()
