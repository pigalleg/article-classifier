#!/usr/bin/env python3
"""Adjudicate divergent PRP and/or RA affinities across model outputs.

Workflow:
- Load merged benchmark PRP/RA affinities (typically merged_prp_affinities.csv and merged_ra_affinities.csv).
- Select divergent (Abstract_Index, Target_ID) pairs.
- Adjudicate selected pairs with LLMAffinityAdjudicatorReasoner.
- Use arithmetic mean for non-selected pairs.
- Save pair-level adjudicated outputs and model-row outputs with final scores for each selected mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CFG_PATH = REPO_ROOT / "src" / "config" / "settings.yaml"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "results" / "affinity_benchmark"


def _load_settings(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _normalize_abstract_index(value: Any) -> str:
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def _normalize_target_id(value: Any) -> str:
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def normalize_abstract_index(value: Any) -> str:
    return _normalize_abstract_index(value)


def normalize_target_id(value: Any) -> str:
    return _normalize_target_id(value)


def _subset_first_n_abstracts(df: pd.DataFrame, first_n: Optional[int]) -> pd.DataFrame:
    if first_n is None:
        return df
    if int(first_n) <= 0:
        return df.iloc[0:0].copy()

    abs_norm = df["Abstract_Index"].apply(_normalize_abstract_index)
    first_ids = abs_norm.drop_duplicates().head(int(first_n)).tolist()
    keep = abs_norm.isin(first_ids)
    return df[keep].copy()


def _discover_run_dir(input_root: Path, run_id: str | None) -> Path:
    if run_id:
        run_dir = input_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    candidates = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {input_root}")
    return candidates[-1]


def _load_abstract_lookup(settings: dict[str, Any]) -> dict[str, str]:
    paths_cfg = settings.get("paths", {})
    runtime_affinity = settings.get("runtime", {}).get("affinity", {})

    data_processed = paths_cfg.get("data_processed", "data/processed")
    abstracts_file = runtime_affinity.get("abstracts_file", "abstracts_cleaned.csv")

    abs_path = (REPO_ROOT / data_processed / abstracts_file).resolve()
    if not abs_path.exists():
        return {}

    try:
        df = pd.read_csv(abs_path)
    except Exception:
        return {}

    text_col = "Abstract_Cleaned" if "Abstract_Cleaned" in df.columns else None
    if text_col is None:
        text_col = "Abstract" if "Abstract" in df.columns else None
    if text_col is None:
        return {}

    if "Source_Index" in df.columns:
        idx_col = "Source_Index"
    elif "Abstract_Index" in df.columns:
        idx_col = "Abstract_Index"
    else:
        idx_col = None

    lookup: dict[str, str] = {}
    if idx_col is None:
        for i, row in df.iterrows():
            k = str(i + 1)
            txt = str(row.get(text_col, "")).strip()
            if txt:
                lookup[k] = txt
        return lookup

    for _, row in df.iterrows():
        k = _normalize_abstract_index(row.get(idx_col))
        txt = str(row.get(text_col, "")).strip()
        if k and txt:
            lookup[k] = txt
    return lookup


def _load_runtime_llm_config(settings: dict[str, Any]) -> tuple[str, int, Optional[str], Optional[str]]:
    llm_cfg = settings.get("models", {}).get("llm", {})

    backend_env_var = str(llm_cfg.get("backend_env_var") or "LLM_BACKEND")
    model_env_var = llm_cfg.get("model_env_var")
    base_url_env_var = llm_cfg.get("base_url_env_var")
    api_key_env_var = llm_cfg.get("api_key_env_var")
    rpm_env_var = llm_cfg.get("rpm_env_var")

    selected_backend = str(os.getenv(backend_env_var) or llm_cfg.get("default_backend") or "local").strip().lower()
    profile = llm_cfg.get("cloud", {}) if selected_backend == "cloud" else llm_cfg.get("local", {})

    effective_model = (os.getenv(str(model_env_var)) if model_env_var else None) or profile.get("model") or llm_cfg.get("model")
    if not effective_model:
        raise ValueError("Missing model configuration for adjudicator")

    effective_base_url = (os.getenv(str(base_url_env_var)) if base_url_env_var else None) or profile.get("base_url")
    if isinstance(effective_base_url, str):
        effective_base_url = effective_base_url.strip() or None

    profile_api_key_env = profile.get("api_key_env") or profile.get("api_key_env_var")
    api_key_source = profile_api_key_env or api_key_env_var
    
    # Check if benchmark script passed a model-specific API key env var name
    # (e.g., GOOGLE_API_KEY for Gemini instead of OPENAI_CLOUD_API_KEY for OpenAI)
    override_api_key_env_var = os.getenv("OPENAI_API_KEY_ENV_VAR")
    if override_api_key_env_var:
        api_key_source = override_api_key_env_var
    
    effective_api_key = (os.getenv(str(api_key_source)) if api_key_source else None) or profile.get("api_key")

    rpm_default = profile.get("requests_per_minute_default", 1)
    effective_rpm = (os.getenv(str(rpm_env_var)) if rpm_env_var else None) or rpm_default
    effective_rpm = _as_int(effective_rpm, 1)

    return str(effective_model), effective_rpm, effective_base_url, effective_api_key

def _affinity_level_fixed(score):
        if pd.isna(score):
            return pd.NA
        try:
            x = float(score)
        except Exception:
            return pd.NA
        if 0 <= x <= 40:
            return 'Low'
        if 40 < x <= 70:
            return 'Moderate'
        if 70 < x <= 100:
            return 'High'
        return pd.NA

def _compute_pair_stats(merged_df: pd.DataFrame, target_id_col: str, target_text_col: str) -> pd.DataFrame:
    base_cols = [
        "Abstract_Index",
        "Abstract_Index_Norm",
        "Document Title",
        target_id_col,
        target_text_col,
    ]
    # merged_df['affinity_level'] = merged_df['LLM_Affinity'].apply(affinity_level_fixed)
    grouped = (
        merged_df.groupby(base_cols, dropna=False, as_index=False)
        .agg(
            n_models=("Model_Slug", "nunique"),
            affinity_mean=("LLM_Affinity", "mean"),
            affinity_std=("LLM_Affinity", "std"),
            affinity_min=("LLM_Affinity", "min"),
            affinity_max=("LLM_Affinity", "max"),
            n_affinity_levels=("affinity_level", "nunique"),
        )
    )
    
    grouped["affinity_std"] = grouped["affinity_std"].fillna(0.0)
    grouped["affinity_range"] = grouped["affinity_max"] - grouped["affinity_min"]

    # Compute per-level model agreement once, then keep only the maximum ratio per pair.
    df_per_affinity_level = (
        merged_df.groupby(base_cols + ["affinity_level"], dropna=False)["Model_Slug"]
        .nunique()
        .reset_index(name="n_models_per_affinity")
    )
    n_models_by_pair = merged_df.groupby(base_cols, dropna=False)["Model_Slug"].nunique()
    pair_keys = pd.MultiIndex.from_frame(df_per_affinity_level[base_cols])
    df_per_affinity_level["n_models"] = pair_keys.map(n_models_by_pair.to_dict())
    df_per_affinity_level["agreement_ratio"] = (
        df_per_affinity_level["n_models_per_affinity"] / df_per_affinity_level["n_models"]
    )
    agreement_stats = (
        df_per_affinity_level.groupby(base_cols, dropna=False, as_index=False)["agreement_ratio"]
        .agg(
            agreement_mean="mean",
            agreement_std="std",
            agreement_min="min",
            agreement_max="max",
        )
    )
    agreement_stats["agreement_std"] = agreement_stats["agreement_std"].fillna(0.0)
    grouped = grouped.merge(agreement_stats, on=base_cols, how="left")

    return grouped


def compute_pair_stats(merged_df: pd.DataFrame, target_id_col: str, target_text_col: str) -> pd.DataFrame:
    return _compute_pair_stats(merged_df, target_id_col=target_id_col, target_text_col=target_text_col)

def affinity_level_fixed(score):
    return _affinity_level_fixed(score)


def _select_divergent_pairs(
    pair_stats: pd.DataFrame,
    min_agreement: float,
    min_agreement_mean: float,
) -> pd.DataFrame:
    out = pair_stats.copy()
    out["needs_adjudication"] = (
        (out["agreement_min"] < float(min_agreement))
        | (out["agreement_mean"] < float(min_agreement_mean))
    )

    return out


def _build_adjudication_targets(rows: pd.DataFrame) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    has_reason = "LLM_Affinity_Reason" in rows.columns

    for prp_name, g in rows.groupby("PRP_Name", sort=False):
        model_evidence: list[dict[str, Any]] = []
        for _, r in g.iterrows():
            ev = {
                "model_slug": r.get("Model_Slug"),
                "llm_affinity": float(r.get("LLM_Affinity")),
            }
            if has_reason:
                reason = r.get("LLM_Affinity_Reason")
                if isinstance(reason, str) and reason.strip():
                    ev["llm_affinity_reason"] = reason.strip()
            model_evidence.append(ev)

        targets.append(
            {
                "id": str(prp_name),
                "text": str(g["PRP_Description"].iloc[0]) if "PRP_Description" in g.columns else str(prp_name),
                "model_evidence": model_evidence,
            }
        )
    return targets


def _build_adjudication_targets_generic(
    rows: pd.DataFrame,
    target_id_col: str,
    target_text_col: str,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    has_reason = "LLM_Affinity_Reason" in rows.columns

    for target_id, g in rows.groupby(target_id_col, sort=False):
        model_evidence: list[dict[str, Any]] = []
        for _, r in g.iterrows():
            ev = {
                "model_slug": r.get("Model_Slug"),
                "llm_affinity": float(r.get("LLM_Affinity")),
            }
            if has_reason:
                reason = r.get("LLM_Affinity_Reason")
                if isinstance(reason, str) and reason.strip():
                    ev["llm_affinity_reason"] = reason.strip()
            model_evidence.append(ev)

        target_text = str(g[target_text_col].iloc[0]) if target_text_col in g.columns else str(target_id)
        targets.append(
            {
                "id": _normalize_target_id(target_id),
                "text": target_text,
                "model_evidence": model_evidence,
            }
        )
    return targets


def _get_reasoner_model_slug(reasoner: Any) -> Optional[str]:
    model = getattr(reasoner, "model", None)
    if model is None:
        return None
    model_slug = str(model).strip()
    return model_slug or None


def _run_adjudication_for_selected(
    merged_prp: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    abstract_lookup: dict[str, str],
    reasoner: Any,
) -> dict[tuple[str, str], dict[str, Any]]:
    selected = selected_pairs[selected_pairs["needs_adjudication"]].copy()
    if selected.empty:
        return {}

    selected_keys = set(zip(selected["Abstract_Index_Norm"].astype(str), selected["PRP_Name"].astype(str)))
    selected_rows = merged_prp[
        merged_prp.apply(
            lambda r: (str(r["Abstract_Index_Norm"]), str(r["PRP_Name"])) in selected_keys,
            axis=1,
        )
    ]

    adjudicated: dict[tuple[str, str], dict[str, Any]] = {}
    grouped_abstracts = list(selected_rows.groupby("Abstract_Index_Norm", sort=False))
    adjudicator_model_slug = _get_reasoner_model_slug(reasoner)
    for abs_norm, abs_group in tqdm(
        grouped_abstracts,
        total=len(grouped_abstracts),
        desc="Adjudicating PRP by abstract",
    ):
        abstract_text = abstract_lookup.get(str(abs_norm), "")
        if not abstract_text:
            abstract_text = str(abs_group["Document Title"].iloc[0])

        targets = _build_adjudication_targets(abs_group)
        out = reasoner.adjudicate_batch(abstract=abstract_text, targets=targets, target_type="PRP")

        for target_id, value in out.items():
            key = (str(abs_norm), str(target_id))
            if isinstance(value, dict):
                adjudicated[key] = {
                    "final_affinity": float(value.get("score", 0.0)),
                    "final_reason": value.get("reason"),
                    "final_method": "adjudicated_llm",
                    "model_slug": adjudicator_model_slug,
                }
            else:
                adjudicated[key] = {
                    "final_affinity": float(value),
                    "final_reason": None,
                    "final_method": "adjudicated_llm",
                    "model_slug": adjudicator_model_slug,
                }

    return adjudicated


def _run_adjudication_for_selected_generic(
    merged_df: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    abstract_lookup: dict[str, str],
    reasoner: Any,
    target_id_col: str,
    target_text_col: str,
    target_type: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    selected = selected_pairs[selected_pairs["needs_adjudication"]].copy()
    if selected.empty:
        return {}

    selected_keys = selected[["Abstract_Index_Norm", "Target_ID_Norm"]].drop_duplicates()
    selected_rows = merged_df.merge(selected_keys, on=["Abstract_Index_Norm", "Target_ID_Norm"], how="inner")

    adjudicated: dict[tuple[str, str], dict[str, Any]] = {}
    grouped_abstracts = list(selected_rows.groupby("Abstract_Index_Norm", sort=False))
    adjudicator_model_slug = _get_reasoner_model_slug(reasoner)
    for abs_norm, abs_group in tqdm(
        grouped_abstracts,
        total=len(grouped_abstracts),
        desc=f"Adjudicating {target_type.upper()} by abstract",
    ):
        abstract_text = abstract_lookup.get(str(abs_norm), "")
        if not abstract_text:
            abstract_text = str(abs_group["Document Title"].iloc[0])

        targets = _build_adjudication_targets_generic(
            rows=abs_group,
            target_id_col=target_id_col,
            target_text_col=target_text_col,
        )
        out = reasoner.adjudicate_batch(abstract=abstract_text, targets=targets, target_type=target_type)

        for target_id, value in out.items():
            key = (str(abs_norm), _normalize_target_id(target_id))
            if isinstance(value, dict):
                adjudicated[key] = {
                    "final_affinity": float(value.get("score", 0.0)),
                    "final_reason": value.get("reason"),
                    "final_method": "adjudicated_llm",
                    "model_slug": adjudicator_model_slug,
                }
            else:
                adjudicated[key] = {
                    "final_affinity": float(value),
                    "final_reason": None,
                    "final_method": "adjudicated_llm",
                    "model_slug": adjudicator_model_slug,
                }

    return adjudicated


def run_prp_adjudication(
    merged_prp: pd.DataFrame,
    abstract_lookup: dict[str, str],
    reasoner: Optional[Any],
    min_agreement: float,
    min_agreement_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = merged_prp.copy()
    required = {"Abstract_Index", "PRP_Name", "LLM_Affinity", "Model_Slug"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Merged PRP input is missing required columns: {missing}")

    if "Document Title" not in df.columns:
        df["Document Title"] = ""
    if "PRP_Description" not in df.columns:
        df["PRP_Description"] = df["PRP_Name"].astype(str)

    df["LLM_Affinity"] = pd.to_numeric(df["LLM_Affinity"], errors="coerce")
    df = df[df["LLM_Affinity"].notna()].copy()
    df["Abstract_Index_Norm"] = df["Abstract_Index"].apply(_normalize_abstract_index)
    df["Target_ID_Norm"] = df["PRP_Name"].apply(_normalize_target_id)
    df["affinity_level"] = df["LLM_Affinity"].apply(_affinity_level_fixed)
    
    pair_stats = _compute_pair_stats(df, target_id_col="PRP_Name", target_text_col="PRP_Description")
    pair_stats["Target_ID_Norm"] = pair_stats["PRP_Name"].apply(_normalize_target_id)
    pair_stats = _select_divergent_pairs(
        pair_stats,
        min_agreement=min_agreement,
        min_agreement_mean=min_agreement_mean,
    )
    adjudicated_map: dict[tuple[str, str], dict[str, Any]] = {}
    selected_count = int(pair_stats["needs_adjudication"].sum())
    print(f"Selected {selected_count} divergent pairs for adjudication")
    if selected_count > 0:
        if reasoner is None:
            raise ValueError("Reasoner is required when selected divergent pairs are present")

        adjudicated_map = _run_adjudication_for_selected(df, pair_stats, abstract_lookup, reasoner)

    def _resolve_pair(row: pd.Series) -> tuple[float, Optional[str], str, Any]:
        key = (str(row["Abstract_Index_Norm"]), str(row["Target_ID_Norm"]))
        adjudicated = adjudicated_map.get(key)
        if adjudicated is not None:
            return (
                float(adjudicated["final_affinity"]),
                adjudicated.get("final_reason"),
                str(adjudicated.get("final_method") or "adjudicated_llm"),
                adjudicated.get("model_slug"),
            )
        return (float(row["affinity_mean"]), None, "ensemble_mean", pd.NA)

    resolved = pair_stats.apply(_resolve_pair, axis=1, result_type="expand")
    resolved.columns = ["final_affinity", "final_reason", "final_method", "adjudication_model_slug"]
    pair_out = pd.concat([pair_stats, resolved], axis=1)

    pair_out = pair_out.rename(
        columns={
            "n_models": "N_Models",
            "affinity_mean": "Affinity_Mean",
            "affinity_std": "Affinity_Std",
            "affinity_min": "Affinity_Min",
            "affinity_max": "Affinity_Max",
            "affinity_range": "Affinity_Range",
            "final_affinity": "Final_Affinity",
            "final_reason": "Final_Reason",
            "final_method": "Final_Method",
            "adjudication_model_slug": "Model_Slug",
        }
    )

    pair_cols = [
        "Abstract_Index",
        "Document Title",
        "PRP_Name",
        "PRP_Description",
        "N_Models",
        "Affinity_Mean",
        "Affinity_Std",
        "Affinity_Min",
        "Affinity_Max",
        "Affinity_Range",
        "agreement_mean",
        "agreement_std",
        "agreement_min",
        "agreement_max",
        "Final_Affinity",
        "Final_Reason",
        "Final_Method",
        "Model_Slug",
    ]
    extra_cols = [c for c in ["Run_ID"] if c in df.columns and c not in pair_cols]
    if extra_cols:
        run_ids = df.groupby(["Abstract_Index_Norm", "Target_ID_Norm"], as_index=False)[extra_cols].first()
        pair_out = pair_out.merge(run_ids, on=["Abstract_Index_Norm", "Target_ID_Norm"], how="left")
        pair_cols.extend(extra_cols)

    join_cols = ["Abstract_Index_Norm", "Target_ID_Norm", "Final_Affinity", "Final_Method"]
    if "Final_Reason" in pair_out.columns:
        join_cols.append("Final_Reason")
    pair_join = pair_out[join_cols].copy()

    pair_out = pair_out.sort_values(["Abstract_Index_Norm", "PRP_Name"], kind="stable")[pair_cols]
    model_rows_out = df.merge(pair_join, on=["Abstract_Index_Norm", "Target_ID_Norm"], how="left")
    model_rows_out = model_rows_out.drop(columns=["Target_ID_Norm"], errors="ignore")

    return pair_out, model_rows_out


def run_ra_adjudication(
    merged_ra: pd.DataFrame,
    abstract_lookup: dict[str, str],
    reasoner: Optional[Any],
    min_agreement: float,
    min_agreement_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = merged_ra.copy()
    required = {"Abstract_Index", "RA2025_ID", "LLM_Affinity", "Model_Slug"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Merged RA input is missing required columns: {missing}")

    if "Document Title" not in df.columns:
        df["Document Title"] = ""
    if "RA_Question" not in df.columns:
        df["RA_Question"] = df["RA2025_ID"].astype(str)

    df["LLM_Affinity"] = pd.to_numeric(df["LLM_Affinity"], errors="coerce")
    df = df[df["LLM_Affinity"].notna()].copy()
    df["Abstract_Index_Norm"] = df["Abstract_Index"].apply(_normalize_abstract_index)
    df["Target_ID_Norm"] = df["RA2025_ID"].apply(_normalize_target_id)
    df["affinity_level"] = df["LLM_Affinity"].apply(_affinity_level_fixed)

    pair_stats = _compute_pair_stats(df, target_id_col="RA2025_ID", target_text_col="RA_Question")
    pair_stats["Target_ID_Norm"] = pair_stats["RA2025_ID"].apply(_normalize_target_id)
    pair_stats = _select_divergent_pairs(
        pair_stats,
        min_agreement=min_agreement,
        min_agreement_mean=min_agreement_mean,
    )

    adjudicated_map: dict[tuple[str, str], dict[str, Any]] = {}
    selected_count = int(pair_stats["needs_adjudication"].sum())
    if selected_count > 0:
        if reasoner is None:
            raise ValueError("Reasoner is required when selected divergent pairs are present")
        adjudicated_map = _run_adjudication_for_selected_generic(
            merged_df=df,
            selected_pairs=pair_stats,
            abstract_lookup=abstract_lookup,
            reasoner=reasoner,
            target_id_col="RA2025_ID",
            target_text_col="RA_Question",
            target_type="RA",
        )

    def _resolve_pair(row: pd.Series) -> tuple[float, Optional[str], str, Any]:
        key = (str(row["Abstract_Index_Norm"]), str(row["Target_ID_Norm"]))
        adjudicated = adjudicated_map.get(key)
        if adjudicated is not None:
            return (
                float(adjudicated["final_affinity"]),
                adjudicated.get("final_reason"),
                str(adjudicated.get("final_method") or "adjudicated_llm"),
                adjudicated.get("model_slug"),
            )
        return (float(row["affinity_mean"]), None, "ensemble_mean", pd.NA)

    resolved = pair_stats.apply(_resolve_pair, axis=1, result_type="expand")
    resolved.columns = ["final_affinity", "final_reason", "final_method", "adjudication_model_slug"]
    pair_out = pd.concat([pair_stats, resolved], axis=1)

    pair_out = pair_out.rename(
        columns={
            "n_models": "N_Models",
            "affinity_mean": "Affinity_Mean",
            "affinity_std": "Affinity_Std",
            "affinity_min": "Affinity_Min",
            "affinity_max": "Affinity_Max",
            "affinity_range": "Affinity_Range",
            "final_affinity": "Final_Affinity",
            "final_reason": "Final_Reason",
            "final_method": "Final_Method",
            "adjudication_model_slug": "Model_Slug",
        }
    )

    pair_cols = [
        "Abstract_Index",
        "Document Title",
        "RA2025_ID",
        "RA_Question",
        "N_Models",
        "Affinity_Mean",
        "Affinity_Std",
        "Affinity_Min",
        "Affinity_Max",
        "Affinity_Range",
        "agreement_mean",
        "agreement_std",
        "agreement_min",
        "agreement_max",
        "Final_Affinity",
        "Final_Reason",
        "Final_Method",
        "Model_Slug",
    ]

    if "Cosine_x100" in df.columns:
        cosine = (
            df.groupby(["Abstract_Index_Norm", "Target_ID_Norm"], as_index=False)["Cosine_x100"]
            .mean()
            .rename(columns={"Cosine_x100": "Cosine_x100_Mean"})
        )
        pair_out = pair_out.merge(cosine, on=["Abstract_Index_Norm", "Target_ID_Norm"], how="left")
        pair_cols.append("Cosine_x100_Mean")

    extra_cols = [c for c in ["Run_ID"] if c in df.columns and c not in pair_cols]
    if extra_cols:
        run_ids = df.groupby(["Abstract_Index_Norm", "Target_ID_Norm"], as_index=False)[extra_cols].first()
        pair_out = pair_out.merge(run_ids, on=["Abstract_Index_Norm", "Target_ID_Norm"], how="left")
        pair_cols.extend(extra_cols)

    join_cols = ["Abstract_Index_Norm", "Target_ID_Norm", "Final_Affinity", "Final_Method"]
    if "Final_Reason" in pair_out.columns:
        join_cols.append("Final_Reason")
    pair_join = pair_out[join_cols].copy()

    pair_out = pair_out.sort_values(["Abstract_Index_Norm", "RA2025_ID"], kind="stable")[pair_cols]
    model_rows_out = df.merge(pair_join, on=["Abstract_Index_Norm", "Target_ID_Norm"], how="left")
    model_rows_out = model_rows_out.drop(columns=["Target_ID_Norm"], errors="ignore")

    return pair_out, model_rows_out


def parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    adjud_cfg = settings.get("runtime", {}).get("affinity", {}).get("adjudication", {})

    parser = argparse.ArgumentParser(description="Adjudicate divergent PRP and/or RA affinities across model outputs")
    parser.add_argument(
        "--mode",
        choices=["both", "prp", "ra"],
        default=str(adjud_cfg.get("mode", "both")),
        help="Run PRP only, RA only, or both",
    )
    parser.add_argument(
        "--input-root",
        default=str(adjud_cfg.get("input_root") or DEFAULT_INPUT_ROOT),
        help="Root containing benchmark run folders",
    )
    parser.add_argument("--run-id", default=None, help="Run id; defaults to latest run folder")
    parser.add_argument(
        "--prp-input-file",
        default=str(adjud_cfg.get("prp_input_file") or "merged_prp_affinities.csv"),
        help="Merged PRP file name under run dir",
    )
    parser.add_argument(
        "--ra-input-file",
        default=str(adjud_cfg.get("ra_input_file") or "merged_ra_affinities.csv"),
        help="Merged RA file name under run dir",
    )
    parser.add_argument(
        "--prp-output-file",
        default=str(adjud_cfg.get("prp_output_file") or "adjudicated_prp_affinities.csv"),
        help="PRP pair-level output file name under run dir",
    )
    parser.add_argument(
        "--prp-output-model-rows-file",
        default=str(adjud_cfg.get("prp_output_model_rows_file") or "adjudicated_prp_model_rows.csv"),
        help="PRP model-row output file name under run dir",
    )
    parser.add_argument(
        "--ra-output-file",
        default=str(adjud_cfg.get("ra_output_file") or "adjudicated_ra_affinities.csv"),
        help="RA pair-level output file name under run dir",
    )
    parser.add_argument(
        "--ra-output-model-rows-file",
        default=str(adjud_cfg.get("ra_output_model_rows_file") or "adjudicated_ra_model_rows.csv"),
        help="RA model-row output file name under run dir",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=_as_float(adjud_cfg.get("min_agreement"), 1.0),
        help="Select pairs where agreement_min is below this threshold",
    )
    parser.add_argument(
        "--min-agreement-mean",
        type=float,
        default=_as_float(adjud_cfg.get("min_agreement_mean"), 1.0),
        help="Select pairs where agreement_mean is below this threshold",
    )
    parser.add_argument(
        "--first-n-abstracts",
        type=int,
        default=adjud_cfg.get("first_n_abstracts", None),
        help="Optional limit to adjudicate only the first N unique abstracts from the merged input",
    )
    parser.add_argument(
        "--enable-affinity-reasons",
        dest="enable_affinity_reasons",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override reason-enabled adjudicator outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected-pair count and exit")
    return parser.parse_args()


def main() -> None:
    settings = _load_settings(CFG_PATH)
    args = parse_args(settings)

    input_root = Path(args.input_root)
    run_dir = _discover_run_dir(input_root=input_root, run_id=args.run_id)
    # Record run metadata (git + settings) for reproducibility
    try:
        from scripts.util.metadata import get_run_metadata, write_run_metadata_file
    except ModuleNotFoundError:
        from util.metadata import get_run_metadata, write_run_metadata_file

    try:
        run_meta = get_run_metadata(repo_root=REPO_ROOT, settings_path=CFG_PATH)
        write_run_metadata_file(run_dir / "run_metadata.json", run_meta)
    except Exception:
        run_meta = {"git": {"commit": None, "commit_short": None, "branch": None}}
    abstract_lookup = _load_abstract_lookup(settings)

    reasoner: Optional[Any] = None
    if not args.dry_run:
        from src.models.llm_affinity_adjudicator_reasoner import LLMAffinityAdjudicatorReasoner

        model, rpm, base_url, api_key = _load_runtime_llm_config(settings)
        reasoner = LLMAffinityAdjudicatorReasoner(
            model=model,
            requests_per_minute=rpm,
            base_url=base_url,
            api_key=api_key,
            enable_affinity_reasons=args.enable_affinity_reasons,
        )

    print(f"Run directory: {run_dir}")
    if args.mode in {"prp", "both"}:
        prp_path = run_dir / args.prp_input_file
        if not prp_path.exists():
            raise FileNotFoundError(f"Merged PRP file not found: {prp_path}")

        merged_prp = pd.read_csv(prp_path)
        merged_prp = _subset_first_n_abstracts(merged_prp, args.first_n_abstracts)
        if args.first_n_abstracts is not None:
            print(f"[PRP] Filtered to first {args.first_n_abstracts} abstracts -> {len(merged_prp)} rows")

        prp_pair_out, prp_model_rows_out = run_prp_adjudication(
            merged_prp=merged_prp,
            abstract_lookup=abstract_lookup,
            reasoner=reasoner,
            min_agreement=args.min_agreement,
            min_agreement_mean=args.min_agreement_mean,
        )

        selected_count = int((prp_pair_out["Final_Method"] == "adjudicated_llm").sum())
        total_pairs = int(len(prp_pair_out))
        print(f"[PRP] Total pairs: {total_pairs}")
        print(f"[PRP] Selected divergent pairs: {selected_count}")

        if not args.dry_run:
            prp_pairs_path = run_dir / args.prp_output_file
            prp_rows_path = run_dir / args.prp_output_model_rows_file
            prp_pair_out.to_csv(prp_pairs_path, index=False)
            prp_model_rows_out.to_csv(prp_rows_path, index=False)
            print(f"[PRP] Saved pair-level output -> {prp_pairs_path}")
            print(f"[PRP] Saved model-row output -> {prp_rows_path}")

    if args.mode in {"ra", "both"}:
        ra_path = run_dir / args.ra_input_file
        if not ra_path.exists():
            raise FileNotFoundError(f"Merged RA file not found: {ra_path}")

        merged_ra = pd.read_csv(ra_path)
        merged_ra = _subset_first_n_abstracts(merged_ra, args.first_n_abstracts)
        if args.first_n_abstracts is not None:
            print(f"[RA] Filtered to first {args.first_n_abstracts} abstracts -> {len(merged_ra)} rows")

        ra_pair_out, ra_model_rows_out = run_ra_adjudication(
            merged_ra=merged_ra,
            abstract_lookup=abstract_lookup,
            reasoner=reasoner,
            min_agreement=args.min_agreement,
            min_agreement_mean=args.min_agreement_mean,
        )

        selected_count = int((ra_pair_out["Final_Method"] == "adjudicated_llm").sum())
        total_pairs = int(len(ra_pair_out))
        print(f"[RA] Total pairs: {total_pairs}")
        print(f"[RA] Selected divergent pairs: {selected_count}")

        if not args.dry_run:
            ra_pairs_path = run_dir / args.ra_output_file
            ra_rows_path = run_dir / args.ra_output_model_rows_file
            ra_pair_out.to_csv(ra_pairs_path, index=False)
            ra_model_rows_out.to_csv(ra_rows_path, index=False)
            print(f"[RA] Saved pair-level output -> {ra_pairs_path}")
            print(f"[RA] Saved model-row output -> {ra_rows_path}")


if __name__ == "__main__":
    main()