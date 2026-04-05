#!/usr/bin/env python3
"""
Compute RA- and PRP-level affinity (0–100) between abstracts and RA questions / PRPs.

Outputs:
- data/results/ra_affinities.csv
- data/results/prp_affinities.csv
- data/results/combined_affinities.xlsx
"""
# How to run the new workflow
#
# Run PRP stage only
# python run_affinity_evaluation.py --mode prp
#
# Run RA stage only with PRP filtering
# python run_affinity_evaluation.py --mode ra --ra-retrieval-mode prp_filter --prp-input prp_affinities.csv --prp-top-n 2
#
# Run RA stage only with strict PRP-only selection
# python run_affinity_evaluation.py --mode ra --ra-retrieval-mode prp_only --prp-input prp_affinities.csv --prp-top-n 2
#
# Run old combined behavior
# python run_affinity_evaluation.py --mode both --ra-retrieval-mode cosine

import argparse
from importlib import import_module
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

# Allow direct execution from scripts/ (python run_affinity_evaluation.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.embeddings import EmbeddingModel
from src.models.classifier import RAClassifier
from src.models.llm_reasoner import LLMReasoner
try:
    # Works when executed from repo root as a package path.
    from scripts.util.execution_time_logger import AffinityTimingLogger
except ModuleNotFoundError:
    # Works when executed directly from inside scripts/.
    from util.execution_time_logger import AffinityTimingLogger

_describe_ollama_gpu_state = import_module("scripts.util.ollama_gpu")._describe_ollama_gpu_state

CFG_PATH = os.path.join("src", "config", "settings.yaml")


def _load_settings(cfg_path: str) -> dict:
    try:
        with open(cfg_path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_str(value, default: str) -> str:
    if value is None:
        return str(default)
    s = str(value).strip()
    return s if s else str(default)


SETTINGS = _load_settings(CFG_PATH)
AFFINITY_DEFAULTS = SETTINGS.get("runtime", {}).get("affinity", {})
PATH_DEFAULTS = SETTINGS.get("paths", {})
MODEL_DEFAULTS = SETTINGS.get("models", {})

DATA_DIR = PATH_DEFAULTS.get("data_processed", "data/processed")
RESULTS_DIR = os.getenv("AFFINITY_RESULTS_DIR") or PATH_DEFAULTS.get("results", "data/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EMBED_MODEL = (
    os.getenv("AFFINITY_EMBED_MODEL")
    or MODEL_DEFAULTS.get("embedding_model")
    or "all-mpnet-base-v2"
)
TOP_K = _as_int(os.getenv("AFFINITY_TOP_K") or AFFINITY_DEFAULTS.get("top_k"), 5)
SAVE_EVERY = _as_int(os.getenv("SAVE_EVERY") or AFFINITY_DEFAULTS.get("save_every"), 50)
ABSTRACTS_FILE = _as_str(
    os.getenv("AFFINITY_ABSTRACTS_FILE") or AFFINITY_DEFAULTS.get("abstracts_file"),
    "abstracts_cleaned.csv",
)

def _normalize_id(value: Any) -> str:
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRP and/or RA affinity evaluation")
    parser.add_argument(
        "--mode",
        choices=["both", "prp", "ra"],
        default=str(AFFINITY_DEFAULTS.get("mode", "both")),
        help="Run PRP only, RA only, or both (default: both)",
    )
    parser.add_argument(
        "--ra-retrieval-mode",
        choices=["cosine", "prp_filter", "prp_only"],
        default=str(AFFINITY_DEFAULTS.get("ra_retrieval_mode", "prp_only")),
        help="RA candidate retrieval strategy (default: prp_only)",
    )
    parser.add_argument(
        "--prp-input",
        default=str(AFFINITY_DEFAULTS.get("prp_input", os.path.join(RESULTS_DIR, "prp_affinities.csv"))),
        help="Path to PRP affinity CSV used by prp_filter/prp_only",
    )
    parser.add_argument(
        "--prp-top-n",
        type=int,
        default=_as_int(AFFINITY_DEFAULTS.get("prp_top_n"), 6),
        help="Number of top PRPs per abstract to use for routing (default: 6)",
    )
    parser.add_argument(
        "--prp-min-affinity",
        type=float,
        default=AFFINITY_DEFAULTS.get("prp_min_affinity", None),
        help="Optional minimum PRP affinity threshold for routing",
    )
    return parser.parse_args()


def load_config(cfg_path: str):
    """Read LLM runtime config with local-by-default backend selection.

    Priority order:
    1) Explicit env overrides (names configurable in settings).
    2) Active backend profile (local/cloud) from settings.
    3) Generic fallbacks.
    """
    llm_cfg = SETTINGS.get("models", {}).get("llm", {})

    backend_env_var = _as_str(llm_cfg.get("backend_env_var"), "LLM_BACKEND")
    model_env_var = _as_str(llm_cfg.get("model_env_var"), "OPENAI_MODEL")
    base_url_env_var = _as_str(llm_cfg.get("base_url_env_var"), "OPENAI_BASE_URL")
    api_key_env_var = _as_str(llm_cfg.get("api_key_env_var"), "OPENAI_API_KEY")
    rpm_env_var = _as_str(llm_cfg.get("rpm_env_var"), "OPENAI_REQUESTS_PER_MINUTE")

    selected_backend = _as_str(
        os.getenv(backend_env_var) or llm_cfg.get("default_backend"),
        "local",
    ).lower()
    active_profile = llm_cfg.get("cloud", {}) if selected_backend == "cloud" else llm_cfg.get("local", {})

    model = os.getenv(model_env_var) or active_profile.get("model") or llm_cfg.get("model") or "gpt-4o-mini"
    base_url = os.getenv(base_url_env_var)
    if base_url is None:
        base_url = active_profile.get("base_url")
    if isinstance(base_url, str):
        base_url = base_url.strip() or None

    api_key = os.getenv(api_key_env_var)
    profile_api_key_env = active_profile.get("api_key_env")
    if api_key is None and profile_api_key_env:
        api_key = os.getenv(str(profile_api_key_env))
    if api_key is None:
        api_key = active_profile.get("api_key")

    rpm = os.getenv(rpm_env_var)
    if rpm is None:
        rpm = active_profile.get("requests_per_minute_default")
    rpm = _as_int(rpm, 3)
    return model, rpm, base_url, api_key


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Optional[str]]:
    abs_path = Path(DATA_DIR) / ABSTRACTS_FILE
    ra_path = Path(DATA_DIR) / "ra_questions_cleaned.csv"
    prp_path = Path(DATA_DIR) / "primary_programmes.csv"

    if not abs_path.exists():
        raise FileNotFoundError(abs_path)
    if not ra_path.exists():
        raise FileNotFoundError(ra_path)
    if not prp_path.exists():
        raise FileNotFoundError(prp_path)

    abstracts = pd.read_csv(abs_path)
    ra = pd.read_csv(ra_path)
    prp = pd.read_csv(prp_path)

    # expected columns (be tolerant with names produced by prepare_data.py)
    if "Abstract_Cleaned" not in abstracts.columns:
        raise ValueError(f"{ABSTRACTS_FILE} must contain 'Abstract_Cleaned'")
    if "Question_Cleaned" not in ra.columns or "RA2025" not in ra.columns:
        raise ValueError("ra_questions_cleaned.csv must contain 'RA2025' and 'Question_Cleaned'")

    # PRP columns
    prp_name_col = "Primary_Programme" if "Primary_Programme" in prp.columns else (
        "Primary Research Programme" if "Primary Research Programme" in prp.columns else str(prp.columns[0])
    )
    prp_desc_col = "Description" if "Description" in prp.columns else (
        "Description_Cleaned" if "Description_Cleaned" in prp.columns else (str(prp.columns[1]) if len(prp.columns) > 1 else None)
    )

    # Build PRP text used in prompts
    if prp_desc_col is None:
        prp["PRP_Text"] = prp[prp_name_col].astype(str)
    else:
        prp["PRP_Text"] = prp[prp_name_col].astype(str) + " — " + prp[prp_desc_col].astype(str)

    return abstracts, ra, prp, prp_name_col, prp_desc_col


def init_models(ra_df: pd.DataFrame) -> tuple[EmbeddingModel, RAClassifier, LLMReasoner]:
    embedder = EmbeddingModel(model_name=EMBED_MODEL)
    classifier = RAClassifier(embedder, ra_df, text_column="Question_Cleaned")
    llm_model, rpm, base_url, api_key = load_config(CFG_PATH)
    reasoner = LLMReasoner(
        model=llm_model,
        temperature=0.0,
        requests_per_minute=rpm,
        base_url=base_url,
        api_key=api_key,
    )
    return embedder, classifier, reasoner


def preflight_llm_affinity(reasoner: LLMReasoner) -> None:
    """Fail fast if the configured model/backend cannot produce affinity outputs."""
    base_url = os.getenv("OPENAI_BASE_URL") or "<openai-cloud>"
    try:
        # Reuse the exact affinity path used in the main loop.
        val = reasoner.rate_affinity(
            abstract="Power systems planning with renewable integration and voltage stability.",
            target_text="Assess grid flexibility and reliability under high renewable penetration.",
            target_type="RA",
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM preflight failed for model '{reasoner.model}' on endpoint '{base_url}'. {e}"
        ) from e

    if val is None:
        raise RuntimeError(
            "LLM preflight returned no numeric affinity (None). "
            f"Model='{reasoner.model}', endpoint='{base_url}'. "
            "Set AFFINITY_DEBUG=1 to inspect raw replies, and verify model availability on the selected backend."
        )


def load_prp_routing_map(
    prp_input: str,
    top_n: int,
    min_affinity: Optional[float],
) -> Dict[str, List[str]]:
    path = Path(prp_input)
    if not path.exists() and not path.is_absolute():
        # Convenience: allow passing just "prp_affinities.csv" from repo root.
        alt = Path(RESULTS_DIR) / path
        if alt.exists():
            path = alt

    if not path.exists():
        raise FileNotFoundError(
            f"PRP routing input not found: {path} (cwd={Path.cwd()}). "
            "Run --mode prp first or provide --prp-input data/results/prp_affinities.csv."
        )

    df = pd.read_csv(path)
    needed = {"Abstract_Index", "PRP_Name", "LLM_Affinity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"PRP routing input missing columns: {sorted(missing)}")

    df = df.dropna(subset=["Abstract_Index", "PRP_Name", "LLM_Affinity"]).copy()
    if min_affinity is not None:
        df = df[df["LLM_Affinity"].astype(float) >= float(min_affinity)]
    df["_abs_key"] = df["Abstract_Index"].astype(str)
    df["_aff"] = df["LLM_Affinity"].astype(float)
    routing: Dict[str, List[str]] = {}
    for abs_key, grp in df.groupby("_abs_key"):
        pairs = []
        for _, row in grp.iterrows():
            name = str(row["PRP_Name"]).strip()
            if not name:
                continue
            pairs.append((name, float(row["_aff"])))
        pairs.sort(key=lambda x: x[1], reverse=True)
        vals = [name for name, _ in pairs]
        deduped = list(dict.fromkeys(vals))
        routing[str(abs_key)] = deduped[:max(1, int(top_n))]
    return routing


def evaluate_ra_affinity_for_abstract(abstract_text: str, doc_title: str, src_idx: Any,
                                      classifier: RAClassifier, reasoner: LLMReasoner, top_k: int,
                                      retrieval_mode: str = "cosine",
                                      top_programmes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    records = []
    candidates = classifier.get_top_k_candidates(
        abstract_text,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        top_programmes=top_programmes,
    )
    # Graceful fallback if PRP-filtered retrieval yields no candidates.
    if not candidates and retrieval_mode in {"prp_filter", "prp_only"}:
        print(f"Warning: No candidates found with retrieval_mode={retrieval_mode}, falling back to cosine similarity")
        candidates = classifier.get_top_k_candidates(abstract_text, top_k=top_k, retrieval_mode="cosine")
    # Batch all RA candidates into a single LLM call (Structured Batch Prompting)
    if candidates:
        ra_targets = [
            {
                "id": _normalize_id(cand.get("RA2025_ID") or cand.get("RA2025")),
                "text": cand.get("Question") or cand.get("Question_Cleaned")
            }
            for cand in candidates
        ]
        affinity_scores = reasoner.rate_affinity_batch(abstract_text, ra_targets, target_type="RA")
    else:
        affinity_scores = {}

    for cand in candidates:
        ra_id = cand.get("RA2025_ID") or cand.get("RA2025")
        question = cand.get("Question") or cand.get("Question_Cleaned")
        cos = float(cand.get("Similarity", np.nan))
        cos100 = float(np.round(cos * 100.0, 4)) if not np.isnan(cos) else np.nan
        
        # Get LLM affinity from batch results
        llm_aff = affinity_scores.get(_normalize_id(ra_id))
        
        records.append({
            "Abstract_Index": src_idx,
            "Document Title": doc_title,
            "RA2025_ID": ra_id,
            "RA_Question": question,
            "Cosine_x100": cos100,
            "LLM_Affinity": None if llm_aff is None else float(np.round(float(llm_aff), 4)),
        })
    return records


def evaluate_prp_affinity_for_abstract(abstract_text: str, doc_title: str, src_idx: Any,
                                       prp: pd.DataFrame, prp_name_col: str, prp_desc_col: str | None,
                                       reasoner: LLMReasoner) -> List[Dict[str, Any]]:
    records = []
    
    # Batch all PRP candidates into a single LLM call (Structured Batch Prompting)
    prp_targets = []
    prp_rows_list = []
    
    for _, prow in prp.iterrows():
        name = prow[prp_name_col]
        desc = prow[prp_desc_col] if prp_desc_col is not None else ""
        prp_text = prow["PRP_Text"]
        target_id = _normalize_id(name)
        
        prp_targets.append({
            "id": target_id,
            "text": prp_text
        })
        prp_rows_list.append({
            "name": name,
            "desc": desc
        })
    
    # Single batch call for all PRPs
    if prp_targets:
        affinity_scores = reasoner.rate_affinity_batch(abstract_text, prp_targets, target_type="PRP")
    else:
        affinity_scores = {}
    
    # Build records from batch results
    for prp_info, target_dict in zip(prp_rows_list, prp_targets):
        name = prp_info["name"]
        desc = prp_info["desc"]
        target_id = _normalize_id(target_dict["id"])
        
        llm_aff = affinity_scores.get(target_id)
        
        records.append({
            "Abstract_Index": src_idx,
            "Document Title": doc_title,
            "PRP_Name": name,
            "PRP_Description": desc,
            "LLM_Affinity": None if llm_aff is None else float(np.round(float(llm_aff), 4)),
        })
    
    return records


def main():
    args = parse_args()
    print(f"Ollama GPU state at launch: {_describe_ollama_gpu_state()}")
    timing_logger = AffinityTimingLogger.from_results_dir(RESULTS_DIR)

    abstracts, ra, prp, prp_name_col, prp_desc_col = load_inputs()
    _, classifier, reasoner = init_models(ra)

    # step_start = time.perf_counter()
    # preflight_llm_affinity(reasoner)
    # startup_timings.append(("preflight_llm_affinity", time.perf_counter() - step_start))

    ra_rows: List[Dict[str, Any]] = []
    prp_rows: List[Dict[str, Any]] = []
    ra_total_elapsed = 0.0
    prp_total_elapsed = 0.0
    ra_total_targets = 0
    prp_total_targets = 0

    # filter out already classified abstracts
    abstracts = abstracts[abstracts["Already_Classified"] != True] 
    n = len(abstracts)
    titles = abstracts["Document Title"] if "Document Title" in abstracts.columns else pd.Series([f"doc_{i}" for i in range(n)])
    abs_texts = abstracts["Abstract_Cleaned"].astype(str)
    prp_routing_map: Dict[str, List[str]] = {}
    print(f"Evaluating affinities for {n} abstracts (mode={args.mode}, ra_retrieval_mode={args.ra_retrieval_mode})...")
    if args.mode in {"ra", "both"} and args.ra_retrieval_mode in {"prp_filter", "prp_only"}:
        prp_routing_map = load_prp_routing_map(
            prp_input=args.prp_input,
            top_n=args.prp_top_n,
            min_affinity=args.prp_min_affinity,
        )
    for i in tqdm(range(n), desc="Evaluating affinities"):
        doc_title = titles.iloc[i]
        src_idx = abstracts.iloc[i]["Source_Index"] if "Source_Index" in abstracts.columns else i
        abstract_text = abs_texts.iloc[i]
        if args.mode in {"ra", "both"}:
            routed_prps = prp_routing_map.get(str(src_idx)) if prp_routing_map else None
            ra_start = time.perf_counter()
            ra_records = evaluate_ra_affinity_for_abstract(
                abstract_text,
                doc_title,
                src_idx,
                classifier,
                reasoner,
                TOP_K,
                retrieval_mode=args.ra_retrieval_mode,
                top_programmes=routed_prps,
            )
            ra_elapsed = time.perf_counter() - ra_start
            ra_total_elapsed += ra_elapsed
            ra_total_targets += len(ra_records)
            ra_rows.extend(ra_records)

        if args.mode in {"prp", "both"}:
            prp_start = time.perf_counter()
            prp_records = evaluate_prp_affinity_for_abstract(
                abstract_text,
                doc_title,
                src_idx,
                prp,
                prp_name_col,
                prp_desc_col,
                reasoner,
            )
            prp_elapsed = time.perf_counter() - prp_start
            prp_total_elapsed += prp_elapsed
            prp_total_targets += len(prp_records)
            prp_rows.extend(prp_records)

        if (i + 1) % SAVE_EVERY == 0:
            if args.mode in {"ra", "both"}:
                pd.DataFrame(ra_rows).to_csv(Path(RESULTS_DIR) / "ra_affinities.csv", index=False)
            if args.mode in {"prp", "both"}:
                pd.DataFrame(prp_rows).to_csv(Path(RESULTS_DIR) / "prp_affinities.csv", index=False)
            
    # final save
    ra_out = Path(RESULTS_DIR) / "ra_affinities.csv"
    prp_out = Path(RESULTS_DIR) / "prp_affinities.csv"
    if args.mode in {"ra", "both"}:
        pd.DataFrame(ra_rows).to_csv(ra_out, index=False)
    if args.mode in {"prp", "both"}:
        pd.DataFrame(prp_rows).to_csv(prp_out, index=False)

    # combined xlsx (best effort)
    combined_out = Path(RESULTS_DIR) / "combined_affinities.xlsx"
    try:
        with pd.ExcelWriter(combined_out) as writer:
            if args.mode in {"ra", "both"}:
                pd.DataFrame(ra_rows).to_excel(writer, sheet_name="RA_Affinities", index=False)
            if args.mode in {"prp", "both"}:
                pd.DataFrame(prp_rows).to_excel(writer, sheet_name="PRP_Affinities", index=False)
    except Exception:
        pass

    # Aggregate timing logs: one entry per stage for the full run.
    if args.mode in {"ra", "both"}:
        timing_logger.log_event(
            stage="ra",
            abstract_index="ALL",
            document_title="ALL_ABSTRACTS",
            targets_evaluated=ra_total_targets,
            records_written=len(ra_rows),
            elapsed_seconds=ra_total_elapsed,
        )
    if args.mode in {"prp", "both"}:
        timing_logger.log_event(
            stage="prp",
            abstract_index="ALL",
            document_title="ALL_ABSTRACTS",
            targets_evaluated=prp_total_targets,
            records_written=len(prp_rows),
            elapsed_seconds=prp_total_elapsed,
        )

    if args.mode in {"ra", "both"}:
        print(f"Saved RA affinities -> {ra_out}")
    if args.mode in {"prp", "both"}:
        print(f"Saved PRP affinities -> {prp_out}")
    print(f"Saved combined (xlsx) -> {combined_out}")


if __name__ == "__main__":
    main()