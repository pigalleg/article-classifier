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
from src.models.llm_affinity_reasoner import LLMAffinityReasoner
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


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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

# Record run metadata (git + settings) for reproducibility
try:
    from scripts.util.metadata import get_run_metadata, write_run_metadata_file
except ModuleNotFoundError:
    from util.metadata import get_run_metadata, write_run_metadata_file

try:
    run_meta = get_run_metadata(repo_root=Path("."), settings_path=CFG_PATH)
    write_run_metadata_file(Path(RESULTS_DIR) / "run_metadata.json", run_meta)
except Exception:
    run_meta = {"git": {"commit": None, "commit_short": None, "branch": None}}

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
PRP_MEMBERSHIP_CONFIDENCE_MIN = _as_float(
    os.getenv("AFFINITY_PRP_MEMBERSHIP_CONFIDENCE_MIN")
    or AFFINITY_DEFAULTS.get("prp_membership_confidence_min"),
    10.0,
)
PRP_SCOPE_FALLBACK_MODE = _as_str(
    os.getenv("AFFINITY_PRP_SCOPE_FALLBACK_MODE")
    or AFFINITY_DEFAULTS.get("prp_scope_fallback_mode"),
    "legacy_scores",
).lower()

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
    parser.add_argument(
        "--prp-membership-confidence-min",
        type=float,
        default=PRP_MEMBERSHIP_CONFIDENCE_MIN,
        help="Minimum confidence for considering abstract in-scope for PRP membership",
    )
    parser.add_argument(
        "--prp-scope-fallback-mode",
        choices=["strict", "legacy_scores"],
        default=PRP_SCOPE_FALLBACK_MODE,
        help="Fallback mode when PRP scope JSON is invalid (strict=fail, legacy_scores=use legacy PRP scoring)",
    )
    parser.add_argument(
        "--enable-affinity-reasons",
        dest="enable_affinity_reasons",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override whether affinity prompts must return a reason per target ID. "
            "Use --enable-affinity-reasons or --no-enable-affinity-reasons."
        ),
    )
    parser.add_argument(
        "--enable-few-shot",
        dest="enable_few_shot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override few-shot prompting for this run. "
            "Use --enable-few-shot or --no-enable-few-shot. "
            "Default: follow AFFINITY_ENABLE_FEW_SHOT env var, then settings.yaml."
        ),
    )
    return parser.parse_args()


def _get_general_prp_description(prp: pd.DataFrame, prp_name_col: str, prp_desc_col: str | None) -> str:
    if prp_desc_col is None:
        raise ValueError("PRP description column is required for strict scope-aware PRP evaluation")

    name_series = prp[prp_name_col].astype(str).str.strip().str.lower()
    general_rows = prp[name_series == "general"]
    if general_rows.empty:
        raise ValueError("General PRP row is required for strict scope-aware PRP evaluation")

    desc_values = general_rows[prp_desc_col].dropna().astype(str).str.strip()
    desc_values = desc_values[desc_values != ""]
    if desc_values.empty:
        raise ValueError("General PRP description is required for strict scope-aware PRP evaluation")

    return str(desc_values.iloc[0])


def load_config(cfg_path: str):
    """Read LLM runtime config with local-by-default backend selection.

    Priority order:
    1) Explicit env overrides (names configurable in settings).
    2) Active backend profile (local/cloud) from settings.
    3) Generic fallbacks.
    """
    llm_cfg = SETTINGS.get("models", {}).get("llm", {})

    backend_env_var = _as_str(llm_cfg.get("backend_env_var"), "LLM_BACKEND")
    model_env_var = llm_cfg.get("model_env_var")
    base_url_env_var = llm_cfg.get("base_url_env_var")
    api_key_env_var = llm_cfg.get("api_key_env_var")
    rpm_env_var = llm_cfg.get("rpm_env_var")

    selected_backend = _as_str(
        os.getenv(backend_env_var) or llm_cfg.get("default_backend"),
        "local",
    ).lower()
    active_profile = llm_cfg.get("cloud", {}) if selected_backend == "cloud" else llm_cfg.get("local", {})

    # Resolve env-var names with profile override first, then llm-global names.
    effective_base_url_env_var = active_profile.get("base_url_env_var") or active_profile.get("base_url_env") or base_url_env_var
    effective_api_key_env_var = active_profile.get("api_key_env_var") or active_profile.get("api_key_env") or api_key_env_var

    model = (os.getenv(str(model_env_var)) if model_env_var else None) or active_profile.get("model") or llm_cfg.get("model")
    if model is None:
        raise ValueError("Missing model configuration: set models.llm.model_env_var or models.llm.<backend>.model in settings.yaml")

    base_url = (os.getenv(str(effective_base_url_env_var)) if effective_base_url_env_var else None) or active_profile.get("base_url")
    if isinstance(base_url, str):
        base_url = base_url.strip() or None

    # Check if benchmark script passed a model-specific API key env var name
    # (e.g., GOOGLE_API_KEY for Gemini instead of OPENAI_CLOUD_API_KEY for OpenAI)
    override_api_key_env_var = os.getenv("OPENAI_API_KEY_ENV_VAR")
    if override_api_key_env_var:
        effective_api_key_env_var = override_api_key_env_var

    api_key = (os.getenv(str(effective_api_key_env_var)) if effective_api_key_env_var else None) or active_profile.get("api_key")

    rpm_default_from_reasoner = SETTINGS.get("runtime", {}).get("llm_reasoner", {}).get(
        "requests_per_minute_default_cloud" if selected_backend == "cloud" else "requests_per_minute_default_local"
    )
    rpm = (os.getenv(str(rpm_env_var)) if rpm_env_var else None) or active_profile.get("requests_per_minute_default") or rpm_default_from_reasoner
    rpm = _as_int(rpm, _as_int(rpm_default_from_reasoner, 1))
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
        prp["PRP_Text"] = prp[prp_desc_col].astype(str)

    return abstracts, ra, prp, prp_name_col, prp_desc_col


def init_models(
    ra_df: pd.DataFrame,
    enable_few_shot: Optional[bool] = None,
    enable_affinity_reasons: Optional[bool] = None,
) -> tuple[EmbeddingModel, RAClassifier, LLMAffinityReasoner]:
    embedder = EmbeddingModel(model_name=EMBED_MODEL)
    classifier = RAClassifier(embedder, ra_df, text_column="Question_Cleaned")
    llm_model, rpm, base_url, api_key = load_config(CFG_PATH)

    reasoner = LLMAffinityReasoner(
        model=llm_model,
        temperature=0.0,
        requests_per_minute=rpm,
        base_url=base_url,
        api_key=api_key,
        enable_few_shot=enable_few_shot,
        enable_affinity_reasons=enable_affinity_reasons,
    )
    return embedder, classifier, reasoner


def _extract_affinity_score_and_reason(value: Any) -> tuple[Optional[float], Optional[str]]:
    if isinstance(value, dict):
        score_raw = value.get("score", value.get("affinity", value.get("value")))
        reason = value.get("reason", value.get("rationale", value.get("explanation")))
    else:
        score_raw = value
        reason = None

    if score_raw is None:
        return None, None

    try:
        score = float(score_raw)
        score = float(np.round(max(0.0, min(100.0, score)), 4))
    except Exception:
        return None, None

    reason_text = None if reason is None else str(reason).strip()
    return score, reason_text or None


# sym:preflight_llm_affinity
def preflight_llm_affinity(reasoner: LLMAffinityReasoner) -> None:
    """DEPRECATED: retained only for manual diagnostics; not used by the main execution path."""
    base_url = reasoner._endpoint_for_logs() if hasattr(reasoner, "_endpoint_for_logs") else "<openai-cloud>"
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
    # For PRP-driven retrieval modes, exclude zero-affinity programmes from routing
    # so RA candidates linked only to zero-affinity PRPs are not retrieved.
    df = df[df["LLM_Affinity"].astype(float) > 0.0]
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
                                      classifier: RAClassifier, reasoner: LLMAffinityReasoner, top_k: int,
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
        llm_aff_raw = affinity_scores.get(_normalize_id(ra_id))
        llm_aff, llm_reason = _extract_affinity_score_and_reason(llm_aff_raw)
        
        records.append({
            "Abstract_Index": src_idx,
            "Document Title": doc_title,
            "RA2025_ID": ra_id,
            "RA_Question": question,
            "Cosine_x100": cos100,
            "LLM_Affinity": llm_aff,
            "LLM_Affinity_Reason": llm_reason,
        })
    return records


def evaluate_prp_affinity_for_abstract(abstract_text: str, doc_title: str, src_idx: Any,
                                       prp: pd.DataFrame, prp_name_col: str, prp_desc_col: str | None,
                                       reasoner: LLMAffinityReasoner,
                                       general_prp_description: str,
                                       membership_confidence_min: float,
                                       scope_fallback_mode: str) -> List[Dict[str, Any]]:
    records = []
    
    # Batch all PRP candidates into a single LLM call (Structured Batch Prompting)
    prp_targets = []
    prp_rows_list = []
    
    for _, prow in prp.iterrows():
        name = prow[prp_name_col]
        if str(name).strip().lower() == "general":
            # "General" is used as scope anchor only and is not scored as a PRP target.
            continue
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
        try:
            prp_eval = reasoner.rate_prp_affinity_with_scope(
                abstract=abstract_text,
                prp_targets=prp_targets,
                general_prp_description=general_prp_description,
            )
            belongs_any_prp = prp_eval["belongs_any_prp"]
            membership_confidence = float(prp_eval["membership_confidence"])
            affinity_scores = prp_eval["scores"]
            out_of_scope = (belongs_any_prp is False) or (membership_confidence < float(membership_confidence_min))
            if out_of_scope and getattr(reasoner, "affinity_debug", False):
                print(
                    f"PRP out-of-scope decision: abstract_index={src_idx}, "
                    f"title={doc_title}, "
                    f"Belong to any PRP? {belongs_any_prp}, confidence={membership_confidence:.4f},  threshold {float(membership_confidence_min):.4f}. "
                )
        except Exception as e:
            if scope_fallback_mode == "legacy_scores":
                print(
                    f"PRP scope parser failed; using legacy PRP scoring fallback: "
                    f"abstract_index={src_idx}, title={doc_title}, reason={e}"
                )
                affinity_scores = reasoner.rate_affinity_batch(abstract_text, prp_targets, target_type="PRP")
                out_of_scope = False
            else:
                raise
    else:
        out_of_scope = False
        affinity_scores = {}
    
    # Build records from batch results
    for prp_info, target_dict in zip(prp_rows_list, prp_targets):
        name = prp_info["name"]
        desc = prp_info["desc"]
        target_id = _normalize_id(target_dict["id"])
        
        llm_aff_raw = 0.0 if out_of_scope else affinity_scores.get(target_id)
        llm_aff, llm_reason = _extract_affinity_score_and_reason(llm_aff_raw)
        
        records.append({
            "Abstract_Index": src_idx,
            "Document Title": doc_title,
            "PRP_Name": name,
            "PRP_Description": desc,
            "LLM_Affinity": llm_aff,
            "LLM_Affinity_Reason": None if out_of_scope else llm_reason,
        })
    
    return records


def main():
    args = parse_args()
    print(f"Ollama GPU state at launch: {_describe_ollama_gpu_state()}")
    timing_logger = AffinityTimingLogger.from_results_dir(RESULTS_DIR)

    abstracts, ra, prp, prp_name_col, prp_desc_col = load_inputs()
    _, classifier, reasoner = init_models(
        ra,
        enable_few_shot=args.enable_few_shot,
        enable_affinity_reasons=args.enable_affinity_reasons,
    )
    general_prp_description = ""
    if args.mode in {"prp", "both"}:
        general_prp_description = _get_general_prp_description(prp, prp_name_col, prp_desc_col)

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
            try:
                prp_records = evaluate_prp_affinity_for_abstract(
                    abstract_text,
                    doc_title,
                    src_idx,
                    prp,
                    prp_name_col,
                    prp_desc_col,
                    reasoner,
                    general_prp_description=general_prp_description,
                    membership_confidence_min=args.prp_membership_confidence_min,
                    scope_fallback_mode=args.prp_scope_fallback_mode,
                )
            except Exception as e:
                print(
                    f"PRP evaluation failed: abstract_index={src_idx}, "
                    f"title={doc_title}, reason={e}"
                )
                raise
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