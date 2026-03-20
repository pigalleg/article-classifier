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
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

from src.models.embeddings import EmbeddingModel
from src.models.classifier import RAClassifier
from src.models.llm_reasoner import LLMReasoner

DATA_DIR = "data/processed"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EMBED_MODEL = os.getenv("AFFINITY_EMBED_MODEL", "all-mpnet-base-v2")
LLM_MODEL = os.getenv("AFFINITY_LLM_MODEL", "gpt-4o-mini")
# REQUESTS_PER_MINUTE = int(os.getenv("AFFINITY_RPM", "20"))
TOP_K = int(os.getenv("AFFINITY_TOP_K", "5"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "50"))
CFG_PATH = os.path.join("src", "config", "settings.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRP and/or RA affinity evaluation")
    parser.add_argument(
        "--mode",
        choices=["both", "prp", "ra"],
        default="both",
        help="Run PRP only, RA only, or both (default: both)",
    )
    parser.add_argument(
        "--ra-retrieval-mode",
        choices=["cosine", "prp_filter", "prp_only"],
        default="prp_only",
        help="RA candidate retrieval strategy (default: cosine)",
    )
    parser.add_argument(
        "--prp-input",
        default=os.path.join(RESULTS_DIR, "prp_affinities.csv"),
        help="Path to PRP affinity CSV used by prp_filter/prp_only",
    )
    parser.add_argument(
        "--prp-top-n",
        type=int,
        default=2,
        help="Number of top PRPs per abstract to use for routing (default: 2)",
    )
    parser.add_argument(
        "--prp-min-affinity",
        type=float,
        default=None,
        help="Optional minimum PRP affinity threshold for routing",
    )
    return parser.parse_args()


def load_config(cfg_path: str):
    """Read LLM config from env or YAML. Env has priority.
    Supports OPENAI_MODEL / OPENAI_REQUESTS_PER_MINUTE; falls back to cfg.models.llm.*.
    """
    # primary env vars (harmonized with run_classification.py)
    model = os.getenv("OPENAI_MODEL") or os.getenv("AFFINITY_LLM_MODEL")
    rpm = os.getenv("OPENAI_REQUESTS_PER_MINUTE") or os.getenv("AFFINITY_RPM")
    try:
        with open(cfg_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
            model = model or cfg.get("models", {}).get("llm", {}).get("model")
            rpm = rpm or cfg.get("models", {}).get("llm", {}).get("requests_per_minute")
    except Exception:
        model = model or "gpt-4o-mini"
        rpm = rpm or 3
    try:
        rpm = int(rpm)
    except Exception:
        rpm = 3
    return model or "gpt-4o-mini", rpm


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Optional[str]]:
    abs_path = Path(DATA_DIR) / "abstracts_cleaned.csv"
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
        raise ValueError("abstracts_cleaned.csv must contain 'Abstract_Cleaned'")
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
    llm_model, rpm = load_config(CFG_PATH)
    reasoner = LLMReasoner(model=llm_model, temperature=0.0, requests_per_minute=rpm)
    return embedder, classifier, reasoner


def preflight_llm_affinity(reasoner: LLMReasoner) -> None:
    """Fail fast if the configured model/backend cannot produce affinity outputs."""
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE_URL") or "<openai-cloud>"
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
        candidates = classifier.get_top_k_candidates(abstract_text, top_k=top_k, retrieval_mode="cosine")

    for cand in candidates:
        ra_id = cand.get("RA2025_ID") or cand.get("RA2025")
        question = cand.get("Question") or cand.get("Question_Cleaned")
        cos = float(cand.get("Similarity", np.nan))
        cos100 = float(np.round(cos * 100.0, 4)) if not np.isnan(cos) else np.nan
        # LLM numeric affinity (0–100)
        llm_aff = reasoner.rate_affinity(abstract_text, question, target_type="RA")
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
    for _, prow in prp.iterrows():
        name = prow[prp_name_col]
        desc = prow[prp_desc_col] if prp_desc_col is not None else ""
        prp_text = prow["PRP_Text"]
        llm_aff = reasoner.rate_affinity(abstract_text, prp_text, target_type="PRP")
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

    abstracts, ra, prp, prp_name_col, prp_desc_col = load_inputs()
    _, classifier, reasoner = init_models(ra)
    preflight_llm_affinity(reasoner)

    ra_rows: List[Dict[str, Any]] = []
    prp_rows: List[Dict[str, Any]] = []

    # filter out already classified abstracts
    abstracts = abstracts[abstracts["Already_Classified"] != True] 
    n = len(abstracts)
    titles = abstracts["Document Title"] if "Document Title" in abstracts.columns else pd.Series([f"doc_{i}" for i in range(n)])
    abs_texts = abstracts["Abstract_Cleaned"].astype(str)

    prp_routing_map: Dict[str, List[str]] = {}
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
            routed_prps = prp_routing_map.get(str(src_idx), []) if prp_routing_map else None
            ra_rows.extend(
                evaluate_ra_affinity_for_abstract(
                    abstract_text,
                    doc_title,
                    src_idx,
                    classifier,
                    reasoner,
                    TOP_K,
                    retrieval_mode=args.ra_retrieval_mode,
                    top_programmes=routed_prps,
                )
            )

        if args.mode in {"prp", "both"}:
            prp_rows.extend(
                evaluate_prp_affinity_for_abstract(
                    abstract_text,
                    doc_title,
                    src_idx,
                    prp,
                    prp_name_col,
                    prp_desc_col,
                    reasoner,
                )
            )

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

    if args.mode in {"ra", "both"}:
        print(f"Saved RA affinities -> {ra_out}")
    if args.mode in {"prp", "both"}:
        print(f"Saved PRP affinities -> {prp_out}")
    print(f"Saved combined (xlsx) -> {combined_out}")


if __name__ == "__main__":
    main()