#!/usr/bin/env python3
"""
Run classification pipeline:
  - Load cleaned CSVs
  - Generate embeddings for abstracts and RA questions
  - Match abstracts to RA questions (top-k)
  - Optionally verify matches with an LLM agent
  - Save classification outputs to data/results/
"""
#!/usr/bin/env python3
"""
Hybrid classification pipeline:
1. Use cosine similarity to retrieve top-k RA2025 candidate questions.
2. Use LLM agent to reason among candidates and pick the best match.
3. Save classification results with explanations.

Outputs:
- data/results/classified_articles_llm.xlsx
- data/results/llm_mismatches.csv
"""

import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import json

from src.models.embeddings import EmbeddingModel
from src.models.classifier import RAClassifier
from src.models.llm_reasoner import LLMReasoner

DATA_DIR = "data/processed"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CFG_PATH = os.path.join("src", "config", "settings.yaml")

TOP_K = 5  # number of candidate RA questions to consider for LLM


def _load_settings(cfg_path: str) -> dict:
    try:
        with open(cfg_path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


SETTINGS = _load_settings(CFG_PATH)
MODEL_DEFAULTS = SETTINGS.get("models", {})

def main():
    # Refactored workflow using helper functions for clarity
    def load_config(cfg_path: str):
        """Read LLM config from YAML with local-by-default backend profiles."""
        model = None
        rpm = None

        try:
            with open(cfg_path, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            cfg = {}

        llm_cfg = cfg.get("models", {}).get("llm", {})
        backend_env_var = str(llm_cfg.get("backend_env_var") or "LLM_BACKEND")
        model_env_var = str(llm_cfg.get("model_env_var") or "OPENAI_MODEL")
        rpm_env_var = str(llm_cfg.get("rpm_env_var") or "OPENAI_REQUESTS_PER_MINUTE")

        selected_backend = str(os.getenv(backend_env_var) or llm_cfg.get("default_backend") or "local").strip().lower()
        active_profile = llm_cfg.get("cloud", {}) if selected_backend == "cloud" else llm_cfg.get("local", {})

        model = os.getenv(model_env_var) or active_profile.get("model") or llm_cfg.get("model") or "gpt-4o-mini"
        rpm = os.getenv(rpm_env_var) or active_profile.get("requests_per_minute_default") or 3

        try:
            rpm = int(rpm)
        except Exception:
            rpm = 3

        return model, rpm


    def prepare_dataframe(ieee_df: pd.DataFrame):
        """Ensure expected columns and types exist on the dataframe."""
        if "Already_Classified" not in ieee_df.columns:
            ieee_df["Already_Classified"] = False
        else:
            ieee_df["Already_Classified"] = ieee_df["Already_Classified"].astype(bool)

        for col in ("LLM_Returned_ID", "RA2025_Final_ID", "LLM_Reason"):
            if col not in ieee_df.columns:
                ieee_df[col] = None
            try:
                ieee_df[col] = ieee_df[col].astype(object)
            except Exception:
                pass

        # New metrics/flags columns
        metrics_float_cols = [
            "Top_Cosine_Sim",
            "LLM_Confidence",
            "Combined_Confidence",
        ]
        for col in metrics_float_cols:
            if col not in ieee_df.columns:
                ieee_df[col] = np.nan
            try:
                ieee_df[col] = pd.to_numeric(ieee_df[col], errors="coerce").astype(float)
            except Exception:
                pass

        prog_cols = [
            "Predicted_Primary_Programme",
            "Predicted_Secondary_Programme",
        ]
        for col in prog_cols:
            if col not in ieee_df.columns:
                ieee_df[col] = None
            try:
                ieee_df[col] = ieee_df[col].astype(object)
            except Exception:
                pass

        if "Low_Confidence_Flag" not in ieee_df.columns:
            ieee_df["Low_Confidence_Flag"] = False
        else:
            try:
                ieee_df["Low_Confidence_Flag"] = ieee_df["Low_Confidence_Flag"].astype(bool)
            except Exception:
                pass

        return ieee_df


    def init_models(ra_df: pd.DataFrame):
        """Initialize embedding/classifier/reasoner objects."""
        embed_model = os.getenv("AFFINITY_EMBED_MODEL") or MODEL_DEFAULTS.get("embedding_model") or "all-mpnet-base-v2"
        embedder = EmbeddingModel(model_name=embed_model)
        classifier = RAClassifier(embedder, ra_df, text_column="Question_Cleaned")
        return embedder, classifier


    def _write_merged(ieee_df: pd.DataFrame, out_path: str):
        """Merge processed rows into existing results while preserving previous rows.

        Behavior:
        - If out_path doesn't exist, write entire dataframe.
        - If existing file has the key column (`Abstract_Cleaned`), update matching rows
          and append new processed rows not already in the previous file.
        - If existing file lacks the key, append processed rows and drop duplicates
          by `Abstract_Cleaned` if available (keep previous rows).
        """
        try:
            key = "Abstract_Cleaned"
            cols_to_update = ["LLM_Returned_ID", "RA2025_Final_ID", "LLM_Reason", "Already_Classified"]

            # Work on processed subset (rows that have been classified in this run)
            processed = ieee_df[ieee_df.get("Already_Classified") == True].copy()

            # If no processed rows, just leave prev as-is (or write ieee_df if no prev)
            if not os.path.exists(out_path):
                ieee_df.to_excel(out_path, index=False)
                return

            try:
                prev = pd.read_excel(out_path)
            except Exception:
                # Can't read prev: write full current df
                ieee_df.to_excel(out_path, index=False)
                return

            # If prev has the key column, perform index-based update + append
            if key in prev.columns:
                prev_idx = prev.set_index(key)
                proc_idx = processed.set_index(key) if (not processed.empty and key in processed.columns) else processed

                # ensure update columns exist and update via dtype-safe helper
                for c in cols_to_update:
                    if c in getattr(proc_idx, 'columns', []):
                        safe_update_column(prev_idx, proc_idx, c)

                # Append any processed rows that are new (not present in prev)
                try:
                    new_idx = proc_idx.index.difference(prev_idx.index)
                except Exception:
                    new_idx = []

                if len(new_idx) > 0:
                    to_append = proc_idx.loc[new_idx]
                    merged_idx = pd.concat([prev_idx, to_append], axis=0)
                else:
                    merged_idx = prev_idx

                merged = merged_idx.reset_index()
            else:
                # prev doesn't have Abstract_Cleaned: preserve previous rows and append processed ones
                try:
                    merged = pd.concat([prev, processed], ignore_index=True)
                    # if Abstract_Cleaned exists, drop duplicates keeping existing prev rows
                    if "Abstract_Cleaned" in merged.columns:
                        merged = merged.drop_duplicates(subset=["Abstract_Cleaned"], keep="first")
                except Exception:
                    merged = ieee_df

            # final write
            merged.to_excel(out_path, index=False)
        except Exception:
            try:
                ieee_df.to_excel(out_path, index=False)
            except Exception:
                pass


    def classify_all(ieee_df: pd.DataFrame, ra_df: pd.DataFrame, reasoner: LLMReasoner, save_every: int):
        """Main loop: classify rows, update dataframe in-place, collect mismatches."""
        results = []
        mismatches = []

        valid_ra_ids = set(map(str, ra_df["RA2025"].tolist()))

        out_path_local = os.path.join(RESULTS_DIR, "classified_articles_llm.xlsx")
        mismatches_out_local = os.path.join(RESULTS_DIR, "llm_mismatches.csv")

        try:
            for i, (_, row) in enumerate(tqdm(ieee_df.iterrows(), total=len(ieee_df), desc="Classifying with LLM")):
                paper_index = row.name

                if ieee_df.at[paper_index, "Already_Classified"]:
                    continue

                abstract = row.get("Abstract_Cleaned")
                candidates = row.get("Candidates") or []

                # Top cosine similarity (normalized to [0,1]) and programmes from best candidate
                top_sim = None
                pred_primary = None
                pred_secondary = None
                if isinstance(candidates, list) and len(candidates) > 0:
                    try:
                        top_sim = float(candidates[0].get("Similarity", np.nan))
                    except Exception:
                        top_sim = None
                    pred_primary = candidates[0].get("Primary_Programme")
                    pred_secondary = candidates[0].get("Secondary_Programme")

                # Normalize cosine to [0,1]
                top_sim_norm = None
                if top_sim is not None and not np.isnan(top_sim):
                    top_sim_norm = (top_sim + 1.0) / 2.0
                    top_sim_norm = float(np.clip(top_sim_norm, 0.0, 1.0))

                res = reasoner.classify_with_reasoning(abstract, candidates)
                if isinstance(res, tuple) and len(res) == 3:
                    ra_id, reason, llm_conf = res
                elif isinstance(res, tuple) and len(res) == 2:
                    ra_id, reason = res
                    llm_conf = None
                else:
                    ra_id, reason, llm_conf = None, str(res), None

                ra_id_str = str(ra_id) if ra_id is not None else None
                exists = ra_id_str in valid_ra_ids if ra_id_str is not None else False

                if not exists:
                    reason = f"[Invalid RA ID returned: {ra_id_str}] {reason}"
                    mismatches.append({
                        "paper_index": paper_index,
                        "returned_ra_id": ra_id_str,
                        "exists": exists,
                        "reason": reason,
                        "candidates": json.dumps(candidates)
                    })

                # Combined confidence: average of available components
                parts = [p for p in [top_sim_norm, llm_conf] if p is not None and not (isinstance(p, float) and np.isnan(p))]
                combined_conf = float(np.mean(parts)) if parts else np.nan
                threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
                low_conf = False
                if parts and combined_conf < threshold:
                    low_conf = True

                # update row in-place
                ieee_df.at[paper_index, "LLM_Returned_ID"] = ra_id_str
                ieee_df.at[paper_index, "RA2025_Final_ID"] = None if (not exists or low_conf) else ra_id_str
                ieee_df.at[paper_index, "LLM_Reason"] = reason
                ieee_df.at[paper_index, "Already_Classified"] = True
                ieee_df.at[paper_index, "Top_Cosine_Sim"] = top_sim
                ieee_df.at[paper_index, "LLM_Confidence"] = llm_conf
                ieee_df.at[paper_index, "Combined_Confidence"] = combined_conf
                ieee_df.at[paper_index, "Predicted_Primary_Programme"] = pred_primary
                ieee_df.at[paper_index, "Predicted_Secondary_Programme"] = pred_secondary
                ieee_df.at[paper_index, "Low_Confidence_Flag"] = bool(low_conf)

                results.append((ra_id_str, None if (not exists or low_conf) else ra_id_str, reason, combined_conf))

                # periodic flush
                if (i + 1) % save_every == 0:
                    try:
                        _write_merged(ieee_df, out_path_local)
                    except Exception:
                        pass
                    try:
                        pd.DataFrame(mismatches).to_csv(mismatches_out_local, index=False)
                    except Exception:
                        pass

        except KeyboardInterrupt:
            print("\n⏸️  Interrupted by user (Ctrl+C). Saving progress...")

        # final write
        try:
            _write_merged(ieee_df, out_path_local)
        except Exception:
            try:
                ieee_df.to_excel(out_path_local, index=False)
            except Exception:
                pass

        try:
            mismatches_df = pd.DataFrame(mismatches)
            if not mismatches_df.empty:
                mismatches_df.to_csv(mismatches_out_local, index=False)
                print(f"⚠️ Some LLM outputs did not match RA IDs. See {mismatches_out_local}")
        except Exception:
            if mismatches:
                with open(mismatches_out_local, "w", encoding="utf-8") as fh:
                    fh.write("paper_index,returned_ra_id,exists,reason,candidates\n")
                    for m in mismatches:
                        line = f"{m['paper_index']},{m['returned_ra_id']},{m['exists']},\"{m['reason']}\",\"{m['candidates']}\"\n"
                        fh.write(line)

        print(f"✅ Classification complete. Results saved to {out_path_local}")
        return results, mismatches


    # ---- Execute refactored workflow ----
    ieee_df = pd.read_csv(os.path.join(DATA_DIR, "abstracts_cleaned.csv"))
    ra_df = pd.read_csv(os.path.join(DATA_DIR, "ra_questions_cleaned.csv"))

    print(f"  → Loaded {len(ieee_df)} abstracts and {len(ra_df)} RA questions")

    ieee_df = prepare_dataframe(ieee_df)

    print("🔹 Initializing embedding model and building RA embeddings...")
    embedder, classifier = init_models(ra_df)

    print(f"🔹 Retrieving top-{TOP_K} candidates for unclassified abstracts...")
    # ensure column exists
    if "Candidates" not in ieee_df.columns:
        ieee_df["Candidates"] = None

    mask = ~ieee_df["Already_Classified"].astype(bool)
    if mask.any():
        ieee_df.loc[mask, "Candidates"] = ieee_df.loc[mask, "Abstract_Cleaned"].apply(
            lambda abs_text: classifier.get_top_k_candidates(abs_text, top_k=TOP_K)
        )
    else:
        print("ℹ️ All rows are already classified; skipping candidate retrieval.")

    print("🔹 Running LLM reasoning for final classification...")
    llm_model, rpm = load_config(CFG_PATH)
    
    reasoner = LLMReasoner(model=llm_model, requests_per_minute=rpm)

    SAVE_EVERY = int(os.getenv("SAVE_EVERY", 50))
    results, mismatches = classify_all(ieee_df, ra_df, reasoner, SAVE_EVERY)

def safe_update_column(prev_df: pd.DataFrame, new_df: pd.DataFrame, col: str) -> None:
    """
    Safely update prev_df[col] with new_df[col], coercing/aligning dtypes to avoid
    pandas FutureWarning about incompatible dtype assignment.

    Rules:
      - Float dest: coerce incoming to numeric, cast to dest float dtype, assign where not NA.
      - Integer dest: upcast dest to float (to allow NaN), then handle as float.
      - Boolean dest: coerce incoming to nullable boolean, assign where not NA.
      - Object dest: assign incoming where not NA.
    """
    if col not in prev_df.columns or col not in new_df.columns:
        return

    # Align source to destination index
    src = new_df[col].reindex(prev_df.index)
    dest_dtype = prev_df[col].dtype

    # If destination is integer and may need NaN, upcast to float
    if pd.api.types.is_integer_dtype(dest_dtype):
        # Prefer float to retain NaN semantics safely
        prev_df[col] = pd.to_numeric(prev_df[col], errors='coerce').astype(float)
        dest_dtype = prev_df[col].dtype  # refresh to float

    # Float or numeric destination: coerce source to numeric and cast to dest dtype
    if pd.api.types.is_float_dtype(dest_dtype):
        numeric = pd.to_numeric(src, errors='coerce')
        # Explicitly cast incoming to destination float dtype to satisfy pandas
        numeric = numeric.astype(dest_dtype)
        mask = numeric.notna()
        if mask.any():
            prev_df.loc[mask, col] = numeric[mask]
        return

    # Boolean destination (numpy bool_ or pandas nullable boolean)
    if pd.api.types.is_bool_dtype(dest_dtype) or str(dest_dtype) == "BooleanDtype":
        # Coerce to pandas nullable boolean to preserve NA
        src_bool = src.astype("boolean")
        mask = src_bool.notna()
        if mask.any():
            prev_df.loc[mask, col] = src_bool[mask]
        return

    # Object or other destination: assign as-is where not NA
    mask = src.notna()
    if mask.any():
        prev_df.loc[mask, col] = src[mask]


if __name__ == "__main__":
    main()