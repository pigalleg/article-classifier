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

def main():
    # Refactored workflow using helper functions for clarity
    def load_config(cfg_path: str):
        """Read LLM config from YAML and allow environment overrides."""
        model = os.getenv("OPENAI_MODEL")
        rpm = os.getenv("OPENAI_REQUESTS_PER_MINUTE")
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

        return ieee_df


    def init_models(ra_df: pd.DataFrame):
        """Initialize embedding/classifier/reasoner objects."""
        embedder = EmbeddingModel(model_name="all-mpnet-base-v2")
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

                # ensure update columns exist with safe dtypes and update values by index
                for c in cols_to_update:
                    if c not in getattr(proc_idx, 'columns', []):
                        continue

                    # Create column in prev if missing with suitable dtype
                    if c not in prev_idx.columns:
                        if c == "Already_Classified":
                            # pandas nullable boolean to allow NA
                            prev_idx[c] = pd.Series([pd.NA] * len(prev_idx), index=prev_idx.index, dtype="boolean")
                        else:
                            prev_idx[c] = pd.Series([None] * len(prev_idx), index=prev_idx.index, dtype=object)

                    # Ensure destination dtype compatible
                    if c == "Already_Classified":
                        if str(prev_idx[c].dtype) != "BooleanDtype":
                            try:
                                prev_idx[c] = prev_idx[c].astype("boolean")
                            except Exception:
                                # fallback to object if conversion fails
                                prev_idx[c] = prev_idx[c].astype(object)
                        incoming = pd.Series(proc_idx[c], copy=False)
                        try:
                            incoming = incoming.astype("boolean")
                        except Exception:
                            incoming = incoming.astype(object)
                    else:
                        # textual/object columns
                        if not pd.api.types.is_object_dtype(prev_idx[c].dtype):
                            try:
                                prev_idx[c] = prev_idx[c].astype(object)
                            except Exception:
                                pass
                        incoming = pd.Series(proc_idx[c], copy=False).astype(object)

                    # assign only to common indices where incoming is not null
                    common_idx = incoming.index.intersection(prev_idx.index)
                    if len(common_idx) > 0:
                        incoming_sub = incoming.loc[common_idx]
                        notna_mask = incoming_sub.notna()
                        if notna_mask.any():
                            prev_idx.loc[common_idx[notna_mask.values], c] = incoming_sub[notna_mask].values
                    # After (explicitly cast before assignment to keep dtype consistent):
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
                candidates = row.get("Candidates")
                ra_id, reason = reasoner.classify_with_reasoning(abstract, candidates)

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

                # update row in-place
                ieee_df.at[paper_index, "LLM_Returned_ID"] = ra_id_str
                ieee_df.at[paper_index, "RA2025_Final_ID"] = ra_id_str if exists else None
                ieee_df.at[paper_index, "LLM_Reason"] = reason
                ieee_df.at[paper_index, "Already_Classified"] = True

                results.append((ra_id_str, ra_id_str if exists else None, reason))

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