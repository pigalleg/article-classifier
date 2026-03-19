#!/usr/bin/env python3
"""
Postprocessing & analysis:
- Load + preprocess classification / affinity outputs
- Compute frequency, alignment & visualization metrics
Output artifacts written to ./outputs/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "data/results"
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------
# Classification (RA) section
# ---------------------------

def load_classification_sources():
    """Load classification Excel if present."""
    path = os.path.join(RESULTS_DIR, "classified_articles_llm.xlsx")
    if not os.path.exists(path):
        return None
    return pd.read_excel(path)


def prepare_classification_frequency(df: pd.DataFrame):
    """Prepare frequency table of RA2025_Final_ID."""
    if df is None or "RA2025_Final_ID" not in df.columns:
        return None
    freq = (df["RA2025_Final_ID"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "RA2025_ID", "RA2025_Final_ID": "Count"})
            .sort_values("RA2025_ID"))
    return freq


def compute_classification_frequency(freq: pd.DataFrame):
    """Compute + persist frequency outputs & plot."""
    if freq is None or freq.empty:
        print("ℹ️ No classification frequency to compute.")
        return
    freq_path = os.path.join(OUTPUTS_DIR, "ra_frequency.csv")
    freq.to_csv(freq_path, index=False)
    print(f"✅ Classification frequency saved: {freq_path}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=freq, x="RA2025_ID", y="Count", color="steelblue")
    plt.title("Distribution of Classified Papers by RA2025 Question")
    plt.xlabel("RA2025 ID")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    out_img = os.path.join(OUTPUTS_DIR, "ra_distribution_llm.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"✅ Classification frequency plot: {out_img}")


def optional_similarity_histogram(df: pd.DataFrame):
    """If raw candidate similarity info exists, plot distribution."""
    if df is None or "Candidates" not in df.columns:
        return
    sims = []
    for val in df["Candidates"].dropna():
        if isinstance(val, str):
            try:
                items = json.loads(val)
                for c in items:
                    if isinstance(c, dict) and "Similarity" in c:
                        sims.append(c["Similarity"])
            except Exception:
                continue
        elif isinstance(val, list):
            for c in val:
                if isinstance(c, dict) and "Similarity" in c:
                    sims.append(c["Similarity"])
    if not sims:
        return
    plt.figure(figsize=(6, 4))
    sns.histplot(sims, bins=20, kde=True)
    plt.title("Candidate Similarity Scores")
    plt.xlabel("Cosine Similarity")
    plt.tight_layout()
    out_img = os.path.join(OUTPUTS_DIR, "similarity_histogram.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"✅ Similarity histogram: {out_img}")


def export_classification_summary(df: pd.DataFrame, freq: pd.DataFrame):
    """Export combined Excel summary."""
    if df is None:
        return
    out_xlsx = os.path.join(OUTPUTS_DIR, "classification_summary_llm.xlsx")
    with pd.ExcelWriter(out_xlsx) as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)
        if freq is not None:
            freq.to_excel(writer, sheet_name="Frequency", index=False)
    print(f"✅ Classification summary Excel: {out_xlsx}")


# ---------------------------
# Affinity alignment section
# ---------------------------

def load_affinity_sources():
    """Load RA & PRP affinity CSVs and RA question metadata; return None if missing."""
    ra_aff = os.path.join(RESULTS_DIR, "ra_affinities.csv")
    prp_aff = os.path.join(RESULTS_DIR, "prp_affinities.csv")
    ra_q = os.path.join(PROCESSED_DIR, "ra_questions_cleaned.csv")
    if not (os.path.exists(ra_aff) and os.path.exists(prp_aff) and os.path.exists(ra_q)):
        return None, None, None
    return pd.read_csv(ra_aff), pd.read_csv(prp_aff), pd.read_csv(ra_q)


def prepare_affinity_alignment(ra_aff: pd.DataFrame, prp_aff: pd.DataFrame, ra_q: pd.DataFrame):
    """Build merged table with alignment metrics; returns merged or None."""
    if any(x is None for x in [ra_aff, prp_aff, ra_q]):
        return None

    prp_col = next((c for c in ["Primary Research Programme", "Primary_Programme"] if c in ra_q.columns), None)
    if prp_col is None or "RA2025" not in ra_q.columns:
        print("⚠️ Missing PRP or RA2025 columns; alignment skipped.")
        return None

    if "Abstract_Index" not in ra_aff.columns or "Abstract_Index" not in prp_aff.columns:
        print("⚠️ Missing Abstract_Index in affinity files; alignment skipped.")
        return None
    if "PRP_Name" not in prp_aff.columns:
        print("⚠️ Missing PRP_Name in prp_affinities; alignment skipped.")
        return None

    ra_prp_map = ra_q.set_index("RA2025")[prp_col].to_dict()

    # Top RA per abstract
    ra_aff["_sort_score"] = ra_aff["LLM_Affinity"].fillna(ra_aff.get("Cosine_x100", np.nan)).fillna(-1e9)
    top_ra = (ra_aff.sort_values(["Abstract_Index", "_sort_score"], ascending=[True, False])
                    .groupby("Abstract_Index", as_index=False)
                    .first())
    top_ra["Selected_PRP"] = top_ra["RA2025_ID"].map(ra_prp_map)
    # Top PRP per abstract
    top_prp = (prp_aff.sort_values(["Abstract_Index", "LLM_Affinity"], ascending=[True, False])
                      .groupby("Abstract_Index", as_index=False)
                      .first()
                      .rename(columns={"PRP_Name": "Direct_Top_PRP", "LLM_Affinity": "Direct_Top_PRP_Aff"}))

    merged = top_ra.merge(top_prp, on="Abstract_Index", how="left")

    # Rank & delta calculations
    def rank_selected(row):

        subset = prp_aff[prp_aff["Abstract_Index"] == row["Abstract_Index"]].sort_values("LLM_Affinity", ascending=False)
        names = subset["PRP_Name"].tolist()
        try:
            return names.index(row["Selected_PRP"]) + 1
        except ValueError:
            return np.nan

    def selected_prp_aff(row):
        subset = prp_aff[(prp_aff["Abstract_Index"] == row["Abstract_Index"]) &
                         (prp_aff["PRP_Name"] == row["Selected_PRP"])]
        return subset["LLM_Affinity"].iloc[0] if not subset.empty else np.nan
    merged["Selected_PRP_Rank"] = merged.apply(rank_selected, axis=1)
    merged["Selected_PRP_Aff"] = merged.apply(selected_prp_aff, axis=1)
    merged["Delta_vs_Best"] = merged["Selected_PRP_Aff"] - merged["Direct_Top_PRP_Aff"]
    merged["PRP_Match"] = merged["Selected_PRP"] == merged["Direct_Top_PRP"]
    return merged


def compute_affinity_alignment(merged: pd.DataFrame):
    """Persist alignment metrics & generate plots."""
    if merged is None or merged.empty:
        print("ℹ️ No affinity alignment to compute.")
        return

    out_csv = os.path.join(OUTPUTS_DIR, "affinity_alignment_analysis.csv")
    merged.to_csv(out_csv, index=False)
    print(f"✅ Alignment table: {out_csv}")

    match_rate = merged["PRP_Match"].mean()
    print(f"   • Match rate: {match_rate:.3f}")

    if merged["Selected_PRP_Rank"].notna().any():
        print(f"   • Rank stats:\n{merged['Selected_PRP_Rank'].describe()}")

    if merged["Delta_vs_Best"].notna().any():
        print(f"   • Δ mean: {merged['Delta_vs_Best'].mean():.2f}")

    # Rank histogram
    if merged["Selected_PRP_Rank"].notna().any():
        plt.figure(figsize=(8, 4))
        sns.histplot(merged["Selected_PRP_Rank"].dropna(), bins=range(1, int(np.nanmax(merged["Selected_PRP_Rank"])) + 2))
        plt.title("Selected PRP Rank Distribution")
        plt.xlabel("Rank (1=best)")
        plt.ylabel("Count")
        plt.tight_layout()
        out_img = os.path.join(OUTPUTS_DIR, "prp_selected_rank_hist.png")
        plt.savefig(out_img, dpi=300)
        plt.close()
        print(f"✅ {out_img}")

    # Δ histogram
    if merged["Delta_vs_Best"].notna().any():
        plt.figure(figsize=(8, 4))
        sns.histplot(merged["Delta_vs_Best"].dropna(), bins=30, kde=True)
        plt.title("Δ Selected PRP Affinity vs Direct Top PRP")
        plt.xlabel("Δ (Selected − Direct Top)")
        plt.ylabel("Count")
        plt.tight_layout()
        out_img = os.path.join(OUTPUTS_DIR, "prp_delta_vs_best_hist.png")
        plt.savefig(out_img, dpi=300)
        plt.close()
        print(f"✅ {out_img}")

    # Confusion heatmap
    conf = (merged.groupby(["Selected_PRP", "Direct_Top_PRP"])
                  .size()
                  .unstack(fill_value=0))
    if not conf.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf, cmap="Blues", cbar=True)
        plt.title("PRP Mapping Consistency")
        plt.xlabel("Direct Top PRP")
        plt.ylabel("RA-derived PRP")
        plt.tight_layout()
        out_img = os.path.join(OUTPUTS_DIR, "prp_confusion_heatmap.png")
        plt.savefig(out_img, dpi=300)
        plt.close()
        conf.to_csv(os.path.join(OUTPUTS_DIR, "prp_confusion_matrix.csv"))
        print(f"✅ {out_img} & prp_confusion_matrix.csv")


# ---------------------------
# Main orchestration
# ---------------------------

def main():
    # Classification frequency
    # cls_df = load_classification_sources()
    # freq_table = prepare_classification_frequency(cls_df)
    # compute_classification_frequency(freq_table)
    # optional_similarity_histogram(cls_df)
    # export_classification_summary(cls_df, freq_table)

    # Affinity alignment
    ra_aff, prp_aff, ra_q = load_affinity_sources()
    merged_alignment = prepare_affinity_alignment(ra_aff, prp_aff, ra_q)
    compute_affinity_alignment(merged_alignment)


if __name__ == "__main__":
    main()
