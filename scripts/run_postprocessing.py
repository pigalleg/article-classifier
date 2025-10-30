#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Analyze and visualize LLM-based classification results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "data/results"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def main():
    # Load classification results
    path = os.path.join(RESULTS_DIR, "classified_articles_llm.xlsx")
    df = pd.read_excel(path)
    print(f"🔹 Loaded {len(df)} classified abstracts")

    # --- Frequency by RA question ---
    freq = df["RA2025_Final_ID"].value_counts().reset_index()
    freq.columns = ["RA2025_ID", "Count"]
    freq = freq.sort_values("RA2025_ID")

    freq_path = os.path.join(OUTPUTS_DIR, "ra_frequency.csv")
    freq.to_csv(freq_path, index=False)
    print(f"✅ Frequency table saved to {freq_path}")

    # --- Bar plot ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=freq, x="RA2025_ID", y="Count", color="skyblue")
    plt.title("Distribution of Classified Papers by RA2025 Question (LLM-based)")
    plt.xlabel("RA2025 Question ID")
    plt.ylabel("Number of Papers")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "ra_distribution_llm.png"), dpi=300)
    plt.close()
    print("✅ Bar chart saved to outputs/ra_distribution_llm.png")

    # --- Similarity summary ---
    if "Candidates" in df.columns:
        sim_values = []
        for cand_list in df["Candidates"].dropna():
            if isinstance(cand_list, str):  # if stored as stringified JSON
                continue
            for c in cand_list:
                sim_values.append(c["Similarity"])
        if sim_values:
            plt.figure()
            sns.histplot(sim_values, bins=20, kde=True)
            plt.title("Distribution of Candidate Similarity Scores")
            plt.xlabel("Cosine Similarity")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUTS_DIR, "similarity_histogram.png"), dpi=300)
            plt.close()
            print("✅ Similarity histogram saved to outputs/similarity_histogram.png")

    # --- Summary export ---
    summary_path = os.path.join(OUTPUTS_DIR, "classification_summary_llm.xlsx")
    with pd.ExcelWriter(summary_path) as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)
        freq.to_excel(writer, sheet_name="Frequency", index=False)
    print(f"✅ Summary Excel exported to {summary_path}")

if __name__ == "__main__":
    main()
