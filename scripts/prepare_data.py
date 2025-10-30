#!/usr/bin/env python3
"""
Prepare and clean IEEE paper abstracts and RA2025 questions.
Outputs:
    - data/processed/abstracts_cleaned.csv
    - data/processed/ra_questions_cleaned.csv
"""

import os
import sys
import pandas as pd
from src.data.preprocess import clean_text


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def main():
        print("🔹 Loading raw data...")
        ieee = pd.read_excel(os.path.join(RAW_DIR, "IEEE_TPS_2024_mapped.xlsx"), sheet_name="main")
        ra = pd.read_excel(os.path.join(RAW_DIR, "Research Agenda 2025.xlsx"), sheet_name="main")

        # --- Clean IEEE abstracts ---
        print("🔹 Cleaning IEEE abstracts...")
        ieee = ieee[["Document Title", "Abstract"]].dropna().reset_index(drop=True)
        ieee["Abstract_Cleaned"] = ieee["Abstract"].apply(clean_text)

        # --- Clean RA questions ---
        print("🔹 Cleaning RA2025 questions...")
        ra = ra[["RA2025", "Questions - long"]].dropna().reset_index(drop=True)
        ra["RA2025"] = ra["RA2025"].astype(str)
        ra["Question_Cleaned"] = ra["Questions - long"].apply(clean_text)

        # --- Save outputs ---
        ieee_out = os.path.join(PROCESSED_DIR, "abstracts_cleaned.csv")
        ra_out = os.path.join(PROCESSED_DIR, "ra_questions_cleaned.csv")

        # after building/cleaning abstracts_df but before writing to disk ensure flag column is present
        if "Already_Classified" not in ieee.columns:
            ieee["Already_Classified"] = False

        ieee.to_csv(ieee_out, index=False)
        ra.to_csv(ra_out, index=False)

        print(f"✅ Cleaned abstracts saved to {ieee_out}")
        print(f"✅ Cleaned RA questions saved to {ra_out}")

if __name__ == "__main__":
        main()

if __name__ == '__main__':
    print('Prepare data (stub)')
