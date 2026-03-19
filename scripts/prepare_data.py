#!/usr/bin/env python3
"""
Prepare and clean IEEE paper abstracts and RA2025 questions.
Outputs:
    - data/processed/abstracts_cleaned.csv
    - data/processed/ra_questions_cleaned.csv
    - data/processed/primary_programmes.xlsx
"""

import os
import pandas as pd
from src.data.preprocess import clean_text


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_primary_research_programmes():
    """Cleans the Primary Research Programme sheet if present."""
    try:
        pri_raw = pd.read_excel(
            os.path.join(RAW_DIR, "Research Agenda 2025.xlsx"),
            sheet_name="Primary Research Programme"
        )
        pri_cols = ["Primary Research Programme", "Long name", "Description"]
        pri_available = [c for c in pri_cols if c in pri_raw.columns]
        pri = pri_raw[pri_available].dropna(subset=["Primary Research Programme"]).reset_index(drop=True)

        # Normalize column names
        rename_map = {}
        if "Primary Research Programme" in pri.columns:
            rename_map["Primary Research Programme"] = "Primary_Programme"
        if "Long name" in pri.columns:
            rename_map["Long name"] = "Long_Name"
        if "Description" in pri.columns:
            rename_map["Description"] = "Description"
        pri = pri.rename(columns=rename_map)

        # Clean description if present
        if "Description" in pri.columns:
            pri["Description_Cleaned"] = pri["Description"].astype(str).apply(clean_text)

        # Persist to CSV for downstream affinity script
        pri_out = os.path.join(PROCESSED_DIR, "primary_programmes.csv")
        pri.to_csv(pri_out, index=False)
        print(f"✅ Cleaned Primary Research Programme saved to {pri_out}")
    except Exception:
        # sheet may be missing; continue
        pass

def clean_RA_questions():
    """Cleans the RA2025 questions sheet."""
    print("🔹 Cleaning RA2025 questions...")
    ra = pd.read_excel(os.path.join(RAW_DIR, "Research Agenda 2025.xlsx"), sheet_name="main")
    ra_cols = [
        "RA2025",
        "Questions - long",
        "Primary Research Programme",
        "Secondary Research Programme",
    ]
    available_cols = [c for c in ra_cols if c in ra.columns]
    ra = ra[available_cols].dropna(subset=["RA2025", "Questions - long"]).reset_index(drop=True)
    ra["RA2025"] = ra["RA2025"].astype(str)
    ra["Question_Cleaned"] = ra["Questions - long"].apply(clean_text)
    ra_out = os.path.join(PROCESSED_DIR, "ra_questions_cleaned.csv")
    ra.to_csv(ra_out, index=False)
    print(f"✅ Cleaned RA questions saved to {ra_out}")

def clean_IEEE_abstracts():
    """Cleans the IEEE abstracts sheet."""
    ieee = pd.read_excel(os.path.join(RAW_DIR, "IEEE_TPS_2024_mapped.xlsx"), sheet_name="main")
    # --- Clean IEEE abstracts ---
    print("🔹 Cleaning IEEE abstracts...")

    ieee_cols  = ["index", "Document Title", "Abstract", "Author Keywords", "IEEE Terms", "Already_Classified"]
    ieee_cols = [c for c in ieee_cols if c in ieee.columns]

    ieee = ieee[ieee_cols].dropna(subset=["Abstract"]).reset_index(drop=True)
    if "index" in ieee.columns:
        ieee = ieee.rename(columns={"index": "Source_Index"})
    ieee["Abstract_Cleaned"] = ieee["Abstract"].apply(clean_text)
    if "Author Keywords" in ieee.columns and "IEEE Terms" in ieee.columns:
        ieee["Keywords_Cleaned"] = ieee[["Author Keywords", "IEEE Terms"]].fillna("").agg("; ".join, axis=1)
        ieee["Keywords_Cleaned"] = ieee["Keywords_Cleaned"].apply(clean_text)
    if "Already_Classified" not in ieee.columns:
        ieee["Already_Classified"] = False

    ieee_out = os.path.join(PROCESSED_DIR, "abstracts_cleaned.csv")
    ieee.to_csv(ieee_out, index=False)
    print(f"✅ Cleaned abstracts saved to {ieee_out}")
    
def main():
    print("🔹 Loading raw data...")
    clean_RA_questions()
    clean_IEEE_abstracts()  
    clean_primary_research_programmes()

if __name__ == "__main__":
    main()
