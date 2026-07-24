#!/usr/bin/env python3
"""
Prepare and clean IEEE paper abstracts and RA2025 questions.
Outputs:
    - data/processed/abstracts_cleaned.csv
    - data/processed/ra_questions_cleaned.csv
    - data/processed/primary_programmes.xlsx
"""

import os
from pathlib import Path

import pandas as pd
from src.data.preprocess import clean_text


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def _default_ieee_input_files() -> list[Path]:
    return [
        Path(RAW_DIR) / "TEC_2000_2026_ieee.csv",
        Path(RAW_DIR) / "TEMPR_2023_2026_ieee.csv",
        Path(RAW_DIR) / "TPWRD_2000_2026_ieee.csv",
        Path(RAW_DIR) / "TPWRS_2000_2026_ieee.csv",
        Path(RAW_DIR) / "TSG_2010_2026_ieee.csv",
        Path(RAW_DIR) / "TSTE_2010_2026_ieee.csv",
    ]


def _normalize_ieee_identifier(ieee: pd.DataFrame) -> pd.DataFrame:
    """Standardize the abstract identifier column to Source_Index."""
    identifier_columns = [column for column in ["doi", "index", "source_index"] if column in ieee.columns]

    if not identifier_columns:
        ieee = ieee.copy()
        ieee["Source_Index"] = ""
        return ieee

    ieee = ieee.copy()
    if "Source_Index" not in ieee.columns:
        ieee["Source_Index"] = pd.NA
    if "doi" in ieee.columns:
        ieee["Source_Index"] = ieee["Source_Index"].combine_first(ieee["doi"])
    if "index" in ieee.columns:
        ieee["Source_Index"] = ieee["Source_Index"].combine_first(ieee["index"])
    if "source_index" in ieee.columns:
        ieee["Source_Index"] = ieee["Source_Index"].combine_first(ieee["source_index"])

    ieee["Source_Index"] = ieee["Source_Index"].fillna("").astype(str).str.strip()
    ieee = ieee.drop(columns=[column for column in ["doi", "index", "source_index"] if column in ieee.columns])
    return ieee

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

def clean_IEEE_abstracts(input_files=None):
    """Cleans one or more IEEE abstracts files and merges them into a single output."""
    input_files = _default_ieee_input_files() if input_files is None else [Path(path) for path in input_files]
    print("🔹 Cleaning IEEE abstracts...")

    cleaned_frames = []
    for input_file in input_files:
        if input_file.suffix.lower() == ".xlsx":
            ieee = pd.read_excel(input_file, sheet_name="main")
        else:
            ieee = pd.read_csv(input_file)

        ieee.columns = ieee.columns.astype(str).str.strip().str.lower()

        ieee_cols = [
            column
            for column in [
                "doi",
                "index",
                "source_index",
                "title",
                "document title",
                "abstract",
                "authors",
                "author keywords",
                "ieee terms",
                "already_classified",
            ]
            if column in ieee.columns
        ]

        ieee = ieee[ieee_cols].copy()
        ieee["abstract"] = ieee["abstract"].fillna("").astype(str).str.strip()
        ieee = ieee[ieee["abstract"] != ""].reset_index(drop=True)
        if "authors" in ieee.columns:
            ieee["authors"] = ieee["authors"].fillna("").astype(str).str.strip()
            ieee = ieee[ieee["authors"] != ""].reset_index(drop=True)
        ieee = _normalize_ieee_identifier(ieee)
        if "document title" not in ieee.columns and "title" in ieee.columns:
            ieee["document title"] = ieee["title"]
        elif "document title" in ieee.columns and "title" in ieee.columns:
            ieee["document title"] = ieee["document title"].fillna(ieee["title"])
        ieee["Document Title"] = ieee["document title"]
        ieee["Abstract_Cleaned"] = ieee["abstract"].apply(clean_text)
        if "author keywords" in ieee.columns and "ieee terms" in ieee.columns:
            ieee["Keywords_Cleaned"] = ieee[["author keywords", "ieee terms"]].fillna("").agg("; ".join, axis=1)
            ieee["Keywords_Cleaned"] = ieee["Keywords_Cleaned"].apply(clean_text)
        if "already_classified" not in ieee.columns:
            ieee["already_classified"] = False
        ieee["Source_File"] = input_file.name
        ieee = ieee.drop(columns=[column for column in ["title", "document title"] if column in ieee.columns])
        cleaned_frames.append(ieee)

    if not cleaned_frames:
        raise FileNotFoundError(f"No IEEE input files found in {RAW_DIR}")

    ieee = pd.concat(cleaned_frames, ignore_index=True)
    column_order = [
        column
        for column in [
            "Source_Index",
            "Document Title",
            "abstract",
            "Abstract_Cleaned",
            "author keywords",
            "ieee terms",
            "Keywords_Cleaned",
            "already_classified",
            "Source_File",
        ]
        if column in ieee.columns
    ]
    remaining_columns = [column for column in ieee.columns if column not in column_order]
    ieee = ieee[column_order + remaining_columns]

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
