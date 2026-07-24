"""Utilities for generating PRP affinity files."""

from __future__ import annotations

import csv
from pathlib import Path


def generate_empty_prp_affinities(output_path: Path, abstracts_path: Path, prp_path: Path) -> None:
    """Generate a fallback PRP affinities file with all abstract-PRP pairs but no LLM scores.
    
    This creates a cartesian product of all abstracts and PRP domains, useful when
    evaluating all possible abstract-PRP pairs without pre-computed affinities.
    
    Args:
        output_path: Path where the generated PRP affinities CSV will be written.
        abstracts_path: Path to abstracts CSV file (must have 'DOI', 'doi', or 'Source_Index' and 'Document Title' columns).
        prp_path: Path to PRP domains CSV file (must have 'Primary_Programme' and 'Description' columns).
    
    Raises:
        RuntimeError: If abstracts or PRP files cannot be read.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load abstracts
    abstracts_data = []
    try:
        with abstracts_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                abstracts_data.append(row)
    except Exception as e:
        raise RuntimeError(f"Failed to load abstracts from {abstracts_path}: {e}")
    
    # Load PRP domains
    prp_data = []
    try:
        with prp_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                prp_data.append(row)
    except Exception as e:
        raise RuntimeError(f"Failed to load PRP domains from {prp_path}: {e}")
    
    # Generate cartesian product
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "Abstract_Index",
            "Document Title",
            "PRP_Name",
            "PRP_Description",
            "LLM_Affinity",
            "LLM_Affinity_Reason",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        
        for abstract in abstracts_data:
            # Skip abstracts that are already classified
            already_classified = abstract.get("Already_Classified", "False").strip().lower()
            if already_classified in {"true", "1", "yes"}:
                continue
            
            abstract_index = abstract.get("DOI") or abstract.get("doi") or abstract.get("Source_Index", "")
            doc_title = abstract.get("Document Title", "")
            
            for prp in prp_data:
                prp_name = prp.get("Primary_Programme", "")
                prp_desc = prp.get("Description", "")
                
                writer.writerow({
                    "Abstract_Index": abstract_index,
                    "Document Title": doc_title,
                    "PRP_Name": prp_name,
                    "PRP_Description": prp_desc,
                    "LLM_Affinity": "",
                    "LLM_Affinity_Reason": "",
                })
