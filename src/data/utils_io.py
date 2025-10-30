"""utils_io.py
File I/O helpers for reading/writing CSV/XLSX.
"""
from pathlib import Path
import pandas as pd

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df, path: Path):
    df.to_csv(path, index=False)
