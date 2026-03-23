#!/usr/bin/env python3
"""
Cluster keywords from data/processed/abstracts_cleaned.csv.
Saves data/processed/keyword_clusters.csv with:
- cluster_name: representative word
- keywords: semicolon-separated normalized keywords in the cluster

Env (optional):
    KEYWORD_EMBED_MODEL=<override models.embedding_model from src/config/settings.yaml>
  KEYWORD_SIM_THRESHOLD=0.75   # cosine similarity threshold (higher = fewer, larger clusters)
  KEYWORD_COL=Keywords_Cleand  # override input column
  KEYWORD_ALIAS_FILE=src/config/keyword_aliases.yaml  # optional manual synonym map
"""

import os
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("Please install 'sentence-transformers' (e.g., pip install sentence-transformers)") from e

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError as e:
    raise SystemExit("Please install 'scikit-learn' (e.g., pip install scikit-learn)") from e


PROCESSED_DIR = Path("data/processed")
INPUT_CSV = PROCESSED_DIR / "abstracts_cleaned.csv"
OUTPUT_CSV = PROCESSED_DIR / "keyword_clusters.csv"


def _settings_embedding_model(default: str = "all-MiniLM-L6-v2") -> str:
    if yaml is None:
        return default
    cfg_path = Path("src/config/settings.yaml")
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return cfg.get("models", {}).get("embedding_model") or default
    except Exception:
        return default

EMBED_MODEL = os.getenv("KEYWORD_EMBED_MODEL") or _settings_embedding_model()
SIM_THRESHOLD = float(os.getenv("KEYWORD_SIM_THRESHOLD", "0.75"))
DIST_THRESHOLD = 1.0 - SIM_THRESHOLD

KEYWORD_COL_ENV = os.getenv("KEYWORD_COL")
ALIAS_FILE = Path(os.getenv("KEYWORD_ALIAS_FILE", "src/config/keyword_aliases.yaml"))

STOPWORDS = {
    "and", "or", "of", "in", "on", "for", "to", "with", "by", "from", "at", "as",
    "using", "use", "via", "a", "an", "the", "based", "study", "system", "approach",
    "analysis", "model", "method", "methods", "results", "paper", "case"
}


def find_keywords_column(df: pd.DataFrame) -> str:
    if KEYWORD_COL_ENV and KEYWORD_COL_ENV in df.columns:
        return KEYWORD_COL_ENV
    candidates = ["Keywords_Cleand", "Keywords_Cleaned", "Keywords", "keywords"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a keywords column. Looked for: {candidates} (or set KEYWORD_COL)")


def normalize_keyword(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s/]", " ", s)     # keep alnum/_/whitespace and '/'
    s = re.sub(r"\s+", " ", s).strip()
    # light plural strip: words length >=4 ending in 's'
    s = re.sub(r"\b(\w{4,})s\b", r"\1", s)
    return s


def load_aliases(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # normalize keys and values
        return {normalize_keyword(k): normalize_keyword(v) for k, v in data.items()}
    except Exception:
        return {}


def extract_all_keywords(df: pd.DataFrame, col: str) -> List[str]:
    out: List[str] = []
    for raw in df[col].fillna(""):
        # primary split on ';'
        parts = str(raw).split(";")
        for p in parts:
            t = p.strip()
            if t:
                out.append(t)
    return out


def unique_with_counts(keywords: List[str]) -> Tuple[List[str], Counter]:
    counts = Counter(keywords)
    uniq = list(counts.keys())
    return uniq, counts


def embed_keywords(phrases: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(phrases, normalize_embeddings=True, show_progress_bar=True)
    return emb


def cluster_embeddings(emb: np.ndarray, dist_threshold: float) -> np.ndarray:
    # average linkage + cosine metric
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_threshold,
    )
    labels = clustering.fit_predict(emb)
    return labels


def representative_word(keywords: List[str]) -> str:
    token_counts = Counter()
    for kw in keywords:
        tokens = re.split(r"[^\w]+", kw)
        for t in tokens:
            if len(t) < 3 or t in STOPWORDS:
                continue
            token_counts[t] += 1
    if token_counts:
        return token_counts.most_common(1)[0][0]
    return ""


def medoid_phrase(indices: List[int], emb: np.ndarray, phrases: List[str]) -> str:
    centroid = emb[indices].mean(axis=0, keepdims=True)
    sims = (emb[indices] @ centroid.T).ravel()
    best_local = int(np.argmax(sims))
    return phrases[indices[best_local]]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)
    kw_col = find_keywords_column(df)

    raw_keywords = extract_all_keywords(df, kw_col)
    if not raw_keywords:
        OUTPUT_CSV.write_text("cluster_name,keywords\n", encoding="utf-8")
        print("No keywords found. Wrote empty header.")
        return

    # normalize and apply aliases
    norm = [normalize_keyword(k) for k in raw_keywords]
    aliases = load_aliases(ALIAS_FILE)
    if aliases:
        norm = [aliases.get(k, k) for k in norm]

    # deduplicate for embedding; keep frequency
    uniq, counts = unique_with_counts(norm)

    # filter out trivial tokens
    uniq = [u for u in uniq if u and u not in STOPWORDS]

    if not uniq:
        OUTPUT_CSV.write_text("cluster_name,keywords\n", encoding="utf-8")
        print("No usable keywords after normalization/stopword filtering.")
        return

    emb = embed_keywords(uniq, EMBED_MODEL)
    labels = cluster_embeddings(emb, DIST_THRESHOLD)

    # group indices by cluster
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)

    rows = []
    for lab, idxs in groups.items():
        cluster_terms = [uniq[i] for i in idxs]
        name = representative_word(cluster_terms)
        if not name:
            name = normalize_keyword(medoid_phrase(idxs, emb, uniq))
        # sort terms by global frequency desc, then alpha
        cluster_terms_sorted = sorted(cluster_terms, key=lambda k: (-counts[k], k))
        rows.append({
            "cluster_name": name,
            "keywords": "; ".join(cluster_terms_sorted),
        })

    out_df = pd.DataFrame(rows).sort_values("cluster_name").reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {len(out_df)} clusters -> {OUTPUT_CSV}")
    print(f"Model={EMBED_MODEL}, sim_threshold={SIM_THRESHOLD:.2f} (dist={DIST_THRESHOLD:.2f}), source_col={kw_col}")


if __name__ == "__main__":
    main()