"""
Module: classifier.py
Retrieves top-k candidate RA2025 questions based on cosine similarity.
"""

import hashlib
import os
from pathlib import Path

import pandas as pd
import numpy as np

class RAClassifier:
    """Retrieve top-k RA questions using embedding similarity."""

    def __init__(self, embedder, ra_df: pd.DataFrame, text_column: str = "Questions - long"):
        self.embedder = embedder
        self.ra_df = ra_df
        self.text_column = text_column
        self._cache_dir = Path(os.getenv("RA_EMBED_CACHE_DIR", "data/processed/cache"))
        self.primary_cols = [
            "Primary Research Programme",
            "Primary_Research_Programme",
            "Primary Programme",
            "Primary_Programme",
        ]
        self.secondary_cols = [
            "Secondary Research Programme",
            "Secondary_Research_Programme",
            "Secondary Programme",
            "Secondary_Programme",
        ]
        print("🔸 Building RA2025 question embeddings...")
        texts = [str(v) for v in ra_df[text_column].tolist()]
        self.ra_embeddings = self._load_or_build_ra_embeddings(texts)

    def _cache_file_for_texts(self, texts: list[str]) -> Path:
        model_name = getattr(self.embedder, "model_name", "unknown-model")
        joined = "\x1f".join(texts)
        digest = hashlib.sha256(f"{model_name}|{self.text_column}|{joined}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"ra_embeddings_{digest}.npy"

    @staticmethod
    def _to_numpy(embeddings):
        if hasattr(embeddings, "detach") and hasattr(embeddings, "cpu"):
            return embeddings.detach().cpu().numpy()
        if hasattr(embeddings, "numpy"):
            return embeddings.numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def _load_or_build_ra_embeddings(self, texts: list[str]):
        cache_file = self._cache_file_for_texts(texts)
        if cache_file.exists():
            print(f"🔸 Loading cached RA embeddings: {cache_file}")
            return np.load(cache_file, allow_pickle=False)

        embeddings = self.embedder.embed_texts(texts)
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self._to_numpy(embeddings), allow_pickle=False)
            print(f"🔸 Saved RA embeddings cache: {cache_file}")
        except Exception as exc:
            print(f"⚠️  Could not save RA embeddings cache: {exc}")
        return embeddings
        

    @staticmethod
    def _first_nonempty(row: pd.Series, cols: list):
        for c in cols:
            if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
                return str(row[c]).strip()
        return None

    def _programme_filtered_indices(self, top_programmes: list[str] | None) -> list[int]:
        if not top_programmes:
            return []
        wanted = {str(p).strip().lower() for p in top_programmes if str(p).strip()}
        if not wanted:
            return []

        matched: list[int] = []
        for idx in range(len(self.ra_df)):
            row = self.ra_df.iloc[idx]
            primary = self._first_nonempty(row, self.primary_cols)
            secondary = self._first_nonempty(row, self.secondary_cols)
            values = {str(v).strip().lower() for v in [primary, secondary] if v is not None}
            if values & wanted:
                matched.append(idx)
        return matched

    def _build_candidates(self, indices: list[int], cos_scores):
        candidates = []
        for idx_int in indices:
            row = self.ra_df.iloc[idx_int]

            sim_val = np.nan
            if cos_scores is not None:
                try:
                    sim_val = float(cos_scores[idx_int])
                except Exception:
                    sim_val = float(np.array(cos_scores)[idx_int])

            cand = {
                "RA2025_ID": str(row["RA2025"]),
                "Question": row[self.text_column],
                "Similarity": sim_val,
            }
            candidates.append(cand)
        return candidates

    def get_top_k_candidates(
        self,
        abstract: str,
        top_k: int = 5,
        retrieval_mode: str = "cosine",
        top_programmes: list[str] | None = None,
    ):
        """Return top-k RA candidates.

        retrieval_mode:
        - cosine: rank globally by cosine similarity
        - prp_filter: filter by top_programmes then rank by cosine within subset
        - prp_only: return first top_k rows in programme-filtered subset (no cosine ranking)
        """
        if not isinstance(abstract, str) or not abstract.strip():
            return []

        mode = str(retrieval_mode or "cosine").strip().lower()
        if mode not in {"cosine", "prp_filter", "prp_only"}:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

        filtered_indices = self._programme_filtered_indices(top_programmes)
        if mode == "prp_only":
            if not filtered_indices:
                return []
            return self._build_candidates(filtered_indices[:top_k], cos_scores=None)

        abs_emb = self.embedder.embed_texts([abstract])
        cos_scores = None
        # Prefer embedder-provided similarity if available
        try:
            cos_scores = self.embedder.cosine_similarities(abs_emb, self.ra_embeddings)
        except Exception:
            # fallback: use util if installed via numpy conversion
            try:
                from sentence_transformers import util  # type: ignore
                cos_scores = util.cos_sim(abs_emb, self.ra_embeddings)[0]
            except Exception:
                # final fallback: compute via numpy
                ra_embs = np.array(self.ra_embeddings)
                q = np.array(abs_emb[0]) if isinstance(abs_emb, (list, tuple)) else np.array(abs_emb)
                # cosine similarity
                denom = np.linalg.norm(ra_embs, axis=1) * (np.linalg.norm(q) + 1e-12)
                cos_scores = (ra_embs @ q) / denom

        cos_np = np.array(cos_scores)
        if mode == "prp_filter":
            if not filtered_indices:
                return []
            ranked = sorted(filtered_indices, key=lambda i: float(cos_np[i]), reverse=True)
            return self._build_candidates(ranked[:top_k], cos_scores=cos_np)

        ranked = cos_np.argsort()[::-1][:top_k]
        return self._build_candidates([int(i) for i in ranked], cos_scores=cos_np)


def match_abstract_to_ras(abstract: str, ra_texts: list):
    """Backward-compatible stub used by tests: when ra_texts is empty return []
    Otherwise, this helper does a naive string-match scoring as a fallback.
    """
    if not ra_texts:
        return []
    # naive fallback: score by shared token count
    scores = []
    tokens = set(abstract.lower().split()) if isinstance(abstract, str) else set()
    for i, t in enumerate(ra_texts):
        t_tokens = set(str(t).lower().split())
        score = len(tokens & t_tokens) / max(1, len(tokens | t_tokens))
        scores.append((i, float(score)))
    # sort desc
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
