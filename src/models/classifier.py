"""
Module: classifier.py
Retrieves top-k candidate RA2025 questions based on cosine similarity.
"""

import pandas as pd
import numpy as np

class RAClassifier:
    """Retrieve top-k RA questions using embedding similarity."""

    def __init__(self, embedder, ra_df: pd.DataFrame, text_column: str = "Questions - long"):
        self.embedder = embedder
        self.ra_df = ra_df
        self.text_column = text_column
        print("🔸 Building RA2025 question embeddings...")
        self.ra_embeddings = embedder.embed_texts(ra_df[text_column].tolist())

    def get_top_k_candidates(self, abstract: str, top_k: int = 5):
        """Return top-k RA question candidates with similarity scores."""
        if not isinstance(abstract, str) or not abstract.strip():
            return []

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

        # Ensure cos_scores is a numpy array or supports indexing
        try:
            # torch tensor has argsort with descending
            top_indices = cos_scores.argsort(descending=True)[:top_k]
        except Exception:
            cos_np = np.array(cos_scores)
            top_indices = cos_np.argsort()[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            idx_int = int(idx)
            sim_val = float(cos_scores[idx]) if not isinstance(cos_scores, (list, tuple)) else float(cos_scores[idx_int])
            candidates.append({
                "RA2025_ID": str(self.ra_df.iloc[idx_int]["RA2025"]),
                "Question": self.ra_df.iloc[idx_int][self.text_column],
                "Similarity": sim_val
            })
        return candidates


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
