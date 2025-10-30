"""
Module: embeddings.py
Embeds texts into semantic vectors using SentenceTransformer.
"""

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


class EmbeddingModel:
    """Encapsulates a sentence-transformer model for semantic similarity tasks.

    If `sentence-transformers` is not available this class provides a lightweight
    fallback that produces simple numeric vectors so the rest of the pipeline can
    run (useful for tests and early development).
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        if _HAS_ST:
            print(f"🔸 Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            print("⚠️  sentence-transformers not found — using fallback embeddings")
            self.model = None

    def embed_texts(self, texts):
        """Generate embeddings for a list of texts.

        Returns a tensor-like object when sentence-transformers is available,
        otherwise returns a list of numeric lists.
        """
        if _HAS_ST and self.model is not None:
            return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        # fallback: use simple length-based features
        return [[float(len(str(t))) for _ in range(8)] for t in texts]

    def cosine_similarities(self, query_emb, corpus_embs):
        """Return cosine similarity scores for a query against corpus.

        For a real model this delegates to sentence-transformers util.cos_sim.
        For the fallback it computes simple cosine similarity using pure Python.
        """
        if _HAS_ST:
            return util.cos_sim(query_emb, corpus_embs)[0]

        # fallback: query_emb expected as list-of-lists with single element
        import math

        # normalize helper
        def norm(v):
            return math.sqrt(sum(x * x for x in v))

        q = list(query_emb[0]) if isinstance(query_emb, (list, tuple)) else list(query_emb)
        qn = norm(q)
        scores = []
        for vec in corpus_embs:
            v = list(vec)
            vn = norm(v)
            if qn == 0 or vn == 0:
                scores.append(0.0)
            else:
                dot = sum(a * b for a, b in zip(q, v))
                scores.append(dot / (qn * vn))
        return scores


def embed_texts(texts):
    """Compatibility helper: return embeddings as a Python list for quick tests.
    This keeps the previous simple API used by some tests/scripts.
    """
    model = EmbeddingModel()
    emb = model.embed_texts(texts)
    # If tensor, convert to list; else return as list
    try:
        return emb.tolist()
    except Exception:
        return list(emb)
