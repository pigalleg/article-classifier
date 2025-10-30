def test_embeddings_stub():
    from src.models.embeddings import embed_texts
    assert isinstance(embed_texts(["a"]), list)
