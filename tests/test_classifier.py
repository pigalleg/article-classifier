def test_classifier_stub():
    from src.models.classifier import match_abstract_to_ras
    assert match_abstract_to_ras("", []) == []
