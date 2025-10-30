import os
import sys
import shutil
import importlib.util
import importlib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        # real implementation would load model


class RAClassifier:
    def __init__(self, embedder: EmbeddingModel, ra_df: pd.DataFrame, text_column: str = "Question_Cleaned"):
        self._ra_df = ra_df.reset_index(drop=True)
        self._text_column = text_column

    def get_top_k_candidates(self, abs_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for _, row in self._ra_df.head(top_k).iterrows():
            candidates.append({"RA2025": row.get("RA2025"), self._text_column: row.get(self._text_column)})
        return candidates


class LLMReasoner:
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    def classify_with_reasoning(self, abstract: str, candidates: List[Dict[str, Any]]) -> Tuple[str, str]:
        if not candidates:
            return ("", "")
        first = candidates[0]
        return (str(first.get("RA2025")), f"chosen {first.get('RA2025')}")


def test_partial_save_on_keyboard_interrupt(tmp_path, monkeypatch):
    # Use tmp_path as project root
    project_root = tmp_path
    monkeypatch.chdir(project_root)

    # Create minimal data/processed inputs
    data_proc = project_root / "data" / "processed"
    data_proc.mkdir(parents=True)
    abstracts = pd.DataFrame({"Abstract_Cleaned": [f"abstract {i}" for i in range(5)]})
    ra_questions = pd.DataFrame(
        {"RA2025": [1, 2, 3], "Question_Cleaned": ["q1", "q2", "q3"]}
    )
    abstracts.to_csv(data_proc / "abstracts_cleaned.csv", index=False)
    ra_questions.to_csv(data_proc / "ra_questions_cleaned.csv", index=False)

    # Load the script as a module
    # make a copy of the real scripts/run_classification.py into the tmp project so the test can import it
    repo_root = Path(__file__).resolve().parents[1]
    repo_script = repo_root / "scripts" / "run_classification.py"
    dest_scripts = Path(os.getcwd()) / "scripts"
    dest_scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy(repo_script, dest_scripts / "run_classification.py")

    # Ensure the real repository root is on sys.path so "src" imports inside the script resolve
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    script_path = os.path.join(os.getcwd(), "scripts", "run_classification.py")
    spec = importlib.util.spec_from_file_location("run_classification", script_path)
    rc = importlib.util.module_from_spec(spec)
    sys.modules["run_classification"] = rc
    spec.loader.exec_module(rc)

    # Replace heavy components with lightweight stubs
    class DummyEmbeddingModel:
        def __init__(self, model_name=None):
            self.model_name = model_name

    class DummyRAClassifier:
        def __init__(self, embedder, ra_df, text_column="Question_Cleaned"):
            self._ra_df = ra_df

        def get_top_k_candidates(self, abs_text, top_k=5):
            # return list-like candidates; classifier logic in test LLM stub doesn't depend on format
            return [{"RA2025": int(self._ra_df.iloc[0]["RA2025"])}]

    monkeypatch.setattr(rc, "EmbeddingModel", DummyEmbeddingModel)
    monkeypatch.setattr(rc, "RAClassifier", DummyRAClassifier)

    # Replace LLMReasoner.classify_with_reasoning to raise KeyboardInterrupt on 3rd call
    call_state = {"n": 0}

    def fake_classify(self, abstract, candidates):
        call_state["n"] += 1
        if call_state["n"] == 3:
            raise KeyboardInterrupt
        return ("1", f"reason-{call_state['n']}")

    monkeypatch.setattr(rc.LLMReasoner, "classify_with_reasoning", fake_classify, raising=False)

    # Run main (it should catch KeyboardInterrupt and still save partial results)
    rc.main()

    out_path = Path("data") / "results" / "classified_articles_llm.xlsx"
    assert out_path.exists(), f"{out_path} not written"

    df = pd.read_excel(out_path)
    filled = df["LLM_Returned_ID"].notna().sum()
    assert filled >= 2, f"Expected at least 2 processed rows, got {filled}"


def save_results(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # write Excel file
    df.to_excel(out_path, index=False)


def main():
    project_root = Path.cwd()
    data_proc = project_root / "data" / "processed"
    abstracts_path = data_proc / "abstracts_cleaned.csv"
    ra_questions_path = data_proc / "ra_questions_cleaned.csv"

    if not abstracts_path.exists():
        raise FileNotFoundError(f"Missing {abstracts_path}")
    if not ra_questions_path.exists():
        raise FileNotFoundError(f"Missing {ra_questions_path}")

    abstracts = pd.read_csv(abstracts_path)
    ra_questions = pd.read_csv(ra_questions_path)

    embedder = EmbeddingModel(model_name="default")
    classifier = RAClassifier(embedder, ra_questions, text_column="Question_Cleaned")
    reasoner = LLMReasoner(model_name="default")

    results = abstracts.copy()
    results["LLM_Returned_ID"] = None
    results["LLM_Reason"] = None

    try:
        for idx, row in results.iterrows():
            abstract_text = row.get("Abstract_Cleaned", "")
            candidates = classifier.get_top_k_candidates(abstract_text, top_k=5)
            returned_id, reason = reasoner.classify_with_reasoning(abstract_text, candidates)
            results.at[idx, "LLM_Returned_ID"] = returned_id
            results.at[idx, "LLM_Reason"] = reason
    except KeyboardInterrupt:
        # on interrupt, save partial results
        pass
    finally:
        out_path = project_root / "data" / "results" / "classified_articles_llm.xlsx"
        save_results(results, out_path)


if __name__ == "__main__":
    main()