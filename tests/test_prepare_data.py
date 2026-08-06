import pandas as pd

from scripts.util import prepare_ra_and_ieee_data


def test_clean_ieee_abstracts_keeps_publication_year(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    input_file = raw_dir / "TEC_2000_2026_ieee.csv"
    pd.DataFrame(
        {
            "doi": ["10.1234/example"],
            "title": ["Sample title"],
            "abstract": ["Sample abstract text."],
            "authors": ["A. Author"],
            "publication_title": ["IEEE Transactions on Energy Conversion"],
            "publication_year": [2024],
        }
    ).to_csv(input_file, index=False)

    monkeypatch.setattr(prepare_ra_and_ieee_data, "RAW_DIR", str(raw_dir))
    monkeypatch.setattr(prepare_ra_and_ieee_data, "PROCESSED_DIR", str(processed_dir))

    prepare_ra_and_ieee_data.clean_IEEE_abstracts([input_file])

    output = pd.read_csv(processed_dir / "abstracts_cleaned.csv")

    assert "publication_year" in output.columns
    assert output.loc[0, "publication_year"] == 2024
    assert "publication_journal" in output.columns
    assert output.loc[0, "publication_journal"] == "TEC"
