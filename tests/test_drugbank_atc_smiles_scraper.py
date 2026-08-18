import json

import pytest

from utils import drugbank_atc_smiles_scraper as scraper


DRUGS = {
    "J05AR13": [
        {"name": "Dolutegravir", "href": "https://go.drugbank.com/drugs/DB08930"},
        {"name": "Lamivudine", "href": "/drugs/DB00709"},
    ],
    "J05AR25": [
        {"name": "Dolutegravir", "href": "/drugs/DB08930"},
    ],
}


def test_drug_id_from_href() -> None:
    assert scraper.drug_id_from_href("https://go.drugbank.com/drugs/DB08930") == "DB08930"
    with pytest.raises(ValueError):
        scraper.drug_id_from_href("https://example.com/no-id")


def test_build_mapping_preserves_multiple_smiles_per_atc() -> None:
    result = scraper.build_atc_smiles_mapping(
        DRUGS, {"DB08930": "SMILES-1", "DB00709": "SMILES-2"}
    )
    assert [record["smiles"] for record in result["J05AR13"]] == [
        "SMILES-1",
        "SMILES-2",
    ]
    assert result["J05AR25"][0]["drug_id"] == "DB08930"


def test_scrape_fetches_each_drug_once_and_resumes(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "drugs.json"
    output_path = tmp_path / "smiles.json"
    input_path.write_text(json.dumps(DRUGS))
    calls = []

    def fake_fetch(drug_id, timeout):
        calls.append((drug_id, timeout))
        return f"SMILES-{drug_id}"

    monkeypatch.setattr(scraper, "fetch_smiles", fake_fetch)
    mapping, errors = scraper.scrape_smiles_mapping(
        input_path, output_path, delay=0, checkpoint_every=1
    )
    assert not errors
    assert calls == [("DB08930", 20.0), ("DB00709", 20.0)]
    assert mapping["J05AR13"][0]["smiles"] == "SMILES-DB08930"

    scraper.scrape_smiles_mapping(input_path, output_path, delay=0)
    assert calls == [("DB08930", 20.0), ("DB00709", 20.0)]
