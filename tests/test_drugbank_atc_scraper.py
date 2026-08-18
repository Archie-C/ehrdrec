import json

import pytest

from utils import drugbank_atc_scraper as scraper


SAMPLE_TREE = [
    {
        "id": "J05AR13",
        "text": "Lamivudine, abacavir and dolutegravir (J05AR13)",
        "state": {"opened": True, "selected": True},
        "children": [
            {
                "id": "J05AR-DB00709",
                "text": "<strong>Lamivudine</strong>",
                "a_attr": {"href": "/drugs/DB00709"},
            },
            {
                "id": "J05AR-DB01048",
                "text": "<strong>Abacavir</strong>",
                "a_attr": {"href": "/drugs/DB01048"},
            },
            {
                "id": "J05AR-DB08930",
                "text": "<strong>Dolutegravir</strong>",
                "a_attr": {"href": "/drugs/DB08930"},
            },
        ],
    }
]


def test_selected_drugs_extracts_names_and_links() -> None:
    assert scraper.selected_drugs(SAMPLE_TREE) == {
        "J05AR13": [
            {"name": "Lamivudine", "href": "/drugs/DB00709"},
            {"name": "Abacavir", "href": "/drugs/DB01048"},
            {"name": "Dolutegravir", "href": "/drugs/DB08930"},
        ]
    }


def test_scrape_requires_the_requested_selected_node(monkeypatch) -> None:
    monkeypatch.setattr(scraper, "fetch_atc_tree", lambda code, timeout: SAMPLE_TREE)
    with pytest.raises(scraper.DrugBankScraperError, match="J05AR14"):
        scraper.scrape_atc_codes(["j05ar14"])


def test_write_mapping(tmp_path) -> None:
    output = tmp_path / "nested" / "mapping.json"
    mapping = scraper.selected_drugs(SAMPLE_TREE, absolute_urls=True)
    scraper.write_mapping(mapping, output)
    assert json.loads(output.read_text()) == mapping



def test_read_atc_codes_deduplicates_and_normalises(tmp_path) -> None:
    csv_path = tmp_path / "mapping.csv"
    csv_path.write_text(
        "drug_id,atc_code\nDDInter1,j05ar13\nDDInter2,J05AR13\nDDInter3,N07BB03\n"
    )
    assert scraper.read_atc_codes(csv_path) == ["J05AR13", "N07BB03"]


def test_scrape_atc_csv_saves_absolute_links_and_resumes(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "mapping.csv"
    csv_path.write_text("drug_id,atc_code\nDDInter1,J05AR13\nDDInter2,J05AR13\n")
    output = tmp_path / "drugs.json"
    calls = []

    def fake_fetch(code, timeout):
        calls.append((code, timeout))
        return SAMPLE_TREE

    monkeypatch.setattr(scraper, "fetch_atc_tree", fake_fetch)
    mapping, errors = scraper.scrape_atc_csv(csv_path, output, delay=0)
    assert not errors
    assert mapping["J05AR13"][0] == {
        "name": "Lamivudine",
        "href": "https://go.drugbank.com/drugs/DB00709",
    }
    assert calls == [("J05AR13", 20.0)]

    scraper.scrape_atc_csv(csv_path, output, delay=0)
    assert calls == [("J05AR13", 20.0)]
