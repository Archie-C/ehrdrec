from __future__ import annotations

import json

from rdkit.Chem import BRICS
import torch

from ehrdrec.requirements import ModelRequirement
from ehrdrec.tasks.medication_set_recommendation.molecular import (
    build_medication_molecule_projection,
    build_medication_substructure_matrix,
    build_molecular_graphs,
    build_molecular_resources,
    resolve_medication_molecules,
)
from ehrdrec.utils import Vocab


def _vocab(*codes: str) -> Vocab:
    tokens = ("PAD", "UNK", "SOS", "EOS", *codes)
    return Vocab(
        token_to_id={token: index for index, token in enumerate(tokens)},
        id_to_token={index: token for index, token in enumerate(tokens)},
    )


def _write_mapping(tmp_path, mapping):
    path = tmp_path / "atc_molecules.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def _record(drug_id: str, smiles: str) -> dict[str, str]:
    return {"drug_id": drug_id, "smiles": smiles}


def test_resolves_direct_atc5_deduplicates_and_skips_invalid(
    tmp_path,
    caplog,
) -> None:
    path = _write_mapping(
        tmp_path,
        {
            "J05AR13": [
                _record("DB1", "CCO"),
                _record("DB1-DUP", "OCC"),
                _record("DB2", "CC(=O)O"),
                _record("DB-BAD", "not-a-smiles"),
            ],
            "J05AR17": [_record("DB3", "c1ccccc1")],
        },
    )
    vocab = _vocab("J05AR13", "MISSING")

    resolved = resolve_medication_molecules(
        medication_vocab=vocab,
        mapping_path=path,
        atc_level=5,
    )

    direct_row = vocab.token_to_id["J05AR13"]
    missing_row = vocab.token_to_id["MISSING"]
    assert len(resolved.canonical_smiles) == 2
    assert len(resolved.medication_molecule_indices[direct_row]) == 2
    assert resolved.medication_molecule_indices[missing_row] == ()
    assert resolved.medication_molecule_indices[vocab.token_to_id["UNK"]] == ()
    assert "atc=J05AR13" in caplog.text
    assert "drugbank_id=DB-BAD" in caplog.text


def test_coarser_atc_uses_union_of_atc5_descendants(tmp_path) -> None:
    path = _write_mapping(
        tmp_path,
        {
            "J05AR13": [_record("DB1", "CCO")],
            "J05AR17": [
                _record("DB1-AGAIN", "OCC"),
                _record("DB2", "c1ccccc1"),
            ],
            "J05AX01": [_record("DB3", "CCN")],
        },
    )
    vocab = _vocab("J05AR")

    resolved = resolve_medication_molecules(
        medication_vocab=vocab,
        mapping_path=path,
        atc_level=4,
    )

    row = vocab.token_to_id["J05AR"]
    assert len(resolved.canonical_smiles) == 2
    assert len(resolved.medication_molecule_indices[row]) == 2


def test_projection_uses_vocab_ids_and_equal_distinct_molecule_weights(
    tmp_path,
) -> None:
    path = _write_mapping(
        tmp_path,
        {
            "A01AA01": [_record("X", "CCO")],
            "B01AA01": [
                _record("Y", "CCN"),
                _record("Y-DUP", "NCC"),
                _record("Z", "CCC"),
            ],
        },
    )
    vocab = _vocab("B01AA01", "A01AA01", "C01AA01")
    resolved = resolve_medication_molecules(
        medication_vocab=vocab,
        mapping_path=path,
        atc_level=5,
    )

    projection = build_medication_molecule_projection(resolved)

    a_row = projection[vocab.token_to_id["A01AA01"]]
    b_row = projection[vocab.token_to_id["B01AA01"]]
    c_row = projection[vocab.token_to_id["C01AA01"]]
    assert projection.shape == (vocab.vocab_size, 3)
    assert torch.count_nonzero(a_row).item() == 1
    assert torch.count_nonzero(b_row).item() == 2
    assert torch.allclose(a_row.sum(), torch.tensor(1.0))
    assert torch.equal(b_row[b_row > 0], torch.tensor([0.5, 0.5]))
    assert torch.count_nonzero(c_row).item() == 0
    assert torch.count_nonzero(projection[vocab.token_to_id["UNK"]]).item() == 0


def test_wl_graphs_cover_isolated_atoms_and_are_deterministic(tmp_path) -> None:
    path = _write_mapping(
        tmp_path,
        {"A01AA01": [_record("IONS", "[Na+].[Cl-]"), _record("ETH", "CCO")]},
    )
    resolved = resolve_medication_molecules(
        medication_vocab=_vocab("A01AA01"),
        mapping_path=path,
        atc_level=5,
    )

    first = build_molecular_graphs(resolved, radius=1)
    second = build_molecular_graphs(resolved, radius=1)

    assert first.n_fingerprints == second.n_fingerprints
    for first_graph, second_graph in zip(first.graphs, second.graphs):
        fingerprints, adjacency, molecular_size = first_graph
        assert len(fingerprints) == molecular_size
        assert adjacency.shape == (molecular_size, molecular_size)
        assert torch.equal(fingerprints, second_graph[0])
        assert torch.equal(adjacency, second_graph[1])
    assert any(graph[2] == 2 for graph in first.graphs)


def test_substructures_union_molecules_and_selective_building(tmp_path) -> None:
    path = _write_mapping(
        tmp_path,
        {
            "A01AA01": [
                _record("DB1", "CCOC(=O)NCC"),
                _record("DB2", "CCN(CC)CC"),
            ],
            "B01AA01": [_record("DB1", "CCOC(=O)NCC")],
        },
    )
    vocab = _vocab("A01AA01", "B01AA01", "C01AA01")
    resolved = resolve_medication_molecules(
        medication_vocab=vocab,
        mapping_path=path,
        atc_level=5,
    )

    first = build_medication_substructure_matrix(resolved)
    second = build_medication_substructure_matrix(resolved)
    expected_a = set()
    for molecule_index in resolved.medication_molecule_indices[
        vocab.token_to_id["A01AA01"]
    ]:
        expected_a.update(BRICS.BRICSDecompose(resolved.molecules[molecule_index]))

    assert first.shape[0] == vocab.vocab_size
    assert int(first[vocab.token_to_id["A01AA01"]].sum()) == len(expected_a)
    assert torch.count_nonzero(first[vocab.token_to_id["C01AA01"]]).item() == 0
    assert torch.equal(first, second)

    resources = build_molecular_resources(
        medication_vocab=vocab,
        model_requirements={ModelRequirement.MEDICATION_SUBSTRUCTURE_MATRIX},
        mapping_path=path,
        atc_level=5,
    )
    assert set(resources) == {"medication_substructure_matrix"}
    assert build_molecular_resources(
        medication_vocab=vocab,
        model_requirements=set(),
        mapping_path=tmp_path / "does-not-exist.json",
        atc_level=5,
    ) == {}
