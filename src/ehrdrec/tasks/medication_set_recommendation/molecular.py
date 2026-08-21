from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
import torch

from ehrdrec.requirements import ModelRequirement
from ehrdrec.utils import Vocab
from ehrdrec.utils.mappings.ndc_atc.normalize import atc_to_level


logger = logging.getLogger(__name__)


MOLECULAR_REQUIREMENTS = {
    ModelRequirement.MOLECULAR_GRAPHS,
    ModelRequirement.MEDICATION_MOLECULE_PROJECTION,
    ModelRequirement.MEDICATION_SUBSTRUCTURE_MATRIX,
}


@dataclass(frozen=True)
class ResolvedMedicationMolecules:
    """Canonical molecules and their medication-vocabulary incidence."""

    canonical_smiles: tuple[str, ...]
    molecules: tuple[Chem.Mol, ...]
    medication_molecule_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MolecularGraphs:
    """SafeDrug-compatible, CPU-side WL molecular graphs."""

    graphs: tuple[tuple[torch.Tensor, torch.Tensor, int], ...]
    n_fingerprints: int


def build_molecular_resources(
    *,
    medication_vocab: Vocab,
    model_requirements: set[ModelRequirement],
    mapping_path: str | Path,
    atc_level: int,
    wl_radius: int = 1,
) -> dict[str, Any]:
    """Build only requested resources from one molecule-resolution pass."""

    requested = model_requirements & MOLECULAR_REQUIREMENTS
    if not requested:
        return {}
    if wl_radius < 0:
        raise ValueError("Molecular WL radius must be non-negative.")

    resolved = resolve_medication_molecules(
        medication_vocab=medication_vocab,
        mapping_path=mapping_path,
        atc_level=atc_level,
    )
    resources: dict[str, Any] = {}

    if ModelRequirement.MOLECULAR_GRAPHS in requested:
        resources["molecular_graphs"] = build_molecular_graphs(
            resolved,
            radius=wl_radius,
        )

    if ModelRequirement.MEDICATION_MOLECULE_PROJECTION in requested:
        resources["medication_molecule_projection"] = (
            build_medication_molecule_projection(resolved)
        )

    if ModelRequirement.MEDICATION_SUBSTRUCTURE_MATRIX in requested:
        resources["medication_substructure_matrix"] = (
            build_medication_substructure_matrix(resolved)
        )

    return resources


def resolve_medication_molecules(
    *,
    medication_vocab: Vocab,
    mapping_path: str | Path,
    atc_level: int,
) -> ResolvedMedicationMolecules:
    """Resolve vocabulary rows against ATC-5 records and canonicalize SMILES."""

    path = Path(mapping_path)
    with path.open(encoding="utf-8") as file:
        mapping = json.load(file)
    if not isinstance(mapping, dict):
        raise ValueError(
            f"ATC molecule mapping must contain a JSON object: {path}"
        )

    canonical_to_molecule: dict[str, Chem.Mol] = {}
    medication_smiles: list[set[str]] = [
        set() for _ in range(medication_vocab.vocab_size)
    ]
    raw_records = 0
    invalid_records = 0
    valid_records = 0

    source_codes = sorted(str(code) for code in mapping)
    for medication_id in range(medication_vocab.vocab_size):
        medication_code = medication_vocab.id_to_token.get(medication_id)
        if medication_code is None:
            continue

        matching_codes = [
            source_code
            for source_code in source_codes
            if atc_to_level(source_code, atc_level) == medication_code
        ]

        for source_code in matching_codes:
            records = mapping[source_code]
            if not isinstance(records, list):
                logger.warning(
                    "Invalid molecular mapping entry: atc=%s reason=expected list",
                    source_code,
                )
                invalid_records += 1
                continue

            for record in records:
                raw_records += 1
                drug_id = (
                    record.get("drug_id")
                    if isinstance(record, dict)
                    else None
                )
                smiles = (
                    record.get("smiles")
                    if isinstance(record, dict)
                    else None
                )

                try:
                    if not isinstance(smiles, str) or not smiles.strip():
                        raise ValueError("missing or empty SMILES")
                    molecule = Chem.MolFromSmiles(smiles)
                    if molecule is None:
                        raise ValueError("RDKit could not parse SMILES")
                    canonical = Chem.MolToSmiles(
                        molecule,
                        canonical=True,
                        isomericSmiles=True,
                    )
                    if not canonical:
                        raise ValueError("RDKit produced an empty canonical SMILES")
                except (TypeError, ValueError, RuntimeError) as exc:
                    invalid_records += 1
                    logger.warning(
                        "Invalid molecular record: atc=%s drugbank_id=%s "
                        "smiles=%r reason=%s",
                        source_code,
                        drug_id,
                        smiles,
                        exc,
                    )
                    continue

                valid_records += 1
                canonical_to_molecule.setdefault(canonical, molecule)
                medication_smiles[medication_id].add(canonical)

    canonical_smiles = tuple(sorted(canonical_to_molecule))
    molecule_to_index = {
        smiles: index for index, smiles in enumerate(canonical_smiles)
    }
    medication_molecule_indices = tuple(
        tuple(molecule_to_index[smiles] for smiles in sorted(smiles_set))
        for smiles_set in medication_smiles
    )
    resolved_medications = sum(
        bool(indices) for indices in medication_molecule_indices
    )
    unique_relationships = sum(
        len(indices) for indices in medication_molecule_indices
    )

    logger.info(
        "Molecule resolution completed: atc_level=%d source=%s "
        "medication_vocab_size=%d medications_resolved=%d "
        "medications_missing=%d raw_records=%d invalid_smiles=%d "
        "valid_records=%d unique_canonical_molecules=%d "
        "duplicate_records_collapsed=%d unique_medication_molecule_links=%d",
        atc_level,
        path,
        medication_vocab.vocab_size,
        resolved_medications,
        medication_vocab.vocab_size - resolved_medications,
        raw_records,
        invalid_records,
        valid_records,
        len(canonical_smiles),
        valid_records - len(canonical_smiles),
        unique_relationships,
    )

    return ResolvedMedicationMolecules(
        canonical_smiles=canonical_smiles,
        molecules=tuple(
            canonical_to_molecule[smiles] for smiles in canonical_smiles
        ),
        medication_molecule_indices=medication_molecule_indices,
    )


def build_medication_molecule_projection(
    resolved: ResolvedMedicationMolecules,
) -> torch.Tensor:
    projection = torch.zeros(
        (
            len(resolved.medication_molecule_indices),
            len(resolved.molecules),
        ),
        dtype=torch.float32,
    )
    for medication_id, molecule_indices in enumerate(
        resolved.medication_molecule_indices
    ):
        if molecule_indices:
            projection[medication_id, list(molecule_indices)] = (
                1.0 / len(molecule_indices)
            )

    nonzero_rows = int((projection.sum(dim=1) > 0).sum().item())
    logger.info(
        "Medication-molecule projection constructed: shape=%s "
        "nonzero_medications=%d",
        tuple(projection.shape),
        nonzero_rows,
    )
    return projection


def build_medication_substructure_matrix(
    resolved: ResolvedMedicationMolecules,
) -> torch.Tensor:
    medication_fragments: list[set[str]] = []
    all_fragments: set[str] = set()

    for molecule_indices in resolved.medication_molecule_indices:
        fragments: set[str] = set()
        for molecule_index in molecule_indices:
            fragments.update(
                str(fragment)
                for fragment in BRICS.BRICSDecompose(
                    resolved.molecules[molecule_index]
                )
            )
        medication_fragments.append(fragments)
        all_fragments.update(fragments)

    fragment_vocab = tuple(sorted(all_fragments))
    fragment_to_index = {
        fragment: index for index, fragment in enumerate(fragment_vocab)
    }
    matrix = torch.zeros(
        (len(medication_fragments), len(fragment_vocab)),
        dtype=torch.float32,
    )
    for medication_id, fragments in enumerate(medication_fragments):
        for fragment in sorted(fragments):
            matrix[medication_id, fragment_to_index[fragment]] = 1.0

    nonzero_rows = int((matrix.sum(dim=1) > 0).sum().item())
    logger.info(
        "Medication substructure matrix constructed: shape=%s "
        "unique_fragments=%d nonzero_medications=%d",
        tuple(matrix.shape),
        len(fragment_vocab),
        nonzero_rows,
    )
    return matrix


def build_molecular_graphs(
    resolved: ResolvedMedicationMolecules,
    *,
    radius: int = 1,
) -> MolecularGraphs:
    atom_ids: dict[Any, int] = {}
    bond_ids: dict[str, int] = {}
    fingerprint_ids: dict[Any, int] = {}
    edge_ids: dict[Any, int] = {}
    graphs: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def identifier(values: dict[Any, int], key: Any) -> int:
        if key not in values:
            values[key] = len(values)
        return values[key]

    for canonical_smiles, base_molecule in zip(
        resolved.canonical_smiles,
        resolved.molecules,
    ):
        try:
            molecule = Chem.AddHs(base_molecule)
            atom_keys: list[Any] = [
                atom.GetSymbol() for atom in molecule.GetAtoms()
            ]
            for atom in molecule.GetAromaticAtoms():
                atom_keys[atom.GetIdx()] = (
                    atom_keys[atom.GetIdx()],
                    "aromatic",
                )
            atoms = [identifier(atom_ids, key) for key in atom_keys]

            bonds: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for bond in molecule.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                bond_id = identifier(bond_ids, str(bond.GetBondType()))
                bonds[begin].append((end, bond_id))
                bonds[end].append((begin, bond_id))
            for neighbors in bonds.values():
                neighbors.sort()

            fingerprints = _extract_wl_fingerprints(
                radius=radius,
                atoms=atoms,
                bonds=bonds,
                fingerprint_ids=fingerprint_ids,
                edge_ids=edge_ids,
                identifier=identifier,
            )
            adjacency = Chem.GetAdjacencyMatrix(molecule)
            atom_count = molecule.GetNumAtoms()
            if len(fingerprints) != atom_count:
                raise ValueError(
                    f"fingerprint count {len(fingerprints)} does not match "
                    f"atom count {atom_count}"
                )
            if adjacency.shape != (atom_count, atom_count):
                raise ValueError(
                    f"adjacency shape {adjacency.shape} does not match "
                    f"atom count {atom_count}"
                )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Failed to construct molecular graph for "
                f"SMILES {canonical_smiles!r}: {exc}"
            ) from exc

        graphs.append(
            (
                torch.tensor(fingerprints, dtype=torch.long),
                torch.tensor(np.asarray(adjacency), dtype=torch.float32),
                atom_count,
            )
        )

    result = MolecularGraphs(
        graphs=tuple(graphs),
        n_fingerprints=len(fingerprint_ids),
    )
    logger.info(
        "Molecular graphs constructed: graphs=%d wl_radius=%d "
        "distinct_fingerprints=%d",
        len(result.graphs),
        radius,
        result.n_fingerprints,
    )
    return result


def _extract_wl_fingerprints(
    *,
    radius: int,
    atoms: list[int],
    bonds: dict[int, list[tuple[int, int]]],
    fingerprint_ids: dict[Any, int],
    edge_ids: dict[Any, int],
    identifier: Callable[[dict[Any, int], Any], int],
) -> list[int]:
    if radius == 0 or len(atoms) == 1:
        return [identifier(fingerprint_ids, atom) for atom in atoms]

    nodes = list(atoms)
    edges = {index: list(bonds.get(index, ())) for index in range(len(atoms))}

    for _ in range(radius):
        next_nodes: list[int] = []
        for index in range(len(nodes)):
            neighbors = tuple(
                sorted((nodes[neighbor], edge) for neighbor, edge in edges[index])
            )
            next_nodes.append(
                identifier(fingerprint_ids, (nodes[index], neighbors))
            )

        next_edges: dict[int, list[tuple[int, int]]] = {
            index: [] for index in range(len(nodes))
        }
        for index in range(len(nodes)):
            for neighbor, edge in edges[index]:
                both_sides = tuple(sorted((nodes[index], nodes[neighbor])))
                next_edge = identifier(edge_ids, (both_sides, edge))
                next_edges[index].append((neighbor, next_edge))
            next_edges[index].sort()

        nodes = next_nodes
        edges = next_edges

    return nodes
