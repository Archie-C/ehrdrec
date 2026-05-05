from dataclasses import dataclass


@dataclass(frozen=True)
class ATCMapping:
    drug_rxcui: str
    ingredient_rxcui: str
    atc_code: str
    atc_name: str | None
    match_types: tuple[str, ...]


@dataclass(frozen=True)
class MappingResult:
    input_ndc: str
    normalised_ndc: str
    mappings: list[ATCMapping]
    mapping_version: str

    @property
    def found(self) -> bool:
        return bool(self.mappings)

    @property
    def atc_codes(self) -> list[str]:
        return sorted({m.atc_code for m in self.mappings})

    @property
    def drug_rxcuis(self) -> list[str]:
        return sorted({m.drug_rxcui for m in self.mappings})

    @property
    def ingredient_rxcuis(self) -> list[str]:
        return sorted({m.ingredient_rxcui for m in self.mappings})