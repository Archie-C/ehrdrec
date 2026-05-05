from pathlib import Path

from .normalise import normalise_ndc
from .store import SQLiteMappingStore
from .models import MappingResult


class NDCATCMapper:
    def __init__(self, store: SQLiteMappingStore):
        self.store = store
        self.metadata = store.get_metadata()

    @classmethod
    def from_file(cls, path: str | Path) -> "NDCATCMapper":
        return cls(SQLiteMappingStore(path))

    @property
    def version(self) -> str:
        return self.metadata["mapping_version"]

    def ndc_to_atc(
        self,
        ndc: str,
        *,
        atc_level: int | None = None,
    ) -> MappingResult:
        normalised_ndc = normalise_ndc(ndc)
        return self.store.lookup_ndc(
            normalised_ndc=normalised_ndc,
            input_ndc=ndc,
            atc_level=atc_level,
        )

    def ndcs_to_atc(
        self,
        ndcs: list[str],
        *,
        atc_level: int | None = None,
    ) -> list[MappingResult]:
        return [
            self.ndc_to_atc(ndc, atc_level=atc_level)
            for ndc in ndcs
        ]

    def close(self) -> None:
        self.store.close()