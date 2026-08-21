import logging
from pathlib import Path
from ehrdrec.utils import MappingBuilder


def main() -> None:
    builder = MappingBuilder(
        umls_dir=Path("/home/cararc/data/rxnorm-2026-april-06"),
        output_path=Path("data/mappings/ndc_atc_mapping.sqlite"),
        mapping_version="umls-2026AA_rxnorm-2026-04-06_atc-2026",
    )

    builder.build()

if __name__ == "__main__":
    main()
    