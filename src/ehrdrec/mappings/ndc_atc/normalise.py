import re

from .exceptions import InvalidNDCError


def normalise_ndc(ndc: str) -> str:
    """
    Normalise NDC to 11-digit HIPAA format where possible.

    Supports:
    - 5-4-2
    - 4-4-2
    - 5-3-2
    - 5-4-1
    - already 11 digits

    Does not guess ambiguous 10-digit NDCs without hyphens.
    """
    if not ndc:
        raise InvalidNDCError("NDC cannot be empty.")

    ndc = ndc.strip()

    if "-" in ndc:
        parts = ndc.split("-")

        if len(parts) != 3:
            raise InvalidNDCError(f"Invalid NDC segment format: {ndc}")

        labeler, product, package = parts

        if not all(part.isdigit() for part in parts):
            raise InvalidNDCError(f"NDC contains non-digit segment: {ndc}")

        if len(labeler) == 5 and len(product) == 4 and len(package) == 2:
            return labeler + product + package

        if len(labeler) == 4 and len(product) == 4 and len(package) == 2:
            return labeler.zfill(5) + product + package

        if len(labeler) == 5 and len(product) == 3 and len(package) == 2:
            return labeler + product.zfill(4) + package

        if len(labeler) == 5 and len(product) == 4 and len(package) == 1:
            return labeler + product + package.zfill(2)

        raise InvalidNDCError(f"Unsupported NDC format: {ndc}")

    digits = re.sub(r"\D", "", ndc)

    if len(digits) == 11:
        return digits

    if len(digits) == 10:
        raise InvalidNDCError(
            f"Ambiguous 10-digit NDC without hyphens: {ndc}"
        )

    raise InvalidNDCError(f"Invalid NDC: {ndc}")


def atc_to_level(atc_code: str, level: int) -> str:
    """
    Convert ATC code to requested level.

    level=1 -> A
    level=2 -> A10
    level=3 -> A10B
    level=4 -> A10BA
    level=5 -> A10BA02
    """
    if level not in {1, 2, 3, 4, 5}:
        raise ValueError("ATC level must be one of: 1, 2, 3, 4, 5")

    lengths = {
        1: 1,
        2: 3,
        3: 4,
        4: 5,
        5: 7,
    }

    return atc_code[: lengths[level]]