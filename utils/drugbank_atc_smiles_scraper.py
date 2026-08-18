"""Map ATC codes to SMILES using a DrugBank ATC-to-drug JSON mapping."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_INPUT = Path("data/ddinter2/mapping/drugbank_atc_drugs.json")
DEFAULT_OUTPUT = Path("data/drugbank/drugbank_atc_smiles.json")
STRUCTURE_URL = "https://go.drugbank.com/structures/small_molecule_drugs/{drug_id}.smiles"
DRUG_ID_PATTERN = re.compile(r"(?:^|/)(DB\d+)(?:$|[/?#])", re.IGNORECASE)


class DrugBankSmilesError(RuntimeError):
    """Raised when a DrugBank SMILES response cannot be retrieved or parsed."""


def drug_id_from_href(href: str) -> str:
    """Extract and normalise a DrugBank identifier from a drug href."""
    match = DRUG_ID_PATTERN.search(href)
    if match is None:
        raise ValueError(f"Could not find a DrugBank ID in href: {href!r}")
    return match.group(1).upper()


def read_drug_mapping(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Read the ATC-to-drug mapping produced by drugbank_atc_scraper.py."""
    mapping_path = Path(path)
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Input mapping {mapping_path} is not a JSON object")
    return payload


def fetch_smiles(drug_id: str, *, timeout: float = 20.0) -> str:
    """Fetch one canonical SMILES string from DrugBank.

    If Cloudflare requires a browser session, the cookie header can be supplied
    at runtime through ``DRUGBANK_COOKIE``. It is deliberately never persisted.
    """
    normalised_id = drug_id.strip().upper()
    if not re.fullmatch(r"DB\d+", normalised_id):
        raise ValueError(f"Invalid DrugBank ID: {drug_id!r}")

    headers = {
        "Accept": "text/html, application/xhtml+xml",
        "Referer": f"https://go.drugbank.com/drugs/{normalised_id}",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
        ),
    }
    cookie = os.environ.get("DRUGBANK_COOKIE")
    if cookie:
        headers["Cookie"] = cookie
    request = Request(STRUCTURE_URL.format(drug_id=normalised_id), headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            smiles = response.read().decode(response.headers.get_content_charset() or "utf-8")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as error:
        hint = (
            "; set DRUGBANK_COOKIE from an active browser session"
            if isinstance(error, HTTPError) and error.code == 403 and not cookie
            else ""
        )
        raise DrugBankSmilesError(
            f"Could not fetch SMILES for {normalised_id}: {error}{hint}"
        ) from error

    smiles = smiles.strip()
    if not smiles or "\n" in smiles or smiles.startswith("<"):
        raise DrugBankSmilesError(f"Unexpected SMILES response for {normalised_id}")
    return smiles


def smiles_cache_from_mapping(
    mapping: Mapping[str, list[Mapping[str, str]]],
) -> dict[str, str]:
    """Recover already fetched values from a prior output for resume support."""
    cache: dict[str, str] = {}
    for drugs in mapping.values():
        if not isinstance(drugs, list):
            continue
        for drug in drugs:
            drug_id = drug.get("drug_id")
            smiles = drug.get("smiles")
            if isinstance(drug_id, str) and isinstance(smiles, str) and smiles:
                cache[drug_id] = smiles
    return cache


def build_atc_smiles_mapping(
    drugs_by_atc: Mapping[str, list[Mapping[str, str]]],
    smiles_by_drug_id: Mapping[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Join an ATC-to-drug mapping with fetched SMILES values."""
    result: dict[str, list[dict[str, str]]] = {}
    for atc_code, drugs in drugs_by_atc.items():
        records: list[dict[str, str]] = []
        for drug in drugs:
            href = drug.get("href")
            name = drug.get("name")
            if not isinstance(href, str) or not isinstance(name, str):
                continue
            try:
                drug_id = drug_id_from_href(href)
            except ValueError:
                continue
            smiles = smiles_by_drug_id.get(drug_id)
            if smiles is not None:
                records.append(
                    {"name": name, "drug_id": drug_id, "href": href, "smiles": smiles}
                )
        result[atc_code] = records
    return result


def write_json(payload: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scrape_smiles_mapping(
    input_path: str | Path,
    output_path: str | Path,
    *,
    timeout: float = 20.0,
    delay: float = 0.2,
    retries: int = 2,
    checkpoint_every: int = 25,
    resume: bool = True,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    """Fetch all unique drugs and save an ATC-to-SMILES mapping."""
    drugs_by_atc = read_drug_mapping(input_path)
    existing = read_drug_mapping(output_path) if resume and Path(output_path).exists() else {}
    smiles_by_id = smiles_cache_from_mapping(existing)

    all_ids: list[str] = []
    seen: set[str] = set()
    invalid_hrefs: dict[str, str] = {}
    for drugs in drugs_by_atc.values():
        for drug in drugs:
            href = drug.get("href")
            if not isinstance(href, str):
                continue
            try:
                drug_id = drug_id_from_href(href)
            except ValueError as error:
                invalid_hrefs[href] = str(error)
                continue
            if drug_id not in seen:
                seen.add(drug_id)
                all_ids.append(drug_id)

    pending = [drug_id for drug_id in all_ids if drug_id not in smiles_by_id]
    errors = dict(invalid_hrefs)
    print(
        f"Found {len(all_ids)} unique drugs; "
        f"{len(all_ids) - len(pending)} already saved, {len(pending)} remaining"
    )

    completed_since_checkpoint = 0
    for position, drug_id in enumerate(pending, start=1):
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                smiles_by_id[drug_id] = fetch_smiles(drug_id, timeout=timeout)
                print(f"[{position}/{len(pending)}] {drug_id}: OK")
                completed_since_checkpoint += 1
                break
            except DrugBankSmilesError as error:
                last_error = error
                if attempt < retries:
                    time.sleep(max(delay, 0) * (attempt + 1))
        else:
            errors[drug_id] = str(last_error)
            print(f"[{position}/{len(pending)}] {drug_id}: ERROR - {last_error}")

        if checkpoint_every > 0 and completed_since_checkpoint >= checkpoint_every:
            write_json(build_atc_smiles_mapping(drugs_by_atc, smiles_by_id), output_path)
            completed_since_checkpoint = 0
        if delay > 0 and position < len(pending):
            time.sleep(delay)

    mapping = build_atc_smiles_mapping(drugs_by_atc, smiles_by_id)
    write_json(mapping, output_path)
    return mapping, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    mapping, errors = scrape_smiles_mapping(
        args.input,
        args.output,
        timeout=args.timeout,
        delay=args.delay,
        retries=args.retries,
        resume=not args.no_resume,
    )
    total = sum(map(len, mapping.values()))
    print(f"Saved {total} ATC-to-SMILES records to {args.output}")
    if errors:
        error_output = Path(args.output).with_suffix(".errors.json")
        write_json(errors, error_output)
        print(f"Could not fetch {len(errors)} drug(s); details saved to {error_output}")


if __name__ == "__main__":
    main()
