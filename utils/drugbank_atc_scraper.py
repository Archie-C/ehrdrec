"""Fetch DrugBank's ATC tree and record the drugs for selected ATC codes.

The endpoint returns a jsTree document.  The node matching the query has
``state.selected == true`` and its children contain the DrugBank drug links.
No browser cookies or CSRF token are required by this read-only endpoint under
normal circumstances.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


TREE_URL = "https://go.drugbank.com/atc/tree.json"
DRUGBANK_URL = "https://go.drugbank.com"
DEFAULT_INPUT = Path("data/ddinter2/mapping/ddinter_atc_codes.csv")
DEFAULT_OUTPUT = Path("data/drugbank/drugbank_atc_drugs.json")


class DrugBankScraperError(RuntimeError):
    """Raised when the ATC tree cannot be fetched or interpreted."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return "".join(parser.parts).strip()


def _walk_nodes(nodes: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for node in nodes:
        yield node
        children = node.get("children")
        if isinstance(children, list):
            yield from _walk_nodes(child for child in children if isinstance(child, Mapping))


def selected_drugs(
    tree: list[Mapping[str, Any]], *, absolute_urls: bool = False
) -> dict[str, list[dict[str, str]]]:
    """Extract ``{ATC code: [{name, href}, ...]}`` from a DrugBank tree.

    Only explicitly selected ATC nodes are returned. Duplicate drug links under
    a node are removed while preserving their response order.
    """
    result: dict[str, list[dict[str, str]]] = {}

    for node in _walk_nodes(tree):
        state = node.get("state")
        if not isinstance(state, Mapping) or state.get("selected") is not True:
            continue

        atc_code = node.get("id")
        if not isinstance(atc_code, str) or not atc_code:
            continue

        drugs: list[dict[str, str]] = []
        seen_hrefs: set[str] = set()
        children = node.get("children")
        if not isinstance(children, list):
            children = []

        for child in _walk_nodes(c for c in children if isinstance(c, Mapping)):
            attributes = child.get("a_attr")
            if not isinstance(attributes, Mapping):
                continue
            href = attributes.get("href")
            text = child.get("text")
            if not isinstance(href, str) or not isinstance(text, str) or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            drugs.append(
                {
                    "name": _plain_text(text),
                    "href": urljoin(DRUGBANK_URL, href) if absolute_urls else href,
                }
            )

        result[atc_code] = drugs

    return result


def fetch_atc_tree(atc_code: str, *, timeout: float = 20.0) -> list[Mapping[str, Any]]:
    """Fetch the jsTree JSON response for one ATC code."""
    code = atc_code.strip().upper()
    if not code:
        raise ValueError("ATC code cannot be empty")

    url = f"{TREE_URL}?{urlencode({'code': '#', 'selected': '', 'query': code})}"
    request = Request(
        url,
        headers={
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "User-Agent": "ehrdrec-drugbank-atc-scraper/1.0",
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise DrugBankScraperError(f"Could not fetch ATC code {code}: {error}") from error

    if not isinstance(payload, list):
        raise DrugBankScraperError(f"Unexpected response for ATC code {code}: expected a list")
    return [node for node in payload if isinstance(node, Mapping)]


def scrape_atc_codes(
    atc_codes: Iterable[str], *, timeout: float = 20.0, absolute_urls: bool = False
) -> dict[str, list[dict[str, str]]]:
    """Fetch several ATC codes and combine their selected drug mappings."""
    mapping: dict[str, list[dict[str, str]]] = {}
    for requested_code in atc_codes:
        code = requested_code.strip().upper()
        tree = fetch_atc_tree(code, timeout=timeout)
        selected = selected_drugs(tree, absolute_urls=absolute_urls)
        if code not in selected:
            raise DrugBankScraperError(
                f"Response for ATC code {code} did not contain a selected node"
            )
        mapping[code] = selected[code]
    return mapping


def read_atc_codes(csv_path: str | Path, *, column: str = "atc_code") -> list[str]:
    """Read and deduplicate ATC codes from a CSV, preserving their order."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"CSV {path} does not contain an {column!r} column")

        codes: list[str] = []
        seen: set[str] = set()
        for row in reader:
            code = (row.get(column) or "").strip().upper()
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def read_mapping(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load an existing output mapping for a resumable batch run."""
    mapping_path = Path(path)
    if not mapping_path.exists():
        return {}
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Existing mapping {mapping_path} is not a JSON object")
    return payload


def scrape_atc_csv(
    csv_path: str | Path,
    output: str | Path,
    *,
    timeout: float = 20.0,
    delay: float = 0.2,
    retries: int = 2,
    checkpoint_every: int = 25,
    resume: bool = True,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    """Scrape every unique ATC code in a CSV and checkpoint the JSON mapping."""
    codes = read_atc_codes(csv_path)
    mapping = read_mapping(output) if resume else {}
    errors: dict[str, str] = {}
    pending = [code for code in codes if code not in mapping]

    print(
        f"Found {len(codes)} unique ATC codes; "
        f"{len(codes) - len(pending)} already saved, {len(pending)} remaining"
    )
    completed_since_checkpoint = 0
    for position, code in enumerate(pending, start=1):
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                selected = selected_drugs(
                    fetch_atc_tree(code, timeout=timeout), absolute_urls=True
                )
                if code not in selected:
                    raise DrugBankScraperError("response did not contain a selected node")
                mapping[code] = selected[code]
                errors.pop(code, None)
                print(f"[{position}/{len(pending)}] {code}: {len(mapping[code])} drug(s)")
                completed_since_checkpoint += 1
                break
            except (DrugBankScraperError, TimeoutError) as error:
                last_error = error
                if attempt < retries:
                    time.sleep(max(delay, 0) * (attempt + 1))
        else:
            errors[code] = str(last_error)
            print(f"[{position}/{len(pending)}] {code}: ERROR - {last_error}")

        if checkpoint_every > 0 and completed_since_checkpoint >= checkpoint_every:
            write_mapping(mapping, output)
            completed_since_checkpoint = 0
        if delay > 0 and position < len(pending):
            time.sleep(delay)

    write_mapping(mapping, output)
    return mapping, errors


def write_mapping(mapping: Mapping[str, Any], output: str | Path) -> None:
    """Write a mapping as readable UTF-8 JSON."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "atc_codes", nargs="*",
        help="Individual ATC codes. If omitted, all codes in --input-csv are used.",
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--absolute-urls", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.atc_codes:
        mapping = scrape_atc_codes(
            args.atc_codes, timeout=args.timeout, absolute_urls=args.absolute_urls
        )
        write_mapping(mapping, args.output)
        errors: dict[str, str] = {}
    else:
        mapping, errors = scrape_atc_csv(
            args.input_csv, args.output, timeout=args.timeout, delay=args.delay,
            retries=args.retries, resume=not args.no_resume,
        )

    print(f"Saved {sum(map(len, mapping.values()))} drug links to {args.output}")
    if errors:
        error_output = Path(args.output).with_suffix(".errors.json")
        write_mapping(errors, error_output)
        print(f"Could not fetch {len(errors)} ATC code(s); details saved to {error_output}")


if __name__ == "__main__":
    main()
