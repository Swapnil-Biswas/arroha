"""
ingestion/inspect_dataset.py
----------------------------
Dataset schema inspector for ai4bharat/MSMARCO-XI.

PURPOSE
-------
Inspect the MSMARCO-XI dataset structure WITHOUT downloading the full corpus.
This script never blocks waiting for HuggingFace Hub API calls that require
authentication or that trigger slow unauthenticated rate-limiting.

APPROACH
--------
Two-phase inspection using only fast, lightweight HTTP calls:

  Phase 1 -- Schema via HF Datasets Server API (JSON endpoint):
      GET https://datasets-server.huggingface.co/info
          ?dataset=ai4bharat/MSMARCO-XI&config=default
      Returns the full feature schema as JSON in milliseconds.
      Zero Parquet bytes transferred.

  Phase 2 -- One row via HF Datasets Server parquet/rows API:
      GET https://datasets-server.huggingface.co/rows
          ?dataset=ai4bharat/MSMARCO-XI&config=default&split=train
          &offset=0&length=1
      Returns exactly 1 row as JSON.
      Zero Parquet bytes transferred.
      The server does all the heavy lifting server-side.

WHAT THIS SCRIPT DOES
---------------------
1. Queries the HF Datasets Server for the dataset schema (feature types).
2. Prints all top-level field names and their HuggingFace feature types.
3. Recursively prints nested struct/sequence sub-field types.
4. Queries the HF Datasets Server for exactly 1 row.
5. Prints field values (truncated at 300 chars).
6. Prints the full language coverage discovered.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- Does NOT download any Parquet files.
- Does NOT save any data locally.
- Does NOT build embeddings, FAISS, or BM25.
- Does NOT decide which fields to index (deferred to next phase).
- Does NOT require a HuggingFace token.
"""

from __future__ import annotations

import json
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID  = "ai4bharat/MSMARCO-XI"
CONFIG_NAME = "default"
SPLIT_NAME  = "train"

# HF Datasets Server base URL (public, no token required for public datasets)
HF_DS_SERVER = "https://datasets-server.huggingface.co"

INDENT = "    "


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 30) -> dict:
    """Fetch a JSON URL and return the parsed dict. Raises on error."""
    req = urllib.request.Request(url, headers={"User-Agent": "hhgoa-rag-inspector/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} from {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 72) -> str:
    return char * width


def _type_label(value: Any) -> str:
    t = type(value)
    if t is dict:
        return f"dict [{len(value)} keys]"
    if t is list:
        inner = type(value[0]).__name__ if value else "empty"
        return f"list [len={len(value)}, item_type={inner}]"
    return t.__name__


def _print_nested(value: Any, depth: int = 1, max_depth: int = 5) -> None:
    pad = INDENT * depth
    if depth > max_depth:
        print(f"{pad}... (max depth {max_depth} reached)")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"{pad}{k!r:35s}  {_type_label(v)}")
            if isinstance(v, (dict, list)) and v:
                _print_nested(v, depth + 1, max_depth)
    elif isinstance(value, list) and value:
        first = value[0]
        print(f"{pad}[0]: {_type_label(first)}")
        if isinstance(first, (dict, list)):
            _print_nested(first, depth + 1, max_depth)


def _safe_repr(value: Any, max_chars: int = 300) -> str:
    raw = repr(value)
    if len(raw) > max_chars:
        return raw[:max_chars] + f"  ...[truncated, full_len={len(raw)}]"
    return raw


def _print_feature_type(feat: dict, depth: int = 0) -> None:
    """Recursively print a HF feature-type descriptor dict."""
    pad = INDENT * depth
    dtype = feat.get("dtype") or feat.get("_type", "?")
    if "_type" in feat:
        ftype = feat["_type"]
        if ftype == "Sequence":
            inner = feat.get("feature", {})
            inner_type = inner.get("_type") or inner.get("dtype", "?")
            print(f"{pad}Sequence of {inner_type}")
            if isinstance(inner, dict) and inner.get("_type") == "Value":
                pass  # leaf
            elif isinstance(inner, dict):
                for k, v in inner.items():
                    if k.startswith("_"):
                        continue
                    sub_feat = v if isinstance(v, dict) else {"dtype": str(v)}
                    print(f"{pad}{INDENT}{k!r:30s}  ", end="")
                    _print_feature_type(sub_feat, depth=0)
        elif ftype == "Value":
            print(f"{pad}Value  dtype={feat.get('dtype', '?')}")
        elif ftype == "ClassLabel":
            names = feat.get("names", [])
            print(f"{pad}ClassLabel  [{len(names)} classes]")
        else:
            print(f"{pad}{ftype}")
    elif "dtype" in feat:
        print(f"{pad}Value  dtype={feat['dtype']}")
    else:
        print(f"{pad}{feat}")


# ---------------------------------------------------------------------------
# Phase 1: Schema from HF Datasets Server
# ---------------------------------------------------------------------------

def inspect_schema(dataset_id: str, config: str) -> tuple[list[str], dict]:
    """
    Fetch the dataset schema from the HF Datasets Server info endpoint.
    Returns (top_level_keys, raw_features_dict).
    Zero Parquet bytes transferred.
    """
    url = (f"{HF_DS_SERVER}/info"
           f"?dataset={urllib.parse.quote(dataset_id)}"
           f"&config={urllib.parse.quote(config)}")

    print(f"[1/4] Fetching schema from Datasets Server...")
    print(f"      {url}")

    data = _get_json(url)

    # The response has shape: {"dataset_info": {"features": {...}, ...}}
    dataset_info = data.get("dataset_info", data)
    features = dataset_info.get("features", {})
    splits   = dataset_info.get("splits", {})

    top_keys = list(features.keys())
    print(f"      OK. {len(top_keys)} top-level fields found.")

    # Print split sizes
    if splits:
        print(f"\n  Available splits:")
        for sname, sinfo in splits.items():
            n = sinfo.get("num_examples", "?")
            print(f"    {sname}: {n:,} examples" if isinstance(n, int) else f"    {sname}: {n} examples")

    print()
    print(_sep("="))
    print("  TOP-LEVEL SCHEMA  (from HF Datasets Server)")
    print(_sep("="))
    print(f"\n  Total top-level fields: {len(top_keys)}\n")

    for key in top_keys:
        feat = features[key]
        ftype = feat.get("_type") or feat.get("dtype", "?")
        print(f"  {key!r:30s}  _type: {ftype}")
        _print_feature_type(feat, depth=2)

    return top_keys, features


# ---------------------------------------------------------------------------
# Phase 2: One row from HF Datasets Server rows endpoint
# ---------------------------------------------------------------------------

def fetch_one_row(dataset_id: str, config: str, split: str) -> dict | None:
    """
    Fetch exactly 1 row via the HF Datasets Server /rows endpoint.
    The server reads the Parquet shards; we receive JSON.
    Zero local Parquet bytes transferred.
    """
    url = (f"{HF_DS_SERVER}/rows"
           f"?dataset={urllib.parse.quote(dataset_id)}"
           f"&config={urllib.parse.quote(config)}"
           f"&split={urllib.parse.quote(split)}"
           f"&offset=0&length=1")

    print(f"\n[2/4] Fetching first row from Datasets Server...")
    print(f"      {url}")

    try:
        data = _get_json(url, timeout=60)
    except RuntimeError as exc:
        print(f"      WARNING: {exc}")
        return None

    rows = data.get("rows", [])
    if not rows:
        print("      WARNING: No rows returned.")
        return None

    example = rows[0].get("row", {})
    print(f"      OK. Row index {rows[0].get('row_idx', '?')} received.")
    return example


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(_sep("="))
    print("  HH Goa RAG -- MSMARCO-XI Dataset Inspector")
    print(f"  Dataset : {DATASET_ID}")
    print(f"  Config  : {CONFIG_NAME}")
    print(f"  Split   : {SPLIT_NAME}")
    print("  Method  : HF Datasets Server API (no Parquet download)")
    print(_sep("="))
    print()

    # --- Phase 1: schema ---
    try:
        top_keys, features = inspect_schema(DATASET_ID, CONFIG_NAME)
    except RuntimeError as exc:
        print(f"\nFATAL: Could not fetch schema.\n  {exc}", file=sys.stderr)
        raise SystemExit(1)

    # --- Phase 2: one row ---
    example = fetch_one_row(DATASET_ID, CONFIG_NAME, SPLIT_NAME)

    if example:
        print("\n[3/4] Nested structure of example fields:")
        print(_sep("="))
        print("  NESTED STRUCTURE DETAIL")
        print(_sep("="))
        for key, val in example.items():
            if isinstance(val, (dict, list)) and val:
                print(f"\n  Field {key!r}  [{_type_label(val)}]:")
                _print_nested(val, depth=1)

        print()
        print(_sep("="))
        print("  FULL EXAMPLE  (first record -- values truncated at 300 chars)")
        print(_sep("="))
        for key, val in example.items():
            repr_val = _safe_repr(val)
            wrapped = textwrap.fill(
                repr_val,
                width=90,
                initial_indent=f"  {key}: ",
                subsequent_indent=" " * (len(key) + 4),
            )
            print(wrapped)
    else:
        print("\n[3/4] Example row not available (server may be indexing).")
        top_keys = top_keys  # use schema keys only

    # --- Summary ---
    print()
    print(_sep("="))
    print("  INSPECTION COMPLETE")
    print("  [OK] No Parquet files downloaded.")
    print("  [OK] No data saved locally.")
    print("  [OK] No embeddings, FAISS, or BM25 built.")
    print("  [OK] Schema and example fetched via lightweight API calls only.")
    print(f"  [OK] Top-level fields: {top_keys}")
    print()
    print("  NOTE: Do not infer which fields to index from this output.")
    print("        Corpus field selection is deferred to the next phase.")
    print(_sep("="))


if __name__ == "__main__":
    main()
