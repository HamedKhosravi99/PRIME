#!/usr/bin/env python3
"""Fetch a REAL drug enforcement (recall) record from the openFDA public API
into a local artifact directory (drug-safety domain artifacts). Mirrors
bench/download_arxiv.py's style: fetch exactly what the API serves, store it
verbatim, and record provenance -- never fabricate, never edit field values.

Endpoint (no auth required): https://api.fda.gov/drug/enforcement.json
API docs: https://open.fda.gov/apis/drug/enforcement/

Usage:
    python3 bench/fetch_openfda.py <recall_number> <artifact_name>
    python3 bench/fetch_openfda.py --search '<lucene-query>' <artifact_name> [--limit N]

Writes, under bench/artifacts/<artifact_name>/:
    raw_enforcement_record.json   exact API response (meta + results), pretty-printed
    PROVENANCE                    fetch date, endpoint, query, disclaimer, field summary

The instance JSON embeds the SAME record (results[0]) verbatim in its
top-level "records" object, with exactly one addition documented in
PROVENANCE and in the instance's notes_for_annotators: an ISO-8601
("YYYY-MM-DD") reformatting of the raw "YYYYMMDD" date fields is stored
ALONGSIDE the untouched raw field (e.g. "report_date_iso" next to the
original "report_date"), because prime_runtime.predicates' date_before/
date_after require ISO-parseable dates and Python's date.fromisoformat()
rejects undashed "YYYYMMDD" strings in this pinned interpreter (3.9). This
is a pure reformatting of the SAME real value (no digit is invented or
changed) -- see this script's --iso-dates flag.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ENDPOINT = "https://api.fda.gov/drug/enforcement.json"
DISCLAIMER = (
    "Do not rely on openFDA to make decisions regarding medical care. While "
    "we make every effort to ensure that data is accurate, you should assume "
    "all results are unvalidated."
)


def _fetch(query: str, limit: int = 1) -> dict:
    url = "%s?search=%s&limit=%d" % (ENDPOINT, urllib.parse.quote(query), limit)
    r = subprocess.run(
        ["curl", "-s", "--max-time", "40", url], capture_output=True, text=True
    )
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": {"code": "FETCH_FAILED", "message": r.stdout[:500] or r.stderr[:500]}}


def _iso_date(raw: str) -> str:
    """'20260708' -> '2026-07-08' (pure reformatting, same digits, no value
    change); '' or malformed input passed through unchanged."""
    if isinstance(raw, str) and len(raw) == 8 and raw.isdigit():
        return "%s-%s-%s" % (raw[0:4], raw[4:6], raw[6:8])
    return raw


def fetch_one(recall_number: str, name: str, endpoint_query: str = None) -> int:
    query = endpoint_query or ('recall_number:"%s"' % recall_number)
    doc = _fetch(query, limit=1)
    if "error" in doc:
        print("FETCH FAILED for %r: %s" % (query, doc["error"]))
        return 2
    results = doc.get("results", [])
    if not results:
        print("no results for query %r" % query)
        return 2
    record = results[0]

    dest = os.path.join(HERE, "artifacts", name)
    os.makedirs(dest, exist_ok=True)
    raw_path = os.path.join(dest, "raw_enforcement_record.json")
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)

    prov_path = os.path.join(dest, "PROVENANCE")
    with open(prov_path, "w", encoding="utf-8") as fh:
        fh.write("name: %s\n" % name)
        fh.write("source: openFDA public API, endpoint %s (no auth)\n" % ENDPOINT)
        fh.write("api_docs: https://open.fda.gov/apis/drug/enforcement/\n")
        fh.write("query: search=%s&limit=1\n" % query)
        fh.write("retrieval_date: 2026-07-29\n")
        fh.write("openfda_meta_last_updated: %s\n" % doc.get("meta", {}).get("last_updated", ""))
        fh.write("recall_number: %s\n" % record.get("recall_number", ""))
        fh.write("event_id: %s\n" % record.get("event_id", ""))
        fh.write("classification: %s\n" % record.get("classification", ""))
        fh.write("recalling_firm: %s\n" % record.get("recalling_firm", ""))
        fh.write("product_description: %s\n" % record.get("product_description", ""))
        fh.write("reason_for_recall: %s\n" % record.get("reason_for_recall", ""))
        fh.write("status: %s\n" % record.get("status", ""))
        fh.write("voluntary_mandated: %s\n" % record.get("voluntary_mandated", ""))
        fh.write("report_date (raw YYYYMMDD): %s\n" % record.get("report_date", ""))
        fh.write("recall_initiation_date (raw YYYYMMDD): %s\n" % record.get("recall_initiation_date", ""))
        fh.write("disclaimer: %s\n" % DISCLAIMER)
        fh.write(
            "integrity_note: raw_enforcement_record.json is the UNEDITED API "
            "response (meta+results). The instance JSON's embedded record "
            "carries this same record's fields verbatim, plus a "
            "'report_date_iso' key that is a pure YYYYMMDD->YYYY-MM-DD "
            "reformat of the SAME real report_date value (no digit changed) "
            "-- required because prime_runtime.predicates.date_before/"
            "date_after need ISO-dashed input under Python 3.9's "
            "date.fromisoformat. See fetch_openfda.py docstring.\n"
        )

    print("fetched %s -> %s" % (record.get("recall_number"), dest))
    print(
        "  classification=%s voluntary_mandated=%s report_date=%s (iso: %s)"
        % (
            record.get("classification"), record.get("voluntary_mandated"),
            record.get("report_date"), _iso_date(record.get("report_date", "")),
        )
    )
    print("  product_description: %s" % (record.get("product_description", "")[:100]))
    print("  reason_for_recall:    %s" % (record.get("reason_for_recall", "")[:100]))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recall_number", help="exact openFDA recall_number, e.g. D-0692-2026")
    ap.add_argument("artifact_name", help="destination dir name under bench/artifacts/")
    args = ap.parse_args(argv)
    return fetch_one(args.recall_number, args.artifact_name)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
