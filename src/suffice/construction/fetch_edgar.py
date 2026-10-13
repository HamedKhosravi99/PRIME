#!/usr/bin/env python3
"""Fetch REAL SEC EDGAR XBRL company-facts data (financial/SEC-XBRL domain
artifacts; mirrors src/suffice/construction/download_arxiv.py's role for the research domain).

Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
(10-digit, zero-padded CIK). SEC REQUIRES a compliant declared User-Agent
identifying the requester -- an unset/generic UA gets HTTP 403. This script
sends the value of the SEC_USER_AGENT environment variable, which must name
the requester and a contact address (e.g. "Jane Doe jane@example.org").

Integrity: we store exactly the ``val``/``end``/``start``/``fy``/``fp``/
``form``/``accn``/``frame`` fields SEC's own API returns for the requested
``us-gaap`` concepts -- nothing computed, edited, or invented. Every instance
built on this data embeds a further-trimmed (but byte-identical on every kept
field) subset directly in the instance JSON (the "records" field read by
the structured-record predicates, src/prime_runtime/predicates.py), so runs stay
offline-replayable; PROVENANCE always points back at the untrimmed file this
script writes so any embedded figure can be checked against the raw API
response.

Usage:
    python3 src/suffice/construction/fetch_edgar.py <CIK> <out_dir> [--concepts C1 C2 ...] \
        [--name entity_short_name]

    python3 src/suffice/construction/fetch_edgar.py 0000320193 data/pilot/artifacts/fin-aapl \
        --concepts Revenues RevenueFromContractWithCustomerExcludingAssessedTax \
                   NetIncomeLoss Assets Liabilities --name aapl

Writes, under <out_dir>/:
    companyfacts_trimmed.json  -- {cik, entityName, retrieved, source_url,
                                    facts: {us-gaap: {<concept>: {units: {USD:
                                    [...every annual 10-K/FY entry, verbatim]}}}}}
    PROVENANCE                 -- source url, retrieval timestamp, entityName,
                                   concepts fetched, raw-response sha256,
                                   annual-entry counts per concept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK%s.json"


def fetch_companyfacts(cik: str, timeout: float = 30.0, max_retries: int = 3) -> dict:
    """GET the real companyfacts JSON for a 10-digit zero-padded CIK.

    CRITICAL: SEC requires a compliant User-Agent header identifying the
    requester (name + contact email); without it every request gets HTTP 403.
    """
    if not USER_AGENT:
        raise RuntimeError(
            "set SEC_USER_AGENT to a name and contact address, e.g. "
            "'Jane Doe jane@example.org'; SEC rejects requests without one")
    cik10 = str(cik).zfill(10)
    url = BASE % cik10
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_exc = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            data = json.loads(raw)
            data["_raw_sha256"] = hashlib.sha256(raw).hexdigest()
            data["_source_url"] = url
            return data
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 403:
                raise RuntimeError(
                    "HTTP 403 from SEC EDGAR -- check the User-Agent header "
                    "is a compliant 'name email' string (got %r)" % USER_AGENT
                ) from exc
            time.sleep(1.5 ** attempt)
        except urllib.error.URLError as exc:
            last_exc = exc
            time.sleep(1.5 ** attempt)
    raise RuntimeError("fetch failed after %d attempts: %r" % (max_retries, last_exc))


def extract_annual(data: dict, concept: str) -> list:
    """Every annual (form=10-K, fp=FY) USD entry for one us-gaap concept,
    VERBATIM (val/start/end/fy/fp/form/accn/frame as SEC returned them),
    deduplicated by (fy, val, end) since SEC repeats prior-year comparatives
    across multiple filings, sorted by fiscal year end date."""
    gaap = data.get("facts", {}).get("us-gaap", {})
    if concept not in gaap:
        return []
    units = gaap[concept].get("units", {}).get("USD", [])
    seen = set()
    out = []
    for u in units:
        if u.get("form") != "10-K" or u.get("fp") != "FY":
            continue
        key = (u.get("fy"), u.get("val"), u.get("end"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "val": u.get("val"),
            "start": u.get("start"),
            "end": u.get("end"),
            "fy": u.get("fy"),
            "fp": u.get("fp"),
            "form": u.get("form"),
            "accn": u.get("accn"),
            "frame": u.get("frame"),
        })
    out.sort(key=lambda e: (e["end"] or "", e["fy"] or 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cik", help="SEC CIK (with or without leading zeros)")
    ap.add_argument("out_dir", help="directory to write companyfacts_trimmed.json + PROVENANCE into")
    ap.add_argument(
        "--concepts", nargs="+",
        default=["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "NetIncomeLoss", "Assets", "Liabilities"],
        help="us-gaap concept names to extract (default: revenue/income/assets/liabilities)",
    )
    args = ap.parse_args(argv)

    cik10 = str(args.cik).zfill(10)
    print("fetching real SEC EDGAR company-facts for CIK %s ..." % cik10, file=sys.stderr)
    data = fetch_companyfacts(cik10)
    entity_name = data.get("entityName", "?")
    print("  entityName = %s" % entity_name, file=sys.stderr)

    trimmed = {
        "cik": cik10,
        "entityName": entity_name,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_url": data["_source_url"],
        "raw_response_sha256": data["_raw_sha256"],
        "facts": {"us-gaap": {}},
    }
    counts = {}
    for concept in args.concepts:
        annual = extract_annual(data, concept)
        if annual:
            trimmed["facts"]["us-gaap"][concept] = {"units": {"USD": annual}}
        counts[concept] = len(annual)
        print("  %-55s %d annual (10-K/FY) entries" % (concept, len(annual)), file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "companyfacts_trimmed.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(trimmed, fh, indent=2, sort_keys=True)

    prov = os.path.join(args.out_dir, "PROVENANCE")
    with open(prov, "w", encoding="utf-8") as fh:
        fh.write("name: %s\n" % entity_name)
        fh.write("cik: %s\n" % cik10)
        fh.write("source_url: %s\n" % data["_source_url"])
        fh.write("retrieved_utc: %s\n" % trimmed["retrieved_utc"])
        fh.write("raw_response_sha256: %s\n" % data["_raw_sha256"])
        fh.write("user_agent_sent: %s\n" % USER_AGENT)
        fh.write("concepts_fetched: %s\n" % ", ".join(args.concepts))
        for c, n in counts.items():
            fh.write("  %s: %d annual (10-K, fp=FY) entries\n" % (c, n))
        fh.write(
            "integrity: every val/start/end/fy/fp/form/accn/frame field in "
            "companyfacts_trimmed.json is copied verbatim from SEC's own API "
            "response (see raw_response_sha256); nothing computed, edited, or "
            "invented. Values embedded in any data/pilot/instances/fin/*.json record "
            "are a further subset of these SAME entries, unedited.\n"
        )
    print("wrote %s and PROVENANCE" % out_json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
