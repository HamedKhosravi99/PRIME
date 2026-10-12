#!/usr/bin/env python3
"""Bulk record harvest for the v5 benchmark generator (docs/EXECUTION_PLAN.md,
Part 4). Key-free public sources only; every record stored with
provenance = source URL + retrieval timestamp + sha256 of the raw response.

Writes bench/harvest/<domain>/records.jsonl (one record per line:
{"record_id", "record", "provenance"}) plus a per-domain manifest with the
sampling frame fields (source population, date window, inclusion rule,
counts harvested).

Domains:
  security   OSV per-ecosystem zip (PyPI) -- one bulk download
  clinical   ClinicalTrials.gov API v2, pageSize=1000 pages
  drugsafety openFDA drug enforcement endpoint, paged
  financial  SEC EDGAR companyfacts per-CIK JSON (compliant User-Agent)

Usage: python3 bench/harvest_records.py [--domains security clinical ...]
       [--target N]   (records per domain to retain; default 1200)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "harvest")
UA = {"User-Agent": os.environ.get(
    "SEC_USER_AGENT", "ObligationBench research harvester (contact: anon@example.org)")}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("failed after retries: %s" % url)


def _write(domain: str, rows: List[Dict[str, Any]], frame: Dict[str, Any]) -> None:
    d = os.path.join(OUT, domain)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "records.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    frame["n_retained"] = len(rows)
    frame["retrieved_at"] = _now()
    with open(os.path.join(d, "sampling_frame.json"), "w", encoding="utf-8") as fh:
        json.dump(frame, fh, indent=2, sort_keys=True)
    print("[%s] wrote %d records" % (domain, len(rows)), file=sys.stderr)


def harvest_security(target: int) -> None:
    url = "https://osv-vulnerabilities.storage.googleapis.com/PyPI/all.zip"
    raw = _get(url, timeout=600)
    h = hashlib.sha256(raw).hexdigest()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = sorted(zf.namelist())
    rows = []
    for name in names:
        if len(rows) >= target:
            break
        try:
            rec = json.loads(zf.read(name))
        except Exception:
            continue
        # inclusion rule: has affected-package info and a summary/details
        if not rec.get("affected") or not (rec.get("summary") or rec.get("details")):
            continue
        rows.append({"record_id": rec.get("id", name),
                     "record": rec,
                     "provenance": {"source": url, "member": name,
                                    "zip_sha256": h, "retrieved_at": _now()}})
    _write("security", rows, {
        "source_population": "OSV PyPI ecosystem bulk zip (all advisories)",
        "url": url, "zip_sha256": h,
        "inclusion_rule": "has affected[] and summary/details",
        "n_available": len(names), "sampling": "lexicographic prefix (deterministic)",
    })


def harvest_clinical(target: int) -> None:
    rows, token = [], None
    base = "https://clinicaltrials.gov/api/v2/studies?pageSize=1000&format=json"
    n_seen = 0
    while len(rows) < target:
        url = base + (("&pageToken=" + token) if token else "")
        raw = _get(url)
        page = json.loads(raw)
        h = hashlib.sha256(raw).hexdigest()
        for st in page.get("studies", []):
            n_seen += 1
            ident = st.get("protocolSection", {}).get("identificationModule", {})
            status = st.get("protocolSection", {}).get("statusModule", {})
            if not ident.get("nctId") or not status.get("overallStatus"):
                continue
            rows.append({"record_id": ident["nctId"], "record": st,
                         "provenance": {"source": url, "page_sha256": h,
                                        "retrieved_at": _now()}})
            if len(rows) >= target:
                break
        token = page.get("nextPageToken")
        if not token:
            break
        time.sleep(1)
    _write("clinical", rows, {
        "source_population": "ClinicalTrials.gov API v2 default ordering",
        "url": base, "inclusion_rule": "has nctId and overallStatus",
        "n_seen": n_seen, "sampling": "API default order, first pages (deterministic at retrieval time)",
    })


def harvest_drugsafety(target: int) -> None:
    rows = []
    skip, n_seen = 0, 0
    while len(rows) < target and skip <= 25000:
        url = ("https://api.fda.gov/drug/enforcement.json?limit=1000&skip=%d" % skip)
        raw = _get(url)
        h = hashlib.sha256(raw).hexdigest()
        page = json.loads(raw)
        for rec in page.get("results", []):
            n_seen += 1
            if not rec.get("recall_number") or not rec.get("classification"):
                continue
            rows.append({"record_id": rec["recall_number"], "record": rec,
                         "provenance": {"source": url, "page_sha256": h,
                                        "retrieved_at": _now()}})
            if len(rows) >= target:
                break
        skip += 1000
        time.sleep(1)
    _write("drugsafety", rows, {
        "source_population": "openFDA drug enforcement endpoint, paged from offset 0",
        "inclusion_rule": "has recall_number and classification",
        "n_seen": n_seen, "sampling": "endpoint default order (deterministic at retrieval time)",
    })


# Deterministic ticker list: alphabetical large-cap set, frozen here.
FIN_CIKS = {
    "AAPL": 320193, "MSFT": 789019, "GOOGL": 1652044, "AMZN": 1018724,
    "NVDA": 1045810, "META": 1326801, "TSLA": 1318605, "BRK-B": 1067983,
    "JPM": 19617, "JNJ": 200406, "V": 1403161, "PG": 80424, "XOM": 34088,
    "UNH": 731766, "HD": 354950, "MA": 1141391, "PFE": 78003, "KO": 21344,
    "PEP": 77476, "COST": 909832, "WMT": 104169, "CSCO": 858877,
    "ORCL": 1341439, "CRM": 1108524, "INTC": 50863, "AMD": 2488,
    "IBM": 51143, "QCOM": 804328, "TXN": 97476, "NKE": 320187,
    "MCD": 63908, "SBUX": 829224, "CAT": 18230, "BA": 12927, "GE": 40545,
    "MMM": 66740, "HON": 773840, "UPS": 1090727, "FDX": 1048911,
    "T": 732717, "VZ": 732712, "DIS": 1744489, "NFLX": 1065280,
    "ADBE": 796343, "PYPL": 1633917, "ABT": 1800, "MRK": 310158,
    "LLY": 59478, "BMY": 14272, "GILD": 882095, "AMGN": 318154,
    "CVX": 93410, "COP": 1163165, "SLB": 87347, "GS": 886982,
    "MS": 895421, "BAC": 70858, "WFC": 72971, "C": 831001, "AXP": 4962,
}


def harvest_financial(target: int) -> None:
    rows = []
    for tk, cik in sorted(FIN_CIKS.items()):
        if len(rows) >= target:
            break
        url = "https://data.sec.gov/api/xbrl/companyfacts/CIK%010d.json" % cik
        try:
            raw = _get(url)
        except RuntimeError:
            continue
        h = hashlib.sha256(raw).hexdigest()
        rec = json.loads(raw)
        if "facts" not in rec:
            continue
        rows.append({"record_id": "CIK%010d" % cik, "record": rec,
                     "provenance": {"source": url, "sha256": h, "ticker": tk,
                                    "retrieved_at": _now()}})
        time.sleep(0.6)
    _write("financial", rows, {
        "source_population": "SEC EDGAR companyfacts for a frozen alphabetical large-cap CIK list (in-script)",
        "inclusion_rule": "companyfacts JSON has 'facts'",
        "n_ciks_attempted": len(FIN_CIKS),
        "sampling": "frozen deterministic list",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+",
                    default=["security", "clinical", "drugsafety", "financial"])
    ap.add_argument("--target", type=int, default=1200)
    args = ap.parse_args()
    fns = {"security": harvest_security, "clinical": harvest_clinical,
           "drugsafety": harvest_drugsafety, "financial": harvest_financial}
    for d in args.domains:
        print("=== harvesting %s ===" % d, file=sys.stderr)
        # financial: per-company records are heavy; ~60 companies is the frame
        fns[d](args.target if d != "financial" else 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
