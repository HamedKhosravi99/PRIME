#!/usr/bin/env python3
"""Fetch a REAL ClinicalTrials.gov study record (clinical-trials domain,
data/pilot/instances/ctgov/) from the public v2 API, no auth required:

    GET https://clinicaltrials.gov/api/v2/studies/<NCT_ID>

Two jobs:

1. ``check <NCT_ID>``  -- fetch the full record and report, field by field,
   whether the atoms this domain's claim ladder needs are actually PRESENT
   (enrollment count and primary outcomes are genuinely sparse in older
   trials -- this is the screening step required before an NCT id is
   accepted as an instance's embedded record).
2. ``fetch <NCT_ID> <out.json>``  -- fetch the full record, extract exactly
   the relevant protocolSection sub-modules (verbatim, not paraphrased) into
   a small embeddable JSON at the SAME key nesting the real API uses
   (protocolSection.designModule.studyType, etc. -- matching the field paths
   listed in REQUIRED_FIELDS below), and writes it to
   ``out.json`` for hand-splicing into an instance's top-level "records"
   field. Also writes ``out.json.raw`` with the full untouched API response,
   for audit.

Integrity: every field written is copied byte-for-byte from the API
response (json.load -> json.dump on a pruned dict, no field value edited,
paraphrased, or invented). Nothing is fabricated; a trial missing a
required field is reported as missing, never backfilled.

Usage:
    python3 src/obligationbench/construction/fetch_ctgov.py check  <NCT_ID>
    python3 src/obligationbench/construction/fetch_ctgov.py fetch  <NCT_ID> <out.json>
"""
from __future__ import annotations

import json
import subprocess
import sys

API = "https://clinicaltrials.gov/api/v2/studies/%s"

# The exact protocolSection sub-modules this domain's atoms read.
REQUIRED_FIELDS = (
    ("protocolSection.designModule.studyType", "study type (interventional/observational)"),
    ("protocolSection.designModule.phases", "trial phase(s)"),
    ("protocolSection.statusModule.overallStatus", "overall recruitment/completion status"),
    ("protocolSection.sponsorCollaboratorsModule.leadSponsor.name", "lead sponsor name"),
    ("protocolSection.designModule.enrollmentInfo.count", "enrollment count (sparse in older trials)"),
    ("protocolSection.outcomesModule.primaryOutcomes", "primary outcomes list (sparse in older trials)"),
)


def _get(d, dotted):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            return None
    return cur


def fetch_raw(nct_id: str):
    r = subprocess.run(
        ["curl", "-s", "--max-time", "40", API % nct_id],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def report_fields(doc) -> bool:
    """Print presence/value of every required field; return True iff all present."""
    all_present = True
    for path, desc in REQUIRED_FIELDS:
        val = _get(doc, path)
        present = val is not None
        all_present = all_present and present
        shown = val
        if isinstance(shown, list) and shown and isinstance(shown[0], dict) and "measure" in shown[0]:
            shown = [o.get("measure") for o in shown][:2]
        print("  %-70s %-9s %s" % (path, "PRESENT" if present else "MISSING", desc if not present else str(shown)[:90]))
    return all_present


def cmd_check(nct_id: str) -> int:
    doc = fetch_raw(nct_id)
    if doc is None:
        print("FAILED to fetch %s" % nct_id)
        return 2
    idm = _get(doc, "protocolSection.identificationModule") or {}
    print("%s  %s" % (nct_id, (idm.get("briefTitle") or "")[:80]))
    ok = report_fields(doc)
    print("ALL REQUIRED FIELDS PRESENT" if ok else "MISSING one or more required fields -- reject this trial")
    return 0 if ok else 3


def cmd_fetch(nct_id: str, out_path: str) -> int:
    doc = fetch_raw(nct_id)
    if doc is None:
        print("FAILED to fetch %s" % nct_id)
        return 2
    ps = doc.get("protocolSection", {})
    pruned = {
        "protocolSection": {
            "identificationModule": {
                k: ps.get("identificationModule", {}).get(k)
                for k in ("nctId", "briefTitle", "organization")
                if k in ps.get("identificationModule", {})
            },
            "statusModule": {
                k: ps.get("statusModule", {}).get(k)
                for k in ("overallStatus", "startDateStruct", "completionDateStruct", "statusVerifiedDate")
                if k in ps.get("statusModule", {})
            },
            "designModule": {
                k: ps.get("designModule", {}).get(k)
                for k in ("studyType", "phases", "enrollmentInfo", "designInfo")
                if k in ps.get("designModule", {})
            },
            "sponsorCollaboratorsModule": {
                k: ps.get("sponsorCollaboratorsModule", {}).get(k)
                for k in ("leadSponsor", "collaborators")
                if k in ps.get("sponsorCollaboratorsModule", {})
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {kk: o.get(kk) for kk in ("measure", "timeFrame") if kk in o}
                    for o in (ps.get("outcomesModule", {}).get("primaryOutcomes") or [])[:2]
                ]
            },
            "conditionsModule": {
                k: ps.get("conditionsModule", {}).get(k)
                for k in ("conditions",)
                if k in ps.get("conditionsModule", {})
            },
        }
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pruned, fh, indent=2, sort_keys=True)
    with open(out_path + ".raw", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    print("wrote %s (pruned, embeddable) and %s (full raw response, audit copy)" % (out_path, out_path + ".raw"))
    report_fields(doc)
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[0]
    if cmd == "check" and len(argv) >= 2:
        return cmd_check(argv[1])
    if cmd == "fetch" and len(argv) >= 3:
        return cmd_fetch(argv[1], argv[2])
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
