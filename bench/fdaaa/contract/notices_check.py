#!/usr/bin/env python3
"""Check the compiled FDAAA deadline logic against FDA's Notices of Noncompliance.

FDA has sent a Notice of Noncompliance for eight trials. This script fetches
those eight registrations from the ClinicalTrials.gov API, applies the
compiled results-reporting deadline (42 CFR 11.44: results due one calendar
year after primary completion, or two years after a certification of delayed
submission filed before that date, with no administrative grace period), and
reports how many it classifies as late or overdue at the snapshot date. None
of the eight is among the 319 trials of the FDAAA test set.

This is the standalone part of contract/agreement_pilot.py, the exploratory
script that also compared deadline variants on 6,000 trials; that comparison
is not used in the paper.

Usage: python3 bench/fdaaa/contract/notices_check.py   (needs network access)
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta

SNAP = date(2026, 8, 3)  # registry snapshot date used throughout the FDAAA family
NOTICES = ["NCT01727336", "NCT02326454", "NCT03013751", "NCT03045406",
           "NCT03052816", "NCT03064438", "NCT03099499", "NCT03785340"]
API = "https://clinicaltrials.gov/api/v2/studies?"


def parse(d, end_of_period):
    """Registry dates come as YYYY-MM-DD, YYYY-MM, or YYYY. A partial date is
    read as the end (deadline side) or start of the period it names."""
    if not d:
        return None
    p = d.split("-")
    if len(p) == 3:
        return date(int(p[0]), int(p[1]), int(p[2]))
    if len(p) == 2:
        y, m = int(p[0]), int(p[1])
        if end_of_period:
            return date(y + m // 12, m % 12 + 1, 1) - timedelta(days=1)
        return date(y, m, 1)
    return date(int(p[0]), 12, 31) if end_of_period else date(int(p[0]), 1, 1)


def plus_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # 29 February
        return d.replace(year=d.year + n, day=28)


def submitted(study):
    st = study["protocolSection"]["statusModule"]
    d = parse(st.get("resultsFirstSubmitDate"), False)
    if d:
        return d
    tracking = study.get("derivedSection", {}).get("miscInfoModule", {}).get("submissionTracking", {})
    return parse(tracking.get("estimatedResultsFirstSubmitDate"), False)


def verdict(primary_completion, certification, submission):
    due = plus_years(primary_completion, 1)
    if certification is not None and certification <= due:
        due = plus_years(certification, 2)
    if submission is not None:
        return "reported" if submission <= due else "reported-late"
    return "overdue" if due < SNAP else "ongoing"


def main() -> int:
    query = urllib.parse.urlencode({
        "filter.ids": ",".join(NOTICES),
        "fields": "protocolSection.identificationModule.nctId,protocolSection.statusModule,"
                  "derivedSection.miscInfoModule.submissionTracking",
        "pageSize": 20})
    with urllib.request.urlopen(API + query, timeout=90) as fh:
        studies = json.load(fh)["studies"]
    adverse = 0
    for s in studies:
        nct = s["protocolSection"]["identificationModule"]["nctId"]
        st = s["protocolSection"]["statusModule"]
        raw = (st.get("primaryCompletionDateStruct", {}).get("date")
               or st.get("completionDateStruct", {}).get("date"))
        v = verdict(parse(raw, True), parse(st.get("dispFirstSubmitDate"), False), submitted(s))
        adverse += v in ("reported-late", "overdue")
        print("%s  completion=%-10s certification=%-10s -> %s"
              % (nct, raw, st.get("dispFirstSubmitDate"), v))
    print("late or overdue: %d of %d" % (adverse, len(NOTICES)))
    return 0 if adverse == len(NOTICES) == len(studies) else 1


if __name__ == "__main__":
    sys.exit(main())
