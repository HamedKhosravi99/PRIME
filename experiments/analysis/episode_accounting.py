#!/usr/bin/env python3
"""Every run's episodes: expected, logged, completed, and failed.

Reads results/**/summary.json (no model is called). For each run folder it counts
the episodes the run logged, the failed ones (an episode whose model calls still
timed out after the client's retries, or that ended on another error, carries no
measurement and is left out of every rate), and, for runs on a SUFFICE
split, the tasks the split holds, so that a missing episode would show. Writes
results/excluded_episodes.json with every failed episode and its error class, and
prints the runs with failures.

Usage: python3 experiments/analysis/episode_accounting.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
SKIP = ("_backup", "tmp", "_smoke_record", "superseded_by_rerun")  # scratch and superseded folders


def error_class(err: str) -> str:
    for name in ("APITimeoutError", "BadRequestError", "SchemaError", "timeout"):
        if name in err:
            return name
    m = re.match(r"(\w+(?:Error|Exception))", err)
    return m.group(1) if m else "other"


def split_size(instance_path: str):
    """Tasks in the SUFFICE split an episode's instance belongs to."""
    # logged runs name bench/instances_v2/; runs from this layout name data/suffice/
    m = re.search(r"(?:instances_v2|suffice)/([a-z]+)/([a-z_]+)/[^/]+\.json$", instance_path or "")
    if not m:
        return None, None
    d = os.path.join(ROOT, "data", "suffice", m.group(1), m.group(2))
    if not os.path.isdir(d):
        return None, None
    return "%s/%s" % (m.group(1), m.group(2)), sum(1 for f in os.listdir(d) if f.endswith(".json"))


def main() -> int:
    runs = []
    for dp, dn, fn in os.walk(RESULTS):
        dn[:] = sorted(d for d in dn if d not in SKIP)
        if "summary.json" not in fn:
            continue
        try:
            with open(os.path.join(dp, "summary.json"), encoding="utf-8") as fh:
                eps = json.load(fh).get("episodes", [])
        except (OSError, ValueError):
            continue
        failed = [{"instance": e.get("instance"), "error_class": error_class(str(e.get("error"))),
                   "error": str(e.get("error"))[:200]} for e in eps if e.get("error")]
        split, expected = split_size(eps[0].get("instance_path", "") if eps else "")
        logged = len({e.get("instance") for e in eps})
        runs.append({"run": os.path.relpath(dp, RESULTS), "split": split,
                     "expected_per_domain_split": expected, "logged": logged,
                     "completed": logged - len(failed), "failed": failed})
    out = {
        "rule": ("an episode whose model calls still timed out after the client's retries, or that "
                 "ended on another error, carries no measurement and is left out of every rate"),
        "runs": runs,
        "totals": {"runs": len(runs), "episodes": sum(r["logged"] for r in runs),
                   "failed": sum(len(r["failed"]) for r in runs),
                   "by_error_class": dict(Counter(f["error_class"] for r in runs for f in r["failed"]))},
    }
    with open(os.path.join(RESULTS, "excluded_episodes.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    short = [r for r in runs if r["expected_per_domain_split"] and r["logged"] < r["expected_per_domain_split"]
             and not r["run"].startswith(("v2dev", "_"))]
    for r in runs:
        if r["failed"]:
            print("%-60s %4d logged %3d failed  %s" % (r["run"], r["logged"], len(r["failed"]),
                                                      dict(Counter(f["error_class"] for f in r["failed"]))))
    print("%(runs)d runs, %(episodes)d episodes, %(failed)d failed: %(by_error_class)s" % out["totals"])
    if short:
        print("runs with fewer logged episodes than their split holds (by design for subsets):",
              ", ".join(r["run"] for r in short[:10]), "..." if len(short) > 10 else "")
    print("wrote results/excluded_episodes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
