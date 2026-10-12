#!/usr/bin/env python3
"""Add cells run after the rerun (new cells, not reruns) to results/.

A new cell is a run directory <folder>/<modeltag>-<domain>/<arm>__<anchor>/
written by experiments/slurm/new_cells.sbatch with the corrected runner. It is copied
into results/ unchanged except for a marker on every summary entry and
per-episode sidecar:

  "new_cell": {"commit": <code commit>, "budget_rule": "own-class", "jobs": [...]}

which tells experiments/analysis/budget_accounting_audit.py that the episode was charged by
the corrected rule (the same check as for a rerun episode). A cell that already
exists in results/ is never overwritten, and a cell with a failed or missing
episode is refused.

Usage: python3 experiments/analysis/import_new_cells.py --src <pulled results dir> --commit <sha> --jobs <id> ... [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def dump(obj, p):
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="the results/ directory pulled from the cluster")
    ap.add_argument("--commit", required=True, help="the code commit the runs used (CODE_COMMIT)")
    ap.add_argument("--jobs", nargs="+", required=True, help="the cluster job ids")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    marker = {"commit": a.commit, "budget_rule": "own-class", "jobs": sorted(a.jobs)}
    runs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.src, "*", "*", "*", "summary.json")))
    bad = 0
    for src in runs:
        rel = os.path.relpath(src, a.src)
        dst = os.path.join(RESULTS, rel)
        s = load(os.path.join(src, "summary.json"))
        eps = s["episodes"]
        expected = len(s["run"]["instances"])
        failed = [e["instance"] for e in eps if e.get("error")]
        if os.path.exists(dst):
            print("exists   %s (left alone)" % rel); bad += 1
            continue
        if failed or len(eps) != expected:
            print("refused  %s: %d of %d episodes, failed %s" % (rel, len(eps), expected, failed)); bad += 1
            continue
        print("import   %s: %d episodes" % (rel, len(eps)))
        if a.dry_run:
            continue
        shutil.copytree(src, dst)
        for e in s["episodes"]:
            e["new_cell"] = marker
        dump(s, os.path.join(dst, "summary.json"))
        for side in sorted(glob.glob(os.path.join(dst, "_episodes", "*.json"))):
            dump(dict(load(side), new_cell=marker), side)
    print("%d cells, %d not imported" % (len(runs), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
