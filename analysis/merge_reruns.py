#!/usr/bin/env python3
"""Replace the episodes of results/rerun_manifest.csv with their reruns.

For every manifest row whose rerun finished (runner/rerun_episodes.py writes
<rerun>/<folder>/<modeltag>-<domain>/<run>/<instance>/), the logged episode is
replaced: its summary.json entry (in place, same position), its trace, its
per-episode sidecar, and, for the verified condition, its verifier record. The
replaced entry carries "rerun": {"commit", "budget_rule": "own-class", ...}, which
tells analysis/budget_accounting_audit.py to check it under the corrected rule.
GLEAN's entry is rebuilt from the rerun trajectory of the fully informed agent
and GLEAN's verdict on it (runner/glean_one.py), the way runner/glean.py builds
it. The files replaced are moved to results_superseded/ with their paths, so the
logged runs stay available. Rows whose rerun is missing or failed are reported
and left alone; rerunning the script after more reruns arrive is safe.

Usage: python3 analysis/merge_reruns.py --rerun <dir pulled from the cluster>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUPERSEDED = os.path.join(ROOT, "results_superseded")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def dump(obj, p):
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def keep_old(path: str) -> None:
    """Move a logged file to results_superseded/ (once; a later merge never overwrites it)."""
    if not os.path.exists(path):
        return
    dst = os.path.join(SUPERSEDED, os.path.relpath(path, ROOT))
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(path, dst)


def replace_entry(summary_path: str, iid: str, new_entry) -> None:
    keep_old(summary_path)
    s = load(summary_path)
    idx = [i for i, e in enumerate(s["episodes"]) if e.get("instance") == iid]
    if len(idx) != 1:
        raise SystemExit("%s: %d entries for %s" % (summary_path, len(idx), iid))
    s["episodes"][idx[0]] = new_entry
    dump(s, summary_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rerun", required=True, help="the rerun/ directory pulled from the cluster")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "results", "rerun_manifest.csv"))
    ap.add_argument("--commit", required=True, help="the code commit the reruns used (CODE_COMMIT)")
    a = ap.parse_args(argv)
    with open(a.manifest, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    marker = {"commit": a.commit, "budget_rule": "own-class",
              "reason": "budget accounting of the logged run (docs/KNOWN_ISSUES.md)"}
    merged = missing = already = 0
    # agents first, GLEAN after (it reads the merged trajectory)
    for r in sorted(rows, key=lambda r: r["entry"] == "glean"):
        iid = r["instance"]
        cell = "%s-%s" % (r["modeltag"], r["domain"])
        src = os.path.join(a.rerun, r["folder"], cell, r["run"], iid)
        dst = os.path.join(ROOT, r["logged_dir"])
        if r["entry"] == "glean":
            verdict_p = os.path.join(src, "glean_verdict.json")
            base_dir = os.path.join(ROOT, "results", "w2a", cell, "react_program_attested__%s" % r["anchor"])
            base = [e for e in load(os.path.join(base_dir, "summary.json"))["episodes"] if e["instance"] == iid]
            if not os.path.exists(verdict_p) or not base or not base[0].get("rerun"):
                print("missing  %s %s %s" % (r["folder"], r["run"], iid)); missing += 1
                continue
            cur = [e for e in load(os.path.join(dst, "summary.json"))["episodes"] if e["instance"] == iid][0]
            if cur.get("rerun", {}).get("commit") == a.commit:
                already += 1
                continue
            e = dict(base[0])
            ext = load(verdict_p)["episodes"][iid]
            e["arm"] = "ext_glean"
            e["trace"] = os.path.join("results", "w2a", cell, r["run"], iid + ".jsonl")
            if ext.get("p_hat") is not None and not ext.get("accepted"):
                e["emitted"] = []
            e["ext"] = ext
            e["rerun"] = dict(marker, glean_verdict="runner/glean_one.py")
            keep_old(os.path.join(dst, iid + ".jsonl"))
            shutil.copy2(os.path.join(base_dir, iid + ".jsonl"), os.path.join(dst, iid + ".jsonl"))
            replace_entry(os.path.join(dst, "summary.json"), iid, e)
            merged += 1
            print("merged   %s %s %s (GLEAN accepted=%s)" % (r["folder"], r["run"], iid, ext.get("accepted")))
            continue
        side = os.path.join(src, "_episodes", iid + ".json")
        if not os.path.exists(side) or load(side).get("error"):
            print("missing  %s %s %s" % (r["folder"], r["run"], iid)); missing += 1
            continue
        cur = [e for e in load(os.path.join(dst, "summary.json"))["episodes"] if e["instance"] == iid][0]
        if cur.get("rerun", {}).get("commit") == a.commit:
            already += 1
            continue
        new = [e for e in load(os.path.join(src, "summary.json"))["episodes"] if e["instance"] == iid][0]
        new = dict(new)
        new["trace"] = os.path.join(r["logged_dir"], iid + ".jsonl")
        new["rerun"] = marker
        for name in (iid + ".jsonl", os.path.join("_episodes", iid + ".json"), os.path.join("_verify", iid + ".json")):
            s, d = os.path.join(src, name), os.path.join(dst, name)
            if os.path.exists(s):
                keep_old(d)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
        side_obj = dict(load(os.path.join(dst, "_episodes", iid + ".json")), trace=new["trace"], rerun=marker)
        dump(side_obj, os.path.join(dst, "_episodes", iid + ".json"))
        replace_entry(os.path.join(dst, "summary.json"), iid, new)
        merged += 1
        print("merged   %s %s %s calls %s -> %s, emitted %s -> %s" % (
            r["folder"], r["run"], iid, cur.get("tool_calls"), new.get("tool_calls"),
            [x["claim"] for x in cur.get("emitted") or []], [x["claim"] for x in new.get("emitted") or []]))
    print("\n%d merged, %d already merged, %d missing (of %d)" % (merged, already, missing, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
