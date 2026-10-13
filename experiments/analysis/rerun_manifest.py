#!/usr/bin/env python3
"""The episodes to rerun with the corrected runner, in priority tiers.

Reads results/budget_accounting_audit.json and lists every affected episode:
those whose charged count differs between the logged and the corrected rule
for an agent that sees the remaining budget, those that would have ended at a
different call for every other agent, and those whose trace and summary come
from different runs. Every other episode is charged identically under both
rules, so its logged run is what the corrected runner produces.

Tiers, by where the paper reports the cell:
  1  Table 2 (judging textual evidence, both rows that can change) and the
     registered guidance comparisons with Qwen2.5-32B-Instruct (Section 6.2)
  2  the rest of Table 3 (closest control systems) and of Figure 2
  3  appendix cells (most of them gpt-oss-20b)

Writes results/rerun_manifest.csv; the cluster driver src/prime_agents/rerun_episodes.py
reads it.
Usage: python3 experiments/analysis/rerun_manifest.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
MODEL_ID = {"qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct", "gptoss-20b": "openai/gpt-oss-20b"}
SPLIT = {"w2a": "heldout_balanced", "w2loc": "heldout_locator", "w2text": "heldout_text",
         "w2seed": "heldout_balanced", "blind_verification": "heldout_text"}
FIELDS = ["tier", "exhibit", "entry", "folder", "modeltag", "model", "domain", "run", "arm", "anchor",
          "temperature", "seed", "instance", "instance_path", "reason", "logged_dir"]


def tier_of(folder: str, tag: str, arm: str, anchor: str):
    if tag == "qwen2.5-32b":
        if folder in ("w2text", "blind_verification"):
            return 1, "Table 2"
        if folder == "w2a" and arm in ("prime_guided", "masked_reactive_v") and anchor in ("bmin", "bmin1", "2bmin", "B20"):
            return 1, "registered guidance comparisons, Figure 2"
        if folder == "w2a" and arm in ("ext_prooforstop", "ext_researchloop", "ext_agentspec",
                                       "ext_agentspec_examine", "ext_hesp") and anchor == "B20":
            return 2, "Table 3"
        if folder in ("w2a", "w2loc") and arm in ("prime_complete", "prime_guided", "masked_reactive_v"):
            return 2, "Figure 2"
    return 3, "appendix"


def main() -> int:
    with open(os.path.join(RESULTS, "budget_accounting_audit.json"), encoding="utf-8") as fh:
        audit = json.load(fh)
    rows = []
    for c in audit["cells"]:
        if not c["affected"]:
            continue
        chosen = set(c["episodes_differ"] if c["sees_budget"] else c["episodes_cap"]) | set(c["episodes_mismatch"])
        for ep in sorted(chosen):
            parts = ep.split("/")             # folder/<tag>-<domain>/<run>/<instance>
            folder, cell, run, iid = parts[0], parts[1], parts[2], parts[3]
            tag, dom = cell.rsplit("-", 1)
            if run.startswith("ext_glean__"):
                entry, arm, anchor = "glean", "ext_glean", run.split("__")[1]
            elif folder == "blind_verification":
                entry, arm, anchor = "blind_verification", "prime_guided_verified", "B20"
            else:
                entry = "run_batch"
                arm, anchor = run.split("__")[0], run.split("__")[-1]
            temperature, seed = ("0.7", run.split("__")[1][len("T07S"):]) if folder == "w2seed" else ("0", "")
            split = SPLIT[folder]
            ipath = os.path.join("data", "suffice", dom, split, iid + ".json")
            if not os.path.exists(os.path.join(ROOT, ipath)):
                raise SystemExit("instance file missing: %s" % ipath)
            why = ("mismatch" if ep in c["episodes_mismatch"] else
                   "differ" if c["sees_budget"] else "cap")
            tier, exhibit = tier_of(folder, tag, arm, anchor)
            logged = os.path.join("experiments", "blind_verification", "results", cell, run) \
                if folder == "blind_verification" else os.path.join("results", folder, cell, run)
            rows.append({"tier": tier, "exhibit": exhibit, "entry": entry, "folder": folder, "modeltag": tag,
                         "model": MODEL_ID[tag], "domain": dom, "run": run, "arm": arm, "anchor": anchor,
                         "temperature": temperature, "seed": seed, "instance": iid, "instance_path": ipath,
                         "reason": why, "logged_dir": logged})
    rows.sort(key=lambda r: (r["tier"], r["modeltag"], r["folder"], r["run"], r["instance"]))
    with open(os.path.join(RESULTS, "rerun_manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    by = Counter((r["tier"], r["modeltag"]) for r in rows)
    for (t, m), n in sorted(by.items()):
        print("tier %d  %-12s %3d" % (t, m, n))
    print("%d episodes -> results/rerun_manifest.csv" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
