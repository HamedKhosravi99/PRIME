#!/usr/bin/env python3
"""Mechanics check for the locator-uncertain split, on partial results.

For one run directory (e.g. results/w2loc/qwen2.5-32b-security/prime_guided__bmin)
it reads every finished episode's trace and checks what the split is
supposed to exercise:

  * the agent's inspect_record calls go to candidate fields (true or decoy),
    not to arbitrary paths, and some of them are decoys (the search is real);
  * an atom is attested only when its TRUE field was read (attestation is
    unchanged by the hint);
  * the strongest stated conclusion is supported (the mask holds);
  * preliminary utility on the finished tasks against the blind candidate
    policy and the oracle-locator policy on the SAME tasks.

Usage: python3 analysis/locator_smoke.py results/w2loc/qwen2.5-32b-security/prime_guided__bmin
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, os.path.join(PILOT, "runner"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402

BENCH = os.path.join(PILOT, "bench", "instances_v2")


def task_of(iid):
    dom = {"sec": "security", "cli": "clinical", "dru": "drugsafety", "fin": "financial"}[iid[:3]]
    return json.load(open(os.path.join(BENCH, dom, "heldout_locator", iid + ".json"), encoding="utf-8")), dom


def main(argv=None) -> int:
    run = (argv or sys.argv[1:])[0].rstrip("/")
    arm_anchor = os.path.basename(run)
    anchor = arm_anchor.rsplit("__", 1)[1]
    sidecars = sorted(glob.glob(os.path.join(run, "_episodes", "*.json")))
    if not sidecars:
        print("no finished episodes in", run)
        return 1
    n = reads = true_reads = decoy_reads = other_reads = 0
    bad_attest = bad_mask = errors = 0
    util = {}
    for p in sidecars:
        e = json.load(open(p, encoding="utf-8"))
        iid = e["instance"]
        if e.get("error"):
            errors += 1
            continue
        task, dom = task_of(iid)
        true_path = {a["id"]: a["resolver"]["params"]["field_path"] for a in task["atoms"] if a.get("resolver")}
        cands = {c for a in task["atoms"] if a.get("spec_hint") for c in a["spec_hint"]["candidates"]}
        trace_path = os.path.join(run, iid + ".jsonl")
        if not os.path.exists(trace_path):
            continue
        rows = [json.loads(l) for l in open(trace_path, encoding="utf-8") if l.strip()]
        read_paths = []
        for r in rows:
            if r.get("type") == "action" and r.get("name") == "inspect_record":
                fp = str((r.get("params") or {}).get("field_path", ""))
                read_paths.append(fp)
                reads += 1
                if fp in true_path.values():
                    true_reads += 1
                elif fp in cands:
                    decoy_reads += 1
                else:
                    other_reads += 1
            if r.get("type") == "attestation":
                att = r.get("attestation") or r
                atom = att.get("atom")
                fp = str((att.get("params") or {}).get("field_path", ""))
                if atom in true_path and fp != true_path[atom]:
                    bad_attest += 1
        f = bw.FACTS.get(iid)
        s = bw.score(e)
        if f and s:
            n += 1
            util[iid] = s["utility"]
            if s["ofar_a"]:
                bad_mask += 1
    # the two policies on the same finished tasks
    pol = {}
    for arm in ("shortest_route_candidates", "shortest_route"):
        vals = []
        for dom in bw.DOMAINS:
            sp = os.path.join(PILOT, "results", "w2loc", "policy-%s" % dom, "%s__%s" % (arm, anchor), "summary.json")
            if not os.path.exists(sp):
                continue
            for pe in json.load(open(sp, encoding="utf-8"))["episodes"]:
                if pe["instance"] in util:
                    ps = bw.score(pe)
                    if ps:
                        vals.append(ps["utility"])
        pol[arm] = sum(vals) / len(vals) if vals else float("nan")
    print("run: %s" % run)
    print("finished episodes: %d scored, %d errors" % (n, errors))
    print("inspect_record reads: %d  (true field %d, decoy candidate %d, other path %d)" % (reads, true_reads, decoy_reads, other_reads))
    print("attestations on a non-true field: %d   unsupported statements (mask): %d" % (bad_attest, bad_mask))
    if n:
        print("preliminary utility on these %d tasks: this arm %.1f | blind candidates policy %.1f | oracle locators %.1f"
              % (n, sum(util.values()) / n, pol["shortest_route_candidates"], pol["shortest_route"]))
    ok = bad_attest == 0 and bad_mask == 0 and other_reads <= 0.1 * max(1, reads)
    print("SMOKE:", "PASS" if ok else "CHECK")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
