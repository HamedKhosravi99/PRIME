#!/usr/bin/env python3
"""When an authored static contract and a committed program are the same object.

The paper concedes that a hand-written static contract matches or beats the
committed program on realized utility in 3 of 5 domains, which reads as
evidence that the apparatus does not earn its place. That concession is
measured entirely on the carried-forward instance family, and this script
asks what the comparison could have shown there.

A static contract in this setting is the CONJUNCTIVE collapse of a claim's
licence family: to license c it demands the union of c's routes. A committed
minimal program licenses c on ANY single satisfied route. Those two objects
differ only when a claim has more than one route. So the question is a census
over the instances, not an experiment:

  * carried-forward family: every claim, how many routes?
  * confirmatory split: same question, and where routes DO branch, how much
    more evidence does the union demand than the cheapest satisfied route?

The result is a scope statement, not a win: the regime where the authored
contract is competitive is identifiable in advance from the specification
view (route topology is spec-level), and it is the regime where the two
contracts coincide. Where they do not coincide, the union costs strictly more
work for the same licence -- which is what "minimal work" names -- but it
licenses exactly the same claims, so the difference is priced in evidence
gathered and NOT in utility retained.

Writes: results/contract_semantics.json
Usage:  python3 experiments/analysis/contract_semantics.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(PILOT, "results")
BENCH = os.path.join(PILOT, "src", "suffice", "construction")

# the five domains of the carried-forward family, in the order the
# static-contract table reports them
CARRIED = (("security", "osv"), ("financial", "fin"), ("drugsafety", "fda"),
           ("clinical", "ctgov"), ("research", "res"))


def _meta(d):
    m = d.get("notes_for_annotators") or {}
    return json.loads(m) if isinstance(m, str) else m


def _route_census(files):
    """Per-claim route multiplicity over a set of instance files."""
    claims = multi = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for _c, rs in (d.get("gold_licenses") or {}).items():
            rs = rs or []
            claims += 1
            if len(rs) > 1:
                multi += 1
    return {"claims": claims, "multi_route": multi}


def _work_census(files):
    """Evidence each contract semantics must gather for the SAME top claim.

    Reported per topology, because topology is the specification-view property
    that predicts which regime an instance is in.
    """
    by = collections.defaultdict(
        lambda: {"n": 0, "min": [], "union": [], "bites": 0})
    sat_profile = collections.Counter()
    licence_gap = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        gv = {a["id"]: a.get("gold_value") for a in d["atoms"]}
        lv = {c["id"]: c["level"] for c in d["claims"]}
        topo = _meta(d).get("topology", "?")

        for _c, rs in (d.get("gold_licenses") or {}).items():
            rs = rs or []
            if len(rs) < 2:
                continue
            sat = sum(1 for r in rs if all(gv.get(x) == "T" for x in r))
            sat_profile["%d_of_%d" % (sat, len(rs))] += 1

        # the strongest claim the committed program can license here
        best = None
        for c, rs in (d.get("gold_licenses") or {}).items():
            rs = rs or []
            sat = [r for r in rs if all(gv.get(x) == "T" for x in r)]
            if sat and (best is None or lv.get(c, 0) > best[0]):
                best = (lv.get(c, 0), rs, sat)
        if best is None:
            continue
        _lvl, rs, sat = best
        union = set().union(*[set(r) for r in rs])
        cheapest = min(sat, key=len)
        # a static conjunctive contract licenses the same claim only if the
        # whole union holds; count where it could not
        if not all(gv.get(x) == "T" for x in union):
            licence_gap += 1
        for k in (topo, "ALL"):
            b = by[k]
            b["n"] += 1
            b["min"].append(len(cheapest))
            b["union"].append(len(union))
            if len(union) > len(cheapest):
                b["bites"] += 1

    out = {}
    for k, b in by.items():
        if not b["n"]:
            continue
        mn, un = statistics.mean(b["min"]), statistics.mean(b["union"])
        bit = [(u, m) for u, m in zip(b["union"], b["min"]) if u > m]
        out[k] = {
            "n": b["n"],
            "min_atoms": round(mn, 2),
            "union_atoms": round(un, 2),
            "extra_atoms": round(un - mn, 2),
            "pct_more_work": round(100.0 * (un - mn) / mn, 1) if mn else 0.0,
            "n_bites": b["bites"],
            "pct_bites": round(100.0 * b["bites"] / b["n"], 1),
            "bites_min_atoms": round(statistics.mean([m for _u, m in bit]), 2) if bit else None,
            "bites_union_atoms": round(statistics.mean([u for u, _m in bit]), 2) if bit else None,
        }
    return out, dict(sat_profile), licence_gap


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    carried = {}
    carried_files = []
    for name, sub in CARRIED:
        fs = [f for f in sorted(glob.glob(
            os.path.join(PILOT, "data", "pilot", "instances", sub, "**", "*.json"),
            recursive=True)) if "openvocab" not in f]
        carried[name] = _route_census(fs)
        carried_files += fs
    carried_all = _route_census(carried_files)

    conf_files = sorted(glob.glob(os.path.join(
        PILOT, "data", "suffice", "*", "heldout_balanced", "*.json")))
    conf = _route_census(conf_files)
    work, sat_profile, licence_gap = _work_census(conf_files)

    out = {
        "carried_forward": {"per_domain": carried, "total": carried_all},
        "confirmatory": conf,
        "work": work,
        "route_satisfaction_profile": sat_profile,
        "licence_gap_instances": licence_gap,
        "_note": ("A static contract is the conjunctive collapse of a licence "
                  "family; it differs from a committed minimal program only "
                  "for multi-route claims. licence_gap_instances counts "
                  "instances where the union is unsatisfiable though some "
                  "route holds -- where the static contract would license "
                  "strictly less. It is 0 here because this generator makes "
                  "the routes of a branching claim jointly live or jointly "
                  "dead, so the benchmark prices the WORK difference and not "
                  "route selection under partial failure."),
    }
    json.dump(out, open(os.path.join(RESULTS, "contract_semantics.json"), "w"),
              indent=1, sort_keys=True)

    print("carried-forward family: %d claims, %d multi-route"
          % (carried_all["claims"], carried_all["multi_route"]))
    print("confirmatory split    : %d claims, %d multi-route"
          % (conf["claims"], conf["multi_route"]))
    print("route satisfaction (multi-route claims):", sat_profile)
    print("instances where union unsatisfiable but a route holds:", licence_gap)
    for k in ("nested", "branching", "ALL"):
        if k in work:
            w = work[k]
            print("  %-10s n=%3d  committed %.2f vs static %.2f atoms "
                  "(+%.1f%%), bites %d (%.1f%%)"
                  % (k, w["n"], w["min_atoms"], w["union_atoms"],
                     w["pct_more_work"], w["n_bites"], w["pct_bites"]))
    print("wrote results/contract_semantics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
