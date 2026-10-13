#!/usr/bin/env python3
"""Score the FDAAA family: the enforcement machinery on a contract we did not author.

Without this family there is no point anywhere in the paper where the
system meets an artifact the authors did not design. This scores the family
that closes that gap.

What is external here, precisely, so the claim can be checked rather than
believed:

  * the CONTRACT -- 42 CFR Part 11, implementing FDAAA 801, written by
    Congress and FDA;
  * the RECORDS -- real ClinicalTrials.gov registrations, frozen, not
    degraded;
  * the VIOLATIONS -- naturally occurring at 56.5% on this split, nothing
    injected and nothing sampled to a target;
  * the CROSS-CHECK -- the Bennett Institute's FDAAA TrialsTracker, an
    independently authored decision procedure over the same registry.

What is NOT external, and the paper must say so: the tracker reads the same
registry fields we do, so the independence is of contract and of decision
procedure, not of data. The only regulator-adjudicated cases are FDA's eight
Notices of Noncompliance.

Scoring reuses experiments/analysis/build_w2a_tables -- the same instance_facts and the
same score() every other family is scored by, with the FDAAA facts injected
into its table. Writing a second scorer for the external family would make
any difference it found uninterpretable.

Writes: results/tableFDAAA.json
Usage:  python3 experiments/analysis/build_fdaaa_table.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw   # noqa: E402  -- the single scorer

BENCH = os.path.join(PILOT, "data", "fdaaa", "instances")
RUNS = os.path.join(PILOT, "results", "fdaaa")
RESULTS = os.path.join(PILOT, "results")

LABEL = {"shortest_route": "ceiling-targeting",
         "cheapest_literal": "cheapest literal",
         "random_masked": "random (masked)",
         "masked_reactive_prefix": "prefix-first, reactive"}


def load_facts():
    """FDAAA instance facts, through the same reader every family uses."""
    facts = {}
    for p in sorted(glob.glob(os.path.join(BENCH, "fdaaa-*.json"))):
        f = bw.instance_facts(p)
        f["domain"] = "fdaaa"
        f["split"] = "external"
        facts[f["id"]] = f
    return facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    man_p = os.path.join(BENCH, "manifest.json")
    if not os.path.exists(man_p):
        print("no FDAAA instances; run src/suffice/construction/fdaaa/build_fdaaa_instances.py")
        return 1
    man = json.load(open(man_p, encoding="utf-8"))

    facts = load_facts()
    bw.FACTS.update(facts)          # inject, do not reimplement

    # the statute's own verdict per instance, carried on the instance
    verdict = {}
    for p in sorted(glob.glob(os.path.join(BENCH, "fdaaa-*.json"))):
        inst = json.load(open(p, encoding="utf-8"))
        meta = inst["notes_for_annotators"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        verdict[inst["id"]] = meta.get("verdict")

    out = {"cells": {}, "manifest": man,
           "_note": "contract authored by Congress and FDA; records real and "
                    "undegraded; violations naturally occurring"}

    run_dirs = sorted(glob.glob(os.path.join(RUNS, "policy-*")))
    # LLM arms on the same 319 registrations: results/fdaaa/<model>/<arm>__B20
    run_dirs += sorted(glob.glob(os.path.join(RUNS, "qwen2.5-32b", "*__B20")))
    for d in run_dirs:
        arm = os.path.basename(d).replace("policy-", "")
        if arm.endswith("__B20"):
            arm = "llm:" + arm[:-5]
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            continue
        rows, per_verdict = [], collections.defaultdict(list)
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            s = bw.score(e)
            if not s:
                continue
            rows.append(s)
            per_verdict[verdict.get(e["instance"])].append(s)
        if not rows:
            continue
        n = len(rows)
        cell = {
            "label": LABEL.get(arm, arm),
            "n_instances": n,
            "n_clusters": len({r["facts"]["family"] for r in rows}),
            "ofar_a": round(100.0 * sum(r["ofar_a"] for r in rows) / n, 1),
            "utility": round(sum(r["utility"] for r in rows) / n, 1),
            "abstain": round(100.0 * sum(r["abstain"] for r in rows) / n, 1),
            "cost": round(sum(r["cost"] for r in rows) / n, 2),
        }
        # The overdue stratum is the abstention test. A trial that never
        # filed leaves the submission atom UNAVAILABLE, not false, so the
        # fail-closed ledger cannot license the top claim however overdue
        # the trial is -- correct behaviour, and the interesting one.
        # "Top claim" means level 3 exactly, i.e. utility 100. An earlier
        # threshold of >66.0 counted level 2 (66.7) as the top claim and
        # reported 100% where the truth is 0% -- the opposite of the
        # finding.
        def top_rate(rs):
            return round(100.0 * sum(1 for r in rs if r["utility"] >= 99.9)
                         / len(rs), 1)
        for name in ("overdue", "late", "compliant"):
            g = per_verdict.get(name) or []
            if not g:
                continue
            cell["%s_n" % name] = len(g)
            cell["%s_top_claim_rate" % name] = top_rate(g)
            cell["%s_ofar" % name] = round(
                100.0 * sum(r["ofar_a"] for r in g) / len(g), 1)
            cell["%s_utility" % name] = round(
                sum(r["utility"] for r in g) / len(g), 1)
        out["cells"][arm] = cell

    with open(os.path.join(RESULTS, "tableFDAAA.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print("FDAAA: %d instances over %d sponsor clusters, adverse rate %.1f%%"
          % (man["n_instances"], man["n_clusters"], man["adverse_rate"]))
    print("       external agreement %.2f%% on %d"
          % (man["external_agreement_pct"], man["external_checked"]))
    for arm, c in sorted(out["cells"].items()):
        print("  %-24s OFAR-A=%4.1f  Util=%5.1f  abstain=%5.1f  cost=%.2f"
              % (c["label"], c["ofar_a"], c["utility"], c["abstain"], c["cost"]))
        for name in ("late", "overdue", "compliant"):
            if "%s_n" % name in c:
                print("      %-10s n=%3d  top claim %5.1f%%  util %5.1f  "
                      "OFAR %.1f%%"
                      % (name, c["%s_n" % name], c["%s_top_claim_rate" % name],
                         c["%s_utility" % name], c["%s_ofar" % name]))
    print("wrote results/tableFDAAA.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
