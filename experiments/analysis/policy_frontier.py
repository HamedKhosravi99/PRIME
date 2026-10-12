#!/usr/bin/env python3
"""Policy frontier and the ordering-vs-direction attribution
(docs/EXECUTION_PLAN.md, Part R4, Q4), all LLM-free.

Two things the confirmatory LLM runs cannot answer on their own:

  1. HOW MUCH PLANNING HEADROOM EXISTS AT ALL. `informed_route_planner` knows
     which routes are all-true but still pays full call-class cost;
     `clairvoyant_ceiling` is the nondeployable regret ceiling. If a simple
     deployable heuristic already reaches them, then no control law can win
     on this benchmark and the E2 null is a property of the instances rather
     than a failure to measure.

  2. ORDERING vs DIRECTION (docs/EXECUTION_PLAN.md, Part R4, Q4, pre-committed
     to run UPFRONT on all branching confirmatory instances).
     `masked_reactive_prefix` is reactive with prefix-first ordering and no
     ceiling targeting; `shortest_route` adds ceiling targeting. Their
     difference isolates direction from ordering with no model in the loop.

Both are scripted or computed arms: real ToolExecutor calls and the real
attested ledger, but action selection is deterministic code, so these numbers
are reproducible without a GPU and cannot drift with a model.

Usage: python3 experiments/analysis/policy_frontier.py [--json]
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
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402

RESULTS = os.path.join(PILOT, "results")
ANCHORS = ("bmin", "bmin1", "2bmin", "B20")
# ordered weakest -> strongest; the last two are not deployable policies
FRONTIER = ("random_masked", "cheapest_literal", "masked_reactive_prefix",
            "shortest_route", "informed_route_planner", "clairvoyant_ceiling")
LABEL = {"random_masked": "random (masked)",
         "cheapest_literal": "cheapest literal",
         "masked_reactive_prefix": "prefix-first, reactive",
         "shortest_route": "ceiling-targeting",
         "informed_route_planner": "outcome-informed planner",
         "clairvoyant_ceiling": "clairvoyant ceiling"}


def collect(split: str = "heldout_balanced", topology: str = None) -> dict:
    util = collections.defaultdict(list)
    lvl = lambda x: 100.0 * x / 3.0
    for p in sorted(glob.glob(os.path.join(RESULTS, "w2a", "policy-*",
                                           "*", "summary.json"))):
        arm, anchor = os.path.basename(os.path.dirname(p)).rsplit("__", 1)
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] != split:
                continue
            if topology and f["topology"] != topology:
                continue
            if arm == "planners":
                util[("informed_route_planner", anchor)].append(
                    lvl(e["informed_route_planner_level"]))
                util[("clairvoyant_ceiling", anchor)].append(
                    lvl(e["clairvoyant_ceiling_level"]))
            else:
                s = bw.score(e)
                if s:
                    util[(arm, anchor)].append(s["utility"])
    return {k: (sum(v) / len(v), len(v)) for k, v in util.items() if v}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--topology", default=None)
    a = ap.parse_args(argv)
    m = collect(topology=a.topology)
    if not m:
        print("no policy runs found")
        return 2

    out = {"frontier": {}, "attribution": {}}
    for arm in FRONTIER:
        out["frontier"][arm] = {k: round(m[(arm, k)][0], 2)
                                for k in ANCHORS if (arm, k) in m}
    for k in ANCHORS:
        pre, sr, cc = m.get(("masked_reactive_prefix", k)), \
            m.get(("shortest_route", k)), m.get(("clairvoyant_ceiling", k))
        if pre and sr:
            out["attribution"][k] = {
                "ordering_only": round(pre[0], 2),
                "with_ceiling_targeting": round(sr[0], 2),
                "direction_effect_pp": round(sr[0] - pre[0], 2),
                "clairvoyant": round(cc[0], 2) if cc else None,
                "headroom_above_deployable_pp":
                    round(cc[0] - sr[0], 2) if cc else None,
                "n": pre[1]}
    # persist as well as print: the manuscript generator reads this, and a
    # figure that exists only on stdout is a number nobody can regenerate
    with open(os.path.join(RESULTS, "frontier_policies.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    if a.json:
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0

    print("Policy frontier -- mean utility, held-out balanced (LLM-free)")
    print("%-26s %s" % ("", "".join("%9s" % k for k in ANCHORS)))
    for arm in FRONTIER:
        row = "".join("%9s" % (("%.2f" % m[(arm, k)][0]) if (arm, k) in m else "-")
                      for k in ANCHORS)
        print("%-26s %s" % (LABEL[arm], row))
    print()
    print("attribution -- ordering vs direction")
    for k in ANCHORS:
        v = out["attribution"].get(k)
        if v:
            print("  %-6s ordering %5.2f -> +ceiling-targeting %5.2f "
                  "(direction %+5.2f pp); headroom above the best deployable "
                  "policy %+5.2f pp"
                  % (k, v["ordering_only"], v["with_ceiling_targeting"],
                     v["direction_effect_pp"],
                     v["headroom_above_deployable_pp"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
