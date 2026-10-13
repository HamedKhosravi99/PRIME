#!/usr/bin/env python3
"""HESP's controller as a scripted policy (results/w2a/policy-<dom>/hesp_controller__<anchor>).

In HESP's full configuration the controller selects every probe and issues
every verdict; the model can only abstain, and in the live 20-call runs it
never did (480 of 480 verdicts by the controller, identical utility). So
HESP's controller is directly comparable with PRIME's schedule run by the
runtime (the ceiling-targeting policy, results/w2a/policy-<dom>/shortest_route__*):
two code controllers with the same probes, resolvers, budgets, and scorer,
differing only in what they optimize. This script runs the ext_hesp arm
with a planner that always asks for a probe, through run_episode (same
summary format and traces), at the four budget anchors.

Usage: python3 experiments/analysis/hesp_controller_runs.py [--prior counted]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_batch  # noqa: E402
from run_episode import run_episode  # noqa: E402

DOMAINS = ("security", "clinical", "drugsafety", "financial")


class AlwaysAct:
    def chat(self, messages, tools=None):
        m = re.search(r"^- (\S+) \| read ", messages[-1]["content"], re.M)
        return {"content": json.dumps({"kind": "action", "action_id": m.group(1) if m else "x",
                                       "reason": "probe"}), "tool_calls": [], "usage": {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior", default="counted", choices=("counted", "counted_lodo", "uniform", "bernoulli"))
    ap.add_argument("--split", default="heldout_balanced")
    a = ap.parse_args(argv)
    os.environ["PRIME_HESP_PRIOR"] = a.prior
    import extsys  # noqa: E402  (reads PRIME_HESP_PRIOR at import)
    extsys.HESP_PRIOR = a.prior
    name = "hesp_controller" if a.prior == "counted" else "hesp_controller_%s" % a.prior
    base = "w2a" if a.split == "heldout_balanced" else "w2loc"
    for anchor in ("bmin", "bmin1", "2bmin", "B20"):
        for dom in DOMAINS:
            out = os.path.join(PILOT, "results", base, "policy-%s" % dom, "%s__%s" % (name, anchor))
            os.makedirs(out, exist_ok=True)
            eps = []
            for p in sorted(glob.glob(os.path.join(PILOT, "data", "suffice", dom, a.split, "*.json"))):
                budget = 20 if anchor == "B20" else run_batch.instance_budget(p, anchor, 20)
                eps.append(run_episode(p, "ext_hesp", "gold", AlwaysAct(), out, max_tool_calls=budget))
            with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump({"run": {"arm": name, "anchor": anchor, "prior": a.prior, "planner": "always_act",
                                   "split": a.split}, "episodes": eps}, fh, indent=1, sort_keys=True)
            print(anchor, dom, len(eps), "errors", sum(1 for e in eps if e.get("error")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
