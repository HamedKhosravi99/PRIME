#!/usr/bin/env python3
"""The live prime_complete cells on the balanced split (results/tableCOMPLETE.json).

For each budget anchor: false authorization, utility, calls per episode, the
share of episodes in which the runtime made completion calls and how many,
the family-mean contrast against PRIME without completion (prime_guided) and
against the ceiling-targeting policy, and, for the record, the replayed
value from results/discharge_replay.json so the live arm can be checked
against the replay it was predicted from.

Usage: python3 experiments/analysis/build_complete_table.py [--model qwen2.5-32b]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402
import policy_frontier as pf  # noqa: E402
import stats_v5 as st  # noqa: E402

ANCHORS = ("bmin", "bmin1", "2bmin", "B20")


def contrast(prs):
    if len(prs) < 2:
        return {"status": "NOT RUN"}
    rz = st.template_cluster_randomization(prs)
    ci = st.hierarchical_cluster_bootstrap(prs)
    return {"delta": round(rz["statistic"], 1), "ci": [round(ci["lo"], 1), round(ci["hi"], 1)],
            "p": round(rz["p"], 4), "discordant": rz["n_discordant_clusters"], "n_clusters": rz["n_clusters"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)
    replay = {}
    rp = os.path.join(bw.RESULTS, "discharge_replay.json")
    if os.path.exists(rp):
        replay = json.load(open(rp, encoding="utf-8")).get("arms", {}).get("prime_guided", {})
    fixed = pf.collect("heldout_balanced")
    out = {"split": "heldout_balanced", "cells": {}, "contrasts": {}}
    for anchor in ANCHORS:
        name = "prime_complete__%s" % anchor
        c = bw.cell(a.model, name)
        if c.get("status"):
            out["cells"][anchor] = c
            continue
        n_disch = tot_disch = 0
        for dom in bw.DOMAINS:
            for e in bw.load(a.model, dom, name):
                d = int(e.get("discharged_calls") or 0)
                n_disch += int(d > 0); tot_disch += d
        c["episodes_with_completion_calls"] = n_disch
        c["completion_calls_per_episode"] = round(tot_disch / c["n_instances"], 2)
        c["replayed_utility"] = replay.get(anchor, {}).get("utility_completed")
        c["ceiling_targeting"] = fixed.get(("shortest_route", anchor), (None, 0))[0]
        out["cells"][anchor] = c
        out["contrasts"]["%s|complete-vs-guided|utility" % anchor] = contrast(
            bw.paired(a.model, name, "prime_guided__%s" % anchor, "utility"))
        # against the ceiling-targeting policy, paired per instance
        live, prs = {}, []
        for dom in bw.DOMAINS:
            for e in bw.load(a.model, dom, name):
                s = bw.score(e)
                if s:
                    live[e["instance"]] = s
            sp = os.path.join(bw.RESULTS, "w2a", "policy-%s" % dom, "shortest_route__%s" % anchor, "summary.json")
            if not os.path.exists(sp):
                continue
            for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
                s = bw.score(e)
                if s and e["instance"] in live:
                    f = live[e["instance"]]["facts"]
                    prs.append({"record": f["record"], "family": f["family"], "domain": dom,
                                "topology": f["topology"], "a": live[e["instance"]]["utility"], "b": s["utility"]})
        out["contrasts"]["%s|complete-vs-ceiling_targeting|utility" % anchor] = contrast(prs)
    path = os.path.join(bw.RESULTS, "tableCOMPLETE.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for anchor, c in out["cells"].items():
        if c.get("status"):
            print(anchor, c["status"]); continue
        print("%-6s n=%d FA=%.1f util=%.1f (replay %s, fixed %s) calls=%.2f completion in %d episodes, %.2f calls each" % (
            anchor, c["n_instances"], c["ofar_a"], c["utility"], c["replayed_utility"], c["ceiling_targeting"],
            c["cost"], c["episodes_with_completion_calls"], c["completion_calls_per_episode"]))
    for k, v in out["contrasts"].items():
        print("%-44s %s" % (k, v))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
