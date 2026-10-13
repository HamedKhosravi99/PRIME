#!/usr/bin/env python3
"""The one-branch-fails split (results/tableOB.json).

In the balanced split a branching task either keeps both alternative routes
to C2 satisfiable or neither, so nothing there tests whether an agent finds
the one surviving alternative. data/suffice/*/heldout_onebranch holds
tasks where exactly one of the two routes to C2 survives and C3 is not
supported (generate_record_instances.py --one-branch-per-domain).

Cells: the fully informed agent (react_program_attested), the mask
(masked_reactive_v) and PRIME (prime_guided) from results/w2ob/<model>-<dom>/
<arm>__B20, plus the LLM-free policies from results/w2ob/policy-<dom>/
<arm>__B20 (src/prime_agents/policies.py). Contrasts use the frozen procedure of
stats_v5 (family-mean paired differences, sign-flip randomization,
hierarchical bootstrap). Also reported: the share of episodes that reached
C2 through the surviving route, which is the quantity the split exists for.

Usage: python3 experiments/analysis/build_onebranch_table.py [--model qwen2.5-32b]
"""
from __future__ import annotations

import argparse
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
import stats_v5 as st  # noqa: E402

RESULTS = os.path.join(PILOT, "results", "w2ob")
SPLIT = "heldout_onebranch"
LLM_ARMS = ("react_program_attested", "masked_reactive_v", "prime_guided")
POLICY_ARMS = ("random_masked", "shortest_route")


def load(model: str, arm: str):
    out = []
    for dom in bw.DOMAINS:
        tag = "policy-%s" % dom if arm in POLICY_ARMS else "%s-%s" % (model, dom)
        p = os.path.join(RESULTS, tag, "%s__B20" % arm, "summary.json")
        if os.path.exists(p):
            out.extend(json.load(open(p, encoding="utf-8"))["episodes"])
    return out


def rows_of(model: str, arm: str):
    rows = {}
    for e in load(model, arm):
        f = bw.FACTS.get(e.get("instance"))
        if not f or f["split"] != SPLIT or e.get("error"):
            continue
        s = bw.score(e)
        if s:
            s["reached_c2"] = int(any(x["claim"] == "C2" for x in (e.get("emitted") or [])))
            rows[e["instance"]] = s
    return rows


def cell(rows) -> dict:
    if not rows:
        return {"status": "NOT RUN"}
    n = len(rows)
    m = lambda k: sum(r[k] for r in rows.values()) / n
    return {"n_instances": n, "n_clusters": len({r["facts"]["family"] for r in rows.values()}),
            "ofar_a": round(100 * m("ofar_a"), 1), "utility": round(m("utility"), 1),
            "emit_rate": round(100 * (1 - m("abstain")), 1),
            "reached_c2": round(100 * m("reached_c2"), 1), "cost": round(m("cost"), 2)}


def contrast(a_rows, b_rows, key) -> dict:
    """a minus b, family-mean, over the tasks both arms ran."""
    rows = []
    for iid in sorted(set(a_rows) & set(b_rows)):
        f = a_rows[iid]["facts"]
        rows.append({"record": f["record"], "family": f["family"], "domain": f["domain"],
                     "topology": f["topology"], "a": a_rows[iid][key], "b": b_rows[iid][key]})
    if len(rows) < 2:
        return {"status": "NOT RUN"}
    r = st.template_cluster_randomization(rows)
    ci = st.hierarchical_cluster_bootstrap(rows)
    scale = 100.0 if key in ("ofar_a", "reached_c2") else 1.0
    return {"delta": round(scale * r["statistic"], 1),
            "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
            "p_randomization": round(r["p"], 4), "discordant_clusters": r["n_discordant_clusters"],
            "n_clusters": r["n_clusters"], "n_instances": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)
    n_tasks = sum(len(glob.glob(os.path.join(PILOT, "data", "suffice", d, SPLIT, "*.json")))
                  for d in bw.DOMAINS)
    out = {"split": SPLIT, "n_tasks": n_tasks, "cells": {}, "contrasts": {}}
    rows = {arm: rows_of(a.model, arm) for arm in LLM_ARMS + POLICY_ARMS}
    for arm, r in rows.items():
        out["cells"][arm] = cell(r)
    pairs = (("masked_reactive_v", "react_program_attested"),
             ("prime_guided", "masked_reactive_v"),
             ("prime_guided", "shortest_route"),
             ("prime_guided", "random_masked"))
    for x, y in pairs:
        for key in ("ofar_a", "utility", "reached_c2"):
            out["contrasts"]["%s-%s|%s" % (x, y, key)] = contrast(rows[x], rows[y], key)
    path = os.path.join(PILOT, "results", "tableOB.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for arm, c in out["cells"].items():
        print("%-24s %s" % (arm, c))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
