#!/usr/bin/env python3
"""HESP on the locator-uncertain split (results/tableLOC_HESP.json).

Experiment C withholds each check's field among three candidates. HESP's
controller (experiments/analysis/hesp_controller_runs.py --split heldout_locator) treats
every candidate read as a probe, tracks which candidate holds each check, and
concludes when a hypothesis reaches posterior 0.8; it cannot read a check's
description. Compared, on paired tasks with the frozen family-cluster
statistics, with PRIME's agent (live), PRIME with completion (live at b_min),
and PRIME's schedule run by code over the listed candidates (blind
enumeration, no model), at the budget anchors.

Usage: python3 experiments/analysis/build_locator_hesp_table.py
"""
from __future__ import annotations

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

ROWS = (("qwen2.5-32b", "prime_guided", "PRIME"),
        ("qwen2.5-32b", "prime_complete", "PRIME with completion"),
        ("policy", "shortest_route_candidates", "PRIME schedule, listed order (no model)"),
        ("policy", "hesp_controller", "HESP controller"),
        ("policy", "shortest_route", "True-locator schedule (no model)"))
ANCHORS = ("bmin", "bmin1", "2bmin", "B20")


def episodes(model: str, name: str):
    out = {}
    for dom in bw.DOMAINS:
        p = os.path.join(bw.RESULTS, "w2loc", "%s-%s" % (model, dom), name, "summary.json")
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            s = bw.score(e)
            if s and s["facts"]["split"] == "heldout_locator":
                out[e["instance"]] = s
    return out


def main() -> int:
    out = {"cells": {}, "contrasts": {}}
    for anchor in ANCHORS:
        data = {arm: episodes(model, "%s__%s" % (arm, anchor)) for model, arm, _ in ROWS}
        for model, arm, label in ROWS:
            d = data[arm]
            if not d:
                continue
            n = len(d)
            out["cells"]["%s|%s" % (arm, anchor)] = {
                "label": label, "n": n,
                "utility": round(sum(r["utility"] for r in d.values()) / n, 1),
                "fa": round(100 * sum(r["ofar_a"] for r in d.values()) / n, 1),
                "abstain": round(100 * sum(r["abstain"] for r in d.values()) / n, 1),
                "cost": round(sum(r["cost"] for r in d.values()) / n, 2)}
        b = data["hesp_controller"]
        for arm in ("prime_guided", "prime_complete", "shortest_route_candidates"):
            a = data[arm]
            common = sorted(set(a) & set(b))
            if len(common) < 2:
                continue
            for key in ("utility", "ofar_a"):
                rows = [{"record": a[i]["facts"]["record"], "family": a[i]["facts"]["family"],
                         "domain": a[i]["facts"]["domain"], "topology": a[i]["facts"]["topology"],
                         "a": a[i][key], "b": b[i][key]} for i in common]
                rz = st.template_cluster_randomization(rows)
                ci = st.hierarchical_cluster_bootstrap(rows)
                scale = 100.0 if key == "ofar_a" else 1.0
                out["contrasts"]["%s-hesp|%s|%s" % (arm, key, anchor)] = {
                    "delta": round(scale * rz["statistic"], 1),
                    "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
                    "p_randomization": rz["p"], "n_clusters": rz["n_clusters"],
                    "discordant_clusters": rz.get("n_discordant_clusters"), "n_instances": len(rows)}
    path = os.path.join(PILOT, "results", "tableLOC_HESP.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for k, c in out["cells"].items():
        print("%-40s n %d util %5.1f  FA %4.1f  abst %5.1f  cost %.2f" % (k, c["n"], c["utility"], c["fa"], c["abstain"], c["cost"]))
    for k, c in out["contrasts"].items():
        print("%-48s %s" % (k, c))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
