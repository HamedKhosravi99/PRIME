#!/usr/bin/env python3
"""The locator-uncertain split (results/tableLOC.json).

heldout_locator holds the 480 balanced tasks with the atoms' exact fields
withheld from the specification view: each atom says what it must read and
lists three candidate fields, the true one among them. Attestation still
fires only on the true field. Code cannot read the description, so the
deterministic route selection (shortest_route_candidates) tries the
candidates in order and pays a call for each miss; an agent that reads the
description can go to the right field. The split therefore tests whether
guidance has value beyond deterministic route selection where discharging
an obligation needs language, and the budget anchors make the price of a
miss bite.

Cells per anchor: the LLM arms from results/w2loc/<model>-<dom>/<arm>__<anchor>
and the policies from results/w2loc/policy-<dom>/<arm>__<anchor>
(shortest_route = the same policy with the true locators, the oracle bound).
Contrasts use the frozen stats_v5 procedure.

Usage: python3 analysis/build_locator_table.py [--model qwen2.5-32b]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, os.path.join(PILOT, "runner"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402
import stats_v5 as st  # noqa: E402

RESULTS = os.path.join(PILOT, "results", "w2loc")
SPLIT = "heldout_locator"
ANCHORS = ("bmin", "bmin1", "2bmin", "B20")
LLM_ARMS = ("masked_reactive_v", "prime_guided", "react_program_attested")
POLICY_ARMS = ("shortest_route_candidates", "shortest_route", "random_masked")


def rows_of(model, arm, anchor):
    out = {}
    for dom in bw.DOMAINS:
        tag = "policy-%s" % dom if arm in POLICY_ARMS else "%s-%s" % (model, dom)
        p = os.path.join(RESULTS, tag, "%s__%s" % (arm, anchor), "summary.json")
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] != SPLIT or e.get("error"):
                continue
            s = bw.score(e)
            if s:
                out[e["instance"]] = s
    return out


def cell(rows):
    if not rows:
        return {"status": "NOT RUN"}
    n = len(rows)
    m = lambda k: sum(r[k] for r in rows.values()) / n
    return {"n_instances": n, "n_clusters": len({r["facts"]["family"] for r in rows.values()}),
            "ofar_a": round(100 * m("ofar_a"), 1), "utility": round(m("utility"), 1),
            "emit_rate": round(100 * (1 - m("abstain")), 1), "cost": round(m("cost"), 2)}


def contrast(a_rows, b_rows, key):
    rows = []
    for iid in sorted(set(a_rows) & set(b_rows)):
        f = a_rows[iid]["facts"]
        rows.append({"record": f["record"], "family": f["family"], "domain": f["domain"],
                     "topology": f["topology"], "a": a_rows[iid][key], "b": b_rows[iid][key]})
    if len(rows) < 2:
        return {"status": "NOT RUN"}
    r = st.template_cluster_randomization(rows)
    ci = st.hierarchical_cluster_bootstrap(rows)
    scale = 100.0 if key == "ofar_a" else 1.0
    return {"delta": round(scale * r["statistic"], 1),
            "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
            "p_randomization": round(r["p"], 4), "discordant_clusters": r["n_discordant_clusters"],
            "n_clusters": r["n_clusters"], "n_instances": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)
    out = {"split": SPLIT, "cells": {}, "contrasts": {}}
    rows = {}
    for anchor in ANCHORS:
        for arm in LLM_ARMS + POLICY_ARMS:
            rows[(arm, anchor)] = rows_of(a.model, arm, anchor)
            out["cells"]["%s|%s" % (arm, anchor)] = cell(rows[(arm, anchor)])
        for x, y in (("prime_guided", "shortest_route_candidates"),
                     ("prime_guided", "masked_reactive_v"),
                     ("prime_guided", "shortest_route"),
                     ("masked_reactive_v", "shortest_route_candidates")):
            for key in ("utility", "ofar_a"):
                out["contrasts"]["%s-%s|%s|%s" % (x, y, anchor, key)] = contrast(
                    rows[(x, anchor)], rows[(y, anchor)], key)
    path = os.path.join(PILOT, "results", "tableLOC.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("%-28s %-6s %s" % ("arm", "anchor", "cell"))
    for (arm, anchor), r in rows.items():
        c = out["cells"]["%s|%s" % (arm, anchor)]
        if not c.get("status"):
            print("%-28s %-6s util %5.1f  FA %4.1f  emit %5.1f  calls %4.2f  (n=%d)" % (
                arm, anchor, c["utility"], c["ofar_a"], c["emit_rate"], c["cost"], c["n_instances"]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
