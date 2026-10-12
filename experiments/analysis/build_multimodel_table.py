#!/usr/bin/env python3
"""The main-text experiments on more than one model (results/tableMULTI.json).

For each model tag with cells under results/w2a/<tag>-<domain>/ and
results/w2loc/<tag>-<domain>/, the rows a reader needs to check the sign of
each headline result: the fully informed agent at 20 calls (false
authorization, utility); PRIME without guidance, PRIME, and PRIME with
completion at b_min, 2b_min, and 20 calls on the balanced split; and on the
locator-uncertain split PRIME at b_min and 2b_min, the fully informed agent,
the mask alone, and PRIME with completion at b_min, against blind enumeration (40.3 / 58.5) and the
true locators (59.2), which are LLM-free and shared by every model. Family-
mean contrasts: guidance (PRIME minus the mask alone), completion against
the ceiling-targeting policy paired per task, and each agent against
blind enumeration at b_min on the locator split.

Usage: python3 experiments/analysis/build_multimodel_table.py [--models qwen2.5-32b gptoss-20b mistral-small-24b]
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
import stats_v5 as st  # noqa: E402

LOC = "heldout_locator"


def cell(model, base, name, split):
    rows = {}
    for dom in bw.DOMAINS:
        sp = os.path.join(bw.RESULTS, base, "%s-%s" % (model, dom), name, "summary.json")
        if not os.path.exists(sp):
            continue
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            s = bw.score(e)
            if s and s["facts"]["split"] == split:
                rows[e["instance"]] = s
    return rows


def policy(name, anchor, base="w2loc"):
    rows = {}
    for dom in bw.DOMAINS:
        sp = os.path.join(bw.RESULTS, base, "policy-%s" % dom, "%s__%s" % (name, anchor), "summary.json")
        if not os.path.exists(sp):
            continue
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            s = bw.score(e)
            if s:
                rows[e["instance"]] = s
    return rows


def summary(rows):
    if not rows:
        return None
    n = len(rows)
    return {"n": n, "ofar_a": round(100.0 * sum(r["ofar_a"] for r in rows.values()) / n, 1),
            "utility": round(sum(r["utility"] for r in rows.values()) / n, 1),
            "calls": round(sum(r["cost"] for r in rows.values()) / n, 2)}


def contrast(A, B, const=None):
    prs = []
    for iid in sorted(A):
        if const is None and iid not in B:
            continue
        f = A[iid]["facts"]
        prs.append({"record": f["record"], "family": f["family"], "domain": f["domain"],
                    "topology": f["topology"], "a": A[iid]["utility"],
                    "b": const if const is not None else B[iid]["utility"]})
    if len(prs) < 2:
        return None
    rz = st.template_cluster_randomization(prs)
    ci = st.hierarchical_cluster_bootstrap(prs)
    return {"delta": round(rz["statistic"], 1), "ci": [round(ci["lo"], 1), round(ci["hi"], 1)],
            "p": round(rz["p"], 4), "discordant": rz["n_discordant_clusters"], "n_clusters": rz["n_clusters"],
            "n_instances": len(prs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["qwen2.5-32b", "gptoss-20b", "mistral-small-24b"])
    a = ap.parse_args(argv)
    out = {"models": {}}
    blind = {an: policy("shortest_route_candidates", an) for an in ("bmin", "2bmin")}
    ceiling = {an: policy("shortest_route", an, base="w2a") for an in ("bmin", "2bmin", "B20")}
    for m in a.models:
        M = {"balanced": {}, "locator": {}, "contrasts": {}}
        fi = cell(m, "w2a", "react_program_attested__B20", "heldout_balanced")
        M["balanced"]["fully_informed_B20"] = summary(fi)
        for an in ("bmin", "2bmin"):
            M["balanced"]["fully_informed_%s" % an] = summary(cell(m, "w2a", "react_program_attested__%s" % an, "heldout_balanced"))
        for an in ("bmin", "2bmin", "B20"):
            mk = cell(m, "w2a", "masked_reactive_v__%s" % an, "heldout_balanced")
            pg = cell(m, "w2a", "prime_guided__%s" % an, "heldout_balanced")
            pc = cell(m, "w2a", "prime_complete__%s" % an, "heldout_balanced")
            M["balanced"]["mask_%s" % an] = summary(mk)
            M["balanced"]["prime_%s" % an] = summary(pg)
            M["balanced"]["complete_%s" % an] = summary(pc)
            if pg and mk:
                M["contrasts"]["%s|prime-vs-mask" % an] = contrast(pg, mk)
            if pc:
                M["contrasts"]["%s|complete-vs-ceiling_targeting" % an] = contrast(pc, ceiling[an])
            if pc and pg:
                M["contrasts"]["%s|complete-vs-prime" % an] = contrast(pc, pg)
        for an in ("bmin", "2bmin"):
            pg = cell(m, "w2loc", "prime_guided__%s" % an, LOC)
            M["locator"]["prime_%s" % an] = summary(pg)
            if pg and blind[an]:
                M["contrasts"]["loc_%s|prime-vs-blind" % an] = contrast(pg, blind[an])
        fl = cell(m, "w2loc", "react_program_attested__bmin", LOC)
        mk = cell(m, "w2loc", "masked_reactive_v__bmin", LOC)
        pc = cell(m, "w2loc", "prime_complete__bmin", LOC)
        M["locator"]["fully_informed_bmin"] = summary(fl)
        M["locator"]["mask_bmin"] = summary(mk)
        M["locator"]["complete_bmin"] = summary(pc)
        if fl and blind["bmin"]:
            M["contrasts"]["loc_bmin|fully_informed-vs-blind"] = contrast(fl, blind["bmin"])
        if mk and blind["bmin"]:
            M["contrasts"]["loc_bmin|mask-vs-blind"] = contrast(mk, blind["bmin"])
        if pc and blind["bmin"]:
            M["contrasts"]["loc_bmin|complete-vs-blind"] = contrast(pc, blind["bmin"])
        out["models"][m] = M
    path = os.path.join(bw.RESULTS, "tableMULTI.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for m, M in out["models"].items():
        print("==", m)
        for k, v in M["balanced"].items():
            print("   balanced %-22s %s" % (k, v))
        for k, v in M["locator"].items():
            print("   locator  %-22s %s" % (k, v))
        for k, v in M["contrasts"].items():
            print("   contrast %-40s %s" % (k, v))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
