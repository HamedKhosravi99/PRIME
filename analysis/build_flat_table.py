#!/usr/bin/env python3
"""The flat-gate ablation (results/tableFLAT.json).

prime_flat is an evidence-gated claim system without PRIME's additions: the
agent may state only the strongest conclusion, gated on its route being
established, and otherwise abstains (no graded conclusions, no residual
obligations, no completion), with the mask-only prompt family. Compared on
the 480 balanced tasks at 20 calls with PRIME without guidance (graded mask,
no obligations) and PRIME (graded mask with obligations), it isolates what
ordered authorization and guidance add over "attest, then gate the finished
claim". Utility on tasks whose ceiling is the strongest conclusion and on the
others is reported separately, since a flat gate can only score on the former.

Usage: python3 analysis/build_flat_table.py [--model qwen2.5-32b]
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

ARMS = (("prime_flat", "Flat gate (top conclusion only)"),
        ("masked_reactive_v", "PRIME, no guidance"),
        ("prime_guided", "PRIME"))


def by_ceiling(model, arm):
    top, rest = [], []
    for dom in bw.DOMAINS:
        for e in bw.load(model, dom, "%s__B20" % arm):
            s = bw.score(e)
            if not s or s["facts"]["split"] != "heldout_balanced":
                continue
            (top if s["facts"]["ceiling"] == 3 else rest).append(s["utility"])
    return ((round(sum(top) / len(top), 1) if top else None, len(top)),
            (round(sum(rest) / len(rest), 1) if rest else None, len(rest)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)
    out = {"split": "heldout_balanced", "anchor": "B20", "cells": {}, "contrasts": {}}
    for arm, label in ARMS:
        c = bw.cell(a.model, "%s__B20" % arm)
        c["label"] = label
        if not c.get("status"):
            (ut, nt), (ur, nr) = by_ceiling(a.model, arm)
            c["utility_ceiling_top"] = ut
            c["n_ceiling_top"] = nt
            c["utility_ceiling_lower"] = ur
            c["n_ceiling_lower"] = nr
        out["cells"][arm] = c
    for x, y in (("prime_guided", "prime_flat"), ("masked_reactive_v", "prime_flat"),
                 ("prime_guided", "masked_reactive_v")):
        for key in ("utility", "ofar_a"):
            prs = bw.paired(a.model, "%s__B20" % x, "%s__B20" % y, key)
            if len(prs) < 2:
                out["contrasts"]["%s-%s|%s" % (x, y, key)] = {"status": "NOT RUN"}
                continue
            rz = st.template_cluster_randomization(prs)
            ci = st.hierarchical_cluster_bootstrap(prs)
            scale = 100.0 if key == "ofar_a" else 1.0
            out["contrasts"]["%s-%s|%s" % (x, y, key)] = {
                "delta": round(scale * rz["statistic"], 1),
                "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
                "p_randomization": rz["p"], "n_clusters": rz["n_clusters"],
                "discordant_clusters": rz.get("n_discordant_clusters"), "n_instances": len(prs)}
    path = os.path.join(PILOT, "results", "tableFLAT.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for arm, c in out["cells"].items():
        if c.get("status"):
            print("%-20s %s" % (arm, c["status"]))
            continue
        print("%-20s FA %.1f  util %.1f  emit %.1f  cost %.2f  | util ceiling=3: %s (n=%d)  lower: %s (n=%d)" % (
            arm, c["ofar_a"], c["utility"], c["emit_rate_avail"], c["cost"],
            c["utility_ceiling_top"], c["n_ceiling_top"], c["utility_ceiling_lower"], c["n_ceiling_lower"]))
    for k, v in out["contrasts"].items():
        print("%-40s %s" % (k, v))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
