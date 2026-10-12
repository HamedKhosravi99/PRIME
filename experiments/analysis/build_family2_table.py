#!/usr/bin/env python3
"""The confirmatory ladder on a second, independent model family.

WHY. The structural guarantee is model-independent by construction, but the
empirical claims -- what the gate costs, what the control law is worth -- are
policy-dependent, and every confirmatory number came from one pretraining
lineage (Qwen2.5 at two scales). gpt-oss-20b is a different lineage, a
different tokenizer, a different tool-calling protocol, and a different
post-training recipe.

WHAT IS AND IS NOT CLAIMED. This is the security domain only: 120
confirmatory instances over 15 template families, the same held-out balanced
split, the same pre-registered arms, the same executable oracle, the same
budget anchor. It is a replication of the enforcement contrast on an
independent family, not a second full confirmatory panel -- the other three
domains are queued and will be reported when they land or reported as absent.
The comparison against Qwen is restricted to the SAME domain so that nothing
in it turns on domain mix.

Writes: results/tableFAM2.json
Usage:  python3 experiments/analysis/build_family2_table.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw     # noqa: E402  -- the single scorer
import stats_v5 as st             # noqa: E402

RESULTS = os.path.join(PILOT, "results")
DOMAIN = "security"
ARMS = ("react_program_attested", "masked_reactive_v", "prime_guided")
MODELS = (("qwen2.5-32b", "results/w2a/qwen2.5-32b-security"),
          ("gptoss-20b", "results/w2a/gptoss-20b-security"))
RUNG = ("react_program_attested", "masked_reactive_v")


def rows_for(model_dir, arm):
    p = os.path.join(PILOT, model_dir, arm + "__B20", "summary.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for e in json.load(open(p, encoding="utf-8"))["episodes"]:
        s = bw.score(e)
        if s and s["facts"]["split"] == "heldout_balanced":
            out[e["instance"]] = s
    return out


def cell(rows):
    if not rows:
        return {"status": "NOT RUN"}
    v = list(rows.values())
    n = len(v)
    avail = [x for x in v if x["facts"]["ceiling"] > 0]
    return {
        "n_instances": n,
        "n_clusters": len({x["facts"]["family"] for x in v}),
        "ofar_a": round(100.0 * sum(x["ofar_a"] for x in v) / n, 1),
        "utility": round(sum(x["utility"] for x in v) / n, 1),
        "emit_rate_avail": round(
            100.0 * sum(1 for x in avail if not x["abstain"]) / len(avail), 1)
        if avail else 0.0,
        "cost": round(sum(x["cost"] for x in v) / n, 2),
    }


def contrast(model_dir, hi_arm, lo_arm, key="ofar_a"):
    a, b = rows_for(model_dir, hi_arm), rows_for(model_dir, lo_arm)
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    pr = [{"record": a[i]["facts"]["record"], "family": a[i]["facts"]["family"],
           "domain": DOMAIN, "topology": a[i]["facts"]["topology"],
           "a": a[i][key], "b": b[i][key]} for i in common]
    r = st.template_cluster_randomization(pr)
    ci = st.hierarchical_cluster_bootstrap(pr)
    scale = 100.0 if key == "ofar_a" else 1.0
    return {"delta": round(scale * r["statistic"], 1),
            "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
            "p_randomization": round(r["p"], 4),
            "discordant_clusters": r["n_discordant_clusters"],
            "n_clusters": r["n_clusters"], "n_instances": len(pr)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    out = {"domain": DOMAIN, "cells": {}, "mask_rung": {}, "control_law": {},
           "_note": "security domain only; a replication of the enforcement "
                    "contrast on an independent model family, not a second "
                    "full confirmatory panel"}
    for label, d in MODELS:
        for arm in ARMS:
            out["cells"]["%s|%s" % (label, arm)] = cell(rows_for(d, arm))
        c = contrast(d, RUNG[0], RUNG[1])
        if c:
            out["mask_rung"][label] = c
        c2 = contrast(d, "prime_guided", "masked_reactive_v", key="utility")
        if c2:
            out["control_law"][label] = c2

    with open(os.path.join(RESULTS, "tableFAM2.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print("second model family -- %s domain" % DOMAIN)
    print("  %-13s %-26s %7s %7s %7s" % ("model", "arm", "OFAR-A", "Util", "Emit"))
    for label, _ in MODELS:
        for arm in ARMS:
            c = out["cells"]["%s|%s" % (label, arm)]
            if c.get("status"):
                continue
            print("  %-13s %-26s %7.1f %7.1f %7.1f"
                  % (label, arm, c["ofar_a"], c["utility"], c["emit_rate_avail"]))
    print("  mask rung (attested -> masked), cluster-level:")
    for label, c in sorted(out["mask_rung"].items()):
        print("     %-13s %+6.1f pp  CI[%+.1f,%+.1f]  p=%.4f  %d/%d discordant"
              % (label, c["delta"], c["ci"][0], c["ci"][1],
                 c["p_randomization"], c["discordant_clusters"], c["n_clusters"]))
    print("  control law (guided - masked), utility:")
    for label, c in sorted(out["control_law"].items()):
        print("     %-13s %+6.1f     CI[%+.1f,%+.1f]  p=%.4f"
              % (label, c["delta"], c["ci"][0], c["ci"][1], c["p_randomization"]))
    print("wrote results/tableFAM2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
