#!/usr/bin/env python3
"""Learned routes (S2), scored end to end over all 480 confirmatory tasks.

build_w2b_table.py scores the induced-program arms over the episodes whose
induced program passed validation (422 without guidance, 420 with guidance).
That conditions on surviving the first way the learned standard can fail. Here
an episode whose program failed validation is scored as an abstention: utility
0, no false authorization. Both views are reported; the contrasts against the
authored program use the frozen procedure (family-mean paired differences,
sign-flip randomization, hierarchical bootstrap) from stats_v5.

Writes results/s2_end_to_end.json.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, HERE, os.path.join(PILOT, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402
import build_w2b_table as w2b  # noqa: E402
import stats_v5 as st  # noqa: E402

MODEL = "qwen2.5-32b"
ARMS = ("masked_reactive_v", "prime_guided")
ABSTAIN = {"ofar_a": 0, "utility": 0.0, "abstain": 1, "cost": 0.0}


def induced_rows(arm: str):
    """One scored row per confirmatory task; invalid programs count as abstention."""
    out = {}
    n_invalid = 0
    for dom in bw.DOMAINS:
        for e in w2b.load(MODEL, dom, arm):
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] != "heldout_balanced":
                continue
            if e.get("error") or e.get("induced_program_valid") is False:
                n_invalid += 1
                out[e["instance"]] = dict(ABSTAIN, facts=f)
            else:
                s = bw.score(e)
                if s is None:
                    n_invalid += 1
                    out[e["instance"]] = dict(ABSTAIN, facts=f)
                else:
                    out[e["instance"]] = s
    return out, n_invalid


def authored_rows(arm: str):
    out = {}
    for dom in bw.DOMAINS:
        for e in bw.load(MODEL, dom, "%s__B20" % arm):
            f = bw.FACTS.get(e.get("instance"))
            if f and f["split"] == "heldout_balanced" and not e.get("error"):
                out[e["instance"]] = bw.score(e)
    return out


def contrast(gold, ind, key):
    rows = []
    for iid in sorted(set(gold) & set(ind)):
        a, b = gold[iid], ind[iid]
        rows.append({"record": a["facts"]["record"], "family": a["facts"]["family"],
                     "domain": a["facts"]["domain"], "topology": a["facts"]["topology"],
                     "a": b[key], "b": a[key]})  # induced minus authored
    r = st.template_cluster_randomization(rows)
    ci = st.hierarchical_cluster_bootstrap(rows)
    scale = 100.0 if key == "ofar_a" else 1.0
    return {"delta": round(scale * r["statistic"], 1),
            "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
            "p_randomization": round(r["p"], 4),
            "discordant_clusters": r["n_discordant_clusters"],
            "n_clusters": r["n_clusters"], "n_instances": len(rows)}


def main() -> int:
    out = {"_note": "induced-route arms scored over all 480 confirmatory tasks; an episode whose "
                    "induced program failed validation counts as an abstention (utility 0, no false "
                    "authorization). 'conditional' repeats the build_w2b_table.py view over valid programs."}
    for arm in ARMS:
        ind, n_invalid = induced_rows(arm)
        gold = authored_rows(arm)
        n = len(ind)
        pooled_all = {"n": n, "n_invalid": n_invalid,
                      "ofar_a": round(100.0 * sum(r["ofar_a"] for r in ind.values()) / n, 1),
                      "utility": round(sum(r["utility"] for r in ind.values()) / n, 1)}
        # conditional view straight from the w2b cell, for exactness
        cell = w2b.cell(MODEL, arm)
        out[arm] = {
            "end_to_end": pooled_all,
            "conditional": {"n": cell["n_instances"], "ofar_a": cell["ofar_a"], "utility": cell["utility"]},
            "contrast_vs_authored_end_to_end": {k: contrast(gold, ind, k) for k in ("ofar_a", "utility")},
            "authored": {"n": len(gold),
                         "ofar_a": round(100.0 * sum(r["ofar_a"] for r in gold.values()) / len(gold), 1),
                         "utility": round(sum(r["utility"] for r in gold.values()) / len(gold), 1)},
        }
    path = os.path.join(PILOT, "results", "s2_end_to_end.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for arm in ARMS:
        o = out[arm]
        print("%-18s authored FA %.1f util %.1f | end-to-end (n=%d, %d invalid) FA %.1f util %.1f | conditional (n=%d) FA %.1f util %.1f"
              % (arm, o["authored"]["ofar_a"], o["authored"]["utility"], o["end_to_end"]["n"], o["end_to_end"]["n_invalid"],
                 o["end_to_end"]["ofar_a"], o["end_to_end"]["utility"], o["conditional"]["n"], o["conditional"]["ofar_a"], o["conditional"]["utility"]))
        for k, c in o["contrast_vs_authored_end_to_end"].items():
            print("   %-8s delta %+.1f  CI [%+.1f, %+.1f]  p=%.4f  discordant %d/%d" % (k, c["delta"], c["ci"][0], c["ci"][1], c["p_randomization"], c["discordant_clusters"], c["n_clusters"]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
