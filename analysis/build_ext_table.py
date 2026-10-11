#!/usr/bin/env python3
"""External comparators on ObligationBench (results/tableEXT.json).

Every arm in the ladder is an ablation of PRIME itself, which three
independent reviews identified as the binding evidence gap: nothing in that
design distinguishes "the mechanism is better" from "the harness is tuned
for it". These six arms are the independent comparators, run end to end on
the confirmatory split under the same model, budget, executor and oracle:

  static_contract  an authored formal contract, conjunctive collapse of the
                   same gold licences -- the nearest prior-art cell in the
                   positioning table
  conformal        elicited per-claim confidence with leave-one-out
                   cross-conformal risk control and graded back-off
  cove             chain-of-verification style self-checking
  reflexion        reflect-and-retry scaffolding
  rubric           the criterion stated in prose, no executable structure
  direct           unconditioned answering

Usage: python3 analysis/build_ext_table.py [--model qwen2.5-32b]
"""
from __future__ import annotations

import argparse
import collections
import glob
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

RESULTS = os.path.join(PILOT, "results")


def _margin(model: str = "qwen2.5-32b") -> float:
    """The one frozen equivalence margin, from the W0 pilot output."""
    p = os.path.join(RESULTS, "w0_pilots_%s.json" % model)
    if not os.path.exists(p):
        return 0.0
    try:
        return float(json.load(open(p, encoding="utf-8"))["MDE_delta"])
    except (OSError, ValueError, KeyError):
        return 0.0
ARMS = ("static_contract", "conformal", "cove", "reflexion", "rubric", "direct")


VARIANT = ""   # "" = published prompts; "spec" = information-matched reruns


def load(model: str, arm: str):
    out = []
    suffix = "__%s" % VARIANT if VARIANT else ""
    for p in glob.glob(os.path.join(RESULTS, "w2ext", "%s-*" % model,
                                    "%s%s__B20" % (arm, suffix), "summary.json")):
        try:
            out.extend(json.load(open(p, encoding="utf-8"))["episodes"])
        except (OSError, ValueError, KeyError):
            continue
    return out


def cell(model: str, arm: str) -> dict:
    rows, errored, calls = [], 0, []
    for e in load(model, arm):
        f = bw.FACTS.get(e.get("instance"))
        if not f or f["split"] != "heldout_balanced":
            continue
        calls.append(e.get("tool_calls") or 0)
        if e.get("error"):
            errored += 1
            continue
        s = bw.score(e)
        if s:
            rows.append(s)
    if not rows:
        return {"status": "NOT RUN"}
    # the same guard the ladder cells carry: an arm that never called a tool
    # never gathered evidence, so its endpoints describe the prompt
    if calls and max(calls) == 0:
        return {"status": "NO TOOL CALLS", "n_attempted": len(calls)}
    n = len(rows)
    m = lambda k: sum(r[k] for r in rows) / n
    avail = [r for r in rows if r["facts"]["ceiling"] > 0]
    emit = (sum(1 for r in avail if not r["abstain"]) / len(avail)) if avail else 0.0
    return {
        "n_instances": n,
        "n_clusters": len({r["facts"]["family"] for r in rows}),
        "n_errored": errored,
        "ofar_a": round(100 * m("ofar_a"), 1),
        "content_false": round(100 * m("content_false"), 1),
        "utility": round(m("utility"), 1),
        "emit_rate_avail": round(100 * emit, 1),
        "degenerate": emit < bw.DEGENERATE_EMIT_RATE,
        "cost": round(sum(calls) / max(1, len(calls)), 2),
    }


GATE_ONLY = "masked_reactive_v"   # the gate WITHOUT the control law


def paired(model: str, ext_arm: str, key: str, our_arm: str = "prime_guided",
           domain: str = None):
    """PRIME against an external arm, instance by instance.

    Signed so that POSITIVE always means PRIME is better, on both endpoints:
    false authorization is external minus PRIME (PRIME emits fewer), utility
    is PRIME minus external (PRIME retains more). One convention for two
    endpoints that point opposite ways is worth the asymmetry here -- the
    alternative is a table where the reader has to remember which column
    inverts.
    """
    ext = {e["instance"]: e for e in load(model, ext_arm)}
    out = []
    for dom in bw.DOMAINS:
        if domain and dom != domain:
            continue
        ours = {e["instance"]: e for e in bw.load(model, dom, our_arm + "__B20")}
        for iid in sorted(set(ext) & set(ours)):
            a, b = bw.score(ours[iid]), bw.score(ext[iid])
            if not (a and b) or a["facts"]["split"] != "heldout_balanced":
                continue
            hi, lo = (b, a) if key == "ofar_a" else (a, b)
            out.append({"record": a["facts"]["record"], "family": a["facts"]["family"],
                        "domain": dom, "topology": a["facts"]["topology"],
                        "a": hi[key], "b": lo[key]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    ap.add_argument("--variant", default="", choices=("", "spec"),
                    help="'spec': the information-matched reruns "
                    "(<arm>__spec__B20 directories, written to tableEXT_spec.json)")
    a = ap.parse_args(argv)
    global VARIANT
    VARIANT = a.variant
    out = {"cells": {}, "contrasts": {}, "variant": a.variant or "published"}
    for arm in ARMS:
        out["cells"][arm] = cell(a.model, arm)
    for arm in ARMS:
        if out["cells"][arm].get("status"):
            continue
        for key in ("ofar_a", "utility"):
            rows = paired(a.model, arm, key)
            if len(rows) < 2:
                continue
            r = st.template_cluster_randomization(rows)
            ci = st.hierarchical_cluster_bootstrap(rows)
            scale = 100.0 if key == "ofar_a" else 1.0
            out["contrasts"]["%s|%s" % (arm, key)] = {
                "delta": round(scale * r["statistic"], 1),
                "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
                "p_randomization": round(r["p"], 4),
                "discordant_clusters": r["n_discordant_clusters"],
                "n_clusters": r["n_clusters"], "n_instances": ci["n_records"]}
    # The gate WITHOUT the control law is the comparison that isolates the
    # architecture from the policy: a scripted planner outperforms the model
    # at action selection on this benchmark, so a claim resting on the
    # control law rests on the weakest part of the system. This one does not.
    out["gate_only"] = {}
    for arm in ARMS:
        if out["cells"][arm].get("status"):
            continue
        for key in ("ofar_a", "utility"):
            rows = paired(a.model, arm, key, our_arm=GATE_ONLY)
            if len(rows) < 2:
                continue
            scale = 100.0 if key == "ofar_a" else 1.0
            # paired() already signs OFAR as external-minus-ours, so both
            # endpoints arrive with positive favouring the gate
            signed = rows
            r = st.template_cluster_randomization(signed)
            ci = st.hierarchical_cluster_bootstrap(signed)
            ent = {"delta": round(scale * r["statistic"], 1),
                   "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
                   "p_randomization": round(r["p"], 4),
                   "discordant_clusters": r["n_discordant_clusters"],
                   "n_clusters": r["n_clusters"], "n_instances": ci["n_records"],
                   "by_domain": {}}
            if key == "utility":
                try:
                    ent["tost_equivalent"] = bool(
                        st.tost(signed, delta=_margin())["equivalent"])
                except Exception:
                    pass
            for dom in bw.DOMAINS:
                sub = paired(a.model, arm, key, our_arm=GATE_ONLY, domain=dom)
                if len(sub) < 3:
                    continue
                rd = st.template_cluster_randomization(sub)
                ent["by_domain"][dom] = {"delta": round(scale * rd["statistic"], 1),
                                         "p": round(rd["p"], 4),
                                         "n_instances": len(sub)}
            out["gate_only"]["%s|%s" % (arm, key)] = ent

    out_name = "tableEXT_%s.json" % a.variant if a.variant else "tableEXT.json"
    with open(os.path.join(RESULTS, out_name), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    got = sum(1 for v in out["cells"].values() if not v.get("status"))
    print("EXT cells populated: %d/%d" % (got, len(ARMS)))
    print("contrasts vs prime_guided: %d" % len(out["contrasts"]))
    print("wrote results/tableEXT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
