#!/usr/bin/env python3
"""W2b table T4: induced programs under the W3 conservative stack.

Scored against the SAME judge-free endpoints as T1 (analysis/build_w2a_tables
supplies them), because the induced condition changes only where the licence
family comes from -- the runtime, the mask and the oracle are untouched.

Three quantities T1 cannot carry:

  validity      fraction of episodes whose induced program passed schema
                validation at all. An invalid program is a measured outcome,
                not an error: the episode abstains.
  eligibility   fraction of claims that survived the W3 commit gates
                (counterfactual validation + typed linter). Reported because
                129 of the 480 confirmatory instances carry a resolver-less
                atom on a gold route, so some claims are ineligible for a
                STRUCTURAL reason rather than an induction-quality one -- the
                W3 exit note in docs/EXECUTION_PLAN.md requires stating that.
  amplification OFAR of the UNGATED guided arm minus the gated one. Guiding
                an induced program toward a claim whose routes never passed
                validation is the configuration that turned 0% false
                authorization into 100% in an earlier wave; the gate is the
                proposed remedy and this is the number that says whether it
                works.

Writes results/tableT4.json.
Usage: python3 analysis/build_w2b_table.py [--models qwen2.5-32b]
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

RESULTS = os.path.join(PILOT, "results")
ARMS = ("masked_reactive_v", "prime_guided_gated", "prime_guided")


def load(model: str, dom: str, arm: str):
    p = os.path.join(RESULTS, "w2b", "%s-%s" % (model, dom),
                     "%s__induced__B20" % arm, "summary.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p, encoding="utf-8"))["episodes"]


def _route_coverage(model: str) -> dict:
    """Claims the induced programs license AT ALL, from the shared cache.

    A schema-valid program can still assign a claim an empty route family,
    and under unanimous aggregation most do: every draw must agree that the
    claim holds in a state before that state is accepted. This is the
    coverage side of the precision/coverage tradeoff and it bounds utility
    directly, since a claim with no route can never be emitted.
    """
    import glob
    tok = model.split("-")[-1].lower()
    n = covered = fams = 0
    widths = {}
    for p in glob.glob(os.path.join(PILOT, "results", "induction_cache",
                                    "*.json")):
        if tok not in os.path.basename(p).lower():
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fams += 1
        for _cid, routes in (d.get("licenses") or {}).items():
            n += 1
            if routes:
                covered += 1
            for r in routes:
                widths[len(r)] = widths.get(len(r), 0) + 1
    if not n:
        return {}
    # counted in the loop: a glob on the model token is case-sensitive and
    # the cache filenames carry "Qwen2.5-32B", so it silently returned 0
    return {"n_claims": n, "n_families": fams,
            "route_coverage": round(100.0 * covered / n, 1),
            "route_widths": dict(sorted(widths.items())),
            "max_route_width": max(widths) if widths else 0}


def cell(model: str, arm: str) -> dict:
    rows, valid, elig_num, elig_den = [], [], 0, 0
    attempted = errored = 0
    for dom in bw.DOMAINS:
        for e in load(model, dom, arm):
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] != "heldout_balanced":
                continue
            attempted += 1
            if e.get("error"):
                errored += 1
            v = e.get("induced_program_valid")
            # An episode that DIED because its induced program failed schema
            # validation carries induced_program_valid=None, so averaging over
            # the non-None entries reported 100% validity for a condition that
            # failed validation on one episode in eight. A crash is the
            # strongest available evidence of an invalid program: count it.
            if v is not None:
                valid.append(1 if v else 0)
            elif e.get("error"):
                valid.append(0)
            el = e.get("induction_eligibility") or {}
            elig_num += sum(1 for ok in el.values() if ok)
            elig_den += len(el)
            s = bw.score(e)
            if s:
                rows.append(s)
    if not rows:
        return {"status": "NOT RUN"}
    n = len(rows)
    m = lambda k: sum(r[k] for r in rows) / n
    avail = [r for r in rows if r["facts"]["ceiling"] > 0]
    emit = (sum(1 for r in avail if not r["abstain"]) / len(avail)) if avail else 0.0
    return {
        "n_instances": n,
        # The loss must be VISIBLE -- build_w2a_tables.cell() says exactly
        # this and T4 was the table that did not do it. n alone cannot
        # distinguish 420 scored from 480 attempted, and three induced arms
        # losing 58/60/60 episodes are three different instance sets sharing
        # one column heading.
        "n_attempted": attempted,
        "n_errored": errored,
        "n_clusters": len({r["facts"]["family"] for r in rows}),
        "program_valid": round(100 * sum(valid) / len(valid), 1) if valid else None,
        "claim_eligibility": round(100 * elig_num / elig_den, 1) if elig_den else None,
        "ofar_a": round(100 * m("ofar_a"), 1),
        "content_false": round(100 * m("content_false"), 1),
        "utility": round(m("utility"), 1),
        "emit_rate_avail": round(100 * emit, 1),
        "degenerate": emit < bw.DEGENERATE_EMIT_RATE,
        "cost": round(m("cost"), 2),
    }


def paired(model: str, arm_a: str, arm_b: str, key: str,
           gate_bound_only: bool = False):
    """Paired rows, optionally restricted to episodes where the eligibility
    gate could actually bind.

    On an instance where NO claim survives the commit gates, the gated arm
    has no target, guidance has nothing to steer toward, and it behaves as
    the unguided masked policy. Those episodes differ between the arms for a
    reason that is not gating, so pooling them answers a different question
    from the one the contrast is meant to ask. The eligibility map is a
    property of the induced program, which both arms share from the same
    per-family cache, so it is identical across arms and conditioning on it
    does not compare post-treatment quantities.
    """
    out = []
    for dom in bw.DOMAINS:
        Ae = {e["instance"]: e for e in load(model, dom, arm_a)}
        Be = {e["instance"]: e for e in load(model, dom, arm_b)}
        A = {i: bw.score(e) for i, e in Ae.items()}
        B = {i: bw.score(e) for i, e in Be.items()}
        for iid in sorted(set(A) & set(B)):
            a, b = A[iid], B[iid]
            if not (a and b) or a["facts"]["split"] != "heldout_balanced":
                continue
            if gate_bound_only:
                el = (Be[iid].get("induction_eligibility")
                      or Ae[iid].get("induction_eligibility") or {})
                # the gate binds only when it removes some target but not all
                if not el or not any(el.values()) or all(el.values()):
                    continue
            out.append({"record": a["facts"]["record"], "family": a["facts"]["family"],
                        "domain": dom, "topology": a["facts"]["topology"],
                        "a": a[key], "b": b[key]})
    return out


def paired_source(model: str, arm: str, key: str):
    """The SAME arm, instance by instance, authored program vs induced one.

    This is the contrast the paper's motivating requirement actually asks
    about, and until now it existed only as a difference of two pooled rates
    quoted in one appendix sentence. It is paired at the instance level and
    clustered at the template family exactly like the ladder contrasts, so it
    carries the same inferential guarantees rather than a weaker informal
    comparison.

    Everything downstream of the licence family is held fixed: same runtime,
    same deterministic mask, same executable oracle, same budget. The only
    thing that varies is where the committed program came from, which is what
    makes this an estimate of the program-quality term Pr(E_prog) rather than
    a mixture of it with enforcement effects.
    """
    out = []
    for dom in bw.DOMAINS:
        gold = {e["instance"]: e for e in bw.load(model, dom, "%s__B20" % arm)}
        ind = {e["instance"]: e for e in load(model, dom, arm)}
        for iid in sorted(set(gold) & set(ind)):
            a, b = bw.score(gold[iid]), bw.score(ind[iid])
            if not (a and b) or a["facts"]["split"] != "heldout_balanced":
                continue
            # a = induced, b = authored, so a positive statistic reads as
            # "induction COSTS this much", which is the direction the text
            # states it in
            out.append({"record": a["facts"]["record"],
                        "family": a["facts"]["family"], "domain": dom,
                        "topology": a["facts"]["topology"],
                        "a": b[key], "b": a[key]})
    return out


def source_contrasts(model: str) -> dict:
    """Authored -> induced, per arm, on both endpoints, plus per domain."""
    out = {}
    for arm in ("masked_reactive_v", "prime_guided"):
        for key in ("ofar_a", "utility"):
            rows = paired_source(model, arm, key)
            if not rows:
                continue
            r = st.template_cluster_randomization(rows)
            ci = st.hierarchical_cluster_bootstrap(rows)
            scale = 100.0 if key == "ofar_a" else 1.0
            ent = {"delta": round(scale * r["statistic"], 1),
                   "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
                   "p_randomization": round(r["p"], 4),
                   "discordant_clusters": r["n_discordant_clusters"],
                   "n_clusters": r["n_clusters"],
                   "n_instances": ci["n_records"], "by_domain": {}}
            for dom in bw.DOMAINS:
                sub = [x for x in rows if x["domain"] == dom]
                if len(sub) < 2:
                    continue
                rd = st.template_cluster_randomization(sub)
                ent["by_domain"][dom] = {
                    "delta": round(scale * rd["statistic"], 1),
                    "p": round(rd["p"], 4), "n_clusters": rd["n_clusters"],
                    "n_instances": len(sub)}
            out["%s|%s" % (arm, key)] = ent
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["qwen2.5-32b", "qwen2.5-7b"])
    a = ap.parse_args(argv)

    T4 = {"_route_coverage": _route_coverage(a.models[0]),
          "_conventions": {
        "amplification_pp": "OFAR-A(prime_guided) - OFAR-A(prime_guided_gated); "
                            "POSITIVE means the W3 eligibility gate prevented "
                            "false authorization that ungated guidance caused",
        "estimand": "cells are instance-level rates; the contrast is the "
                    "cluster-level estimand, so they do not reconcile"},
        "cells": {}, "amplification": {}}
    for model in a.models:
        for arm in ARMS:
            T4["cells"]["%s|%s" % (model, arm)] = cell(model, arm)
        sc = source_contrasts(model)
        if sc:
            T4.setdefault("source", {})[model] = sc
        bound = paired(model, "prime_guided", "prime_guided_gated", "ofar_a",
                       gate_bound_only=True)
        if bound:
            rb = st.template_cluster_randomization(bound)
            cb = st.hierarchical_cluster_bootstrap(bound)
            T4.setdefault("amplification_gate_bound", {})[model] = {
                "amplification_pp": round(100 * rb["statistic"], 1),
                "ci": [round(100 * cb["lo"], 1), round(100 * cb["hi"], 1)],
                "p_randomization": round(rb["p"], 4),
                "discordant_clusters": rb["n_discordant_clusters"],
                "n_clusters": rb["n_clusters"], "n_instances": cb["n_records"]}
        rows = paired(model, "prime_guided", "prime_guided_gated", "ofar_a")
        if rows:
            r = st.template_cluster_randomization(rows)
            ci = st.hierarchical_cluster_bootstrap(rows)
            T4["amplification"][model] = {
                "amplification_pp": round(100 * r["statistic"], 1),
                "ci": [round(100 * ci["lo"], 1), round(100 * ci["hi"], 1)],
                "p_randomization": round(r["p"], 4),
                "discordant_clusters": r["n_discordant_clusters"],
                "n_clusters": r["n_clusters"], "n_instances": ci["n_records"]}
    with open(os.path.join(RESULTS, "tableT4.json"), "w") as fh:
        json.dump(T4, fh, indent=1, sort_keys=True)
    got = sum(1 for v in T4["cells"].values() if v.get("status") != "NOT RUN")
    print("T4 cells populated: %d/%d" % (got, len(T4["cells"])))
    print("amplification contrasts: %d" % len(T4["amplification"]))
    print("wrote results/tableT4.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
