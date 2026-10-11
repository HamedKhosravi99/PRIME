#!/usr/bin/env python3
"""The text-evidence split (results/tableTEXT.json).

heldout_text (clinical, security; 120 tasks each) has atoms whose evidence is
free text and whose resolver is the model: it reads the text and attests the
atom with a quote. Ground truth comes from structured twins hidden from the
agent, so no judge is involved, and a wrong attestation is a resolver error
that can authorize an unsupported conclusion. Per arm this reports false
authorization, utility, emission, the resolver-error rate (attested value
against the gold value, over accepted attestations), the share of episodes
with at least one wrong attestation (Pr(E_res), the bound of Theorem 4), and
how many false authorizations are explained by a wrong attestation.

Usage: python3 analysis/build_text_table.py [--model qwen2.5-32b]
"""
from __future__ import annotations

import argparse
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

RESULTS = os.path.join(PILOT, "results", "w2text")
SPLIT = "heldout_text"
ARMS = ("react_program_attested", "masked_reactive_v", "prime_guided")
DOMS = ("clinical", "security")


def episodes(model, arm):
    for dom in DOMS:
        run = os.path.join(RESULTS, "%s-%s" % (model, dom), "%s__B20" % arm)
        sp = os.path.join(run, "summary.json")
        if not os.path.exists(sp):
            continue
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            yield dom, run, e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)
    out = {"split": SPLIT, "cells": {}, "contrasts": {}}
    rows_by_arm = {}
    for arm in ARMS:
        rows = {}
        att_total = att_wrong = 0
        ep_with_wrong = fa_with_wrong = 0
        for dom, run, e in episodes(a.model, arm):
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] != SPLIT or e.get("error"):
                continue
            s = bw.score(e)
            if not s:
                continue
            gold = {x["id"]: x["gold_value"] for x in json.load(open(
                os.path.join(bw.BENCH, dom, SPLIT, e["instance"] + ".json"), encoding="utf-8"))["atoms"]}
            wrong = 0
            tp = os.path.join(run, e["instance"] + ".jsonl")
            if os.path.exists(tp):
                for r in (json.loads(l) for l in open(tp, encoding="utf-8") if l.strip()):
                    if r.get("type") == "attestation":
                        at = r.get("attestation") or r
                        if at.get("tool") == "attest_atom":
                            att_total += 1
                            if at.get("value") != gold.get(at.get("atom")):
                                att_wrong += 1
                                wrong += 1
            ep_with_wrong += int(wrong > 0)
            fa_with_wrong += int(s["ofar_a"] and wrong > 0)
            s["wrong_attestations"] = wrong
            rows[e["instance"]] = s
        rows_by_arm[arm] = rows
        if not rows:
            out["cells"][arm] = {"status": "NOT RUN"}
            continue
        n = len(rows)
        m = lambda k: sum(r[k] for r in rows.values()) / n
        n_fa = sum(r["ofar_a"] for r in rows.values())
        out["cells"][arm] = {
            "n_instances": n, "n_clusters": len({r["facts"]["family"] for r in rows.values()}),
            "ofar_a": round(100 * m("ofar_a"), 1), "utility": round(m("utility"), 1),
            "emit_rate": round(100 * (1 - m("abstain")), 1), "cost": round(m("cost"), 2),
            "attestations": att_total,
            "resolver_error_pct": round(100.0 * att_wrong / att_total, 1) if att_total else None,
            "episodes_with_wrong_attestation_pct": round(100.0 * ep_with_wrong / n, 1),
            "false_auth_explained_by_wrong_attestation": "%d of %d" % (fa_with_wrong, n_fa),
        }
    for x, y in (("prime_guided", "masked_reactive_v"), ("prime_guided", "react_program_attested"),
                 ("masked_reactive_v", "react_program_attested")):
        for key in ("utility", "ofar_a"):
            prs = []
            for iid in sorted(set(rows_by_arm.get(x, {})) & set(rows_by_arm.get(y, {}))):
                f = rows_by_arm[x][iid]["facts"]
                prs.append({"record": f["record"], "family": f["family"], "domain": f["domain"],
                            "topology": f["topology"], "a": rows_by_arm[x][iid][key], "b": rows_by_arm[y][iid][key]})
            if len(prs) < 2:
                continue
            # one family per domain here, so the family-level test has two
            # clusters; report the record-level paired bootstrap as well
            rz = st.template_cluster_randomization(prs)
            ci = st.hierarchical_cluster_bootstrap(prs)
            # record-level view: each record its own cluster (percentile
            # bootstrap over records), plus the exact McNemar test for the
            # binary endpoint
            rec_rows = [dict(p, family=p["record"]) for p in prs]
            rci = st.hierarchical_cluster_bootstrap(rec_rows)
            scale = 100.0 if key == "ofar_a" else 1.0
            diffs = [p["a"] - p["b"] for p in prs]
            entry = {
                "delta": round(scale * sum(diffs) / len(diffs), 1),
                "family_ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
                "p_randomization": rz["p"], "n_clusters": rz["n_clusters"],
                "record_ci": [round(scale * rci["lo"], 1), round(scale * rci["hi"], 1)],
                "n_instances": len(prs)}
            if key == "ofar_a":
                entry["mcnemar"] = st.exact_mcnemar(prs)
            out["contrasts"]["%s-%s|%s" % (x, y, key)] = entry
    path = os.path.join(PILOT, "results", "tableTEXT.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for arm, c in out["cells"].items():
        print("%-24s %s" % (arm, c))
    for k, v in out["contrasts"].items():
        print("%-44s %s" % (k, v))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
