#!/usr/bin/env python3
"""Endpoints the emission rule does not optimise (results/tableALT.json).

PRIME emits the arg max of the supplied utility over authorized claims, and
utility retained is what the paper scores, so among authorized claims the
system optimises the metric by construction and the reactive arms do not.
That circularity does not touch the false-authorization endpoints, which are
oracle-decided, but it does touch every utility comparison -- which is the
paper's only positive headline.

The fix is not a friendlier utility. Any U the emission rule still arg maxes
is circular in the same way, including a reweighting of the claim levels,
because a monotone reweighting has the same arg max. These two endpoints are
determined by what the agent RESOLVED and what it SPENT, so no choice the
emission rule makes can raise them:

  evidence_recall   fraction of the atoms named anywhere in the instance's
                    gold licences that the agent actually resolved (value in
                    {T,F} in its own attested ledger). An agent that gathers
                    the relevant evidence scores high whether it emits a
                    claim, the wrong claim, or nothing at all.
  utility_per_call  supported utility divided by tool calls spent. PRIME
                    optimises the numerator and is indifferent to the
                    denominator; a policy that reaches the same claim more
                    cheaply beats it here.

Usage: python3 analysis/alt_metrics.py [--model qwen2.5-32b]
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
import build_ext_table as be  # noqa: E402
import stats_v5 as st  # noqa: E402

RESULTS = os.path.join(PILOT, "results")

_INSTANCE_INDEX = {}
for _p in glob.glob(os.path.join(PILOT, "bench", "instances_v2",
                                 "*", "*", "*.json")):
    _INSTANCE_INDEX.setdefault(os.path.basename(_p), []).append(_p)


def _gold_atoms(episode) -> set:
    """Every atom named in the instance's gold licences."""
    p = episode.get("instance_path") or ""
    if not os.path.exists(p):
        hits = _INSTANCE_INDEX.get(os.path.basename(p), [])
        p = hits[0] if hits else ""
    if not p or not os.path.exists(p):
        return set()
    try:
        gl = json.load(open(p, encoding="utf-8"))["gold_licenses"]
    except (OSError, ValueError, KeyError):
        return set()
    out = set()
    for routes in gl.values():
        for r in routes or []:
            out.update(r)
    return out


def episode_metrics(episode):
    """(evidence_recall, utility_per_call) or None when not scoreable."""
    s = bw.score(episode)
    if not s:
        return None
    atoms = _gold_atoms(episode)
    if not atoms:
        return None
    ledger = episode.get("final_ledger") or {}
    resolved = sum(1 for a in atoms if ledger.get(a) in ("T", "F"))
    recall = 100.0 * resolved / len(atoms)
    calls = episode.get("tool_calls") or 0
    # +1 so an arm that answers without spending is not divided by zero and
    # is not rewarded infinitely for guessing
    per_call = s["utility"] / (1.0 + calls)
    return recall, per_call


def collect(model, arm, loader, splits=("heldout_balanced",)):
    rows = []
    for dom in bw.DOMAINS:
        for e in loader(model, dom, arm):
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] not in splits or e.get("error"):
                continue
            m = episode_metrics(e)
            if m is None:
                continue
            rows.append({"record": f["record"], "family": f["family"],
                         "domain": dom, "topology": f["topology"],
                         "instance": e["instance"],
                         "evidence_recall": m[0], "utility_per_call": m[1]})
    return rows


def _ladder_loader(model, dom, arm):
    return bw.load(model, dom, arm + "__B20")


def _ext_loader(model, dom, arm):
    return [e for e in be.load(model, arm)
            if (bw.FACTS.get(e.get("instance")) or {}).get("domain") == dom]


def cell(rows):
    if not rows:
        return {"status": "NOT RUN"}
    n = len(rows)
    return {"n_instances": n,
            "n_clusters": len({r["family"] for r in rows}),
            "evidence_recall": round(sum(r["evidence_recall"] for r in rows) / n, 1),
            "utility_per_call": round(sum(r["utility_per_call"] for r in rows) / n, 2)}


def paired(ours, theirs, key):
    a = {r["instance"]: r for r in ours}
    b = {r["instance"]: r for r in theirs}
    return [{"record": a[i]["record"], "family": a[i]["family"],
             "domain": a[i]["domain"], "topology": a[i]["topology"],
             "a": a[i][key], "b": b[i][key]}
            for i in sorted(set(a) & set(b))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)
    # Two reference arms, kept apart on purpose. `contrasts` is the gate-only
    # arm and is what Table 9's advantage column reports; `contrasts_guided`
    # is deployed PRIME. A range drawn from the first is easily misread as
    # describing the second, so each block records its own reference arm and
    # the abstract reads the reference off the file rather than off a caption
    # written somewhere else.
    out = {"cells": {}, "contrasts": {}, "contrasts_guided": {},
           "control_law_unoptimised": {},
           "reference": "masked_reactive_v",
           "reference_guided": "prime_guided",
           "_note": "endpoints the emission rule does not optimise; positive "
                    "delta favours the gated arm"}
    ours = {}
    for arm in ("masked_reactive_v", "prime_guided", "react",
                "react_program_attested"):
        rows = collect(a.model, arm, _ladder_loader)
        ours[arm] = rows
        out["cells"][arm] = cell(rows)
    theirs = {}
    for arm in be.ARMS:
        rows = collect(a.model, arm, _ext_loader)
        theirs[arm] = rows
        out["cells"][arm] = cell(rows)

    # THE CONTROL LAW ON AN ENDPOINT PRIME DOES NOT OPTIMISE.
    #
    # PRIME emits the arg max of the supplied utility, which is also what we
    # score, so its selection step optimises the scoring function by
    # construction and the reactive arms' does not -- which confounds every
    # utility comparison, including the control law. The clean answer is to
    # re-run the SAME contrast on an endpoint the emission rule cannot be
    # optimising: evidence recall is fixed by what an arm gathered, not by what
    # it chose to emit, and utility-per-call has a denominator PRIME is
    # indifferent to.
    #
    # This is the control-law contrast, cluster-level, same tests, on
    # endpoints outside the confound.
    gate_rows = ours.get("masked_reactive_v") or []
    guided_rows = ours.get("prime_guided") or []
    if gate_rows and guided_rows:
        for key in ("evidence_recall", "utility_per_call"):
            pr = paired(guided_rows, gate_rows, key)
            if len(pr) < 2:
                continue
            r = st.template_cluster_randomization(pr)
            ci = st.hierarchical_cluster_bootstrap(pr)
            out["control_law_unoptimised"]["%s" % key] = {
                "delta": round(r["statistic"], 2),
                "ci": [round(ci["lo"], 2), round(ci["hi"], 2)],
                "p_randomization": round(r["p"], 4),
                "discordant_clusters": r["n_discordant_clusters"],
                "n_clusters": r["n_clusters"], "n_instances": ci["n_records"]}

    for bucket, ref in (("contrasts", "masked_reactive_v"),
                        ("contrasts_guided", "prime_guided")):
        ours_rows = ours.get(ref) or []
        for arm, rows in theirs.items():
            if not rows or not ours_rows:
                continue
            for key in ("evidence_recall", "utility_per_call"):
                pr = paired(ours_rows, rows, key)
                if len(pr) < 2:
                    continue
                r = st.template_cluster_randomization(pr)
                ci = st.hierarchical_cluster_bootstrap(pr)
                out[bucket]["%s|%s" % (arm, key)] = {
                    "delta": round(r["statistic"], 2),
                    "ci": [round(ci["lo"], 2), round(ci["hi"], 2)],
                    "p_randomization": round(r["p"], 4),
                    "discordant_clusters": r["n_discordant_clusters"],
                    "n_clusters": r["n_clusters"], "n_instances": ci["n_records"]}
    with open(os.path.join(RESULTS, "tableALT.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    got = sum(1 for v in out["cells"].values() if v.get("status") != "NOT RUN")
    print("ALT cells populated: %d/%d" % (got, len(out["cells"])))
    print("contrasts vs the gate-only arm: %d" % len(out["contrasts"]))
    print("contrasts vs guided PRIME:      %d" % len(out["contrasts_guided"]))
    for k, v in sorted(out["control_law_unoptimised"].items()):
        print("  control law on %-18s delta=%+7.2f CI[%+.2f,%+.2f] p=%.4f disc=%d/%d"
              % (k, v["delta"], v["ci"][0], v["ci"][1],
                 v["p_randomization"], v["discordant_clusters"],
                 v["n_clusters"]))
    print("wrote results/tableALT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
