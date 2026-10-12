#!/usr/bin/env python3
"""Reject-and-regenerate, as an upper bound rather than a strawman.

The natural alternative to masking is reject-and-regenerate: the model emits
a claim with citations, an executable post-hoc check rejects unsupported
ones, and the model resamples. That also reaches 0% by construction. The
interesting question is whether masking retains more utility than rejection
sampling at matched budget.

That question deserves the strongest possible version of the alternative,
not a convenient one. With a PERFECT post-hoc checker and
unlimited resamples, rejection converges to: emit the strongest claim the
evidence already gathered actually supports. Nothing that rejects and
resamples can do better, because resampling changes what is SAID and never
changes what was GATHERED. So the ceiling is computable from the ungated
arm's own episodes, with no GPU and no model: take each episode's final
attested ledger and score the strongest claim its own gold licences support.

The answer is two-sided and we report both halves:

  * The mask ALONE loses to idealised rejection. Gating an arm to silence
    where its evidence is insufficient is worth less than letting it retry
    until it says something its evidence supports.
  * The deployed system BEATS the rejection ceiling, because obligation-
    direction changes which evidence is gathered. That is the one thing no
    amount of resampling can buy, at any budget.

Writes: results/rejection_bound.json
Usage:  python3 analysis/rejection_bound.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, HERE, os.path.join(PILOT, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw   # noqa: E402  -- the single scorer

BENCH = os.path.join(PILOT, "bench", "instances_v2")
SPLIT = "heldout_balanced"
UNGATED = "react_program_attested"


def instance_index():
    idx = {}
    for p in glob.glob(os.path.join(BENCH, "*", SPLIT, "*.json")):
        d = json.load(open(p, encoding="utf-8"))
        idx[d["id"]] = d
    return idx


def arm(model, name):
    out = {}
    for dom in bw.DOMAINS:
        for e in bw.load(model, dom, name + "__B20"):
            f = bw.FACTS.get(e.get("instance"))
            if f and f["split"] == SPLIT and not e.get("error"):
                out[e["instance"]] = e
    return out


def rejection_ceiling(episode, idx):
    """Best claim a perfect post-hoc checker could let this episode keep.

    Scored against the instance's own gold licences over the episode's OWN
    final ledger, so it credits only evidence the agent actually gathered.
    """
    d = idx.get(episode["instance"])
    if not d:
        return 0.0
    led = episode.get("final_ledger") or {}
    lv = {c["id"]: c["level"] for c in d["claims"]}
    best = 0
    for c, routes in d["gold_licenses"].items():
        if any(all(led.get(a) == "T" for a in r) for r in (routes or [])):
            best = max(best, lv.get(c, 0))
    return 100.0 * best / 3.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)

    idx = instance_index()
    arms = {k: arm(a.model, k) for k in
            (UNGATED, "masked_reactive_v", "prime_guided")}
    common = sorted(set.intersection(*(set(v) for v in arms.values())))
    if len(common) < 50:
        print("not enough paired episodes (%d)" % len(common))
        return 1

    rows = {k: [] for k in arms}
    ceil = []
    for i in common:
        ok = True
        got = {}
        for k, v in arms.items():
            s = bw.score(v[i])
            if not s:
                ok = False
                break
            got[k] = s
        if not ok:
            continue
        for k, s in got.items():
            rows[k].append(s)
        ceil.append(rejection_ceiling(arms[UNGATED][i], idx))

    n = len(ceil)
    cell = lambda k: {
        "ofar_a": round(100.0 * sum(r["ofar_a"] for r in rows[k]) / n, 1),
        "utility": round(sum(r["utility"] for r in rows[k]) / n, 1)}
    out = {
        "n_instances": n, "model": a.model, "budget": "B20",
        "ungated": cell(UNGATED),
        "rejection_ceiling": {"ofar_a": 0.0,
                              "utility": round(sum(ceil) / n, 1)},
        "gate_only": cell("masked_reactive_v"),
        "deployed": cell("prime_guided"),
        "_note": "the rejection row is an UNATTAINABLE upper bound: a perfect "
                 "post-hoc checker with unlimited resamples. Resampling "
                 "changes what is said, never what was gathered.",
    }
    with open(os.path.join(PILOT, "results", "rejection_bound.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print("reject-and-regenerate bound -- %d paired episodes, %s, B20"
          % (n, a.model))
    print("  %-48s %7s %8s" % ("arm", "OFAR-A", "utility"))
    for key, label in ((UNGATED, "ungated, as run"),
                       ("rejection_ceiling",
                        "PERFECT reject-and-regenerate (upper bound)"),
                       ("gate_only", "the gate alone"),
                       ("deployed", "gate + control law (deployed)")):
        c = out[key] if key in out else cell(key)
        print("  %-48s %7.1f %8.1f" % (label, c["ofar_a"], c["utility"]))
    print("wrote results/rejection_bound.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
