#!/usr/bin/env python3
"""The enforcement contrast as a function of corruption prevalence.

WHY. The false-authorization endpoint is identifiable only because records
were degraded until claims lost support, so a single reported gap is the
enforcement gap AT a 41.5% corruption rate, with no calibration to anything.
A single engineered operating point cannot support a claim about how much
enforcement is worth.

That is answerable without regenerating the benchmark. The
confirmatory split carries a per-instance flag for whether a route is broken
by a genuinely FALSE atom (false-load-bearing) rather than a missing one.
Resampling the SAME episodes at different mixtures of those two strata gives
the contrast at any prevalence between them, so the single point becomes a
curve.

WHAT IT IS. Pooled instance-level rates, resampled without replacement, 200
draws per prevalence, seeded. This is deliberately NOT the cluster-mean
paired contrast the paper's confirmatory tests use -- those are a different
estimand and are not resampled here -- so the number at the observed
prevalence is the pooled rate difference, which is why it reads 26.7 rather
than the cluster-mean 28.0. Both are correct; they measure different things,
and the caption says so.

Writes: results/prevalence_sweep.json
Usage:  python3 experiments/analysis/prevalence_sweep.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw   # noqa: E402  -- the single scorer

BENCH = os.path.join(PILOT, "data", "obligationbench")
SPLIT = "heldout_balanced"
SEED = 20260801                     # the benchmark's own frozen seed
DRAWS = 200
GRID = (0.10, 0.20, 0.30, 0.60, 0.80)
UNGATED = "react_program_attested"  # the strongest arm without the mask
GATED = "masked_reactive_v"         # the same arm with the mask, nothing else


def flb_flags():
    out = {}
    for p in sorted(glob.glob(os.path.join(BENCH, "*", SPLIT, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        m = d["notes_for_annotators"]
        m = json.loads(m) if isinstance(m, str) else m
        out[d["id"]] = bool(m.get("false_mech"))
    return out


def arm_rows(arm, model="qwen2.5-32b"):
    out = {}
    for dom in bw.DOMAINS:
        for e in bw.load(model, dom, arm + "__B20"):
            s = bw.score(e)
            if s:
                out[e["instance"]] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)

    flb = flb_flags()
    hi, lo = arm_rows(UNGATED, a.model), arm_rows(GATED, a.model)
    common = sorted(set(hi) & set(lo) & set(flb))
    if len(common) < 50:
        print("not enough paired episodes (%d)" % len(common))
        return 1

    P = [i for i in common if flb[i]]
    N = [i for i in common if not flb[i]]
    observed = len(P) / float(len(common))
    rng = random.Random(SEED)

    rows = []
    for target in sorted(set(GRID) | {round(observed, 3)}):
        n = min(int(len(P) / target) if target else len(N),
                int(len(N) / (1 - target)) if target < 1 else len(P))
        k = int(round(n * target))
        if k > len(P) or (n - k) > len(N) or n < 40:
            continue
        g = []
        for _ in range(DRAWS):
            s = rng.sample(P, k) + rng.sample(N, n - k)
            ung = 100.0 * sum(hi[i]["ofar_a"] for i in s) / len(s)
            gat = 100.0 * sum(lo[i]["ofar_a"] for i in s) / len(s)
            g.append((ung, gat))
        rows.append({
            "prevalence": round(target, 3),
            "is_observed": abs(target - observed) < 1e-3,
            "n_resampled": n,
            "ungated": round(sum(x[0] for x in g) / len(g), 1),
            "gated": round(sum(x[1] for x in g) / len(g), 1),
            "gap_pp": round(sum(x[0] - x[1] for x in g) / len(g), 1),
        })

    out = {"rows": rows, "observed_prevalence": round(100 * observed, 1),
           "n_instances": len(common), "draws": DRAWS, "seed": SEED,
           "ungated_arm": UNGATED, "gated_arm": GATED,
           "_note": "pooled instance-level rates, resampled; NOT the "
                    "cluster-mean paired contrast"}
    with open(os.path.join(PILOT, "results", "prevalence_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print("enforcement contrast vs corruption prevalence "
          "(observed %.1f%%, n=%d)" % (100 * observed, len(common)))
    print("  %-12s %9s %8s %9s" % ("prevalence", "ungated", "gated", "gap"))
    for r in rows:
        print("  %-12s %9.1f %8.1f %9.1f%s"
              % ("%.0f%%" % (100 * r["prevalence"]), r["ungated"], r["gated"],
                 r["gap_pp"], "   <- observed" if r["is_observed"] else ""))
    print("wrote results/prevalence_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
