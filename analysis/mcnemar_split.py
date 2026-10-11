#!/usr/bin/env python3
"""The carried-forward SWE comparison, split by when the instances were authored.

WHY. Table~\\ref{tab:mcnemar} reported an exact McNemar p over 21 instances and
noted, in its own caption, that the pool had been scaled up from 9 after the
n=9 comparison came out marginal (p=0.062 on gpt-4.1). Extending a sample
after seeing a marginal p and then quoting the pooled p at face value is
optional stopping, and it is the one practice this paper's confirmatory design
exists to avoid. Disclosing it is not the same as correcting it.

The correction needs no new runs, because the 12 instances authored later
(swe-0008..swe-0019) are an independent sample: scoring them ALONE is a clean
replication test of the effect first seen on the original 9. This emits all
three: the original pre-extension subset, the independent extension, and the
pooled figure, so a reader can apply whichever standard they hold.

Note on the exact test's floor. With b discordant pairs against PRIME and c
for it, the two-sided exact p cannot fall below 2 * 0.5^(b+c). With five
discordant pairs and none against PRIME, p = 0.0625 is the SMALLEST attainable
value: "marginal" there is a statement about the sample size, not about the
strength of the effect.

Writes: results/mcnemar_split.json
Usage:  python3 analysis/mcnemar_split.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, HERE, os.path.join(PILOT, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from end_to_end_tables import exact_mcnemar_p  # noqa: E402  -- the one test

RESULTS = os.path.join(PILOT, "results")

# The pool as it stood before the scale-up, read from the paper's own account
# of the extension (swe-0008..swe-0019 were authored afterwards).
PRE_EXTENSION = {
    "swe-0001", "swe-0001__noev", "swe-0002", "swe-0003", "swe-0004",
    "swe-0005", "swe-0005__noev", "swe-0006", "swe-0007",
}
MODELS = ("gpt-4.1", "gpt-4o-mini")
BASELINE = "react"


def _pairs(table, model, arm):
    key = "%s|%s|gold" % (model, arm)
    cell = table.get(key) or {}
    return {e["instance"]: e for e in cell.get("episodes", [])}


def _test(target, base, ids):
    b = c = 0  # b: PRIME worse on this instance; c: baseline worse
    for i in ids:
        t, bl = target[i]["episode_ofar"], base[i]["episode_ofar"]
        if t > bl:
            b += 1
        elif bl > t:
            c += 1
    n = b + c
    return {
        "n_instances": len(ids),
        "prime_worse_pairs": b,
        "baseline_worse_pairs": c,
        "discordant": n,
        "exact_mcnemar_p": exact_mcnemar_p(b, c),
        # the smallest p this many discordant pairs could ever produce
        "p_floor": (2.0 * 0.5 ** n) if n else 1.0,
    }


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    table = json.load(open(os.path.join(RESULTS, "tableC.json"), encoding="utf-8"))

    out = {"baseline": BASELINE, "models": {}}
    for model in MODELS:
        P, R = _pairs(table, model, "prime"), _pairs(table, model, BASELINE)
        shared = sorted(set(P) & set(R))
        pre = [i for i in shared if i in PRE_EXTENSION]
        new = [i for i in shared if i not in PRE_EXTENSION]
        out["models"][model] = {
            "pre_extension": _test(P, R, pre),
            "extension_only": _test(P, R, new),
            "pooled": _test(P, R, shared),
        }
    json.dump(out, open(os.path.join(RESULTS, "mcnemar_split.json"), "w"),
              indent=1, sort_keys=True)

    print("%-12s %-15s %5s %5s %5s %10s %9s"
          % ("model", "subset", "n", "b", "c", "exact p", "p floor"))
    for model, m in out["models"].items():
        for label in ("pre_extension", "extension_only", "pooled"):
            r = m[label]
            print("%-12s %-15s %5d %5d %5d %10.4f %9.4f"
                  % (model, label, r["n_instances"], r["prime_worse_pairs"],
                     r["baseline_worse_pairs"], r["exact_mcnemar_p"],
                     r["p_floor"]))
    print("wrote results/mcnemar_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
