#!/usr/bin/env python3
"""Where the induced support logic actually fails: the aggregation rule.

WHY. The paper closes on "producing the contract is the measurable open
problem", and the evidence for that is one induction procedure -- spec-view
prompting, three draws, UNANIMOUS aggregation -- reaching 25.8 utility against
the authored program's 59.2. A reviewer asked the right question: is that a
property of S2, or of this aggregator? If the latter, the closing claim is
partly an indictment of our own procedure.

It cannot be settled without inference, because only the AGGREGATED licence
family is cached, not the individual draws; re-aggregating under a weaker rule
needs the draws back, and that needs a GPU. What CAN be measured from the
cached programs is the shape of the failure, and the shape is decisive enough
to scope the claim honestly.

A claim whose induced route family is EMPTY is not mislicensed -- it is
unlicensable: no evidence state can satisfy it, so the arm is silent on it
whatever it gathers. Unanimity is exactly the rule that manufactures empty
families, and it does so at a rate that grows with route width, because
admitting a width-w route over n atoms requires every draw to agree on all
2^(n-w) completions. If the empty families concentrate on the strongest
claims, then the utility collapse is mostly the aggregator refusing to license
anything hard, not the model failing to learn support structure.

Writes: results/induction_aggregation.json
Usage:  python3 analysis/induction_aggregation.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, HERE, os.path.join(PILOT, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import induction_policy_lab as lab  # noqa: E402  -- the same loader the lab uses

RESULTS = os.path.join(PILOT, "results")


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    insts = lab.load_instances()
    induced = lab.load_cached()          # policy=unanimous, draws=3
    if not induced:
        print("no cached induced programs")
        return 1

    total = collections.Counter()
    empty = collections.Counter()
    n_claims = n_empty = 0
    fully_licensable = 0
    covered_instances = 0

    for inst in insts:
        fam = induced.get(inst["family"])
        if not fam:
            continue
        covered_instances += 1
        ok = True
        for cid, level in inst["levels"].items():
            n_claims += 1
            total[level] += 1
            if not (fam.get(cid) or []):
                n_empty += 1
                empty[level] += 1
                ok = False
        if ok:
            fully_licensable += 1

    levels = sorted(total)
    out = {
        "policy": "unanimous", "draws": 3,
        "n_instances": covered_instances,
        "n_claims": n_claims,
        "n_claims_empty": n_empty,
        "pct_claims_empty": round(100.0 * n_empty / n_claims, 1) if n_claims else 0.0,
        "by_level": {
            str(l): {
                "n": total[l], "empty": empty[l],
                "pct_empty": round(100.0 * empty[l] / total[l], 1) if total[l] else 0.0,
            } for l in levels
        },
        "strongest_level": str(max(levels)) if levels else None,
        "n_instances_fully_licensable": fully_licensable,
        "_note": ("An empty induced route family makes a claim UNLICENSABLE: "
                  "no ledger state satisfies it, so the arm is silent on that "
                  "claim at any budget. This measures the aggregation rule's "
                  "contribution to the S2 utility gap. It does not measure "
                  "what a weaker rule would recover, which needs the "
                  "individual draws and therefore new inference."),
    }
    json.dump(out, open(os.path.join(RESULTS, "induction_aggregation.json"), "w"),
              indent=1, sort_keys=True)

    print("induced programs (unanimous/3) over %d instances, %d claims"
          % (covered_instances, n_claims))
    print("  claims with an EMPTY route family: %d (%.1f%%)"
          % (n_empty, out["pct_claims_empty"]))
    for l in levels:
        print("    level %d: %d/%d empty (%.1f%%)"
              % (l, empty[l], total[l], out["by_level"][str(l)]["pct_empty"]))
    print("  instances where every claim is licensable at all: %d"
          % fully_licensable)
    print("wrote results/induction_aggregation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
