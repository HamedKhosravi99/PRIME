#!/usr/bin/env python3
"""Where an authored static contract beats the full apparatus.

If a hand-written contract matches the learned-and-committed program at
equal safety in half the domains, the case for the machinery is weaker, and
that conclusion belongs in the main text, not only in an appendix table.

The paper already prints the per-domain table in the appendix. What it did not
do is state the conclusion in the body. This computes it from the episode
records so the sentence cannot drift from the table.

Both arms hold 0.0% false authorization in every domain here, so the
comparison is purely about realized utility: what a hand-written conjunctive
checklist retains against a committed claim--evidence program.

Writes: results/static_contract_domains.json
Usage:  python3 experiments/analysis/static_contract_domains.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(PILOT, "results")
MODEL = "gpt-4.1"
MAX_LEVEL = 3.0
DOMAINS = (("security", "security"), ("financial", "financial"),
           ("drugsafety", "drug safety"), ("clinical", "clinical"),
           ("research", "research"))


def arm_utility(tbl, want):
    """Mean supported utility, on the paper's 0-100 scale, for one arm."""
    for k, v in tbl.items():
        if not k.startswith(MODEL + "|") or "gold" not in k:
            continue
        if not want(v.get("arm", "")):
            continue
        eps = v.get("episodes") or []
        if not eps:
            continue
        return 100.0 * sum(e["u_supported"] for e in eps) / (MAX_LEVEL * len(eps))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    rows, wins = [], 0
    for key, label in DOMAINS:
        p = os.path.join(RESULTS, "tableC_%s.json" % key)
        if not os.path.exists(p):
            continue
        tbl = (json.load(open(p, encoding="utf-8")) or {}).get("table") or {}
        st = arm_utility(tbl, lambda a: "static" in a)
        pr = arm_utility(tbl, lambda a: a == "prime")
        if st is None or pr is None:
            continue
        w = st >= pr
        wins += w
        rows.append({"domain": label, "static": round(st, 1),
                     "prime_gold": round(pr, 1), "static_wins": bool(w)})

    out = {"model": MODEL, "rows": rows, "n_domains": len(rows),
           "static_wins": wins,
           "_note": "both arms hold 0.0% false authorization in every domain "
                    "here, so this compares realized utility only; the "
                    "software domain is in the carried-forward table and is "
                    "not recomputed here"}
    with open(os.path.join(RESULTS, "static_contract_domains.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print("authored static contract vs PRIME (gold), realized utility, %s" % MODEL)
    print("  %-12s %8s %12s" % ("domain", "static", "PRIME(gold)"))
    for r in rows:
        print("  %-12s %8.1f %12.1f  %s"
              % (r["domain"], r["static"], r["prime_gold"],
                 "static wins" if r["static_wins"] else ""))
    print("  static matches or beats PRIME in %d of %d domains" % (wins, len(rows)))
    print("wrote results/static_contract_domains.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
