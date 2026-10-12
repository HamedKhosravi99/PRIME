#!/usr/bin/env python3
"""Measure the DEPLOYMENT-AVAILABLE refinement signal space -- the question
behind the oracle-supervised refinement upper bound.

The oracle-supervised CEGIS loop consumes exact ground-truth counterexamples
from the authored support oracle F* -- an object a genuinely unseen task
does not have. What a deployment DOES have, before any episode evidence, is
purely structural, no-truth-content information about the induced program
itself:

  (a) schema validity (order-coherence, dominated/subsumed sibling routes,
      unknown atom or claim ids) -- ALREADY consumed by the deployed
      pipeline: `induce_program`'s compiler-message retry loop feeds the
      validator error back, and a still-invalid program is a measured
      abstention;
  (b) residual structural lint on programs that PASS the validator: any
      dominated sibling route, coherence violation, claim left route-less,
      or vocabulary atom used by no route.

This script measures (b) on every stored induced draw of the de-confounded
Table B panel (results/induction_table_b_v2.json: 5 models x 9 instances x
5 draws), i.e. it asks: after the validity gate that one-shot induction
already consumes, is there ANY structural signal left for a no-oracle
refinement loop to use?

Writes results/deployment_signal_audit.json. No LLM call, no judge; pure
offline computation over stored programs.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
RESULTS = os.path.join(PILOT, "results")
if PILOT not in sys.path:
    sys.path.insert(0, PILOT)


def audit_program(licenses: Dict[str, List[List[str]]]) -> Dict[str, int]:
    dominated = 0
    routeless = 0
    used = set()
    for cid, routes in licenses.items():
        sets = [frozenset(map(str, r)) for r in routes if isinstance(r, list)]
        if not sets:
            routeless += 1
            continue
        for i, a in enumerate(sets):
            used |= a
            if any(i != j and b < a for j, b in enumerate(sets)):
                dominated += 1  # a strictly contains a sibling: redundant
    return {"dominated_routes": dominated, "routeless_claims": routeless,
            "atoms_used": len(used)}


def main() -> int:
    src = os.path.join(RESULTS, "induction_table_b_v2.json")
    d = json.load(open(src, encoding="utf-8"))
    out: Dict[str, Any] = {"source": os.path.basename(src), "models": {}}
    for row in d["raw"]:
        model = row["model"]
        stats = {"n_draws": 0, "n_valid": 0,
                 "valid_with_dominated_routes": 0,
                 "valid_with_routeless_claims": 0,
                 "invalid_draws": 0}
        for pi in row["per_instance"]:
            stats["n_draws"] += 1
            if not pi.get("valid"):
                stats["invalid_draws"] += 1
                continue
            stats["n_valid"] += 1
            a = audit_program(pi.get("induced_licenses") or {})
            if a["dominated_routes"]:
                stats["valid_with_dominated_routes"] += 1
            if a["routeless_claims"]:
                stats["valid_with_routeless_claims"] += 1
        out["models"][model] = stats
    tot_valid = sum(m["n_valid"] for m in out["models"].values())
    tot_dom = sum(m["valid_with_dominated_routes"] for m in out["models"].values())
    tot_rl = sum(m["valid_with_routeless_claims"] for m in out["models"].values())
    out["total"] = {"valid_programs": tot_valid,
                    "with_dominated_routes": tot_dom,
                    "with_routeless_claims": tot_rl}
    dst = os.path.join(RESULTS, "deployment_signal_audit.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    print("wrote %s" % dst, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
