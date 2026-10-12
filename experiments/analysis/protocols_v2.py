#!/usr/bin/env python3
"""Protocol pair F*_A / F*_B for the v2 benchmark (docs/EXECUTION_PLAN.md,
Part R4, Q1).

F*_A  = the generated gold licenses (the authored protocol).
F*_B  = the MECHANICAL conjunctive collapse of A: per claim, one route =
        the union of all atoms across A's alternative routes (every atom A
        treats as optional becomes required). Strictly stronger by
        construction.

Endpoints derived from the pair (the term "protocol-robust unsupported" is
retired -- it reduces to OFAR-A):
  OFAR-A, OFAR-B, protocol disagreement rate, protocol sensitivity
  (OFAR_B - OFAR_A), content-false (from claim denotations).

Realizability gate (Part R4, Q1): Phi AND p_B must be satisfiable. For the v2
instance class (positive literals, phi = ()), unsatisfiability can only
arise from complementary literals or a phi forbid-pattern matched by the
forced assignment; both are checked mechanically. An unrealizable collapsed
route makes F*_B UNDEFINED for that claim (reported, never silently
claim-rejecting).

Freezing: this script canonicalizes and hashes both protocols over a split
and writes results/protocols_v2.json. The held-out hashes are recorded in
docs/ANALYSIS_PLAN.md BEFORE any model output on held-out v2 instances is
inspected.

Usage: python3 experiments/analysis/protocols_v2.py [--splits heldout_balanced heldout_natural dev val]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
BENCH = os.path.join(PILOT, "data", "obligationbench")
RESULTS = os.path.join(PILOT, "results")


def collapse(routes: List[List[str]]) -> List[str]:
    """Conjunctive collapse: union of atoms across alternative routes,
    deterministic order."""
    out = sorted({x for r in routes for x in r})
    return out


def realizable(route: List[str], phi: List[Dict[str, Any]]) -> bool:
    """Phi AND route satisfiable, for the restricted literal language:
    a route with both 'X' and '!X' is unsatisfiable; a phi forbid-pattern
    fully forced by the route's polarity assignment is a violation."""
    pos = {x for x in route if not x.startswith("!")}
    neg = {x[1:] for x in route if x.startswith("!")}
    if pos & neg:
        return False
    forced = {**{a: "T" for a in pos}, **{a: "F" for a in neg}}
    for pat in phi or []:
        forbid = pat.get("forbid", pat) if isinstance(pat, dict) else {}
        if forbid and all(forced.get(a) == v for a, v in forbid.items()):
            return False
    return True


def protocol_pair(inst: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    A = {cid: [sorted(r) for r in routes]
         for cid, routes in inst["gold_licenses"].items()}
    B: Dict[str, Any] = {}
    for cid, routes in inst["gold_licenses"].items():
        c = collapse(routes)
        B[cid] = {"route": c, "realizable": realizable(c, inst.get("phi", []))}
    return A, B


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+",
                    default=["dev", "val", "heldout_balanced", "heldout_natural"])
    args = ap.parse_args()

    out: Dict[str, Any] = {"per_split_hashes": {}, "unrealizable": []}
    for split in args.splits:
        blobA, blobB = [], []
        n = 0
        for path in sorted(glob.glob(os.path.join(BENCH, "*", split, "*.json"))):
            inst = json.load(open(path, encoding="utf-8"))
            A, B = protocol_pair(inst)
            blobA.append((inst["id"], A))
            blobB.append((inst["id"], B))
            n += 1
            for cid, spec in B.items():
                if not spec["realizable"]:
                    out["unrealizable"].append({"instance": inst["id"], "claim": cid})
        out["per_split_hashes"][split] = {
            "n_instances": n,
            "FA_sha256": hashlib.sha256(canonical(blobA).encode()).hexdigest(),
            "FB_sha256": hashlib.sha256(canonical(blobB).encode()).hexdigest(),
        }
    os.makedirs(RESULTS, exist_ok=True)
    dst = os.path.join(RESULTS, "protocols_v2.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps(out["per_split_hashes"], indent=1))
    print("unrealizable collapsed routes:", len(out["unrealizable"]))
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
