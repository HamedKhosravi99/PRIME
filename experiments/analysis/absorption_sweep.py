#!/usr/bin/env python3
"""Exercise the exact absorption check on every committed program
(results/tableABS.json).

The exact robust route-absorption check (the paper's main formal result) is
stated, proved, and then never exercised: the deployed system runs under a
full in-episode freeze, so nothing revises a program, and the three
unsound shortcuts are falsified only by three hand-built fixtures. That
leaves the theorem governing a capability the system disables.

This sweeps it over the committed programs of every confirmatory instance.
For each claim it enumerates the revisions the runtime would actually have
to decide -- remove a literal from a route, add a route, add a literal,
delete a route -- and evaluates four checkers on each:

and reports what the exact check decides, by revision kind, at scale.

It does NOT re-derive the three unsound shortcuts. Reproducing them
faithfully is subtle -- R1 turns on ontic entailment over complete
realizable assignments rather than the epistemic closure the runtime uses,
and R2/R3 turn on conflict masks -- and a straw-man checker that is merely
stricter than the exact one would produce a flattering comparison that means
nothing. The shortcuts stay falsified where they were falsified: by the
pinned R1/R2/R3 witnesses in tests, which run on every build.
What this adds is the part that was missing, namely that the exact check
governs the real committed programs rather than three fixtures.

Usage: python3 experiments/analysis/absorption_sweep.py [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime import absorption as ab  # noqa: E402
from prime_runtime.schema import make_program  # noqa: E402

RESULTS = os.path.join(PILOT, "results")
BENCH = os.path.join(PILOT, "data", "obligationbench")


def program_of(inst):
    claims = [c["id"] for c in inst["claims"]]
    atoms = [a["id"] for a in inst["atoms"]]
    return make_program(claims, [tuple(p) for p in inst.get("order", [])],
                        inst["gold_licenses"], inst.get("scope", ""), atoms)


def candidate_revisions(prog, cid, routes):
    """The revision kinds the runtime must decide."""
    out = []
    all_atoms = list(prog.atoms)
    for r in routes:
        r = list(r)
        for lit in r:                                   # literal removal
            out.append(("remove_literal", tuple(x for x in r if x != lit)))
        for extra in all_atoms[:3]:                     # literal addition
            if extra not in r:
                out.append(("add_literal", tuple(r + [extra])))
    for r in routes:                                    # route deletion
        rest = [x for x in routes if x != r]
        if rest:
            out.append(("delete_route", None))
            break
    if routes:                                          # route addition
        base = list(routes[0])
        for k in range(1, min(len(base), 3) + 1):
            for sub in itertools.combinations(base, k):
                if len(sub) < len(base):
                    out.append(("add_route", tuple(sub)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(BENCH, "*", "heldout_balanced",
                                          "*.json")))
    if a.limit:
        files = files[:a.limit]
    tally = {"exact": {"accept": 0, "reject": 0}}
    examples = {}
    n_rev = n_inst = 0
    by_kind = {}

    for path in files:
        try:
            inst = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prog = program_of(inst)
        phi = inst.get("phi") or []
        n_inst += 1
        for cid in prog.claims:
            routes = [tuple(r) for r in prog.routes(cid)]
            if not routes:
                continue
            family = ab.absorption_family(cid, prog)
            for kind, new_route in candidate_revisions(prog, cid, routes):
                n_rev += 1
                lic = {c: [list(r) for r in prog.routes(c)]
                       for c in prog.claims}
                if kind == "delete_route":
                    lic[cid] = [list(r) for r in routes[1:]]
                    probe = None
                elif kind == "add_route":
                    lic[cid] = [list(r) for r in routes] + [list(new_route)]
                    probe = new_route
                else:
                    lic[cid] = [list(new_route)] + [list(r) for r in routes[1:]]
                    probe = new_route
                pn = make_program(list(prog.claims),
                                  [tuple(p) for p in inst.get("order", [])],
                                  lic, inst.get("scope", ""), list(prog.atoms))
                ok = ab.check_patch(prog, pn, phi)
                tally["exact"]["accept" if ok else "reject"] += 1
                slot = by_kind.setdefault(kind, {"n": 0, "exact_accept": 0})
                slot["n"] += 1
                slot["exact_accept"] += 1 if ok else 0
                if probe is not None and not ok and kind not in examples:
                    examples[kind] = {
                        "instance": inst.get("id"), "claim": cid,
                        "revised_route": list(probe),
                        "committed_routes": [list(r) for r in routes],
                        "absorption_family": [sorted(q) for q in family]}
    out = {"n_instances": n_inst, "n_revisions": n_rev,
           "tally": tally, "by_kind": by_kind, "rejected_examples": examples}
    with open(os.path.join(RESULTS, "tableABS.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("instances %d, candidate revisions %d" % (n_inst, n_rev))
    v = tally["exact"]
    print("  exact check: accept %d, reject %d" % (v["accept"], v["reject"]))
    for k, s2 in sorted(by_kind.items()):
        print("    %-16s n=%5d accepted=%5d (%.1f%%)"
              % (k, s2["n"], s2["exact_accept"],
                 100.0 * s2["exact_accept"] / max(1, s2["n"])))
    print("wrote results/tableABS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
