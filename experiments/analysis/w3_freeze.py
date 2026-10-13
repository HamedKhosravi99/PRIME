#!/usr/bin/env python3
"""W3 freeze gate: prove the MECHANICAL half of the induction stack is exact
on the whole benchmark before any GPU is spent on the W2b induced runs.

experiments/analysis/induction_v5.py has two halves. The LLM half (state labelling) is
what W2b actually measures. The mechanical half -- realizable-state
enumeration, exact DNF/prime-implicant route compilation, and the typed
linter -- is pure code, and if it is not exact then every induced program in
W2b is compiled wrong and the whole workstream is void regardless of how good
the labels are.

So we feed the compiler PERFECT labels, derived from each instance's own gold
licences, and require that it recovers a route family that is exactly
equivalent to gold on every realizable state. This is a round-trip identity,
not a similarity score: any single state where the compiled family and the
gold family disagree is a hard failure. The unit tests check this on one
fixture; this checks it on all 480 confirmatory instances.

The gold licences are read HERE only as an oracle for validating the
compiler. Nothing in the induction path itself reads them.

Usage:
  python3 experiments/analysis/w3_freeze.py                    # confirmatory splits
  python3 experiments/analysis/w3_freeze.py --split dev --limit 50
Exit status is nonzero if any instance fails, so it can gate a run.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import induction_v5 as iv  # noqa: E402

BENCH = os.path.join(PILOT, "data", "suffice")
DOMAINS = ("security", "clinical", "drugsafety", "financial")


def satisfied(routes, state) -> bool:
    """Claim is supported at a state iff some route's literals all hold."""
    return any(all(a in state for a in r) for r in routes)


def check(path: str) -> dict:
    inst = iv.load_instance(path)
    states = iv.realizable_states(inst)
    atoms = [a.id for a in inst.atoms]
    gold = {c.id: [list(r) for r in inst.gold_licenses.get(c.id, [])]
            for c in inst.claims}

    out = {"id": inst.id, "n_states": len(states), "claims": {}}
    for cid, groutes in gold.items():
        # perfect labels: exactly the states where gold licenses the claim
        accepted = {s for s in states if satisfied(groutes, s)}
        compiled = iv.compile_routes_exact(accepted, states, atoms)
        # round-trip: agree on EVERY realizable state, not just on average
        bad = [sorted(s) for s in states
               if satisfied(compiled, s) != (s in accepted)]
        out["claims"][cid] = {
            "gold_routes": len(groutes), "compiled_routes": len(compiled),
            "mismatched_states": len(bad),
            "example": bad[0] if bad else None,
            # a compiled family must never be larger than gold in width:
            # prime implicants are minimal, so a wider route means a
            # non-minimal compile
            "max_width_gold": max([len(r) for r in groutes] + [0]),
            "max_width_compiled": max([len(r) for r in compiled] + [0]),
        }
    lint = iv.lint_program(inst, {c: [list(r) for r in gold[c]] for c in gold})
    # Only findings that NAME A CLAIM gate guidance (induction_v5.eligibility),
    # so only those can invalidate a program. `unused_vocabulary_atom` names an
    # atom and fires on gold by construction: SUFFICE seeds distractor
    # atoms the gold routes deliberately do not use, which is the discrimination
    # the benchmark is testing. Counting it as a defect would fail every
    # instance in the corpus.
    # `unresolvable_atom_in_route` names a claim and so DOES gate guidance,
    # but on gold it fires on the benchmark's evidence-unavailable stratum:
    # 146 atoms across the balanced split carry no resolver, gold_value F, and
    # produce exactly the 235 content=unavailable claims. Nothing can ever
    # establish them, so refusing to guide toward those claims is the
    # fail-closed behaviour working, not a defect -- and W2b must report the
    # count, because those claims are guidance-ineligible for a STRUCTURAL
    # reason rather than an induction-quality one.
    noresolver = {a.id for a in inst.atoms if a.resolver is None}
    expected, blocking = [], []
    for f in lint:
        if not f.get("claim"):
            continue
        if (f.get("lint") == "unresolvable_atom_in_route"
                and f.get("atom") in noresolver):
            expected.append(f)
        else:
            blocking.append(f)
    out["lint_blocking"] = blocking
    out["lint_unavailable_stratum"] = expected
    out["lint_informational"] = [f for f in lint if not f.get("claim")]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="heldout_balanced")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(BENCH, "*", a.split, "*.json")))
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        print("no instances for split %s" % a.split)
        return 2

    fails, lintfails, widths = [], [], collections.Counter()
    info = collections.Counter()
    strat = strat_inst = 0
    results = []
    for p in paths:
        r = check(p)
        results.append(r)
        for cid, c in r["claims"].items():
            if c["mismatched_states"]:
                fails.append((r["id"], cid, c["mismatched_states"], c["example"]))
            widths[c["max_width_compiled"] - c["max_width_gold"]] += 1
        if r["lint_blocking"]:
            lintfails.append((r["id"], r["lint_blocking"][:2]))
        info.update(f["lint"] for f in r["lint_informational"])
        strat += len(r["lint_unavailable_stratum"])
        if r["lint_unavailable_stratum"]:
            strat_inst += 1

    n_claims = sum(len(r["claims"]) for r in results)
    print("W3 mechanical freeze gate -- split=%s" % a.split)
    print("  instances checked        : %d" % len(results))
    print("  claims round-tripped     : %d" % n_claims)
    print("  realizable states total  : %d" % sum(r["n_states"] for r in results))
    print("  compile mismatches       : %d" % len(fails))
    print("  blocking lint on gold    : %d" % len(lintfails))
    print("  informational lint       : %s (expected: distractor atoms)"
          % dict(info))
    print("  unavailable-stratum lint : %d findings over %d instances "
          "(guidance-ineligible by construction, not a defect)"
          % (strat, strat_inst))
    print("  compiled-vs-gold width   : %s"
          % dict(sorted(widths.items())))
    for f in fails[:5]:
        print("    FAIL %s/%s: %d states disagree, e.g. %s" % f)
    for f in lintfails[:5]:
        print("    LINT %s: %s" % f)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print("  wrote %s" % a.out)

    ok = not fails and not lintfails
    print("  VERDICT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
