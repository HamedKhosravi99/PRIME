#!/usr/bin/env python3
"""Score an induced claim--evidence program without spending a GPU.

WHY THIS EXISTS. The missing piece is an induction procedure whose committed
programs keep meaningful utility at near-zero false authorization. Today
induction gives OFAR-A 5.5 at utility 16.0 against the authored program's 0.0
at 45.5. Searching for a better induction procedure by re-running episodes
costs GPU-hours per candidate, which is why only one procedure was ever tried.

The search does not need episodes. A committed program is enforced by a
deterministic mask over an attested ledger, so given the program and the
instance's atom values the emission is a FUNCTION -- no model in the loop.
This module evaluates that function directly:

    for each instance:
        ledger  <- every atom resolved to its true value (the ceiling case)
        emitted <- MaxAuth(induced program, ledger)
        supported? <- does the GOLD licence family support each emitted claim

so it reports, per induction policy:

    ofar_a   fraction of instances emitting a claim the oracle does not support
    utility  mean strongest supported level emitted, on the paper's 0/33/67/100 scale
    silence  fraction of instances that license nothing at all

WHAT THIS IS AND IS NOT. This is an upper bound on utility (the agent is
assumed to resolve every atom it needs, which a budgeted episode may not) and
an exact measure of program SOUNDNESS (a route that licenses an unsupported
claim does so regardless of budget). It is a program-quality instrument, not
an end-to-end result, and any candidate it likes must still be confirmed by
real episodes before it goes in the paper. It exists to make the search
cheap, not to replace the measurement.

The gold licences are read ONLY to score. Nothing here feeds induction.

Usage:
  python3 experiments/analysis/induction_policy_lab.py                 # score every policy
  python3 experiments/analysis/induction_policy_lab.py --policy authored
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
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BENCH = os.path.join(PILOT, "data", "obligationbench")
CACHE = os.path.join(PILOT, "results", "induction_cache")
SPLIT = "heldout_balanced"


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------
def load_instances():
    out = []
    for p in sorted(glob.glob(os.path.join(BENCH, "*", SPLIT, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        meta = d["notes_for_annotators"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        out.append({
            "id": d["id"],
            "family": meta["family"],
            "levels": {c["id"]: c["level"] for c in d["claims"]},
            "gold": d["gold_licenses"],
            "atoms": {a["id"]: a["gold_value"] for a in d["atoms"]},
            "order": d.get("order") or [],
        })
    return out


def load_cached(policy="unanimous", draws=3):
    """family -> induced licence family, as committed today."""
    fams = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("policy") == policy and d.get("draws") == draws:
            fams[d["family"]] = d["licenses"]
    return fams


# ---------------------------------------------------------------------------
# the emission function
# ---------------------------------------------------------------------------
def satisfied(route, atoms):
    """A route fires when every literal on it is established TRUE.

    Fail-closed: 'U' (unavailable) and 'F' both fail to establish. This is
    the deployed rule, not a relaxation of it.
    """
    return bool(route) and all(atoms.get(a) == "T" for a in route)


def licensed(program, atoms):
    """Claims the committed program licenses at this ledger state."""
    return {c for c, routes in program.items()
            if any(satisfied(r, atoms) for r in (routes or []))}


def emitted_level(program, inst):
    """Strongest level the mask would permit, and the claims it permits.

    MaxAuth is the maximal authorized antichain; for the chain orders in this
    benchmark the strongest licensed claim is what the arm emits.
    """
    lic = licensed(program, inst["atoms"])
    if not lic:
        return 0, set()
    lvl = max(inst["levels"].get(c, 0) for c in lic)
    return lvl, lic


def score(programs, insts, label):
    """programs: family -> licences. Missing family => nothing licensed."""
    n = len(insts)
    ofar = util = silent = 0
    per_family = collections.defaultdict(list)
    for i in insts:
        prog = programs.get(i["family"])
        if not prog:
            silent += 1
            per_family[i["family"]].append((0, 0))
            continue
        lvl, lic = emitted_level(prog, i)
        # Unsupported iff the GOLD licences do not license that claim at this
        # ledger state. This is the oracle, used only for scoring.
        bad = any(not any(satisfied(r, i["atoms"]) for r in (i["gold"].get(c) or []))
                  for c in lic)
        ofar += int(bad)
        # utility credits only the strongest SUPPORTED claim, as the paper does
        sup = [i["levels"][c] for c in lic
               if any(satisfied(r, i["atoms"]) for r in (i["gold"].get(c) or []))]
        u = 100.0 * (max(sup) if sup else 0) / 3.0
        util += u
        if not lic:
            silent += 1
        per_family[i["family"]].append((int(bad), u))
    return {
        "policy": label,
        "n_instances": n,
        "n_families": len(per_family),
        "ofar_a": round(100.0 * ofar / n, 1),
        "utility": round(util / n, 1),
        "silence": round(100.0 * silent / n, 1),
    }


# ---------------------------------------------------------------------------
# width awareness
# ---------------------------------------------------------------------------
# The ceiling scoring above assumes the agent resolves every atom a route
# needs. It does not, and the gap is not a budget effect: measured over the
# confirmatory episodes the agent resolves 3.11-3.51 atoms whether its budget
# is the per-instance minimum or 20. So the probability a route FIRES falls
# off a cliff in its width, and a lab that ignores width will happily
# recommend wider programs -- which is precisely backwards, because widening
# is the only sound repair available without new elicitation.
#
# P(k atoms resolved >= w), measured from prime_guided at B20, n=480:
FIRE_P = {1: 0.996, 2: 0.925, 3: 0.679, 4: 0.356,
          5: 0.115, 6: 0.025, 7: 0.010, 8: 0.000}


def fire_prob(width):
    if width <= 0:
        return 0.0
    return FIRE_P.get(width, 0.0 if width > 8 else 1.0)


def score_width_aware(programs, insts, label):
    """A SEARCH HEURISTIC ONLY. Refuted as a predictor -- do not report it.

    It charges route width at P(agent resolves >= w atoms). An adversarial
    audit refuted three of its assumptions: (i) prefix-first gathering is
    active for 50% of authored and 4% of induced episodes but would be 100%
    under a nested program, so both calibration points sit outside the
    regime being predicted; (ii) atoms-resolved is not a sufficient
    statistic for utility -- authored utility FALLS 52.6 -> 45.5 from the
    tightest budget to B20 while atoms-resolved barely moves; (iii) the
    decay is non-monotonic in width, because per-atom truth is not uniform,
    so a wide route of always-true atoms outfires a narrow route containing
    a rarely-true one.

    It is retained because it still stops a search from preferring programs
    the runtime cannot execute, which is what an unweighted lab does. No
    number it produces belongs in the paper.
    """
    n = len(insts)
    util = ofar = 0.0
    for i in insts:
        prog = programs.get(i["family"])
        if not prog:
            continue
        best_u, p_bad = 0.0, 0.0
        for c, routes in prog.items():
            widths = [len(r) for r in (routes or []) if satisfied(r, i["atoms"])]
            if not widths:
                continue
            p = fire_prob(min(widths))
            ok = any(satisfied(g, i["atoms"]) for g in (i["gold"].get(c) or []))
            if ok:
                best_u = max(best_u, p * 100.0 * i["levels"].get(c, 0) / 3.0)
            else:
                p_bad = max(p_bad, p)
        util += best_u
        ofar += p_bad
    return {"policy": label, "ofar_a": round(100.0 * ofar / n, 1),
            "utility": round(util / n, 1)}


def authored_programs(insts):
    """The gold licence family per template family, as the ceiling reference."""
    out = {}
    for i in insts:
        out.setdefault(i["family"], i["gold"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    insts = load_instances()
    induced = load_cached()
    rows = [score(authored_programs(insts), insts, "authored (ceiling)"),
            score(induced, insts, "induced, unanimous/3 (as committed today)")]

    print("Program-quality lab -- %d instances, %d families, %s split"
          % (len(insts), len({i["family"] for i in insts}), SPLIT))
    print("upper bound on utility (every needed atom assumed resolved); "
          "exact on soundness")
    print()
    print("  %-44s %7s %8s %8s" % ("policy", "OFAR-A", "utility", "silence"))
    for r in rows:
        print("  %-44s %7.1f %8.1f %8.1f"
              % (r["policy"], r["ofar_a"], r["utility"], r["silence"]))
    if a.json:
        print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
