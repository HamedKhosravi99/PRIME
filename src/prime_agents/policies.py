#!/usr/bin/env python3
"""Deterministic (no-LLM) policy arms and computed planners for the v5
control-law study (docs/EXECUTION_PLAN.md, Part 1 and Part R4).

Policy arms -- run REAL episodes (real ToolExecutor calls, real attested
ledger, same MaxAuth emission), only ACTION SELECTION is scripted:

  shortest_route         resolve the open atoms of the currently cheapest
                         (call-class cost) live route of the HIGHEST claim
                         with any live route; recompute each step.
  cheapest_literal       resolve the globally cheapest unresolved literal
                         among live routes (call-class dedup), ignoring
                         claim structure.
  random_masked          uniformly random eligible evidence call (seeded by
                         instance id; reproducible).
  masked_reactive_prefix reactive = no ceiling targeting; resolves open
                         route literals in PREFIX-FIRST order (route-
                         membership depth desc, then atom id). The
                         attribution arm for the ordering-vs-direction
                         question; runs UPFRONT on all branching
                         confirmatory instances (docs/EXECUTION_PLAN.md,
                         Part R4, Q4).

Computed planners -- no episodes at all, pure DP over the committed program
and the instance's verified gold values:

  clairvoyant_ceiling      sees true outcomes; regret ceiling, NONDEPLOYABLE.
  informed_route_planner   knows WHICH routes are all-true (outcome-informed)
                           but pays full call-class resolution cost for every
                           literal on the chosen route.

Every policy arm episode writes the same summary shape as runner arms
(instance, emitted, tool_calls, ...), scored by the same downstream
machinery.

Usage:
  python3 src/prime_agents/policies.py --instances 'data/obligationbench/security/dev/*.json' \
      --arm shortest_route --budget 20 --out results/policy_runs/...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

_PILOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUNNER = os.path.dirname(os.path.abspath(__file__))
for _p in (_PILOT, os.path.join(_PILOT, "src"), _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
from prime_runtime import goldledger, guidance, status  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import Instance, load_instance  # noqa: E402
from prime_runtime import auth as prime_auth  # noqa: E402

POLICY_ARMS = ("shortest_route", "cheapest_literal", "random_masked",
               "masked_reactive_prefix", "shortest_route_candidates",
               "random_masked_candidates")
# random_masked_candidates: random_masked on the locator-uncertain split with
# the locators withheld: a random open atom, then a random candidate field
# not yet read. Every policy without "_candidates" uses the true locators.
# shortest_route_candidates: shortest_route on the locator-uncertain split,
# where an atom's spec view lists candidate fields but not which one is
# right. Code cannot read the description, so it tries the candidates in
# the listed order, one call each, until the atom is attested or the list
# is exhausted. With the true locators (every other split) it IS
# shortest_route.


def _call_class_of(atom_id: str, resolvers) -> Any:
    spec = resolvers.get(atom_id, {}) or {}
    return spec.get("call_class", atom_id)


def _live_open(program, ledger, resolvers) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """claim -> [(route, open_atoms)] over non-dead routes."""
    out: Dict[str, List[Tuple[List[str], List[str]]]] = {}
    for cid in program.claims:
        rs = []
        for route in program.routes(cid):
            open_atoms = guidance._route_open_atoms(route, ledger, resolvers)
            if open_atoms is not None:
                rs.append((list(route), list(open_atoms)))
        if rs:
            out[cid] = rs
    return out


def _prefix_depth(program):
    depth: Dict[str, int] = {}
    for cid in program.claims:
        for r in program.routes(cid):
            for lit in r:
                a = lit.lstrip("!")
                depth[a] = depth.get(a, 0) + 1
    return depth


def pick_action(arm: str, instance: Instance, program, ledger, resolvers,
                levels, rng: random.Random) -> Optional[str]:
    """Return the next atom id to resolve, or None (stop)."""
    live = _live_open(program, ledger, resolvers)
    all_open = sorted({a.lstrip("!") for rs in live.values()
                       for _r, opens in rs for a in opens})
    if not all_open:
        return None
    if arm in ("random_masked", "random_masked_candidates"):
        return rng.choice(all_open)
    if arm == "cheapest_literal":
        # cheapest = literal whose call class resolves the most open atoms
        # (cost 1 call, max information), tie-break atom id
        gain = {}
        for a in all_open:
            cc = _call_class_of(a, resolvers)
            gain[a] = sum(1 for b in all_open if _call_class_of(b, resolvers) == cc)
        return sorted(all_open, key=lambda a: (-gain[a], a))[0]
    if arm in ("shortest_route", "shortest_route_candidates"):
        best = None  # (cost, -level, claim) -> opens; satisfied routes are
        # not candidates (nothing to resolve on them)
        for cid, rs in live.items():
            for route, opens in rs:
                if not opens:
                    continue
                cost = len({_call_class_of(a.lstrip("!"), resolvers) for a in opens})
                key = (cost, -levels.get(cid, 0), cid)
                if best is None or key < best[0]:
                    best = (key, opens)
        if best is None:
            return None
        return sorted(a.lstrip("!") for a in best[1])[0]
    if arm == "masked_reactive_prefix":
        depth = _prefix_depth(program)
        return sorted(all_open, key=lambda a: (-depth.get(a, 0), a))[0]
    raise ValueError(arm)


def run_policy_episode(path: str, arm: str, budget: int) -> Dict[str, Any]:
    instance = load_instance(path)
    program = instance.gold_program()
    resolvers = status.resolver_table(instance)
    levels = {c.id: c.level for c in instance.claims}
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    executor = arms_mod.ToolExecutor(instance)
    rng = random.Random(instance.id)
    by_id = {a.id: a for a in instance.atoms}
    calls = 0
    step = 0
    tried_candidates: Dict[str, int] = {}
    while calls < budget:
        nxt = pick_action(arm, instance, program, ledger, resolvers, levels, rng)
        if nxt is None:
            break
        atom = by_id[nxt]
        if atom.resolver is not None and atom.resolver.tool == "attest_atom":
            # text-evidence atom: no code can resolve it; a scripted policy treats it as impossible
            resolvers = dict(resolvers)
            resolvers[nxt] = {**(resolvers.get(nxt) or {}), "impossible": True, "enabled": False, "tool": None}
            continue
        hint = getattr(atom, "hint", None)
        if arm.endswith("_candidates") and hint and atom.resolver is not None:
            cands = list(hint.get("candidates", []))
            if arm == "random_masked_candidates":
                # a fixed random order per (task, atom), seeded like the policy
                cands = sorted(cands, key=lambda c: random.Random("%s|%s|%s" % (instance.id, nxt, c)).random())
            k = tried_candidates.get(nxt, 0)
            if k >= len(cands):
                # every listed field tried without attestation: give up on it
                resolvers = dict(resolvers)
                resolvers[nxt] = {**(resolvers.get(nxt) or {}), "impossible": True,
                                  "enabled": False, "tool": None}
                continue
            tried_candidates[nxt] = k + 1
            step += 1
            params = dict(atom.resolver.params, field_path=cands[k])
            obs = executor.execute(atom.resolver.tool, params)
            if not executor.last_was_cached:
                calls += 1
            for att in goldledger.delta(instance, ledger, atom.resolver.tool, params, obs, step):
                ledger.apply(att)
            continue
        if atom.resolver is None:
            # unresolvable atom: mark via goldledger path (no call consumed);
            # exclude it from future picks by attesting F? The ledger has no
            # tool to move it; treat as dead-end: stop considering it by
            # applying an unavailability conflict is NOT in the schema, so
            # simply never pick it again via a local tabu.
            resolvers = dict(resolvers)
            resolvers[nxt] = {**(resolvers.get(nxt) or {}), "impossible": True,
                              "enabled": False, "tool": None}
            continue
        step += 1
        obs = executor.execute(atom.resolver.tool, dict(atom.resolver.params))
        if not executor.last_was_cached:
            calls += 1
        for att in goldledger.delta(instance, ledger, atom.resolver.tool,
                                    dict(atom.resolver.params), obs, step):
            ledger.apply(att)
    maximal = prime_auth.max_auth(program, ledger)
    emitted = [{"claim": cid, "step": step, "ledger_version": ledger.version,
                "level": levels.get(cid)} for cid in sorted(maximal)]
    return {"instance": instance.id, "instance_path": os.path.abspath(path),
            "arm": arm, "license_source": "gold", "budget": budget,
            "emitted": emitted, "tool_calls": calls,
            "final_states": ledger.states(), "error": ""}


# ---------------------------------------------------------------------------
# computed planners (DP over call classes; uses verified gold values)
# ---------------------------------------------------------------------------
def plan_bounds(path: str, budget: int) -> Dict[str, Any]:
    inst_raw = json.load(open(path, encoding="utf-8"))
    instance = load_instance(path)
    resolvers = status.resolver_table(instance)
    levels = {c.id: c.level for c in instance.claims}
    gv = {a["id"]: a.get("gold_value") for a in inst_raw["atoms"]}
    best_clair = 0
    best_informed = 0
    for cid, routes in instance.gold_licenses.items():
        for r in routes:
            atoms = [x.lstrip("!") for x in r]
            if not all(gv.get(a) == "T" for a in atoms):
                continue
            cost = len({_call_class_of(a, resolvers) for a in atoms})
            if cost <= budget:
                # informed: pays the full route resolution cost
                best_informed = max(best_informed, levels[cid])
            # clairvoyant: knows outcomes; still must "cite" the route --
            # by convention the ceiling pays the same cost floor, so the
            # ceiling differs from informed only when budget < cost of every
            # all-true route of the top supported claim; the TRUE ceiling is
            # the strongest supported claim regardless of budget:
            best_clair = max(best_clair, levels[cid])
    return {"instance": instance.id, "clairvoyant_ceiling_level": best_clair,
            "informed_route_planner_level": best_informed, "budget": budget}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--arm", choices=POLICY_ARMS + ("planners",), required=True)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--budget-anchor", default="fixed",
                    choices=("fixed", "bmin", "bmin1", "2bmin"),
                    help="spec-only per-instance anchors (docs/ANALYSIS_PLAN.md, F4), "
                         "computed by the SAME function run_batch uses. "
                         "Without this the policy family can only be run at "
                         "a fixed budget, and at B=20 every arm reaches the "
                         "ceiling so the comparison is vacuous.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(args.instances))
    os.makedirs(args.out, exist_ok=True)
    episodes = []
    from run_batch import instance_budget
    for p in paths:
        b = instance_budget(p, args.budget_anchor, args.budget)
        if args.arm == "planners":
            episodes.append(plan_bounds(p, b))
        else:
            episodes.append(run_policy_episode(p, args.arm, b))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"run": {"arm": args.arm, "budget": args.budget,
                           "budget_anchor": args.budget_anchor,
                           "instances": paths}, "episodes": episodes},
                  fh, indent=1, sort_keys=True, default=str)
    print("wrote %s (%d episodes)" % (os.path.join(args.out, "summary.json"),
                                      len(episodes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
