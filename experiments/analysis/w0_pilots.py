#!/usr/bin/env python3
"""W0-exit pilot analysis (docs/ANALYSIS_PLAN.md, F3 and F6 slots):
computed from the v2dev LLM runs + the computed planners.

Outputs (per model tag):
  1. G_dev      mean utility-regret gap masked_reactive_v ->
                informed_route_planner, DEV BRANCHING stratum, frozen
                budgets -> MDE delta = max(5.0, 0.20 * G_dev) [0-100 scale].
  2. TOST power at delta, from dev paired-difference variance with
                template-family clustering (analytic normal approximation
                on cluster means) -> decides verdict-vs-CI language.
  3. Instruction effect: react_program vs react_program_verify (OFAR +
                utility deltas) -- destined for the paper body either way.
  4. E1 degeneracy check: react_program_attested vs masked_reactive_v
                discordant-cluster count on dev.

Judge-free: utility from verified gold values; regret vs the computed
informed_route_planner.

Usage: python3 experiments/analysis/w0_pilots.py --tag qwen2.5-7b   (or 32b / untagged)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS = os.path.join(PILOT, "results")
BENCH = os.path.join(PILOT, "data", "suffice")
DOMAINS = ("security", "clinical", "drugsafety", "financial")


def inst_meta(path: str) -> Dict[str, Any]:
    inst = json.load(open(path, encoding="utf-8"))
    meta = json.loads(inst["notes_for_annotators"])
    gv = {a["id"]: a["gold_value"] for a in inst["atoms"]}
    levels = {c["id"]: c["level"] for c in inst["claims"]}
    supported = {cid: any(all(gv.get(x) == "T" for x in r) for r in routes)
                 for cid, routes in inst["gold_licenses"].items()}
    ceiling = max([levels[c] for c, ok in supported.items() if ok] + [0])
    return {"topology": meta["topology"], "family": meta["family"],
            "levels": levels, "supported": supported, "ceiling": ceiling,
            "record": inst["seed"]}


META: Dict[str, Dict[str, Any]] = {}
for dom in DOMAINS:
    for p in glob.glob(os.path.join(BENCH, dom, "dev", "*.json")):
        META[json.load(open(p))["id"]] = inst_meta(p)


def epi_utility(e: Dict[str, Any], m: Dict[str, Any]) -> float:
    """Supported utility on [0,100]: best emitted claim that is oracle-
    supported, level/3 x100; unsupported emissions contribute 0 here (OFAR
    is tracked separately)."""
    best = 0
    for x in e.get("emitted") or []:
        cid = x["claim"]
        if m["supported"].get(cid):
            best = max(best, m["levels"].get(cid, 0))
    return 100.0 * best / 3.0


def epi_ofar(e: Dict[str, Any], m: Dict[str, Any]) -> int:
    return int(any(not m["supported"].get(x["claim"], False)
                   for x in (e.get("emitted") or [])))


def load_eps(tag: str, dom: str, arm: str, B: int) -> Optional[List[Dict[str, Any]]]:
    cands = [os.path.join(RESULTS, "v2dev", "%s-%s" % (tag, dom), "%s__B%d" % (arm, B)),
             os.path.join(RESULTS, "v2dev", dom, "%s__B%d" % (arm, B))]
    for c in cands:
        p = os.path.join(c, "summary.json")
        if os.path.exists(p):
            return json.load(open(p))["episodes"]
    return None


def planner_levels(dom: str, B: int) -> Dict[str, int]:
    """informed_route_planner level per dev instance at budget B (computed
    fresh; cached per call)."""
    import policies
    out = {}
    for p in glob.glob(os.path.join(BENCH, dom, "dev", "*.json")):
        r = policies.plan_bounds(p, B)
        out[r["instance"]] = r["informed_route_planner_level"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--budgets", nargs="+", type=int, default=[3, 4, 20])
    args = ap.parse_args()
    tag = args.tag

    out: Dict[str, Any] = {"tag": tag}

    # ---- 1. G_dev on the branching stratum --------------------------------
    gaps = []  # per (instance, budget): planner_util - mrv_util
    cluster_diffs = defaultdict(list)  # family -> diffs (for TOST variance)
    for dom in DOMAINS:
        for B in args.budgets:
            eps = load_eps(tag, dom, "masked_reactive_v", B)
            if eps is None:
                continue
            plan = planner_levels(dom, B)
            for e in eps:
                m = META.get(e["instance"])
                if m is None or m["topology"] != "branching":
                    continue
                u_m = epi_utility(e, m)
                u_p = 100.0 * plan.get(e["instance"], 0) / 3.0
                gaps.append(u_p - u_m)
                cluster_diffs[m["family"]].append(u_p - u_m)
    if gaps:
        G = sum(gaps) / len(gaps)
        out["G_dev"] = round(G, 2)
        out["n_gap_obs"] = len(gaps)
        out["n_gap_clusters"] = len(cluster_diffs)
        out["MDE_delta"] = round(max(5.0, 0.20 * G), 2)

    # ---- 2. TOST power at delta (cluster-mean normal approximation) -------
    if cluster_diffs:
        cm = [sum(v) / len(v) for v in cluster_diffs.values()]
        k = len(cm)
        mean = sum(cm) / k
        sd = math.sqrt(sum((x - mean) ** 2 for x in cm) / max(1, k - 1))
        se = sd / math.sqrt(k)
        delta = out.get("MDE_delta", 5.0)
        # TOST power under true effect 0: P(|Z| within both one-sided tests)
        # approx: power = Phi((delta - 1.645*se)/se) - Phi((-delta + 1.645*se)/se)
        z = (delta - 1.645 * se) / se if se > 0 else 99
        from math import erf
        Phi = lambda x: 0.5 * (1 + erf(x / math.sqrt(2)))
        power = max(0.0, Phi(z) - Phi(-z))
        out["tost"] = {"clusters": k, "cluster_sd": round(sd, 2),
                       "se": round(se, 2), "delta": delta,
                       "power_at_zero_effect": round(power, 3),
                       "verdict_language": bool(power >= 0.8)}

    # ---- 3. instruction effect (react_program pair, B=20) -----------------
    instr = {}
    for dom in DOMAINS:
        a = load_eps(tag, dom, "react_program", 20)
        b = load_eps(tag, dom, "react_program_verify", 20)
        if not (a and b):
            continue
        for name, eps in (("uninstructed", a), ("instructed", b)):
            of = sum(epi_ofar(e, META[e["instance"]]) for e in eps if e["instance"] in META)
            ut = sum(epi_utility(e, META[e["instance"]]) for e in eps if e["instance"] in META)
            n = sum(1 for e in eps if e["instance"] in META)
            d = instr.setdefault(dom, {})
            d[name] = {"ofar": round(of / n, 3), "utility": round(ut / n, 1), "n": n}
    out["instruction_effect"] = instr

    # ---- 4. E1 degeneracy: discordant clusters on dev ---------------------
    disc = 0
    tot_cl = set()
    for dom in DOMAINS:
        a = load_eps(tag, dom, "react_program_attested", 20)
        b = load_eps(tag, dom, "masked_reactive_v", 20)
        if not (a and b):
            continue
        am = {e["instance"]: epi_ofar(e, META[e["instance"]]) for e in a if e["instance"] in META}
        bm = {e["instance"]: epi_ofar(e, META[e["instance"]]) for e in b if e["instance"] in META}
        fam_diff = defaultdict(int)
        for iid in set(am) & set(bm):
            fam = META[iid]["family"]
            tot_cl.add(fam)
            fam_diff[fam] += am[iid] - bm[iid]
        disc += sum(1 for v in fam_diff.values() if v != 0)
    out["e1_dev"] = {"discordant_clusters": disc, "total_clusters": len(tot_cl)}

    dst = os.path.join(RESULTS, "w0_pilots_%s.json" % tag)
    json.dump(out, open(dst, "w"), indent=2, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
