#!/usr/bin/env python3
"""Score the `conformal` arm: leave-one-out cross-conformal risk control
over model-elicited claim confidences -- the instantiation of the
conformal/calibrated-threshold family (Conformal Factuality, CSS) that this
benchmark CAN support without an external calibration set.

Mechanism (faithful to the family, judge-free):
  score        s(claim) = the model's own elicited confidence (0-100),
               logged by the `conformal` arm after free evidence gathering;
  labels       y(claim) = executable-oracle support at full evidence
               (verified per-atom gold values; no judge);
  calibration  LEAVE-ONE-OUT over instances within a (model, domain) cell:
               for test instance i, the threshold tau_i is the smallest
               candidate score such that, among all calibration (score,
               label) pairs from the OTHER instances with score >= tau_i,
               the unsupported fraction is <= alpha (risk-controlling
               selection, RCPS-style);
  emission     the strongest claim of instance i with s >= tau_i, else
               abstain (the family's graded back-off).

Honest small-n statement, printed and stored with every cell: with n_cal
calibration pairs the empirical-risk guarantee is only meaningful down to
alpha >= 1/(n_cal+1); cells where the requested alpha is below that floor
are flagged `alpha_below_floor` and must be read as best-effort, not as a
coverage guarantee.

Writes results/tableC_conformal.json.
Usage: python3 experiments/analysis/build_conformal.py --models qwen2.5-32b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime.schema import load_instance  # noqa: E402

RESULTS = os.path.join(PILOT, "results")
DOMAIN_SUB = {"research": "res", "security": "osv", "drugsafety": "fda",
              "financial": "fin", "clinical": "ctgov", "software": "swe"}


def gold_support_and_levels(instance_path: str) -> Tuple[Dict[str, bool], Dict[str, int]]:
    inst = load_instance(instance_path)
    raw = json.load(open(instance_path, encoding="utf-8"))
    gv = {a["id"]: a.get("gold_value") for a in raw["atoms"]}
    supported = {
        cid: any(all(gv.get(x) == "T" for x in route) for route in routes)
        for cid, routes in inst.gold_licenses.items()
    }
    return supported, {c.id: c.level for c in inst.claims}


def risk_threshold(cal_pairs: List[Tuple[float, bool]], alpha: float) -> Optional[float]:
    """Smallest candidate score tau such that among calibration pairs with
    score >= tau the unsupported fraction is <= alpha; None if no threshold
    achieves the bound (=> always abstain). Candidates are the observed
    scores themselves (thresholding between observed values changes
    nothing)."""
    if not cal_pairs:
        return None
    best = None
    for tau in sorted({s for s, _y in cal_pairs}):
        sel = [(s, y) for s, y in cal_pairs if s >= tau]
        if not sel:
            continue
        bad = sum(1 for _s, y in sel if not y) / len(sel)
        if bad <= alpha and (best is None or tau < best):
            best = tau
    return best


def score_cell(episodes: List[Dict[str, Any]], labels: Dict[str, Dict[str, bool]],
               levels: Dict[str, Dict[str, int]], alpha: float) -> Dict[str, Any]:
    """LOO cross-conformal over one (model, domain) cell."""
    usable = [e for e in episodes if e.get("claim_confidences")]
    rows = []
    for i, e in enumerate(usable):
        iid = e["instance"]
        cal_pairs = [
            (float(s), bool(labels[o["instance"]].get(c, False)))
            for j, o in enumerate(usable) if j != i
            for c, s in o["claim_confidences"].items()
        ]
        tau = risk_threshold(cal_pairs, alpha)
        lv = levels[iid]
        sup = labels[iid]
        emitted = None
        if tau is not None:
            cands = [c for c, s in e["claim_confidences"].items() if float(s) >= tau]
            if cands:
                emitted = max(cands, key=lambda c: (lv.get(c, 0), c))
        ceiling = max((lv[c] for c, ok in sup.items() if ok), default=0)
        ofar = int(emitted is not None and not sup.get(emitted, False))
        util = (lv.get(emitted, 0) / ceiling) if (emitted and sup.get(emitted) and ceiling) else 0.0
        rows.append({"instance": iid, "tau": tau, "emitted": emitted,
                     "episode_ofar": ofar, "utility": util,
                     "abstained": int(emitted is None), "n_cal": len(cal_pairs)})
    n = len(rows)
    n_cal_min = min((r["n_cal"] for r in rows), default=0)
    return {
        "alpha": alpha,
        "n_episodes_scored": n,
        "n_with_confidences": len(usable),
        "n_episodes_total": len(episodes),
        "episode_ofar": (sum(r["episode_ofar"] for r in rows) / n) if n else None,
        "utility": (sum(r["utility"] for r in rows) / n) if n else None,
        "abstained": (sum(r["abstained"] for r in rows) / n) if n else None,
        "min_n_cal": n_cal_min,
        "alpha_floor": (1.0 / (n_cal_min + 1)) if n_cal_min else None,
        "alpha_below_floor": bool(n_cal_min and alpha < 1.0 / (n_cal_min + 1)),
        "episodes": rows,
    }


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["qwen2.5-32b"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2])
    args = ap.parse_args(argv)

    out: Dict[str, Any] = {}
    for model in args.models:
        for dom, sub in DOMAIN_SUB.items():
            path = os.path.join(RESULTS, "%s-%s" % (model, dom), "conformal__gold", "summary.json")
            if not os.path.exists(path):
                continue
            episodes = json.load(open(path, encoding="utf-8"))["episodes"]
            labels, levels = {}, {}
            for p in sorted(glob.glob(os.path.join(PILOT, "data", "pilot", "instances", sub, "*.json"))):
                iid = os.path.splitext(os.path.basename(p))[0]
                labels[iid], levels[iid] = gold_support_and_levels(p)
            for alpha in args.alphas:
                key = "%s|%s|a%.2f" % (model, dom, alpha)
                out[key] = score_cell(episodes, labels, levels, alpha)
                c = out[key]
                print("%-40s OFAR=%s util=%s abst=%s n=%s floor=%s%s" % (
                    key,
                    "-" if c["episode_ofar"] is None else "%.3f" % c["episode_ofar"],
                    "-" if c["utility"] is None else "%.3f" % c["utility"],
                    "-" if c["abstained"] is None else "%.3f" % c["abstained"],
                    c["n_episodes_scored"],
                    "-" if c["alpha_floor"] is None else "%.3f" % c["alpha_floor"],
                    "  [ALPHA BELOW FLOOR]" if c["alpha_below_floor"] else ""),
                    file=sys.stderr)
    dst = os.path.join(RESULTS, "tableC_conformal.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote %s" % dst, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
