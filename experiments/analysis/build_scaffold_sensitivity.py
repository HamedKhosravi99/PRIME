#!/usr/bin/env python3
"""Aggregate the scaffold-variant runs into a sensitivity band.

Reads results/<model>-<domain>/<arm>_v{0,1,2}__gold/summary.json (produced by
the sbatch `compare` configset; variant 0 is byte-identical to the published
scaffold, pinned by test_conformal_and_scaffold.py) and reports, per
(model, domain, arm), the across-variant range of episode OFAR and supported
utility -- scored exactly like every other arm, against verified gold values.

The point: adapter/prompt choices of ours could confound a mechanism
comparison. This measures how much those choices move the numbers.

Writes results/scaffold_sensitivity.json.
Usage: python3 experiments/analysis/build_scaffold_sensitivity.py --models qwen2.5-32b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_conformal import gold_support_and_levels  # noqa: E402

RESULTS = os.path.join(PILOT, "results")
DOMAIN_SUB = {"research": "res", "security": "osv", "drugsafety": "fda",
              "financial": "fin", "clinical": "ctgov", "software": "swe"}
ARMS = ("rubric", "reflexion", "cove")


def score_summary(path: str, labels, levels) -> Dict[str, float]:
    eps = json.load(open(path, encoding="utf-8"))["episodes"]
    n = 0
    ofar = 0
    util = 0.0
    for e in eps:
        if e.get("error"):
            continue
        n += 1
        iid = e["instance"]
        sup, lv = labels[iid], levels[iid]
        ceiling = max((lv[c] for c, ok in sup.items() if ok), default=0)
        emitted = [x["claim"] for x in (e.get("emitted") or [])]
        bad = any(not sup.get(c, False) for c in emitted)
        good_lv = max((lv.get(c, 0) for c in emitted if sup.get(c, False)), default=0)
        ofar += int(bad)
        util += (good_lv / ceiling) if ceiling else 0.0
    return {"n": n, "episode_ofar": (ofar / n) if n else None,
            "utility": (util / n) if n else None}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["qwen2.5-32b"])
    args = ap.parse_args(argv)

    out: Dict[str, Any] = {}
    for model in args.models:
        for dom, sub in DOMAIN_SUB.items():
            labels, levels = {}, {}
            for p in sorted(glob.glob(os.path.join(PILOT, "data", "pilot", "instances", sub, "*.json"))):
                iid = os.path.splitext(os.path.basename(p))[0]
                labels[iid], levels[iid] = gold_support_and_levels(p)
            for arm in ARMS:
                per_variant = {}
                for v in (0, 1, 2, 3):  # v3 = verbatim released Reflexion prompt (reflexion only)
                    path = os.path.join(RESULTS, "%s-%s" % (model, dom),
                                        "%s_v%d__gold" % (arm, v), "summary.json")
                    if os.path.exists(path):
                        per_variant["v%d" % v] = score_summary(path, labels, levels)
                if not per_variant:
                    continue
                ofars = [x["episode_ofar"] for x in per_variant.values()
                         if x["episode_ofar"] is not None]
                utils = [x["utility"] for x in per_variant.values()
                         if x["utility"] is not None]
                out["%s|%s|%s" % (model, dom, arm)] = {
                    "variants": per_variant,
                    "n_variants": len(per_variant),
                    "ofar_range": [min(ofars), max(ofars)] if ofars else None,
                    "utility_range": [min(utils), max(utils)] if utils else None,
                }
    dst = os.path.join(RESULTS, "scaffold_sensitivity.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    for k, v in sorted(out.items()):
        print("%-38s variants=%d OFAR range=%s util range=%s"
              % (k, v["n_variants"], v["ofar_range"], v["utility_range"]),
              file=sys.stderr)
    print("wrote %s" % dst, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
