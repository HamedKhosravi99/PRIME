#!/usr/bin/env python3
"""LaTeX rows for the comparator table with both views (results/tableEXT.json,
results/tableEXT_spec.json): each mechanism with the prompt its authors
proposed and with the specification view added, plus the enforced arms.

Usage: python3 analysis/ext_table_tex.py
"""
from __future__ import annotations

import json
import os

PILOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(PILOT, "results")
NAMES = [("static_contract", "Checklist"), ("conformal", "Conformal"), ("cove", "CoVe"),
         ("reflexion", "Reflexion"), ("rubric", "Rubric"), ("direct", "Direct")]


def main() -> int:
    own = json.load(open(os.path.join(R, "tableEXT.json")))["cells"]
    spec = json.load(open(os.path.join(R, "tableEXT_spec.json")))
    t1 = json.load(open(os.path.join(R, "tableT1.json")))
    rows = []
    for arm, label in NAMES:
        a, b = own[arm], spec["cells"][arm]
        if a.get("status") or b.get("status"):
            rows.append("%s & -- & -- & -- & -- & -- & -- \\\\" % label)
            continue
        rows.append("%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\" % (
            label, a["ofar_a"], a["utility"], a["emit_rate_avail"], b["ofar_a"], b["utility"], b["emit_rate_avail"]))
    print("\n".join(rows))
    print("== spec-view contrasts vs prime_guided (positive favours PRIME)")
    for k, v in sorted(spec.get("contrasts", {}).items()):
        if v.get("status"):
            continue
        print("%-26s %+6.1f [%+.1f, %+.1f] p=%.4f (%d/%d)" % (k, v["delta"], v["ci"][0], v["ci"][1], v["p_randomization"], v["discordant_clusters"], v["n_clusters"]))
    print("== spec-view gate-only contrasts (vs masked_reactive_v)")
    for k, v in sorted(spec.get("gate_only", {}).items()):
        if v.get("status"):
            continue
        print("%-26s %+6.1f [%+.1f, %+.1f] p=%.4f tost=%s" % (k, v["delta"], v["ci"][0], v["ci"][1], v["p_randomization"], v.get("tost_equivalent")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
