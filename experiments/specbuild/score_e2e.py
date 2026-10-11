#!/usr/bin/env python3
"""Test phase, part 2 (PROTOCOL.md): end-to-end scoring against the tracker.
The only step that reads test labels; run after every episode.

For each program (A: the paper's, B: the frozen protocolized one) and policy
(the ceiling-targeting policy with no LLM on all 1,999 registrations; the LLM
policies on the 600-registration agent subset): unsupported-conclusion rate
(stated level above the tracker level), supported utility, coverage (any
conclusion stated), abstention, and the same by tracker status; plus each
program's full-evidence level agreement with the tracker.

Usage: python3 score_e2e.py
Writes results/e2e.json and results/e2e.md.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

ARMS = [("react_program_attested", "fully informed agent (unenforced)"), ("prime_guided", "PRIME"), ("prime_complete", "PRIME with completion")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def metrics(stated, ref, ids):
    """stated: id -> list of stated levels ([] = abstain)."""
    n = len(ids); un = sum(1 for i in ids if any(l > ref[i]["level"] for l in stated.get(i, [])))
    util = sum(100 * max([l for l in stated.get(i, []) if l <= ref[i]["level"]] + [0]) / 3 for i in ids) / n
    cov = sum(1 for i in ids if stated.get(i)) / n
    by = {}
    for st in ("reported", "reported-late", "overdue", "ongoing"):
        sub = [i for i in ids if ref[i]["status"] == st or (st == "overdue" and ref[i]["status"] == "overdue-cancelled")]
        if sub:
            by[st] = {"n": len(sub), "unsupported": 100 * sum(1 for i in sub if any(l > ref[i]["level"] for l in stated.get(i, []))) / len(sub),
                      "coverage": 100 * sum(1 for i in sub if stated.get(i)) / len(sub),
                      "utility": sum(100 * max([l for l in stated.get(i, []) if l <= ref[i]["level"]] + [0]) / 3 for i in sub) / len(sub)}
    return {"n": n, "unsupported": 100 * un / n, "unsupported_k": un, "unsupported_ci": wilson(un, n), "utility": util,
            "coverage": 100 * cov, "abstain": 100 * (1 - cov), "by_status": by}


def main() -> int:
    split = json.load(open(os.path.join(HERE, "data", "test_split.json")))
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(os.environ.get("SPECBUILD_TEST_RECORDS") or os.path.join(common.PILOT, "experiments", "fdaaa_heldout", "results", "records"), "NCT*.json")))
    ref = common.reference(ids, "test")
    ids = [i for i in ids if i in ref]; subset = [i for i in split["agent_subset"] if i in ref]
    out = {"n_test": len(ids), "n_subset": len(subset), "reference_levels": dict(collections.Counter(ref[i]["level"] for i in ids)), "programs": {}}
    md = ["# Specification construction: end-to-end evaluation on held-out registrations", "",
          "Reference: FDAAA TrialsTracker (level 3 reported late, 2 results due, 1 not yet due). Test registrations: %d; agent subset: %d." % (len(ids), len(subset)), ""]
    for tag, name in (("A", "paper's program (coding assistant, readings chosen on the development pool)"), ("B", "protocolized program (open model, validation-selected, frozen)")):
        inst = {}
        for p in glob.glob(os.path.join(HERE, "data", "test_%s" % tag, "*.json")):
            x = json.load(open(p)); inst[x["seed"]] = x
        full = {i: common.full_level(inst[i]) if i in inst else 0 for i in ids}
        fixed = {i: ([full[i]] if full[i] > 0 else []) for i in ids}
        prog = {"instances": len(inst), "level_agreement": 100 * sum(1 for i in ids if full[i] == ref[i]["level"]) / len(ids),
                "fixed_all": metrics(fixed, ref, ids), "fixed_subset": metrics(fixed, ref, subset), "arms": {}}
        for arm, aname in ARMS:
            p = os.path.join(HERE, "results", tag, arm + "__B20", "summary.json")
            if not os.path.exists(p):
                continue
            eps = {e["instance"]: e for e in json.load(open(p))["episodes"]}
            stated, errors = {}, 0
            for i in subset:
                e = eps.get(("fdaaa-%s" if tag == "A" else "specB-%s") % i)
                if e is None:
                    stated[i] = []  # no instance: the program abstains
                    continue
                if e.get("error"):
                    errors += 1
                stated[i] = [common.LEVEL[x["claim"]] for x in e.get("emitted", []) if x.get("claim") in common.LEVEL]
            m = metrics(stated, ref, subset); m["episode_errors"] = errors
            m["calls"] = sum(eps[k].get("tool_calls", 0) for k in eps) / max(1, len(eps)); prog["arms"][arm] = m
        out["programs"][tag] = prog
        md += ["## (%s) %s" % (tag, name), "", "Instances built: %d of %d; full-evidence level agreement with the tracker: %.1f%%." % (len(inst), len(ids), prog["level_agreement"]), "",
               "| policy | registrations | unsupported % [95% CI] | utility | coverage % |", "|---|---|---|---|---|"]
        rows = [("ceiling-targeting, no LLM (all)", prog["fixed_all"]), ("ceiling-targeting, no LLM (subset)", prog["fixed_subset"])] + \
               [(aname, prog["arms"][arm]) for arm, aname in ARMS if arm in prog["arms"]]
        for nm, m in rows:
            md.append("| %s | %d | %.1f [%.1f, %.1f] | %.1f | %.1f |" % (nm, m["n"], m["unsupported"], m["unsupported_ci"][0], m["unsupported_ci"][1], m["utility"], m["coverage"]))
        md += ["", "By tracker status, ceiling-targeting on all registrations (unsupported % / coverage % / utility):", ""]
        for st, b in prog["fixed_all"]["by_status"].items():
            md.append("- %s (n=%d): %.1f / %.1f / %.1f" % (st, b["n"], b["unsupported"], b["coverage"], b["utility"]))
        md.append("")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "e2e.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "e2e.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
