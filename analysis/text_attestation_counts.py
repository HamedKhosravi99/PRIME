#!/usr/bin/env python3
"""Attestation calls on the text-evidence split (results/text_attestation_counts.json).

For each agent on the 240 text-evidence tasks (Qwen2.5-32B, 20 calls), counted
from the episode traces of results/w2text/ (with the rerun episodes in place):

  total      attest_atom calls the runtime executed (an observation of attest_atom)
  rejected   those it refused: the atom is not a text-evidence atom, the value is
             not T or F, the source text was not read earlier in the episode, or
             the quote is not a verbatim span of at least eight characters
             (ToolExecutor._tool_attest_atom in runner/arms.py)
  admitted   attestations written to the ledger (the attestation records that
             analysis/build_text_table.py counts as "attestations")
  wrong      admitted attestations whose value differs from the task's gold value,
             by domain, atom, and direction

total - rejected must equal admitted; any call that is neither is listed under
"unaccounted". The same counts for the logged originals of the rerun episodes
(results_superseded/, when present) trace each change the rerun made.

Usage: python3 analysis/text_attestation_counts.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
# the logged originals of rerun episodes: results_superseded/ in the research
# tree, results/superseded_by_rerun/ in the released logs
SUPERSEDED = next((p for p in (os.path.join(ROOT, "results_superseded", "results"),
                               os.path.join(RESULTS, "superseded_by_rerun", "results")) if os.path.isdir(p)),
                  os.path.join(ROOT, "results_superseded", "results"))
BENCH = os.path.join(ROOT, "bench", "instances_v2")
MODEL = "qwen2.5-32b"
DOMS = ("clinical", "security")
SPLIT = "heldout_text"
ARMS = ("react_program_attested", "masked_reactive_v", "prime_guided")
# what each text-evidence atom states (the "what" of its specification hint)
ATOM = {("clinical", "E2"): "adults only (minimum age 18 or more)",
        ("clinical", "E3"): "one sex only",
        ("clinical", "E4"): "healthy volunteers accepted",
        ("security", "E2"): "names the affected package",
        ("security", "E3"): "states a fixed version",
        ("security", "E4"): "states the first affected version"}


def gold_values(dom: str, iid: str):
    with open(os.path.join(BENCH, dom, SPLIT, iid + ".json"), encoding="utf-8") as fh:
        return {a["id"]: a["gold_value"] for a in json.load(fh)["atoms"]}


def count_trace(path: str, dom: str, iid: str) -> dict:
    gold = gold_values(dom, iid)
    c = {"total": 0, "rejected": 0, "admitted": 0, "wrong": [], "rejected_reasons": Counter()}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") == "observation" and r.get("action") == "attest_atom":
                c["total"] += 1
                d = r.get("data") or {}
                if not d.get("accepted"):
                    c["rejected"] += 1
                    reason = str(d.get("reason", ""))
                    c["rejected_reasons"]["source text not read" if reason.startswith("read the source text")
                                          else reason] += 1
            elif r.get("type") == "attestation":
                at = r.get("attestation") or r
                if at.get("tool") == "attest_atom":
                    c["admitted"] += 1
                    if at.get("value") != gold.get(at.get("atom")):
                        c["wrong"].append((at.get("atom"), at.get("value"), gold.get(at.get("atom"))))
    return c


def episodes(base: str, arm: str):
    for dom in DOMS:
        run = os.path.join(base, "w2text", "%s-%s" % (MODEL, dom), "%s__B20" % arm)
        sp = os.path.join(run, "summary.json")
        if not os.path.exists(sp):
            continue
        with open(sp, encoding="utf-8") as fh:
            for e in json.load(fh)["episodes"]:
                if e.get("error"):
                    continue
                tp = os.path.join(run, e["instance"] + ".jsonl")
                if os.path.exists(tp):
                    yield dom, e, tp


def main() -> int:
    out = {"model": MODEL, "split": SPLIT, "budget": 20, "arms": {}, "rerun_episodes": {}}
    for arm in ARMS:
        tot = Counter()
        reasons = Counter()
        wrong = Counter()
        unaccounted = []
        n = 0
        for dom, e, tp in episodes(RESULTS, arm):
            n += 1
            c = count_trace(tp, dom, e["instance"])
            for k in ("total", "rejected", "admitted"):
                tot[k] += c[k]
            reasons.update(c["rejected_reasons"])
            for atom, val, g in c["wrong"]:
                wrong["%s %s (%s): attested %s, gold %s" % (dom, atom, ATOM.get((dom, atom), "?"), val, g)] += 1
            if c["total"] - c["rejected"] != c["admitted"]:
                unaccounted.append({"instance": e["instance"], "total": c["total"], "rejected": c["rejected"],
                                    "admitted": c["admitted"]})
            if e.get("rerun"):
                old_tp = os.path.join(SUPERSEDED, os.path.relpath(tp, RESULTS))
                old = count_trace(old_tp, dom, e["instance"]) if os.path.exists(old_tp) else None
                out["rerun_episodes"]["%s %s" % (arm, e["instance"])] = {
                    "new": {k: c[k] for k in ("total", "rejected", "admitted")} | {"wrong": len(c["wrong"])},
                    "old": ({k: old[k] for k in ("total", "rejected", "admitted")} | {"wrong": len(old["wrong"])})
                    if old else None}
        out["arms"][arm] = {"episodes": n, "total": tot["total"], "rejected": tot["rejected"],
                            "admitted": tot["admitted"], "wrong_admitted": sum(wrong.values()),
                            "wrong_by_atom": dict(sorted(wrong.items())),
                            "rejected_reasons": dict(sorted(reasons.items())),
                            "unaccounted": unaccounted}
    path = os.path.join(RESULTS, "text_attestation_counts.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for arm, a in out["arms"].items():
        print("%-24s n %d  total %d  rejected %d  admitted %d  wrong %d  unaccounted %d"
              % (arm, a["episodes"], a["total"], a["rejected"], a["admitted"], a["wrong_admitted"], len(a["unaccounted"])))
        for k, v in a["wrong_by_atom"].items():
            print("    %3d  %s" % (v, k))
    for k, v in out["rerun_episodes"].items():
        print("rerun %-36s old %s  new %s" % (k, v["old"], v["new"]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
