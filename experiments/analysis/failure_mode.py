#!/usr/bin/env python3
"""How each arm fails, not just how often.

OFAR says an arm authorized something unsupported. It does not say whether
the arm over-reached for a strong claim it could not license, or asserted a
weak one on an instance where nothing at all was supported and abstention
was the correct behaviour. Those are different defects and the ladder's
information rungs exhibit different ones, so the paper should not report
them under a single rate.

Two decompositions, both judge-free and derived from the same gold licences
the endpoints use:

  by level     emissions and unsupported emissions at each claim level, so
               over-reach (unsupported mass concentrated at the strongest
               level) is visible as such
  on dead      emissions on instances whose support ceiling is 0, where no
               claim is licensed and abstention is the only correct action
  knew         emissions the arm's OWN attested ledger already contradicted:
               every route to the emitted claim contains an atom the agent
               had itself established FALSE. This is the sharpest form of
               the gap -- not missing information, but ignoring information
               the agent produced and was shown

  lost         tool calls that returned "field not found": the arm could not
               LOCATE the evidence, which is a different failure from judging
               it wrongly. Record schemas differ sharply in guessability --
               OSV exposes flat paths like `published`, ClinicalTrials.gov
               nests `protocolSection.identificationModule.briefTitle` -- so
               an arm without the specification view can be crippled in one
               domain and fine in another for reasons that have nothing to do
               with support logic. Reported so the ladder's first rung is not
               credited with fixing judgement when it fixed addressing.

This is the Rubric-to-Work Gap made quantitative: an agent that is told the
criterion, or handed the vocabulary, asserts it rather than performing it.

Usage: python3 experiments/analysis/failure_mode.py [--model qwen2.5-32b] [--json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402


def _routes(episode, claim):
    """Gold routes for a claim, read from the instance the episode names."""
    p = episode.get("instance_path") or ""
    if not os.path.exists(p):
        hits = [q for q in _INSTANCE_INDEX.get(os.path.basename(p), [])]
        p = hits[0] if hits else ""
    if not p or not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))["gold_licenses"].get(claim)
    except (OSError, ValueError, KeyError):
        return None


_INSTANCE_INDEX = {}
for _p in __import__("glob").glob(os.path.join(
        PILOT, "data", "suffice", "*", "*", "*.json")):
    _INSTANCE_INDEX.setdefault(os.path.basename(_p), []).append(_p)


def _lost_calls(model: str, arm: str, dom: str):
    """(failed-lookup calls, total calls) from the episode traces."""
    d = os.path.join(PILOT, "results", "w2a", "%s-%s" % (model, dom),
                     "%s__B20" % arm)
    bad = tot = 0
    for t in __import__("glob").glob(os.path.join(d, "*.jsonl")):
        try:
            for line in open(t, encoding="utf-8"):
                rec = json.loads(line)
                if rec.get("type") != "observation":
                    continue
                tot += 1
                if isinstance(rec.get("data"), dict) and rec["data"].get("error"):
                    bad += 1
        except (OSError, ValueError):
            continue
    return bad, tot


def profile(model: str, arm: str, splits=("heldout_balanced",)) -> dict:
    by_level = collections.Counter()
    bad_level = collections.Counter()
    dead_n = dead_emit = 0
    knew, examples = 0, []
    scored = 0
    for dom in bw.DOMAINS:
        for e in bw.load(model, dom, "%s__B20" % arm):
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] not in splits or e.get("error"):
                continue
            scored += 1
            emitted = [x["claim"] for x in (e.get("emitted") or [])]
            if f["ceiling"] == 0:
                dead_n += 1
                if emitted:
                    dead_emit += 1
            ledger = e.get("final_ledger") or {}
            for c in emitted:
                lv = f["levels"].get(c, 0)
                by_level[lv] += 1
                if not f["supp_a"].get(c):
                    bad_level[lv] += 1
                    routes = f.get("routes", {}).get(c) or _routes(e, c)
                    if routes and all(
                            any(ledger.get(a) == "F" for a in r) for r in routes):
                        knew += 1
                        if len(examples) < 3:
                            examples.append({"instance": f["id"], "claim": c,
                                             "routes": routes,
                                             "false_atoms": sorted(
                                                 {a for r in routes for a in r
                                                  if ledger.get(a) == "F"})})
    lost_bad = lost_tot = 0
    by_dom_lost = {}
    for dom in bw.DOMAINS:
        b, t = _lost_calls(model, arm, dom)
        lost_bad += b
        lost_tot += t
        if t:
            by_dom_lost[dom] = round(100.0 * b / t, 1)
    if not scored:
        return {"status": "NOT RUN"}
    top = max(by_level) if by_level else 0
    return {
        "n_scored": scored,
        "emissions": sum(by_level.values()),
        "by_level": {str(k): by_level[k] for k in sorted(by_level)},
        "unsupported_by_level": {str(k): bad_level.get(k, 0) for k in sorted(by_level)},
        "overreach_share": (round(bad_level.get(top, 0) / by_level[top], 3)
                            if by_level.get(top) else None),
        "top_level": top,
        "n_dead": dead_n,
        "emitted_on_dead": dead_emit,
        "dead_emit_rate": round(dead_emit / dead_n, 3) if dead_n else None,
        "lost_call_rate": round(lost_bad / lost_tot, 3) if lost_tot else None,
        "lost_by_domain": by_dom_lost,
        "emitted_against_own_ledger": knew,
        "knew_examples": examples,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = {}
    for arm in bw.LADDER:
        p = profile(a.model, arm)
        if p.get("status") != "NOT RUN":
            out[arm] = p
    if a.json:
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    if not out:
        print("no runs for %s" % a.model)
        return 2
    print("Failure-mode decomposition (%s, held-out balanced)" % a.model)
    print("%-26s %7s %9s %13s %13s %s" %
          ("arm", "emits", "unsup@top", "on-dead", "vs-ledger", "lost-calls"))
    for arm, p in out.items():
        print("%-26s %7d %8s%% %8s (%d/%d) %10d  %s" %
              (arm, p["emissions"],
               "-" if p["overreach_share"] is None
               else "%.0f" % (100 * p["overreach_share"]),
               "-" if p["dead_emit_rate"] is None
               else "%.0f%%" % (100 * p["dead_emit_rate"]),
               p["emitted_on_dead"], p["n_dead"],
               p["emitted_against_own_ledger"],
               "-" if p["lost_call_rate"] is None
               else "%.0f%% %s" % (100 * p["lost_call_rate"],
                                   p["lost_by_domain"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
