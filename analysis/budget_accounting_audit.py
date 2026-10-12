#!/usr/bin/env python3
"""How the logged runs' budget accounting could have moved each reported number.

The agent loops of the logged runs charged a call against the budget unless the
previously executed call had been a cache hit; the released runner charges a
call unless the call itself repeats an executed call class (docs/KNOWN_ISSUES.md).
Runtime completion and the scripted policies charged correctly in both. This
script replays every logged trace under both rules (no model is called) and, for
every reported cell (model, split, agent, budget, pooled over domains), counts:

  n          completed episodes
  repeat     episodes with a repeated call (the only ones the rules can treat differently)
  differ     episodes whose charged count differs between the rules at some step
  cap        episodes in which the budget would have ended at a different call:
             the corrected count reaches the cap before the logged trajectory ends,
             or the logged run ended at the cap and the corrected count had not

Bound. Agents that see or use the remaining budget (PRIME's guided agent and
PRIME with completion, through guidance and the completion budget; HESP, whose
prompt shows it) can take another course in any `differ` episode. Every other
agent receives identical messages under both rules until the cap ends the
episode, so only its `cap` episodes can change. An episode changes a binary
endpoint (false authorization, premature conclusion, emission) by at most one
and supported utility by at most 100, so no rate or utility of a cell can move
by more than 100 x affected / n, where affected is `differ` for the first group
and `cap` for the second (n is the rate's denominator: the episodes, except on
the regulation study's held-out sample, where program A's rates are over all 600
registrations, of which it built tasks for 161).

Unverified episodes. Where the old rule replayed on the trace does not reproduce
the call count of the episode's summary entry (`mismatch`), the trace and the
summary come from different runs of the same episode (a rerun, or two jobs that
ran the same cell at once); such an episode cannot be replayed faithfully and is
added to `affected`, which keeps the bound hard.

Rerun episodes. An episode replaced by a rerun with the corrected runner
(analysis/merge_reruns.py marks it "rerun") was charged by the corrected rule;
the audit checks that replaying the corrected rule on its trace reproduces its
logged count (`rerun_mismatch` otherwise) and does not count it as affected.

Writes results/budget_accounting_audit.json.
Usage: python3 analysis/budget_accounting_audit.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
REPORTED = ("w2a", "w2loc", "w2text", "w2seed", "blind_verification", "fdaaa", "specbuild_e2e")
# The regulation study's end-to-end episodes ship in results/specbuild_e2e/; the
# research tree keeps them in experiments/specbuild/results/.
SOURCES = {top: os.path.join(RESULTS, top) for top in REPORTED}
if not os.path.isdir(SOURCES["specbuild_e2e"]):
    SOURCES["specbuild_e2e"] = os.path.join(ROOT, "experiments", "specbuild", "results")
# The verified condition of the text-evidence table (PRIME with a blind second judgment).
SOURCES["blind_verification"] = os.path.join(ROOT, "experiments", "blind_verification", "results")
SPLIT = {"w2a": "balanced", "w2loc": "locator", "w2text": "text", "w2seed": "balanced (sampled)",
         "blind_verification": "text (verified)",
         "fdaaa": "regulation, 319 development registrations",
         "specbuild_e2e": "regulation, held-out sample of 600"}
# Rates whose denominator is not the number of episodes: on the held-out sample,
# program A (the assistant-built program) built tasks for 161 of the 600
# registrations and abstains on the rest, and its rates are over all 600.
DENOMINATOR = {("specbuild_e2e", "program A"): 600}
SKIP_RUN_PREFIXES = ("natural__",)          # the natural split is not reported
SEES_BUDGET = ("prime_guided", "prime_complete", "ext_hesp")
DOMAIN = re.compile(r"-(security|clinical|drugsafety|financial)$")


def replay(trace_path: str, cap: int):
    """Charged counts under the logged and the corrected rule, step by step."""
    records = []
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    executed_steps = {r.get("step") for r in records if r.get("type") == "observation"}
    runtime_steps = {r.get("step") for r in records if r.get("type") == "discharge"}
    seen, flag = set(), False
    old = new = 0
    first_diff = None
    new_hit_cap_before_end = False
    actions = [r for r in records if r.get("type") == "action"]
    agent_actions = [a for a in actions if a.get("step") not in runtime_steps]
    last_agent_step = agent_actions[-1].get("step") if agent_actions else None
    repeat = False
    for a in actions:
        step = a.get("step")
        key = (a.get("name"), tuple(sorted((str(k), str(v)) for k, v in (a.get("params") or {}).items())))
        executed = step in executed_steps
        cached_now = executed and key in seen
        repeat = repeat or cached_now
        runtime = step in runtime_steps
        old += int((not cached_now) if runtime else (not flag))
        new += int(not cached_now)
        if old != new and first_diff is None:
            first_diff = step
        if not runtime and new >= cap and step != last_agent_step:
            new_hit_cap_before_end = True
        if executed:
            flag = cached_now
            seen.add(key)
    return old, new, repeat, first_diff is not None, new_hit_cap_before_end


def cell_key(rel_run: str):
    """(folder, model or program, split, run). The regulation runs used
    Qwen2.5-32B-Instruct; their second component names the program."""
    parts = rel_run.split(os.sep)
    top = parts[0]
    if top == "specbuild_e2e":
        return (top, "program " + parts[1], SPLIT[top], parts[-1])
    if top == "fdaaa":
        return (top, "assistant-built program", SPLIT[top], parts[-1])
    return (top, DOMAIN.sub("", parts[1]), SPLIT[top], parts[-1])


def main() -> int:
    cells = defaultdict(lambda: {"n": 0, "repeat": 0, "differ": 0, "cap": 0, "count_mismatch": 0,
                                 "rerun": 0, "rerun_mismatch": 0,
                                 "episodes_differ": [], "episodes_mismatch": [], "episodes_cap": []})
    for top in REPORTED:
        base = SOURCES[top]
        for dp, dn, fn in os.walk(base):
            dn.sort()   # walk order differs between filesystems; the output must not
            if "summary.json" not in fn:
                continue
            rel = os.path.join(top, os.path.relpath(dp, base))
            if os.path.basename(dp).startswith(SKIP_RUN_PREFIXES) or "/policy-" in "/" + rel:
                continue
            with open(os.path.join(dp, "summary.json"), encoding="utf-8") as fh:
                episodes = json.load(fh).get("episodes", [])
            key = cell_key(rel)
            c = cells[key]
            for e in episodes:
                trace = os.path.join(dp, str(e.get("instance")) + ".jsonl")
                if e.get("error") or not os.path.exists(trace):
                    continue
                cap = int(e.get("max_tool_calls") or 20)
                old, new, repeat, differ, early = replay(trace, cap)
                c["n"] += 1
                if e.get("rerun"):
                    c["rerun"] += 1
                    c["rerun_mismatch"] += int(new != e.get("tool_calls"))
                    continue
                c["repeat"] += repeat
                mismatch = old != e.get("tool_calls")
                c["count_mismatch"] += mismatch
                if mismatch:
                    c["episodes_mismatch"].append(os.path.join(rel, str(e.get("instance"))))
                if differ:
                    c["differ"] += 1
                    c["episodes_differ"].append(os.path.join(rel, str(e.get("instance"))))
                    logged_capped = old >= cap
                    if early or (logged_capped and new < cap):
                        c["cap"] += 1
                        c["episodes_cap"].append(os.path.join(rel, str(e.get("instance"))))
    rows = []
    for (top, model, split, run), c in sorted(cells.items()):
        if not c["n"]:
            continue
        arm = run.rsplit("__", 1)[0]
        sees = arm.startswith(SEES_BUDGET)
        base = set(c["episodes_differ"] if sees else c["episodes_cap"])
        affected = len(base | set(c["episodes_mismatch"]))
        denominator = DENOMINATOR.get((top, model), c["n"])
        rows.append({"folder": top, "model": model, "split": split, "run": run, "n": c["n"],
                     "denominator": denominator,
                     "repeat": c["repeat"], "differ": c["differ"], "cap": c["cap"],
                     "sees_budget": sees, "affected": affected,
                     "bound_points": round(100.0 * affected / denominator, 2),
                     "replay_count_mismatch": c["count_mismatch"],
                     "rerun": c["rerun"], "rerun_mismatch": c["rerun_mismatch"],
                     "episodes_differ": sorted(c["episodes_differ"]), "episodes_cap": sorted(c["episodes_cap"]),
                     "episodes_mismatch": sorted(c["episodes_mismatch"])})
    out = {"rule": __doc__.split("\n\n")[1], "cells": rows,
           "totals": {k: sum(r[k] for r in rows) for k in ("n", "repeat", "differ", "cap", "affected",
                                                            "replay_count_mismatch", "rerun", "rerun_mismatch")}}
    with open(os.path.join(RESULTS, "budget_accounting_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("%-9s %-18s %-9s %-34s %5s %6s %6s %4s %8s %8s" % ("folder", "model", "split", "run", "n", "repeat",
                                                           "differ", "cap", "bound", "mismatch"))
    for r in rows:
        if r["repeat"] or r["replay_count_mismatch"]:
            print("%-9s %-18s %-9s %-34s %5d %6d %6d %4d %7.2f%s %8d" % (
                r["folder"], r["model"], r["split"], r["run"], r["n"], r["repeat"], r["differ"], r["cap"],
                r["bound_points"], "*" if r["sees_budget"] else " ", r["replay_count_mismatch"]))
    t = out["totals"]
    print("\n%d cells, %d episodes: %d with a repeated call, %d with a different charged count, %d with a "
          "different end at the cap; in %d episodes the trace and the summary come from different runs"
          % (len(rows), t["n"], t["repeat"], t["differ"], t["cap"], t["replay_count_mismatch"]))
    print("bound: largest possible change of any rate or utility in a cell, in points (* = agent sees the budget)")
    print("max bound over cells: %.2f points" % max(r["bound_points"] for r in rows))
    print("rerun with the corrected runner: %d episodes, %d whose logged count the corrected rule does not reproduce"
          % (t["rerun"], t["rerun_mismatch"]))
    print("wrote results/budget_accounting_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
