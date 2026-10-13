#!/usr/bin/env python3
"""Reproduce the paper's per-agent endpoints from the episode logs with the
standalone SUFFICE scorer.

For every agent of the information ladder (the paper's Section 6.1) and six
further baselines that the paper does not report, on the 480 held-out balanced tasks with Qwen2.5-32B-Instruct and
20 tool calls, this script rebuilds the agent's predictions from its logged
traces (the conclusions it stated and the record fields it read), scores them
with src/suffice/__init__.py, and checks false authorization, the premature-
conclusion rate, supported utility, the emission rate, and evidence value
against the reference values: those the paper prints (Table 1 and Figure 3) for
the ladder, and the logged values for the six baselines. It exits nonzero on any
difference larger than 0.05.

The predictions it rebuilds use only the trace's actions, never an agent's own
account of its evidence, so the check also shows that the scorer needs nothing
but the stated conclusions and the reads.

Every agent has a logged episode for each of the 480 tasks. An episode whose
model calls still timed out after the client's retries carries no measurement
and is left out of every rate, so an agent's denominator is its completed
episodes; ``EXCLUDED`` lists those episodes (24 of the 7,200 checked here), and
the check fails if any other episode is absent or failed.

Usage
  python3 experiments/reproduce_tables.py                       # check against the paper
  python3 experiments/reproduce_tables.py --write-predictions reference_predictions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the repository root
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import suffice as ob  # noqa: E402

DOMAINS = ("security", "clinical", "drugsafety", "financial")
MODEL = "qwen2.5-32b"
BUDGET = "B20"

# Agent id in the logs -> (name in the paper, results folder,
#   expected: false authorization %, premature %, utility, emission %, evidence value;
#   None where the paper prints no value)
AGENTS = {
    "react": ("ReAct", "w2a", (13.0, 56.9, 17.9, 59.0, None)),
    "react_verify": ("ReAct +verify", "w2a", (19.9, 47.6, 10.0, 49.6, None)),
    "react_spec": ("ReAct +spec", "w2a", (21.9, 37.9, 28.3, 92.2, None)),
    "react_spec_verify": ("ReAct +spec+verify", "w2a", (22.3, 33.8, 26.9, 89.9, None)),
    "react_program": ("ReAct +program", "w2a", (22.9, 30.2, 31.9, 99.7, None)),
    "react_program_verify": ("ReAct +program+verify", "w2a", (22.3, 25.2, 32.7, 98.5, None)),
    "react_program_attested": ("Fully informed, unenforced", "w2a", (26.7, 28.3, 38.0, 99.0, 43.5)),
    "masked_reactive_v": ("PRIME without guidance", "w2a", (0.0, 0.0, 30.1, 99.0, 30.1)),
    "prime_guided": ("PRIME", "w2a", (0.0, 0.0, 45.5, 100.0, 45.5)),
    "static_contract": ("Static checklist", "w2ext", (0.4, 0.4, 2.5, 9.6, 3.1)),
    "conformal": ("Conformal (uncalibrated)", "w2ext", (13.3, 55.6, 17.5, 59.1, 2.4)),
    "cove": ("CoVe", "w2ext", (10.1, 32.1, 11.3, 35.0, 2.7)),
    "reflexion": ("Reflexion", "w2ext", (9.7, 38.2, 13.4, 41.5, 2.9)),
    "rubric": ("Rubric", "w2ext", (24.6, 58.8, 17.6, 61.6, 1.4)),
    "direct": ("Direct", "w2ext", (42.1, 92.9, 30.3, 92.9, 0.0)),
}
KEYS = ("false_authorization_pct", "premature_pct", "utility", "emission_rate_pct", "evidence_value")

# The episodes left out of the rates, per agent: each ended because the model
# server did not answer within the client's retries ("LLM request failed after
# retries: APITimeoutError"). experiments/analysis/episode_accounting.py lists such episodes
# for every run.
EXCLUSION_RULE = ("the episode's model calls timed out after the client's retries; "
                  "it carries no measurement and is left out of every rate")
EXCLUDED = {
    "react": ("dru-v2-0123", "dru-v2-0147", "sec-v2-0170", "sec-v2-0178"),
    "react_verify": ("dru-v2-0080", "dru-v2-0164", "dru-v2-0178"),
    "react_spec_verify": ("cli-v2-0119",),
    "static_contract": ("sec-v2-0194",),
    "conformal": ("cli-v2-0148", "cli-v2-0193", "dru-v2-0123", "sec-v2-0027", "sec-v2-0058"),
    "cove": ("dru-v2-0084", "dru-v2-0147", "fin-v2-0222", "sec-v2-0088"),
    "reflexion": ("cli-v2-0148", "cli-v2-0185", "cli-v2-0193", "dru-v2-0084", "dru-v2-0170", "sec-v2-0164"),
}
TIMEOUT = "APITimeoutError"


def heldout_tasks() -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    for dom in DOMAINS:
        tasks.update(ob.load_tasks(os.path.join(ROOT, "data", "suffice", dom, "heldout_balanced")))
    return tasks


def predictions(agent: str, folder: str, tasks: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], set, Dict[str, str]]:
    """One prediction per completed episode: the stated conclusions and the
    inspect_record reads, in order, from the agent's trace. Also returns the
    task ids with a logged episode and the failed episodes' errors."""
    out = []
    logged: set = set()
    failed: Dict[str, str] = {}
    for dom in DOMAINS:
        run = os.path.join(ROOT, "results", folder, "%s-%s" % (MODEL, dom), "%s__%s" % (agent, BUDGET))
        with open(os.path.join(run, "summary.json"), encoding="utf-8") as fh:
            episodes = json.load(fh)["episodes"]
        for e in episodes:
            if e.get("instance") not in tasks:
                continue
            logged.add(e["instance"])
            if e.get("error"):
                failed[e["instance"]] = str(e["error"])
                continue  # a failed episode carries no measurement
            reads = []
            with open(os.path.join(run, e["instance"] + ".jsonl"), encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("type") == "action" and r.get("name") == "inspect_record":
                        p = r.get("params") or {}
                        reads.append([str(p.get("record_id", "")), str(p.get("field_path", ""))])
            out.append({"task_id": e["instance"], "agent": agent, "model": MODEL,
                        "budget": BUDGET, "stated": [x["claim"] for x in e.get("emitted", [])],
                        "reads": reads, "calls": e.get("tool_calls", 0)})
    return out, logged, failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-predictions", default=None,
                    help="also write each agent's predictions as <dir>/%s/<agent>.jsonl" % MODEL)
    a = ap.parse_args(argv)
    tasks = heldout_tasks()
    print("%d held-out balanced tasks; %s, %s tool calls" % (len(tasks), MODEL, BUDGET[1:]))
    head = "%-28s %5s %5s %5s %8s %8s %8s %8s %8s" % ("agent", "tasks", "excl.", "n", "false", "prem.",
                                                       "utility", "emit", "evid.")
    print(head)
    bad = []
    for agent, (name, folder, expected) in AGENTS.items():
        preds, logged, failed = predictions(agent, folder, tasks)
        absent = sorted(set(tasks) - logged)
        if absent:
            bad.append("%s: no logged episode for %d tasks, e.g. %s" % (name, len(absent), absent[:3]))
        declared = set(EXCLUDED.get(agent, ()))
        if set(failed) != declared:
            bad.append("%s: failed episodes %s, declared %s" % (name, sorted(failed), sorted(declared)))
        not_timeout = sorted(t for t, err in failed.items() if TIMEOUT not in err)
        if not_timeout:
            bad.append("%s: episodes failed for another reason: %s" % (name, not_timeout))
        scored = ob.score({p["task_id"]: tasks[p["task_id"]] for p in preds}, preds)["overall"]
        got = tuple(round(scored[k], 1) for k in KEYS)
        print("%-28s %5d %5d %5d %8.1f %8.1f %8.1f %8.1f %8.1f" % ((name, len(tasks), len(failed), scored["n"]) + got))
        for k, g, x in zip(KEYS, got, expected):
            if x is not None and abs(g - x) > 0.05:
                bad.append("%s %s: got %.1f, expected %.1f" % (name, k, g, x))
        if a.write_predictions:
            d = os.path.join(a.write_predictions, MODEL)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, agent + ".jsonl"), "w", encoding="utf-8") as fh:
                for p in preds:
                    fh.write(json.dumps(p, sort_keys=True) + "\n")
    if a.write_predictions:
        with open(os.path.join(a.write_predictions, MODEL, "EXCLUDED.json"), "w", encoding="utf-8") as fh:
            json.dump({"rule": EXCLUSION_RULE, "tasks_per_agent": len(tasks),
                       "excluded": {k: list(v) for k, v in EXCLUDED.items()}}, fh, indent=1, sort_keys=True)
    if bad:
        print("MISMATCH:\n  " + "\n  ".join(bad))
        return 1
    print("all %d agents match their reference values; %d episodes excluded (model-server timeouts), "
          "no other episode missing" % (len(AGENTS), sum(len(v) for v in EXCLUDED.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
