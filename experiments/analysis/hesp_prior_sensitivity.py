#!/usr/bin/env python3
"""HESP's controller without a model, under three priors (results/hesp_prior_sensitivity.json).

In HESP's full configuration the controller picks every probe and issues
every verdict; the model can only abstain. Running the same arm
(src/prime_agents/extsys.py, ext_hesp) with a planner that always asks for an action
therefore gives the controller's own behaviour, and repeating it under
different priors shows how much the verdicts depend on the prior:

  counted    the primary prior: uniform over feasible hypotheses, check base
             rates counted on the development split within each (HESP's
             counted development tables)
  uniform    uniform over check assignments within each hypothesis
  bernoulli  independent Bernoulli(0.5) checks

Scored with the release scorer on the 480 balanced tasks at 20 calls and at
b_min. No GPU and no model calls.

Usage: python3 experiments/analysis/hesp_prior_sensitivity.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), os.path.join(PILOT, "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import extsys  # noqa: E402
import suffice as ob  # noqa: E402
import run_batch  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

DOMAINS = ("security", "clinical", "drugsafety", "financial")


class AlwaysAct:
    """A planner that always asks for a probe (the controller picks which)."""

    def chat(self, messages, tools=None):
        m = re.search(r"^- (\S+) \| read ", messages[-1]["content"], re.M)
        return {"content": json.dumps({"kind": "action", "action_id": m.group(1) if m else "x",
                                       "reason": "probe"}), "tool_calls": [], "usage": {}}


def run(prior: str, anchor: str, tmp: str):
    tasks = {}
    for dom in DOMAINS:
        tasks.update(ob.load_tasks(os.path.join(PILOT, "data", "suffice", dom, "heldout_balanced")))
    preds = []
    for p in sorted(glob.glob(os.path.join(PILOT, "data", "suffice", "*", "heldout_balanced", "*.json"))):
        inst = load_instance(p)
        budget = 20 if anchor == "B20" else run_batch.instance_budget(p, anchor, 20)
        tp = os.path.join(tmp, "%s-%s-%s.jsonl" % (prior, anchor, inst.id))
        with TraceLogger(tp) as tr:
            ep = extsys.Episode("ext_hesp", inst, AlwaysAct(), tr, budget)
            extsys._run_hesp_with_prior(ep, prior)
            res = ep.finish()
        reads = [[r["params"]["record_id"], r["params"]["field_path"]]
                 for r in map(json.loads, open(tp)) if r.get("type") == "action" and r.get("name") == "inspect_record"]
        preds.append({"task_id": inst.id, "stated": [e["claim"] for e in res.emitted], "reads": reads})
    s = ob.score(tasks, preds)["overall"]
    return {k: round(v, 2) for k, v in s.items()}


def main() -> int:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for prior in ("counted", "uniform", "bernoulli"):
            for anchor in ("B20", "bmin"):
                out["%s|%s" % (prior, anchor)] = r = run(prior, anchor, tmp)
                print("%-10s %-5s FA %5.1f  prem %5.1f  util %5.1f  abst %5.1f  reads %.2f" % (
                    prior, anchor, r["false_authorization_pct"], r["premature_pct"], r["utility"],
                    r["abstention_pct"], r["mean_reads"]))
    path = os.path.join(PILOT, "results", "hesp_prior_sensitivity.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
