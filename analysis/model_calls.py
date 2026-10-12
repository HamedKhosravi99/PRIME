#!/usr/bin/env python3
"""Model calls per episode for the PRIME runs and the fully informed agent,
counted from the logged trace and summary entry.

The external systems record every model request as it is made
(``ext.llm_calls``, runner/extsys.py). The PRIME runs and the fully informed
agent did not, but their agent loop (``run_arm`` in runner/arms.py) makes
requests in a fixed pattern that the trace and the summary entry determine:

  * one request per turn that returned a tool call; each such turn logs an
    action (the runtime's own completion calls log an action too, but with a
    "discharge" record at the same step, and are not model calls);
  * one request per reminder after a silent stop (``guided_stalls``);
  * one request for the turn that ended the loop by stopping silently: the
    loop ends that way unless its last turn was an accepted emit_claim (an
    emission logged at that action's step) or the budget was spent
    (``tool_calls`` minus ``discharged_calls`` reached ``max_tool_calls``);
  * one request for the forced final question (``_force_claim_selection``),
    asked when nothing was stated: for the fully informed agent whenever the
    loop did not end on an accepted emit_claim; for the masked arms when the
    trace holds no emission at all (the runtime states the maximal authorized
    conclusions first, and when nothing is authorized the forced answer
    cannot be stated either).

Completion (prime_complete's scripted discharge) adds no model calls.
prime_runtime/tests/test_model_calls.py checks this count against a counting
model on the real agent loop.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

MASKED = ("prime", "prime_nonminimal", "prime_guided", "prime_guided_gated", "masked_reactive",
          "masked_reactive_v", "prime_complete", "prime_flat")


def model_calls(entry: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> int:
    """Model requests of one episode of ``run_arm`` (summary entry, trace records)."""
    records = list(records)
    discharge_steps = {r.get("step") for r in records if r.get("type") == "discharge"}
    actions = [r for r in records if r.get("type") == "action" and r.get("step") not in discharge_steps]
    emission_steps = {r.get("step") for r in records if r.get("type") == "emission"}
    n = len(actions) + int(entry.get("guided_stalls") or 0)
    last = actions[-1] if actions else None
    if last is not None and last.get("name") == "emit_claim" and last.get("step") in emission_steps:
        return n
    loop_charged = int(entry.get("tool_calls") or 0) - int(entry.get("discharged_calls") or 0)
    if loop_charged < int(entry["max_tool_calls"]):
        n += 1  # the turn that stopped without a tool call
    if entry.get("arm") not in MASKED or not emission_steps:
        n += 1  # the forced final question
    return n


def trace_records(path: str):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
