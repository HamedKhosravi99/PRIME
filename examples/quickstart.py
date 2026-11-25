#!/usr/bin/env python3
"""PRIME on the paper's running example (Figure 1), with no model and no data.

A company accepts a vulnerability report for investigation if it has a
description and either
  p1: a severity rating and an advisory reviewed by a curator, or
  p2: a published advisory reference with a URL.

The script commits these routes, records check results one at a time, and
prints after each step what PRIME authorizes and which checks it asks for.
It uses three runtime calls: make_program (commit), Ledger.apply (record a
check), and auth_ids / reachable_ceiling (what may be reported, what to check).

Run:  python3 examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from prime_runtime import auth, guidance  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import make_program  # noqa: E402

CHECKS = ["description", "severity", "curator_review", "advisory_reference", "url"]
WEAK, STRONG = "has_description", "ready_for_investigation"

# 1. Commit: the evidence routes are fixed before any check runs.
program = make_program(
    claims=[WEAK, STRONG],
    order=[(WEAK, STRONG)],  # (weaker, stronger)
    licenses={
        WEAK: [["description"]],
        STRONG: [["description", "severity", "curator_review"],      # route p1
                 ["description", "advisory_reference", "url"]],      # route p2
    },
    scope="one vulnerability report",
    atoms=CHECKS,
)
levels = {WEAK: 1, STRONG: 2}
resolvers = {c: {"tool": "inspect_record"} for c in CHECKS}
ledger = Ledger(CHECKS)
step = 0


def record(check, holds):
    """2. Record: the runtime, not the agent, writes each check result."""
    global step
    step += 1
    ledger.apply({"atom": check, "value": "T" if holds else "F", "tool": "inspect_record", "step": step})


def report(title):
    """3. Gate and guide: what may be reported now, and what to check next."""
    authorized = auth.auth_ids(program, ledger)
    g = guidance.reachable_ceiling(program, ledger, resolvers, levels, remaining_budget=10)
    print(title)
    print("   authorized conclusions :", ", ".join(authorized) or "none")
    print("   may report '%s'? %s" % (STRONG, "yes" if STRONG in authorized else "no, the gate refuses it"))
    if g["blocking_atoms"]:
        print("   PRIME targets '%s' and asks for: %s" % (g["ceiling_claim"], ", ".join(g["blocking_atoms"])))
    else:
        print("   nothing left to check")
    print()


report("Step 0. Nothing has been checked.")

record("description", True)
record("severity", True)
record("curator_review", False)
report("Step 1. The description and the severity hold, and the curator review failed. Route p1 is blocked.")

record("advisory_reference", True)
record("url", True)
report("Step 2. The advisory reference and its URL hold. Route p2 is complete.")
