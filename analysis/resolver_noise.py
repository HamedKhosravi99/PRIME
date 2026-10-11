#!/usr/bin/env python3
"""Resolver error and the bound of the decomposition theorem, by replay.

The decomposition bounds the user-visible error of an enforced agent by the
program-error term plus the resolver-error term. With the benchmark's own
standard the program term is zero, so the bound says: false authorization can
only come from resolvers that return wrong values, and it is at most the share
of episodes in which some settled atom was resolved wrongly.

This script tests that on the logged episodes of the two enforced arms. It
takes each episode's final ledger (the atoms the agent's reads settled), flips
every settled value independently with probability eps while holding the
reads fixed, applies the authored program to the corrupted ledger, takes the
strongest authorized conclusion (the worst case among the outputs the runtime
would allow), and scores it against the true support. It reports, per eps,
the false-authorization rate and the share of episodes with at least one
flipped settled atom (the resolver-error term), averaged over repetitions.

Writes results/resolver_noise.json.
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, HERE, os.path.join(PILOT, "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import obligationbench as ob  # noqa: E402

MODEL = "qwen2.5-32b"
DOMAINS = ("security", "clinical", "drugsafety", "financial")
ARMS = ("masked_reactive_v", "prime_guided")
EPS = (0.0, 0.01, 0.05, 0.10, 0.20)
REPS = 20


def episodes(arm):
    tasks = {}
    for dom in DOMAINS:
        tasks.update(ob.load_tasks(os.path.join(PILOT, "bench", "instances_v2", dom, "heldout_balanced")))
    out = []
    for dom in DOMAINS:
        run = os.path.join(PILOT, "results", "w2a", "%s-%s" % (MODEL, dom), "%s__B20" % arm)
        with open(os.path.join(run, "summary.json"), encoding="utf-8") as fh:
            for e in json.load(fh)["episodes"]:
                if e.get("error") or e["instance"] not in tasks:
                    continue
                ledger = {a: v for a, v in (e.get("final_ledger") or {}).items() if v in ("T", "F")}
                out.append((tasks[e["instance"]], ledger))
    return out


def strongest_authorized(task, ledger):
    levels = {c["id"]: int(c["level"]) for c in task["claims"]}
    est = ob.established(task, ledger)
    if not est:
        return None
    return max(est, key=lambda c: levels[c])


def run_arm(arm, rng):
    eps_results = {}
    ledgers = episodes(arm)
    for eps in EPS:
        fa = res = 0.0
        for _ in range(REPS):
            for task, led in ledgers:
                supp = ob.supported(task)
                corrupted = {}
                flipped = False
                for a, v in led.items():
                    if rng.random() < eps:
                        corrupted[a] = "F" if v == "T" else "T"
                        flipped = True
                    else:
                        corrupted[a] = v
                c = strongest_authorized(task, corrupted)
                fa += int(c is not None and not supp.get(c, False))
                res += int(flipped)
        n = REPS * len(ledgers)
        eps_results[str(eps)] = {"false_authorization_pct": round(100.0 * fa / n, 2),
                                 "resolver_error_pct": round(100.0 * res / n, 2),
                                 "n_episodes": len(ledgers)}
    return eps_results


def main() -> int:
    rng = random.Random(20260801)
    out = {"_note": __doc__.strip().split("\n\n")[1]}
    for arm in ARMS:
        out[arm] = run_arm(arm, rng)
        print(arm)
        for eps, r in out[arm].items():
            print("  eps=%-5s FA %.2f%%  Pr(resolver error) %.2f%%  (n=%d, %d reps)" % (eps, r["false_authorization_pct"], r["resolver_error_pct"], r["n_episodes"], REPS))
    path = os.path.join(PILOT, "results", "resolver_noise.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
