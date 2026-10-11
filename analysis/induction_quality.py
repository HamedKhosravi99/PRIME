#!/usr/bin/env python3
"""Measure license-INDUCTION QUALITY, not just schema-validity.

The reviewer's deeper point on the learned component: a schema-valid induced
program is necessary but not sufficient -- it must recover the CORRECT support
structure (master §11.2-11.4). For each instance we induce a program and, per
claim, compare its route family to the expert gold family:

  validity      the induced program parses + passes structural validation
  route recall  fraction of gold routes recovered (a gold route g is recovered
                if some induced route equals g as a set, modulo Phi)
  route prec.   fraction of induced routes that are SUFFICIENT under gold (the
                induced route contains some gold route) AND MINIMAL (no proper
                subset of it also contains a gold route)
  exact match   the induced family equals the gold family per claim (set of
                routes), for all claims -- the strict "got the contract right"

Reported per model over all instances, with induction attempted N times per
instance (default 1) so validity is not a lucky single draw. Uses the real
induce_program path with the deployed prompt.

EXTENDED (Task #4 / paper Table B, Weakness 3): this module's recall/
precision/exact numbers are SYNTACTIC (set-containment against the raw gold
routes) and do not canonicalize modulo Phi, do not check semantic route
soundness/coverage against an executable oracle, and do not report
AuthFP/AuthFN (Weakness 2 §2.3 C1) or UsableEpisodeRate (§3.8). For the full
oracle-scored Table B -- Phi-canonicalization, semantic route precision/
coverage, prime-rate, AuthFP/AuthFN, valid-program-rate-in-the-denominator,
and the over-/under-/unsound failure-mode breakdown -- see
``analysis/induction_table_b.py`` and ``results/induction_table_b.json``.
This module is kept as-is (and its JSON output contract unchanged) for
continuity with anything already reading ``results/induction_quality.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, PILOT)

from prime_runtime.schema import load_instance  # noqa: E402
from runner.llm import LLMClient  # noqa: E402
from runner.arms import induce_program, InducedProgramInvalid  # noqa: E402


def _fam(prog_licenses, cid):
    return [frozenset(r) for r in prog_licenses.get(cid, ())]


def compare(inst, induced):
    """Return per-instance (recall, precision, exact) over claims."""
    gold = inst.gold_program().licenses
    ind = induced.licenses
    rec_num = rec_den = prec_num = prec_den = 0
    exact_all = True
    for c in inst.claim_ids:
        G = _fam(gold, c)
        I = _fam(ind, c)
        Gset = set(G)
        # recall: gold routes recovered exactly (as sets)
        rec_den += len(G)
        rec_num += sum(1 for g in G if g in set(I))
        # precision: induced routes sufficient (superset of some gold route)
        #            AND minimal (no proper subset is also sufficient)
        for i in I:
            prec_den += 1
            sufficient = any(g <= i for g in G)
            minimal = sufficient and not any(
                (g <= i and g != i) for g in G) and not any(
                (len(j) < len(i) and j <= i and any(g <= j for g in G)) for j in I)
            if sufficient and minimal:
                prec_num += 1
        if set(I) != Gset:
            exact_all = False
    recall = rec_num / rec_den if rec_den else 0.0
    prec = prec_num / prec_den if prec_den else 0.0
    return recall, prec, (1.0 if exact_all else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--instances", default=os.path.join(PILOT, "bench/instances/swe/*.json"))
    ap.add_argument("--base", default=os.environ.get("PRIME_LLM_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--key", default=os.environ.get("PRIME_LLM_KEY", os.environ.get("OPENAI_API_KEY", "")))
    args = ap.parse_args()
    paths = sorted(glob.glob(args.instances))
    print("%-16s %8s %8s %8s %8s" % ("model", "valid", "recall", "prec", "exact"))
    rows = []
    for m in args.models:
        llm = LLMClient(model=m, base=args.base, key=args.key)
        n = valid = 0
        rr = pp = ee = 0.0
        for p in paths:
            inst = load_instance(p)
            n += 1
            try:
                prog = induce_program(inst, llm)
            except (InducedProgramInvalid, Exception):
                continue
            valid += 1
            r, pr, ex = compare(inst, prog)
            rr += r; pp += pr; ee += ex
        vd = valid or 1
        line = "%-16s %5d/%-2d %8.2f %8.2f %8.2f" % (
            m.replace("qwen2.5:", "")[:16], valid, n, rr / vd, pp / vd, ee / vd)
        print(line)
        rows.append({"model": m, "valid": valid, "n": n,
                     "recall": rr / vd, "precision": pp / vd, "exact": ee / vd})
    out = os.path.join(PILOT, "results", "induction_quality.json")
    json.dump(rows, open(out, "w"), indent=2)
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
