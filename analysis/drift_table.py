#!/usr/bin/env python3
"""Revision-freedom -> standard-drift table (mechanism experiment 4).

Reads results/<model>/drift__<mode>/summary.json plus the fixed deployed arm
(prime__gold), and reports per revision-freedom mode: drift-event rate,
gold-ledger false-license rate (FLR), and revisions accepted -- with
seed-level cluster-bootstrap CIs. Proposition 4 predicts drift and FLR
increasing with granted freedom, both zero in the deployed fixed arm.

FLR here is protocol-based (emitted claim unauthorized on the reconstructed
gold ledger at emission) -- the agent's self-weakened program cannot move the
gold ledger, which is the whole point: drift shows up as gold-unsupported
claims the agent's OWN program was talked into permitting.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate import cluster_bootstrap  # noqa: E402

MODE_ORDER = ["fixed", "strengthen_only", "unfrozen_lattice", "arbitrary", "posthoc"]
MODE_LABEL = {
    "fixed": "Fully fixed (deployed)",
    "strengthen_only": "Strengthen-only",
    "unfrozen_lattice": "Unfrozen lattice",
    "arbitrary": "Arbitrary revision",
    "posthoc": "Fully post-hoc",
}


def load_mode_episodes(results_root, model_dir, prefix="drift__", fixed_from="prime__gold"):
    """mode -> list of per-episode metric dicts (seed-keyed).

    prefix/fixed_from switch between the behavioral runs (drift__*, deployed
    fixed arm read from prime__gold) and the adversarial-reviser runs
    (advdrift__*, which include their own advdrift__fixed)."""
    out = defaultdict(list)
    base = os.path.join(results_root, model_dir)
    sources = [(p, os.path.basename(os.path.dirname(p)).replace(prefix, ""))
               for p in glob.glob(os.path.join(base, prefix + "*", "summary.json"))]
    fixed_path = os.path.join(base, fixed_from, "summary.json")
    if fixed_from and os.path.exists(fixed_path) and not any(m == "fixed" for _, m in sources):
        sources = [(fixed_path, "fixed")] + sources
    for path, mode in sources:
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        for ep in s.get("episodes", []):
            if ep.get("error"):
                continue
            em = ep.get("emitted", [])
            unsup = sum(1 for e in em if not e.get("complete_auth_gold", 0))
            out[mode].append({
                "seed": ep.get("seed", ep.get("instance", "?")),
                "drift": float(ep.get("drift_events", 0)),
                "drift_any": 1.0 if ep.get("drift_events", 0) else 0.0,
                "flr": 1.0 if unsup else 0.0,
                "rev_acc": float(ep.get("revisions_accepted", 0)),
                "rev_prop": float(ep.get("revisions_proposed", 0)),
            })
    return out


def _boot(rows, key):
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r[key])
    return cluster_bootstrap(by_seed)


def render(results_root, model_dir, prefix="drift__", fixed_from="prime__gold"):
    modes = load_mode_episodes(results_root, model_dir, prefix, fixed_from)
    if not modes:
        return None, None
    md = ["| Revision arm | n | drift-event rate | mean drift/ep | gold FLR | revisions accepted |",
          "|---|---|---|---|---|---|"]
    tex = []
    for mode in MODE_ORDER:
        rows = modes.get(mode)
        if not rows:
            continue
        n = len(rows)
        da_pt, da_lo, da_hi = _boot(rows, "drift_any")
        dm_pt, _, _ = _boot(rows, "drift")
        fl_pt, fl_lo, fl_hi = _boot(rows, "flr")
        ra_pt, _, _ = _boot(rows, "rev_acc")
        md.append("| %s | %d | %.2f [%.2f,%.2f] | %.2f | %.2f [%.2f,%.2f] | %.1f |"
                  % (MODE_LABEL[mode], n, da_pt, da_lo, da_hi, dm_pt, fl_pt, fl_lo, fl_hi, ra_pt))
        tex.append("%s & %d & %.2f\\,{\\scriptsize[%.2f,%.2f]} & %.2f & %.2f\\,{\\scriptsize[%.2f,%.2f]} & %.1f \\\\"
                   % (MODE_LABEL[mode], n, da_pt, da_lo, da_hi, dm_pt, fl_pt, fl_lo, fl_hi, ra_pt))
    return "\n".join(md), "\n".join(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.normpath(os.path.join(HERE, "..", "results")))
    ap.add_argument("--model-dir", default="qwen2.5-7b-q4-local")
    args = ap.parse_args()
    md, tex = render(args.results, args.model_dir)
    if md is None:
        print("no drift runs found under %s/%s" % (args.results, args.model_dir))
        return 2
    print("### %s\n" % args.model_dir)
    print(md)
    print("\n% LaTeX rows:\n" + (tex or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
