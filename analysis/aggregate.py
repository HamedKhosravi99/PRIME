#!/usr/bin/env python3
"""Aggregate pilot results into the paper's tables.

Reads results/<model>/<arm>__<license-source>/summary.json files produced by
runner/run_batch.py (see pilot/SPEC.md), computes the preregistered endpoints,
and emits (a) a markdown report and (b) LaTeX table rows for paper/prime.tex.

Statistical plan (PRIME-master.md v15 §12.7, §9.6):
  - The SEED TASK is the primary resampling unit: episodes sharing a seed are
    dependent (controlled variants), so all CIs are cluster bootstrap over seeds.
  - Metrics are computed exclusively from gold-ledger CompleteAuth values that
    the runner derived via goldledger.reconstruct from the logged trace —
    never from an arm's self-report.
  - No numeric cell in the paper may come from anywhere but this script's
    output on real run directories. This script refuses to run on empty input.

Endpoints per arm x model x license-source:
  gap_rate          fraction of emitted evaluative claims with CompleteAuth=0
                    at emission (split into zero_work / partial_work when the
                    runner recorded per-criterion work overlap)
  strong_unsupported fraction of episodes emitting a top-level (max strength)
                    claim with CompleteAuth=0
  supported_rate    fraction of episodes whose final emission set is fully
                    gold-authorized at emission time
  completion        mean per-episode max obligation completion (gold program)
  tool_calls        mean tool calls per episode
  tokens            mean completion tokens per episode
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List

# Any results/<model>-<domain>/ directory for the four record-oracle domains
# added 2026-07-29 (security/OSV, financial/SEC-XBRL, drug-safety/openFDA,
# clinical-trials/CTgov -- see prime_runtime/RECORD_ORACLE_REPORT.md and
# analysis/{SECURITY,FINANCIAL,DRUGSAFETY,CLINICAL}_DOMAIN_REPORT.md). These
# share model names (e.g. "gpt-4.1") with the software-review directory, so
# without a path exclusion they would silently pool into the same
# (model, arm, license_source) group as the software-review runs -- exactly
# the tab:pilot contamination bug this exclusion fixes. research/ is the
# pre-existing second domain and is excluded the same way for the same
# reason.
_DOMAIN_DIR_RE = re.compile(
    r"(?:^|/)[^/]+-(research|security|financial|drugsafety|clinical)(?:/|$)"
)


def load_summaries(results_root: str) -> List[dict]:
    rows = []
    for path in glob.glob(os.path.join(results_root, "**", "summary.json"), recursive=True):
        if "_smoke_record" in path:
            # Synthetic structured-record infra smoke-test proof run (see
            # bench/instances/_smoke_record/rec-0001.json and
            # prime_runtime/RECORD_ORACLE_REPORT.md) -- never a real
            # experimental condition; excluded by path so it can never reach
            # any aggregate table, mirroring the drift/adversarial/frontier
            # exclusions fill_paper.py applies to its own loader.
            continue
        if _DOMAIN_DIR_RE.search(path):
            # New-domain result dir (see comment above) -- these get their
            # own per-domain tables (tableC_research.json, tableC_security.json,
            # etc.) via analysis/build_tableC_*.py / end_to_end_tables.py, not
            # this general-purpose aggregator.
            continue
        with open(path) as f:
            s = json.load(f)
        run = s.get("run", {})  # canonical: metadata nested under "run"
        for ep in s.get("episodes", []):
            if ep.get("error"):
                continue  # failed episodes carry no measurements — never zeros
            ep["_model"] = run.get("model") or s.get("model", "?")
            ep["_arm"] = run.get("arm") or s.get("arm", "?")
            ep["_lsrc"] = run.get("license_source") or s.get("license_source", "?")
            rows.append(ep)
    return rows


def episode_metrics(ep: dict) -> dict:
    # canonical episode schema (runner/run_episode.py): "emitted" entries carry
    # claim, level, step, complete_auth_gold; zero-work split = unsupported
    # emission with zero tool calls before its step.
    emissions = ep.get("emitted", ep.get("emissions", []))
    n_claims = len(emissions)
    unsupported = [e for e in emissions if not e.get("complete_auth_gold", e.get("complete_auth", 0))]
    inst_max_level = ep.get("max_level", 3)
    strong_unsup = any(
        (not e.get("complete_auth_gold", 0)) and e.get("level", 1) >= inst_max_level
        for e in emissions
    )
    zero_work = ep.get("tool_calls", 0) == 0 and bool(unsupported)
    return {
        "seed": ep.get("seed", ep.get("instance", "?")),
        "gap_any": 1.0 if unsupported else 0.0,
        "gap_zero_work": 1.0 if zero_work else 0.0,
        "gap_partial_work": 1.0 if (unsupported and not zero_work) else 0.0,
        "strong_unsupported": 1.0 if strong_unsup else 0.0,
        "supported": 1.0 if (n_claims > 0 and not unsupported) else 0.0,
        "refused": float(ep.get("refused_emissions", 0)),
        "completion": float(ep.get("max_completion") or 0.0),
        "tool_calls": float(ep.get("tool_calls", 0)),
        "tokens": float((ep.get("tokens") or {}).get("completion_tokens", 0)),
    }


def cluster_bootstrap(values_by_seed: Dict[str, List[float]], iters: int = 5000, seed: int = 7) -> tuple:
    """Mean and 95% CI, resampling SEEDS with replacement (cluster bootstrap)."""
    seeds = sorted(values_by_seed)
    if not seeds:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = _mean([v for s in seeds for v in values_by_seed[s]])
    stats = []
    for _ in range(iters):
        sample = [rng.choice(seeds) for _ in seeds]
        vals = [v for s in sample for v in values_by_seed[s]]
        stats.append(_mean(vals))
    stats.sort()
    lo = stats[int(0.025 * iters)]
    hi = stats[min(int(0.975 * iters), iters - 1)]
    return point, lo, hi


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-tex", default=None)
    args = ap.parse_args()

    rows = load_summaries(args.results)
    if not rows:
        print("REFUSING: no summary.json found under %s — tables stay TBD." % args.results)
        sys.exit(2)

    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ep in rows:
        m = episode_metrics(ep)
        key = (ep["_model"], ep["_arm"], ep["_lsrc"])
        for k, v in m.items():
            if k == "seed":
                continue
            groups[key][k][m["seed"]].append(v)

    md = ["| model | arm | lic | gap rate | strong-unsup | supported | completion | tools | tokens |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for key in sorted(groups):
        model, arm, lsrc = key
        cells_md, cells_tex = [], []
        for metric in ["gap_any", "strong_unsupported", "supported", "completion", "tool_calls", "tokens"]:
            pt, lo, hi = cluster_bootstrap(groups[key][metric])
            if metric in ("tool_calls", "tokens"):
                cells_md.append(f"{pt:.1f}")
                cells_tex.append(f"{pt:.1f}")
            else:
                cells_md.append(f"{pt:.2f} [{lo:.2f},{hi:.2f}]")
                cells_tex.append(f"{pt:.2f} \\,[{lo:.2f},{hi:.2f}]")
        md.append("| " + " | ".join([model.split('/')[-1], arm, lsrc] + cells_md) + " |")
        tex.append(" & ".join([model.split('/')[-1], arm, lsrc] + cells_tex) + r" \\")

    report = "\n".join(md)
    print(report)
    if args.out_md:
        open(args.out_md, "w").write(report + "\n")
    if args.out_tex:
        open(args.out_tex, "w").write("\n".join(tex) + "\n")
        print("\nLaTeX rows -> %s" % args.out_tex)


if __name__ == "__main__":
    main()
