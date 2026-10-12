#!/usr/bin/env python3
"""Open-weight (self-hosted vLLM) rung of the guided-completion ablation --
unguided-PRIME vs guided-PRIME (prime_guided arm, src/prime_runtime/guidance.py)
at gold and induced license source, plus static_contract -- oracle-scored
with exactly the same machinery as every other Table C script.

This is the free-compute companion to the API-model scripts
(experiments/analysis/pilot/build_tableC_guided_{security,drugsafety,financial}.py): the runs
it reads are produced by experiments/slurm/serve_and_run_guided.sbatch on a free
cluster queue, with no paid API key anywhere in the pipeline.

ADDITIVE, like every other experiments/analysis/pilot/build_tableC_guided_*.py: it imports
end_to_end_tables.py and tableC_guided_common.py without editing them, and
writes only its own output files. It differs from the per-domain scripts
only in being parameterized over (models, domains) instead of hardcoding
one domain -- an open-weight rung is run across several domains at once, so
six near-identical hardcoded files would be the wrong shape.

Results-dir convention (unchanged): results/<model>-<domain>/<arm>__<lsrc>.

Writes, one file per domain:
  results/tableC_guided_openweight_<domain>.json  ({"table":..., "comparisons":...})

Usage:
  python3 experiments/analysis/pilot/build_tableC_guided_openweight.py
  python3 experiments/analysis/pilot/build_tableC_guided_openweight.py --models qwen2.5-32b \\
      --domains research:res security:osv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import end_to_end_tables as e2e  # noqa: E402
import tableC_guided_common as g  # noqa: E402

RESULTS = e2e.RESULTS

# Mirrors serve_and_run_guided.sbatch's DOMAINS default: (label, bench subdir).
DEFAULT_DOMAINS: Tuple[Tuple[str, str], ...] = (
    ("research", "res"),
    ("security", "osv"),
    ("drugsafety", "fda"),
    ("financial", "fin"),
)
DEFAULT_MODELS = ["qwen2.5-7b", "qwen2.5-32b"]

DOMAIN_LABELS = {
    "research": "Research-review",
    "security": "Security/OSV",
    "drugsafety": "Drug-safety/openFDA",
    "financial": "Financial/SEC-XBRL",
    "software": "Software-review",
    "clinical": "Clinical-trials/CTgov",
}


def _parse_domains(specs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for spec in specs:
        if ":" not in spec:
            raise SystemExit("bad --domains entry %r (want '<label>:<bench-subdir>')" % spec)
        label, sub = spec.split(":", 1)
        out.append((label, sub))
    return out


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="results-dir model tags (MODELTAG in the sbatch)")
    ap.add_argument("--domains", nargs="+", default=None,
                    help="'<label>:<bench-subdir>' pairs, e.g. research:res")
    args = ap.parse_args(argv)

    domains = _parse_domains(args.domains) if args.domains else list(DEFAULT_DOMAINS)
    models = list(args.models)

    all_missing: List[str] = []
    wrote: List[str] = []
    for label, sub in domains:
        bench_glob = os.path.join(PILOT, "data", "pilot", "instances", sub, "*.json")
        instance_paths = sorted(glob.glob(bench_glob))
        if not instance_paths:
            print("# no instances at %s -- skipping domain %s" % (bench_glob, label), file=sys.stderr)
            continue
        print("Computing per-instance oracle ceilings for %d %s instances..."
              % (len(instance_paths), label), file=sys.stderr)
        ceilings = e2e.compute_oracle_ceilings(instance_paths)

        table, missing = g.build_guided_table_c(ceilings, models, label)
        comparisons = g.build_guided_comparisons(table, models)

        out = {"table": table, "comparisons": comparisons}
        out_path = os.path.join(RESULTS, "tableC_guided_openweight_%s.json" % label)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        wrote.append(out_path)

        pretty = "%s (open-weight: %s)" % (DOMAIN_LABELS.get(label, label), ", ".join(models))
        print("\n" + g.render_guided_markdown(table, models, pretty))
        print("\n## %s domain (open-weight): guided-completion pairwise comparisons\n" % label)
        print(g.render_comparisons_markdown(comparisons))
        all_missing.extend(missing)

    if all_missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in all_missing:
            print("  - %s" % m, file=sys.stderr)
    for p in wrote:
        print("Wrote %s" % p, file=sys.stderr)
    return 1 if all_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
