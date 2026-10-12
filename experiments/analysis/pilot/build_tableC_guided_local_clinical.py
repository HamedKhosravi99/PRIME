#!/usr/bin/env python3
"""Clinical-trial domain (data/pilot/instances/ctgov/), FREE LOCAL MODEL rung:
guided-completion ablation Table C -- unguided-PRIME vs guided-PRIME
(prime_guided arm, src/prime_runtime/guidance.py) at gold license source, plus
static_contract, on qwen2.5:14b-instruct-q4_K_M via ollama, oracle-scored,
exactly the same scoring machinery as experiments/analysis/pilot/build_tableC_guided_clinical.py
(this is an additive companion script for the local-model rung --
results/tableC_guided_clinical.json and the gpt-4.1/gpt-4o-mini script it
mirrors are both untouched).

Results-dir convention: ``results/<model>-clinical/<arm>__<lsrc>``, e.g.
``results/qwen2.5-14b-q4-local-clinical/prime_guided__gold`` -- the same
per-domain suffix convention every non-software domain script uses.

Writes:
  results/tableC_guided_local_clinical.json   ({"table": ..., "comparisons": ...})

Usage: python3 experiments/analysis/pilot/build_tableC_guided_local_clinical.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import end_to_end_tables as e2e  # noqa: E402
import tableC_guided_common as g  # noqa: E402

RESULTS = e2e.RESULTS
BENCH_GLOB = os.path.join(PILOT, "data", "pilot", "instances", "ctgov", "*.json")
MODELS = ["qwen2.5-14b-q4-local"]


def main() -> int:
    instance_paths = sorted(glob.glob(BENCH_GLOB))
    print("Computing per-instance oracle ceilings for %d clinical-trial instances..." % len(instance_paths),
          file=sys.stderr)
    ceilings = e2e.compute_oracle_ceilings(instance_paths)

    table, missing = g.build_guided_table_c(ceilings, MODELS, "clinical")
    comparisons = g.build_guided_comparisons(table, MODELS)

    out = {"table": table, "comparisons": comparisons}
    out_path = os.path.join(RESULTS, "tableC_guided_local_clinical.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n" + g.render_guided_markdown(table, MODELS, "Clinical-trial (local qwen2.5-14b)"))
    print("\n## Clinical domain (local): guided-completion pairwise comparisons\n")
    print(g.render_comparisons_markdown(comparisons))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print("\nWrote %s" % out_path, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
