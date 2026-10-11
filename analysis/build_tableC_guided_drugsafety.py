#!/usr/bin/env python3
"""Drug-safety/openFDA domain (bench/instances/fda/): guided-completion
ablation Table C -- unguided-PRIME vs guided-PRIME (prime_guided arm,
prime_runtime/guidance.py) at gold and induced license source, plus
static_contract, oracle-scored, exactly the same scoring machinery as
analysis/build_tableC_drugsafety.py (this is an additive companion script,
not a replacement -- results/tableC_drugsafety.json is untouched).

Writes:
  results/tableC_guided_drugsafety.json   ({"table": ..., "comparisons": ...})

Usage: python3 analysis/build_tableC_guided_drugsafety.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import end_to_end_tables as e2e  # noqa: E402
import tableC_guided_common as g  # noqa: E402

RESULTS = e2e.RESULTS
BENCH_GLOB = os.path.join(PILOT, "bench", "instances", "fda", "*.json")
MODELS = ["gpt-4.1", "gpt-4o-mini"]


def main() -> int:
    instance_paths = sorted(glob.glob(BENCH_GLOB))
    print("Computing per-instance oracle ceilings for %d drug-safety instances..." % len(instance_paths),
          file=sys.stderr)
    ceilings = e2e.compute_oracle_ceilings(instance_paths)

    table, missing = g.build_guided_table_c(ceilings, MODELS, "drugsafety")
    comparisons = g.build_guided_comparisons(table, MODELS)

    out = {"table": table, "comparisons": comparisons}
    out_path = os.path.join(RESULTS, "tableC_guided_drugsafety.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n" + g.render_guided_markdown(table, MODELS, "Drug-safety (openFDA)"))
    print("\n## Drug-safety domain: guided-completion pairwise comparisons\n")
    print(g.render_comparisons_markdown(comparisons))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print("\nWrote %s" % out_path, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
