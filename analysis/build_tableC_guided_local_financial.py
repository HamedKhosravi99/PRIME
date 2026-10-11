#!/usr/bin/env python3
"""Financial/SEC-XBRL domain (bench/instances/fin/), FREE LOCAL MODEL rung:
guided-completion ablation Table C on qwen2.5:14b-instruct-q4_K_M via ollama,
mirroring analysis/build_tableC_guided_financial.py (additive companion
script for the local-model rung -- results/tableC_guided_financial.json and
the gpt-4.1/gpt-4o-mini script it mirrors are both untouched).

Writes:
  results/tableC_guided_local_financial.json   ({"table": ..., "comparisons": ...})

Usage: python3 analysis/build_tableC_guided_local_financial.py
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
BENCH_GLOB = os.path.join(PILOT, "bench", "instances", "fin", "*.json")
MODELS = ["qwen2.5-14b-q4-local"]


def main() -> int:
    instance_paths = sorted(glob.glob(BENCH_GLOB))
    print("Computing per-instance oracle ceilings for %d financial instances..." % len(instance_paths),
          file=sys.stderr)
    ceilings = e2e.compute_oracle_ceilings(instance_paths)

    table, missing = g.build_guided_table_c(ceilings, MODELS, "financial")
    comparisons = g.build_guided_comparisons(table, MODELS)

    out = {"table": table, "comparisons": comparisons}
    out_path = os.path.join(RESULTS, "tableC_guided_local_financial.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n" + g.render_guided_markdown(table, MODELS, "Financial/SEC-XBRL (local qwen2.5-14b)"))
    print("\n## Financial domain (local): guided-completion pairwise comparisons\n")
    print(g.render_comparisons_markdown(comparisons))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print("\nWrote %s" % out_path, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
