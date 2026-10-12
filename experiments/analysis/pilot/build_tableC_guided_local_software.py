#!/usr/bin/env python3
"""Software-review domain (data/pilot/instances/swe/), FREE LOCAL MODEL rung:
guided-completion ablation Table C -- unguided-PRIME vs guided-PRIME
(prime_guided arm, src/prime_runtime/guidance.py) at gold license source, plus
static_contract, on qwen2.5:14b-instruct-q4_K_M via ollama, oracle-scored,
exactly the same scoring machinery as experiments/analysis/pilot/build_tableC_guided_software.py
(this is an additive companion script for the local-model rung -- neither
results/tableC_guided_software.json nor the gpt-4.1/gpt-4o-mini script it
mirrors is touched).

Results-dir convention: UNSUFFIXED (``results/<model>/<arm>__<lsrc>``), same
as build_tableC_guided_software.py uses for the software domain -- i.e.
``results/qwen2.5-14b-q4-local/prime_guided__gold`` etc.

Only the induced-license rows are included when actually run; when not run,
they show as NOT RUN in the printed table/comparisons (never fabricated).

Writes:
  results/tableC_guided_local_software.json   ({"table": ..., "comparisons": ...})

Usage: python3 experiments/analysis/pilot/build_tableC_guided_local_software.py
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
BENCH_GLOB = os.path.join(PILOT, "data", "pilot", "instances", "swe", "*.json")
MODELS = ["qwen2.5-14b-q4-local"]


def build_guided_table_c_software_local(ceilings, models):
    """Same GUIDED_CONFIGS method list as the other guided scripts, but the
    UNSUFFIXED dir_for convention (results/<model>/<arm>__<lsrc>) -- this
    domain's own pre-existing convention (matches build_tableC_guided_software.py)."""
    return e2e.build_table_c(
        ceilings, models, g.GUIDED_CONFIGS,
        dir_for=lambda model, arm, lsrc: os.path.join(
            RESULTS, model, "%s__%s" % (arm, lsrc)
        ),
    )


def main() -> int:
    instance_paths = sorted(glob.glob(BENCH_GLOB))
    print("Computing per-instance oracle ceilings for %d software instances..." % len(instance_paths),
          file=sys.stderr)
    ceilings = e2e.compute_oracle_ceilings(instance_paths)

    table, missing = build_guided_table_c_software_local(ceilings, MODELS)
    comparisons = g.build_guided_comparisons(table, MODELS)

    out = {"table": table, "comparisons": comparisons}
    out_path = os.path.join(RESULTS, "tableC_guided_local_software.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n" + g.render_guided_markdown(table, MODELS, "Software-review (local qwen2.5-14b)"))
    print("\n## Software domain (local): guided-completion pairwise comparisons\n")
    print(g.render_comparisons_markdown(comparisons))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print("\nWrote %s" % out_path, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
