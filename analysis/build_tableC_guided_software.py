#!/usr/bin/env python3
"""Software-review domain (bench/instances/swe/): guided-completion ablation
Table C -- unguided-PRIME vs guided-PRIME (prime_guided arm,
prime_runtime/guidance.py) at gold and induced license source, plus
static_contract, oracle-scored, exactly the same scoring machinery as
analysis/end_to_end_tables.py's main software-review Table C (this is an
additive companion script, not a replacement -- results/tableC.json is
untouched).

Results-dir convention: UNSUFFIXED (``results/<model>/<arm>__<lsrc>``,
e2e.build_table_c's default), matching where this domain's existing
prime__gold / prime__induced / static_contract__gold runs already live
(unlike security/drugsafety/financial/research/clinical, which each have
their own ``results/<model>-<domain>/`` tree). The ``prime_guided`` runs
added by this ablation are written into that SAME unsuffixed tree
(``results/<model>/prime_guided__{gold,induced}``). This is safe w.r.t.
analysis/fill_paper.py's software-review cascade: that script's arm filter
only pools ``arm in (direct, rubric, react, prime)`` into the cascade table,
so ``prime_guided`` rows are excluded on arm name alone, independent of
directory (fill_paper.py is not modified by this script).

Writes:
  results/tableC_guided_software.json   ({"table": ..., "comparisons": ...})

Usage: python3 analysis/build_tableC_guided_software.py
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
BENCH_GLOB = os.path.join(PILOT, "bench", "instances", "swe", "*.json")
MODELS = ["gpt-4.1", "gpt-4o-mini"]


def build_guided_table_c_software(ceilings, models):
    """Same GUIDED_CONFIGS method list as the other guided scripts, but the
    UNSUFFIXED dir_for convention (results/<model>/<arm>__<lsrc>) -- this
    domain's own pre-existing convention, not the ``<model>-<domain>``
    convention the other five guided scripts use."""
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

    table, missing = build_guided_table_c_software(ceilings, MODELS)
    comparisons = g.build_guided_comparisons(table, MODELS)

    out = {"table": table, "comparisons": comparisons}
    out_path = os.path.join(RESULTS, "tableC_guided_software.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n" + g.render_guided_markdown(table, MODELS, "Software-review"))
    print("\n## Software domain: guided-completion pairwise comparisons\n")
    print(g.render_comparisons_markdown(comparisons))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print("\nWrote %s" % out_path, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
