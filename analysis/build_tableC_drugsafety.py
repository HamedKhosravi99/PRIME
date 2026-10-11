#!/usr/bin/env python3
"""Drug-safety/openFDA domain Table C -- oracle-scored, mirroring
analysis/end_to_end_tables.py's research-domain Table C exactly (same 9-config
method list, same matched budget=20 tool calls, same oracle-scoring machinery,
same exact-McNemar comparison), only the instance glob and the results-dir
naming convention differ:
    results/<model>-drugsafety/<arm>__<lsrc>/summary.json

Written as a STANDALONE script (imports, does not edit,
analysis/end_to_end_tables.py) to avoid touching that shared/concurrently-
edited file -- every function reused here (compute_oracle_ceilings,
build_table_c, compare_mcnemar, the render_* helpers) is exactly the same
code path the software- and research-review domains' Table C already run
through, so drug-safety numbers are produced by IDENTICAL scoring machinery,
not a reimplementation.

SCORING: every emitted claim is scored against the EXECUTABLE ORACLE
(prime_runtime/oracle.py oracle_authorization), never PRIME's own gold-ledger
field -- see end_to_end_tables.py's module docstring for the full rationale.

Writes:
  results/tableC_drugsafety.json            ({"table": ..., "mcnemar": ...})
  results/oracle_ceiling_drugsafety.json    (per-instance oracle ceiling)

Usage:
    python3 analysis/build_tableC_drugsafety.py
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

from end_to_end_tables import (  # noqa: E402  (analysis dir)
    RESULTS,
    TABLE_C_CONFIGS,
    build_table_c,
    compare_mcnemar,
    compute_oracle_ceilings,
    render_mcnemar_markdown,
    render_table_c_latex,
    render_table_c_markdown,
)

DRUGSAFETY_BENCH_GLOB = os.path.join(PILOT, "bench", "instances", "fda", "*.json")
DRUGSAFETY_TABLE_C_MODELS = ["gpt-4.1", "gpt-4o-mini"]
BASELINE_ARMS = ["direct", "rubric", "react", "cove", "reflexion", "static_contract"]


def build_drugsafety_table_c(ceilings):
    return build_table_c(
        ceilings, DRUGSAFETY_TABLE_C_MODELS, TABLE_C_CONFIGS,
        dir_for=lambda model, arm, lsrc: os.path.join(
            RESULTS, "%s-drugsafety" % model, "%s__%s" % (arm, lsrc)
        ),
    )


def main() -> int:
    instance_paths = sorted(glob.glob(DRUGSAFETY_BENCH_GLOB))
    print(
        "Computing per-instance oracle ceilings for %d drug-safety instances..."
        % len(instance_paths),
        file=sys.stderr,
    )
    ceilings = compute_oracle_ceilings(instance_paths)
    with open(os.path.join(RESULTS, "oracle_ceiling_drugsafety.json"), "w", encoding="utf-8") as fh:
        json.dump(ceilings, fh, indent=2, sort_keys=True)
    for iid, c in sorted(ceilings.items()):
        print("  %-16s ceiling_level=%d (%s)" % (iid, c["ceiling_level"], c["ceiling_claim"]), file=sys.stderr)

    table_c, missing = build_drugsafety_table_c(ceilings)
    mcnemar = compare_mcnemar(table_c, DRUGSAFETY_TABLE_C_MODELS, BASELINE_ARMS)

    out = {"table": table_c, "mcnemar": mcnemar}
    with open(os.path.join(RESULTS, "tableC_drugsafety.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n# Drug-safety (openFDA) domain Table C (matched budget = 20 tool calls; oracle-scored)\n")
    print(render_table_c_markdown(table_c, DRUGSAFETY_TABLE_C_MODELS, TABLE_C_CONFIGS))
    print("\n## Drug-safety Table C LaTeX rows\n")
    print(render_table_c_latex(table_c, DRUGSAFETY_TABLE_C_MODELS, TABLE_C_CONFIGS))
    print("\n# Drug-safety domain: exact McNemar, PRIME (induced) vs strongest baseline\n")
    print(render_mcnemar_markdown(mcnemar))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print(
        "\nWrote results/tableC_drugsafety.json, results/oracle_ceiling_drugsafety.json",
        file=sys.stderr,
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
