#!/usr/bin/env python3
"""Financial/SEC-XBRL domain (data/pilot/instances/fin/): end-to-end oracle-scored
Table C, mirroring the research-review domain's Table C in
experiments/analysis/pilot/end_to_end_tables.py (build_research_table_c) -- SAME 9-config
method list, SAME matched budget=20, SAME oracle-scoring machinery
(src/prime_runtime/oracle.py oracle_authorization on emission-time evidence
reconstructed from the trace: benchmark truth is F*, never PRIME's own
routes) -- only the instance glob and the results-dir naming
convention differ:
  results/<model>-financial/<arm>__<lsrc>/summary.json

This is a NEW, standalone script that IMPORTS the shared scoring/table/
McNemar machinery from experiments/analysis/pilot/end_to_end_tables.py rather than modifying
that shared file (task instruction: do not modify shared infra code).

Writes results/tableC_financial.json, shape-matching results/tableC_research.
json exactly: {"table": {...}, "mcnemar": {...}}.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUNNER = os.path.join(PILOT, "src", "prime_agents")
for _p in (PILOT, os.path.join(PILOT, "src"), RUNNER, HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import end_to_end_tables as e2e  # noqa: E402  (analysis dir; shared, unmodified)

RESULTS = os.path.join(PILOT, "results")
FINANCIAL_BENCH_GLOB = os.path.join(PILOT, "data", "pilot", "instances", "fin", "*.json")
FINANCIAL_TABLE_C_MODELS = ["gpt-4.1", "gpt-4o-mini"]


def build_financial_table_c(ceilings):
    return e2e.build_table_c(
        ceilings, FINANCIAL_TABLE_C_MODELS, e2e.TABLE_C_CONFIGS,
        dir_for=lambda model, arm, lsrc: os.path.join(
            RESULTS, "%s-financial" % model, "%s__%s" % (arm, lsrc)
        ),
    )


def main() -> int:
    instance_paths = sorted(glob.glob(FINANCIAL_BENCH_GLOB))
    print("Computing per-instance oracle ceilings (U(C_oracle-max)) for %d financial instances..."
          % len(instance_paths), file=sys.stderr)
    ceilings = e2e.compute_oracle_ceilings(instance_paths)
    with open(os.path.join(RESULTS, "oracle_ceiling_financial.json"), "w", encoding="utf-8") as fh:
        json.dump(ceilings, fh, indent=2, sort_keys=True)
    for iid, c in sorted(ceilings.items()):
        print("  %-20s ceiling_level=%d (%s)" % (iid, c["ceiling_level"], c["ceiling_claim"]), file=sys.stderr)

    table_c, missing = build_financial_table_c(ceilings)
    baseline_arms = ["direct", "rubric", "react", "cove", "reflexion", "static_contract"]
    mcnemar = e2e.compare_mcnemar(table_c, FINANCIAL_TABLE_C_MODELS, baseline_arms)

    out = {"table": table_c, "mcnemar": mcnemar}
    out_path = os.path.join(RESULTS, "tableC_financial.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n# Financial (SEC-XBRL) domain Table C (matched budget = 20 tool calls; oracle-scored)\n")
    print(e2e.render_table_c_markdown(table_c, FINANCIAL_TABLE_C_MODELS, e2e.TABLE_C_CONFIGS))
    print("\n## Financial Table C LaTeX rows\n")
    print(e2e.render_table_c_latex(table_c, FINANCIAL_TABLE_C_MODELS, e2e.TABLE_C_CONFIGS))
    print("\n# Financial domain: exact McNemar, PRIME (induced) vs strongest baseline\n")
    print(e2e.render_mcnemar_markdown(mcnemar))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print("\nWrote %s and results/oracle_ceiling_financial.json" % out_path, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
