#!/usr/bin/env python3
"""Security/supply-chain domain (data/pilot/instances/osv/): oracle-scored Table C,
exactly parallel to the research-review domain's
``experiments/analysis/pilot/end_to_end_tables.py:build_research_table_c`` / ``tableC_research.json``
-- SAME 9-config method list (matched action budget = 20), SAME oracle-scoring
machinery (``src/prime_runtime/oracle.py:oracle_authorization`` against facts
reconstructed from the trace, never PRIME's own gold-ledger field), SAME exact
McNemar pairing vs. the strongest non-PRIME baseline.

This is a NEW, additive script -- it does not modify
``experiments/analysis/pilot/end_to_end_tables.py`` (shared infra used by the software-review
and research-review domains); it only IMPORTS that module's already-general
building blocks (``compute_oracle_ceilings``, ``build_table_c``,
``compare_mcnemar``, the render_* helpers) and supplies this domain's own
instance glob, models, and results-dir naming convention
(``results/<model>-security/<arm>__<lsrc>``, mirroring the pre-existing
``results/<model>-research/<arm>__<lsrc>`` convention).

Reads:
  data/pilot/instances/osv/*.json
  results/<model>-security/<arm>__<lsrc>/summary.json   (9 configs x 2 models)

Writes:
  results/tableC_security.json   ({"table": ..., "mcnemar": ...})
  results/oracle_ceiling_security.json

Usage: python3 experiments/analysis/pilot/build_tableC_security.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUNNER = os.path.join(PILOT, "src", "prime_agents")
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.dirname(HERE), RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import end_to_end_tables as e2e  # noqa: E402  (analysis dir; unmodified shared module)

RESULTS = os.path.join(PILOT, "results")

SECURITY_BENCH_GLOB = os.path.join(PILOT, "data", "pilot", "instances", "osv", "*.json")
SECURITY_TABLE_C_MODELS = ["gpt-4.1", "gpt-4o-mini"]
SECURITY_BASELINE_ARMS = ["direct", "rubric", "react", "cove", "reflexion", "static_contract"]


def build_security_table_c(ceilings):
    return e2e.build_table_c(
        ceilings, SECURITY_TABLE_C_MODELS, e2e.TABLE_C_CONFIGS,
        dir_for=lambda model, arm, lsrc: os.path.join(
            RESULTS, "%s-security" % model, "%s__%s" % (arm, lsrc)
        ),
    )


def main() -> int:
    import glob
    instance_paths = sorted(glob.glob(SECURITY_BENCH_GLOB))
    print("Computing per-instance oracle ceilings (U(C_oracle-max)) for %d security instances..."
          % len(instance_paths), file=sys.stderr)
    ceilings = e2e.compute_oracle_ceilings(instance_paths)
    with open(os.path.join(RESULTS, "oracle_ceiling_security.json"), "w", encoding="utf-8") as fh:
        json.dump(ceilings, fh, indent=2, sort_keys=True)
    for iid, c in sorted(ceilings.items()):
        print("  %-20s ceiling_level=%d (%s)" % (iid, c["ceiling_level"], c["ceiling_claim"]), file=sys.stderr)

    table_c, missing = build_security_table_c(ceilings)
    mcnemar = e2e.compare_mcnemar(table_c, SECURITY_TABLE_C_MODELS, SECURITY_BASELINE_ARMS)

    out = {"table": table_c, "mcnemar": mcnemar}
    with open(os.path.join(RESULTS, "tableC_security.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n# Security/supply-chain domain Table C (matched budget = 20 tool calls; oracle-scored)\n")
    print(e2e.render_table_c_markdown(table_c, SECURITY_TABLE_C_MODELS, e2e.TABLE_C_CONFIGS))
    print("\n## Security Table C LaTeX rows\n")
    print(e2e.render_table_c_latex(table_c, SECURITY_TABLE_C_MODELS, e2e.TABLE_C_CONFIGS))
    print("\n# Security domain: exact McNemar, PRIME (induced) vs strongest baseline\n")
    print(e2e.render_mcnemar_markdown(mcnemar))

    if missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in missing:
            print("  - %s" % m, file=sys.stderr)
    print("\nWrote results/tableC_security.json, results/oracle_ceiling_security.json", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
