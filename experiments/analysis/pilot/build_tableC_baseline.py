#!/usr/bin/env python3
"""The fair-baseline comparison: does PRIME's
evidence-SEEKING control law improve over straightforward execution under an
EQUALLY EXPRESSIVE evidence gate?

`static_contract` alone is not a fair baseline: it takes the union of
literals across all gold routes and requires all of them, which destroys the
disjunctive alternatives and predictably causes abstention. It is an
over-conservative reference, not a strong formal-contract baseline.

`masked_reactive` (src/prime_agents/arms.py) is the baseline that isolates the
question. It keeps, bit-for-bit:
  * the same committed gold DNF routes (alternatives intact),
  * the same graded claim lattice and MaxAuth fallback,
  * the same enforced interpreter mask,
and removes ONLY obligation-directed action selection and minimal-route
planning. So `prime` vs `masked_reactive` varies exactly one thing: the
control law. `react` is included as the unconstrained reference (no gate at
all), and `static_contract` is retained so the new baseline can be compared
against the old one on identical episodes.

Additive, like every other experiments/analysis/pilot/build_tableC_*.py: imports
end_to_end_tables.py and tableC_guided_common.py without editing them.

Writes: results/tableC_baseline_<domain>.json
Usage:  python3 experiments/analysis/pilot/build_tableC_baseline.py --models qwen2.5-7b
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

BASELINE_CONFIGS: Tuple[Tuple[str, str, str], ...] = (
    ("react", "gold", "ReAct (no gate)"),
    ("static_contract", "gold", "Static contract (conjunctive collapse)"),
    ("masked_reactive", "gold", "Masked reactive (same gate, no control law)"),
    ("prime", "gold", "PRIME (gate + control law)"),
    ("masked_reactive", "induced", "Masked reactive (induced)"),
    ("prime", "induced", "PRIME (induced)"),
)

DEFAULT_DOMAINS = (("research", "res"), ("security", "osv"),
                   ("drugsafety", "fda"), ("financial", "fin"))
DOMAIN_LABELS = {"research": "Research-review", "security": "Security/OSV",
                 "drugsafety": "Drug-safety/openFDA", "financial": "Financial/SEC-XBRL",
                 "software": "Software-review", "clinical": "Clinical-trials"}


def build(ceilings, models, domain):
    return e2e.build_table_c(
        ceilings, models, BASELINE_CONFIGS,
        dir_for=lambda model, arm, lsrc: os.path.join(
            RESULTS, "%s-%s" % (model, domain), "%s__%s" % (arm, lsrc)),
    )


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["qwen2.5-7b"])
    ap.add_argument("--domains", nargs="+", default=None)
    args = ap.parse_args(argv)

    domains = ([tuple(d.split(":", 1)) for d in args.domains] if args.domains
               else list(DEFAULT_DOMAINS))
    missing_all: List[str] = []
    for label, sub in domains:
        paths = sorted(glob.glob(os.path.join(PILOT, "data", "pilot", "instances", sub, "*.json")))
        if not paths:
            continue
        print("Computing oracle ceilings for %d %s instances..." % (len(paths), label),
              file=sys.stderr)
        ceilings = e2e.compute_oracle_ceilings(paths)
        table, missing = build(ceilings, args.models, label)

        comparisons = {}
        for m in args.models:
            comparisons[m] = {
                "prime_vs_masked_reactive_gold": g.pairwise_mcnemar(
                    table, m, ("prime", "gold"), ("masked_reactive", "gold")),
                "masked_reactive_vs_static_gold": g.pairwise_mcnemar(
                    table, m, ("masked_reactive", "gold"), ("static_contract", "gold")),
                "masked_reactive_vs_react_gold": g.pairwise_mcnemar(
                    table, m, ("masked_reactive", "gold"), ("react", "gold")),
                "prime_vs_masked_reactive_induced": g.pairwise_mcnemar(
                    table, m, ("prime", "induced"), ("masked_reactive", "induced")),
            }

        out_path = os.path.join(RESULTS, "tableC_baseline_%s.json" % label)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"table": table, "comparisons": comparisons}, fh, indent=2, sort_keys=True)

        print("\n# %s -- fair-baseline comparison (matched budget 20; oracle-scored)\n"
              % DOMAIN_LABELS.get(label, label))
        print(e2e.render_table_c_markdown(table, args.models, BASELINE_CONFIGS))
        print("\n## %s: pairwise comparisons\n" % label)
        print(g.render_comparisons_markdown(comparisons))
        missing_all.extend(missing)

    if missing_all:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):", file=sys.stderr)
        for m in missing_all:
            print("  - %s" % m, file=sys.stderr)
    return 1 if missing_all else 0


if __name__ == "__main__":
    raise SystemExit(main())
