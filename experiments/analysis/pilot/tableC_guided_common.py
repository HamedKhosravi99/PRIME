#!/usr/bin/env python3
"""Shared building blocks for the three `experiments/analysis/pilot/build_tableC_guided_*.py`
domain scripts (security/osv, drug-safety/fda, financial/fin) -- the
guided-vs-unguided PRIME ablation on the three domains where PRIME under-
claims (see src/prime_runtime/guidance.py).

IMPORTS, never edits, `experiments/analysis/pilot/end_to_end_tables.py` (scoring/table/render
building blocks already shared by the software-, research-, security-,
drug-safety-, and financial-domain Table C scripts) -- consistent with the
existing "new, additive script" convention used throughout `experiments/analysis/`.

Adds only what those scripts do not already provide: the 5-config method
list for this ablation (unguided-PRIME/guided-PRIME x gold/induced, plus
static_contract) and a direct pairwise exact-McNemar comparison between two
SPECIFIC configs (rather than `end_to_end_tables.compare_mcnemar`'s
"best-baseline-among-many" search), so guided-vs-unguided and
guided-vs-static_contract can each be reported on their own terms.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUNNER = os.path.join(PILOT, "src", "prime_agents")
for _p in (PILOT, os.path.join(PILOT, "src"), RUNNER, HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import end_to_end_tables as e2e  # noqa: E402  (analysis dir; unmodified shared module)

RESULTS = e2e.RESULTS

# (arm, license_source, label) -- the guided-completion ablation's 5-config
# method list: unguided PRIME (existing arm, untouched) vs guided PRIME (the
# new prime_guided arm), each at gold and induced license source, plus the
# static_contract baseline PRIME loses to on utility in these domains.
GUIDED_CONFIGS: Tuple[Tuple[str, str, str], ...] = (
    ("prime", "gold", "PRIME (gold, unguided)"),
    ("prime_guided", "gold", "PRIME (gold, guided)"),
    ("prime", "induced", "PRIME (induced, unguided)"),
    ("prime_guided", "induced", "PRIME (induced, guided)"),
    ("static_contract", "gold", "Static formal contract"),
)


def build_guided_table_c(ceilings, models: List[str], domain_dir_suffix: str):
    """Table C restricted to GUIDED_CONFIGS, results dir
    ``results/<model>-<domain_dir_suffix>/<arm>__<lsrc>`` (the same
    per-domain naming convention every other domain script already uses)."""
    return e2e.build_table_c(
        ceilings, models, GUIDED_CONFIGS,
        dir_for=lambda model, arm, lsrc: os.path.join(
            RESULTS, "%s-%s" % (model, domain_dir_suffix), "%s__%s" % (arm, lsrc)
        ),
    )


def pairwise_mcnemar(
    table: Dict[str, Any], model: str, key_a: Tuple[str, str], key_b: Tuple[str, str]
) -> Dict[str, Any]:
    """Exact McNemar on episode OFAR, paired by instance id, between two
    SPECIFIC (arm, license_source) configs for one model -- e.g. guided vs
    unguided PRIME(gold), or guided PRIME vs static_contract. Also reports
    the paired supported-utility and tool-call sums (ratio-of-sums cost is
    read off the group stats already in ``table``, not re-derived here)."""
    ak, bk = "%s|%s|%s" % (model, key_a[0], key_a[1]), "%s|%s|%s" % (model, key_b[0], key_b[1])
    arow, brow = table.get(ak), table.get(bk)
    if arow is None or brow is None or arow.get("stats") is None or brow.get("stats") is None:
        return {"n_paired": 0, "note": "NOT RUN: %s" % (ak if arow is None else bk)}
    a_by_inst = {e["instance"]: e for e in arow["episodes"]}
    b_by_inst = {e["instance"]: e for e in brow["episodes"]}
    shared = sorted(set(a_by_inst) & set(b_by_inst))
    b_cnt = c_cnt = 0  # b: a-worse (a_ofar=1,b_ofar=0); c: b-worse (a_ofar=0,b_ofar=1)
    a_u = b_u = 0.0
    a_tc = b_tc = 0.0
    for inst in shared:
        a_ofar, b_ofar = a_by_inst[inst]["episode_ofar"], b_by_inst[inst]["episode_ofar"]
        if a_ofar > b_ofar:
            b_cnt += 1
        elif b_ofar > a_ofar:
            c_cnt += 1
        a_u += a_by_inst[inst]["u_supported"]
        b_u += b_by_inst[inst]["u_supported"]
        a_tc += a_by_inst[inst]["tool_calls"]
        b_tc += b_by_inst[inst]["tool_calls"]
    return {
        "a": ak, "b": bk, "n_paired": len(shared),
        "a_worse_ofar_pairs": b_cnt, "b_worse_ofar_pairs": c_cnt,
        "exact_mcnemar_p": e2e.exact_mcnemar_p(b_cnt, c_cnt),
        "sum_u_supported_a": a_u, "sum_u_supported_b": b_u,
        "mean_u_supported_a": a_u / len(shared) if shared else float("nan"),
        "mean_u_supported_b": b_u / len(shared) if shared else float("nan"),
        "mean_tool_calls_a": a_tc / len(shared) if shared else float("nan"),
        "mean_tool_calls_b": b_tc / len(shared) if shared else float("nan"),
    }


def build_guided_comparisons(table: Dict[str, Any], models: List[str]) -> Dict[str, Any]:
    """The three comparisons the task's decision-grade summary needs, per
    model: guided vs unguided (gold), guided vs unguided (induced), guided
    vs static_contract (gold)."""
    out: Dict[str, Any] = {}
    for model in models:
        out[model] = {
            "guided_vs_unguided_gold": pairwise_mcnemar(
                table, model, ("prime_guided", "gold"), ("prime", "gold")
            ),
            "guided_vs_unguided_induced": pairwise_mcnemar(
                table, model, ("prime_guided", "induced"), ("prime", "induced")
            ),
            "guided_vs_static_contract": pairwise_mcnemar(
                table, model, ("prime_guided", "gold"), ("static_contract", "gold")
            ),
        }
    return out


def render_guided_markdown(table: Dict[str, Any], models: List[str], domain_label: str) -> str:
    return (
        "# %s domain -- guided-completion ablation Table C "
        "(matched budget = 20 tool calls; oracle-scored)\n\n" % domain_label
        + e2e.render_table_c_markdown(table, models, GUIDED_CONFIGS)
    )


def render_comparisons_markdown(comparisons: Dict[str, Any]) -> str:
    lines = [
        "| Model | Comparison | n paired | A-worse-OFAR | B-worse-OFAR | Exact McNemar p | "
        "mean U(supp) A / B | mean tool_calls A / B |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, comps in comparisons.items():
        for label, r in comps.items():
            if r.get("n_paired", 0) == 0:
                lines.append("| %s | %s | 0 | - | - | - | NOT RUN (%s) | - |" % (model, label, r.get("note", "")))
                continue
            lines.append(
                "| %s | %s | %d | %d | %d | %s | %.2f / %.2f | %.2f / %.2f |"
                % (
                    model, label, r["n_paired"], r["a_worse_ofar_pairs"], r["b_worse_ofar_pairs"],
                    ("%.3f" % r["exact_mcnemar_p"]) if r["exact_mcnemar_p"] >= 0.001 else "<0.001",
                    r["mean_u_supported_a"], r["mean_u_supported_b"],
                    r["mean_tool_calls_a"], r["mean_tool_calls_b"],
                )
            )
    return "\n".join(lines)
