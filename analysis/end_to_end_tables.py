#!/usr/bin/env python3
"""End-to-end oracle-scored Table C (method list, matched resources, primary
metrics), Table E (the 2^4 mechanism-interaction ablation), and the E3
cost-support frontier.

SCORING: every emitted claim in every episode is scored against the
EXECUTABLE BENCHMARK ORACLE (prime_runtime/oracle.py) -- oracle_authorization
evaluated on the emission-time evidence facts reconstructed method-
independently from the trace -- NOT against PRIME's own gold-ledger
complete_auth field. Benchmark truth is F*, NOT PRIME's routes: a claim
counts as oracle-supported iff
``oracle.oracle_authorization(instance, claim_id, facts_at_emission)`` is
True; PRIME's own routes never enter the scoring path (only the induced/gold/
nonminimal *program* is under test, exactly like every other arm).

Reads (schema: a "run" block + an "episodes" list, per run_batch.py /
run_episode.py / run_ablation_batch.py):
  results/<model>/<arm>__<lsrc>/summary.json                    (Table C)
  results/gpt-4.1/frontier/<arm>__<lsrc>__b<N>/summary.json      (E3)
  results/gpt-4.1/ablation__m<0|1>_f<0|1>_k<0|1>_g<0|1>/summary.json (Table E)

Writes:
  results/tableC.json, results/tableE.json, results/frontier.json
  results/oracle_ceiling.json (per-instance oracle ceiling, cached/audit trail)
  markdown + LaTeX table bodies to stdout.

Never fabricates a missing cell: a config whose summary.json does not exist
(experiment not yet run, or every episode errored) is reported as
"NOT RUN" / excluded from aggregates, and listed explicitly at the end.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
RUNNER = os.path.join(PILOT, "runner")
for _p in (PILOT, RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aggregate import cluster_bootstrap  # noqa: E402  (analysis dir)

from prime_runtime import goldledger, oracle  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

import arms as arms_mod  # noqa: E402  (ToolExecutor, for the oracle ceiling)

RESULTS = os.path.join(PILOT, "results")
BENCH_GLOB = os.path.join(PILOT, "bench", "instances", "swe", "*.json")

# --------------------------------------------------------------------------
# Research-review domain (second evaluation domain, bench/instances/res/):
# same 9-config Table C method list, same matched budget=20, same oracle-
# scoring machinery -- only the instance glob and the results-dir naming
# convention differ ("<model>-research/<arm>__<lsrc>" vs "<model>/<arm>__<lsrc>").
# --------------------------------------------------------------------------
RESEARCH_BENCH_GLOB = os.path.join(PILOT, "bench", "instances", "res", "*.json")
RESEARCH_TABLE_C_MODELS = ["gpt-4.1", "gpt-4o-mini"]

EPS = 1e-9

# --------------------------------------------------------------------------
# Table C method list: 9 configs, matched budget=20, gpt-4.1
# primary / gpt-4o-mini secondary.
# --------------------------------------------------------------------------
TABLE_C_MODELS = ["gpt-4.1", "gpt-4o-mini"]
TABLE_C_CONFIGS: Tuple[Tuple[str, str, str], ...] = (
    # (arm, license_source, display label)
    ("direct", "gold", "Direct"),
    ("rubric", "gold", "Rubric (fixed checklist, no gating)"),
    ("react", "gold", "ReAct"),
    ("cove", "gold", "CoVe (verify-then-revise)"),
    ("reflexion", "gold", "Reflexion (self-critique-then-revise)"),
    ("static_contract", "gold", "Static formal contract"),
    ("prime", "gold", "PRIME (gold)"),
    ("prime", "induced", "PRIME (induced)"),
    ("prime_nonminimal", "gold", "PRIME (gold, nonminimal routes)"),
)

# --------------------------------------------------------------------------
# E3 frontier method list: gpt-4.1 only, budgets 4/8/16 (+20 reused
# from Table C, so at least four action budgets).
# --------------------------------------------------------------------------
FRONTIER_CONFIGS: Tuple[Tuple[str, str, str], ...] = (
    ("direct", "gold", "Direct"),
    ("react", "gold", "ReAct"),
    ("prime", "gold", "PRIME (gold)"),
    ("prime", "induced", "PRIME (induced)"),
)
FRONTIER_BUDGETS = (4, 8, 16, 20)
FRONTIER_MODEL = "gpt-4.1"


# ==========================================================================
# I/O
# ==========================================================================


def _load_summary(dir_path: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(dir_path, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_INSTANCE_CACHE: Dict[str, Any] = {}


def _instance_for(episode: Dict[str, Any]):
    path = episode.get("instance_path")
    if not path or not os.path.exists(path):
        # Fall back to the canonical bench path by instance id. This fires
        # whenever the recorded absolute instance_path does not exist on THIS
        # machine -- most importantly for episodes produced on a cluster,
        # whose summaries carry absolute cluster paths.
        #
        # Resolve by searching every bench/instances/<domain>/ subdirectory
        # rather than hardcoding a prefix->directory map: the map used to
        # cover only swe-/res- and silently pointed osv-/fin-/fda-/ctgov-
        # ids at bench/instances/swe/, which raised FileNotFoundError on the
        # first cluster-produced record-domain run.
        iid = episode["instance"]
        matches = sorted(glob.glob(
            os.path.join(PILOT, "bench", "instances", "*", "%s.json" % iid)
        ))
        if not matches:
            raise FileNotFoundError(
                "no bench instance file for id %r (searched "
                "bench/instances/*/%s.json); recorded instance_path %r does "
                "not exist on this machine" % (iid, iid, episode.get("instance_path"))
            )
        path = matches[0]
    if path not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[path] = load_instance(path)
    return _INSTANCE_CACHE[path]


# ==========================================================================
# Per-instance oracle ceiling: U(C_oracle-max) -- computed ONCE per instance
# by exhaustively resolving every atom that has a resolver (no LLM, no
# policy: the ceiling is a property of the instance's real evidence, not of
# what any arm happened to explore) and taking the strongest oracle-
# supported claim under those FULLY-resolved facts.
# ==========================================================================


def _resolve_all_atoms(instance) -> Dict[str, str]:
    executor = arms_mod.ToolExecutor(instance)
    led = Ledger(instance.atom_ids, instance.phi_patterns)
    step = 0
    for atom in instance.atoms:
        r = atom.resolver
        if r is None:
            continue  # terminal-unavailability atom: stays 'U', by design
        step += 1
        observation = executor.execute(r.tool, dict(r.params))
        goldledger.delta(instance, led, r.tool, dict(r.params), observation, step)
    return led.states()


def compute_oracle_ceilings(instance_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """{instance_id: {"ceiling_level": int, "ceiling_claim": str|None,
    "facts": {...}}} -- U(C_oracle-max) per instance, from FULLY resolved
    real evidence (not from any episode's partial exploration)."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(instance_paths):
        inst = load_instance(path)
        facts = _resolve_all_atoms(inst)
        best_level, best_claim = 0, None
        for c in inst.claims:
            if oracle.oracle_authorization(inst, c.id, facts) and c.level >= best_level:
                best_level, best_claim = c.level, c.id
        out[inst.id] = {
            "ceiling_level": best_level,
            "ceiling_claim": best_claim,
            "facts": facts,
        }
    return out


# ==========================================================================
# Per-episode oracle scoring (score via F*, not PRIME's routes)
# ==========================================================================


def score_episode(episode: Dict[str, Any], ceilings: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if episode.get("error"):
        return None  # failed episodes carry no measurement -- never a fabricated zero
    instance = _instance_for(episode)
    trace_path = episode.get("trace")
    if not trace_path or not os.path.exists(trace_path):
        trace_path = os.path.join(os.path.dirname(episode.get("instance_path", "")), "..")
    replay = goldledger.reconstruct(trace_path, instance)

    per_claim = []
    for e in episode.get("emitted", []):
        facts = replay.states_at(e["step"])
        supported = oracle.oracle_authorization(instance, e["claim"], facts)
        level = e.get("level")
        if level is None:
            try:
                level = instance.claim(e["claim"]).level
            except KeyError:
                level = 0
        per_claim.append({"claim": e["claim"], "level": level, "oracle_supported": supported})

    n = len(per_claim)
    n_unsupported = sum(1 for c in per_claim if not c["oracle_supported"])
    ceiling = ceilings.get(instance.id, {}).get("ceiling_level", 0)
    u_emitted = max((c["level"] or 0) for c in per_claim) if per_claim else 0
    supported_levels = [c["level"] or 0 for c in per_claim if c["oracle_supported"]]
    u_supported = max(supported_levels) if supported_levels else 0
    tool_calls = float(episode.get("tool_calls", 0))
    zero_work = bool(tool_calls == 0 and n_unsupported > 0)
    partial_work = bool(n_unsupported > 0 and not zero_work)

    return {
        "seed": episode.get("seed", episode.get("instance", "?")),
        "instance": instance.id,
        "n_emitted": n,
        "n_unsupported": n_unsupported,
        "episode_ofar": 1.0 if n_unsupported > 0 else 0.0,
        "claim_ofar_num": float(n_unsupported),
        "claim_ofar_den": float(n),
        "u_emitted": float(u_emitted),
        "u_supported": float(u_supported),
        "u_ceiling": float(ceiling),
        "supported_utility_rate": (u_supported / ceiling) if ceiling > 0 else (1.0 if u_supported == 0 else 0.0),
        "utility_regret": float(ceiling - u_supported),  # >=0: shortfall from the ceiling
        "mou": float(u_emitted - ceiling),  # signed: >0 only if an unsupported overclaim exceeded the ceiling
        "tool_calls": tool_calls,
        "zero_work_gap": 1.0 if zero_work else 0.0,
        "partial_work_gap": 1.0 if partial_work else 0.0,
        "abstained": 1.0 if n == 0 else 0.0,
        "refused_emissions": float(episode.get("refused_emissions", 0)),
        "induced_program_valid": episode.get("induced_program_valid"),
        "drift_events": float(episode.get("drift_events", 0)),
        "revisions_accepted": float(episode.get("revisions_accepted", 0)),
    }


def _group_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_seed: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if k in ("seed", "instance", "induced_program_valid"):
                continue
            by_seed[k][r["seed"]].append(v)
    out: Dict[str, Any] = {}
    for metric, per_seed in by_seed.items():
        pt, lo, hi = cluster_bootstrap(per_seed)
        out[metric] = {"point": pt, "lo": lo, "hi": hi}
    # ratio-of-sums cost-per-supported-utility (more stable than mean-of-ratios
    # when u_supported is 0 for some episodes)
    total_cost = sum(r["tool_calls"] for r in rows)
    total_u_supported = sum(r["u_supported"] for r in rows)
    out["cost_per_supported_utility"] = {
        "point": (total_cost / total_u_supported) if total_u_supported > EPS else float("inf"),
        "lo": float("nan"), "hi": float("nan"),
    }
    # claim-level (not episode-level) OFAR: ratio of sums
    total_unsup = sum(r["n_unsupported"] for r in rows)
    total_emit = sum(r["n_emitted"] for r in rows)
    out["claim_ofar"] = {
        "point": (total_unsup / total_emit) if total_emit > 0 else float("nan"),
        "lo": float("nan"), "hi": float("nan"),
    }
    out["n_episodes"] = len(rows)
    n_ind = [r for r in rows if r.get("induced_program_valid") is not None]
    if n_ind:
        out["induction_valid_rate"] = sum(1 for r in n_ind if r["induced_program_valid"]) / len(n_ind)
        out["n_induced_episodes"] = len(n_ind)
    return out


# ==========================================================================
# Table C
# ==========================================================================


def build_table_c(
    ceilings,
    models: Optional[List[str]] = None,
    configs: Optional[Tuple[Tuple[str, str, str], ...]] = None,
    dir_for: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Table C for one domain. ``dir_for(model, arm, lsrc) -> results dir``
    defaults to the software-review convention (``results/<model>/<arm>__<lsrc>``);
    the research domain passes ``results/<model>-research/<arm>__<lsrc>`` (see
    ``build_research_table_c``). Each row also keeps a compact per-episode
    ``episodes`` list (instance id, episode_ofar, u_supported, tool_calls) so
    McNemar pairing (research domain) can be computed without re-scoring."""
    models = models if models is not None else TABLE_C_MODELS
    configs = configs if configs is not None else TABLE_C_CONFIGS
    dir_for = dir_for or (lambda model, arm, lsrc: os.path.join(RESULTS, model, "%s__%s" % (arm, lsrc)))
    missing: List[str] = []
    table: Dict[str, Any] = {}
    for model in models:
        for arm, lsrc, label in configs:
            dir_path = dir_for(model, arm, lsrc)
            summary = _load_summary(dir_path)
            key = "%s|%s|%s" % (model, arm, lsrc)
            if summary is None:
                missing.append("Table C: %s" % dir_path)
                continue
            episodes = summary.get("episodes", [])
            n_errors = sum(1 for e in episodes if e.get("error"))
            scored = [score_episode(e, ceilings) for e in episodes]
            scored = [s for s in scored if s is not None]
            table[key] = {
                "model": model, "arm": arm, "license_source": lsrc, "label": label,
                "n_episodes_total": len(episodes), "n_errors": n_errors,
                "n_episodes_scored": len(scored),
                "stats": _group_stats(scored) if scored else None,
                "episodes": [
                    {
                        "instance": s["instance"], "episode_ofar": s["episode_ofar"],
                        "u_supported": s["u_supported"], "tool_calls": s["tool_calls"],
                    }
                    for s in scored
                ],
            }
    return table, missing


def build_research_table_c(ceilings) -> Tuple[Dict[str, Any], List[str]]:
    return build_table_c(
        ceilings, RESEARCH_TABLE_C_MODELS, TABLE_C_CONFIGS,
        dir_for=lambda model, arm, lsrc: os.path.join(
            RESULTS, "%s-research" % model, "%s__%s" % (arm, lsrc)
        ),
    )


# ==========================================================================
# Exact McNemar test (paired binary outcomes, e.g. episode OFAR by instance)
# ==========================================================================


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant-pair counts
    (b = A-worse-than-B pairs, c = B-worse-than-A pairs); concordant pairs
    (both-0 / both-1) do not enter the test. n = b + c."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def compare_mcnemar(
    table: Dict[str, Any], models: List[str], baseline_arms: List[str],
    target: Tuple[str, str] = ("prime", "induced"),
) -> Dict[str, Any]:
    """Per model: pick the best-supported-utility and best-(lowest)-OFAR
    baseline among ``baseline_arms`` (each license_source='gold'), then run
    exact McNemar (paired by instance id) between ``target`` (default
    PRIME induced) and each baseline reading on episode OFAR (1 = episode
    emitted >=1 oracle-unsupported claim), for any instance set."""
    out: Dict[str, Any] = {}
    for model in models:
        tkey = "%s|%s|%s" % (model, target[0], target[1])
        trow = table.get(tkey)
        if trow is None or trow.get("stats") is None:
            continue
        t_by_inst = {e["instance"]: e for e in trow["episodes"]}

        cand_rows = []
        for arm in baseline_arms:
            bkey = "%s|%s|gold" % (model, arm)
            brow = table.get(bkey)
            if brow is None or brow.get("stats") is None:
                continue
            cand_rows.append((arm, brow))
        if not cand_rows:
            continue
        best_utility = max(cand_rows, key=lambda ab: ab[1]["stats"]["supported_utility_rate"]["point"])
        best_ofar = min(
            cand_rows,
            key=lambda ab: (ab[1]["stats"]["episode_ofar"]["point"],
                             -ab[1]["stats"]["supported_utility_rate"]["point"]),
        )
        readings = {"best_supported_utility": best_utility, "best_ofar": best_ofar}
        if best_utility[0] == best_ofar[0]:
            readings = {"best_supported_utility_and_ofar": best_utility}

        model_out = {}
        for reading_label, (arm, brow) in readings.items():
            b_by_inst = {e["instance"]: e for e in brow["episodes"]}
            shared = sorted(set(t_by_inst) & set(b_by_inst))
            b_cnt = c_cnt = 0  # b: target-worse (t=1,base=0); c: baseline-worse (t=0,base=1)
            t_u = base_u = 0.0
            for inst in shared:
                t_ofar = t_by_inst[inst]["episode_ofar"]
                b_ofar = b_by_inst[inst]["episode_ofar"]
                if t_ofar > b_ofar:
                    b_cnt += 1
                elif b_ofar > t_ofar:
                    c_cnt += 1
                t_u += t_by_inst[inst]["u_supported"]
                base_u += b_by_inst[inst]["u_supported"]
            model_out[reading_label] = {
                "baseline_arm": arm, "n_paired": len(shared),
                "target_worse_pairs": b_cnt, "baseline_worse_pairs": c_cnt,
                "exact_mcnemar_p": exact_mcnemar_p(b_cnt, c_cnt),
                "sum_u_supported_target": t_u, "sum_u_supported_baseline": base_u,
            }
        out[model] = model_out
    return out


# ==========================================================================
# E3 cost-support frontier
# ==========================================================================


def build_frontier(ceilings) -> Tuple[Dict[str, Any], List[str]]:
    missing: List[str] = []
    points: List[Dict[str, Any]] = []
    for arm, lsrc, label in FRONTIER_CONFIGS:
        for budget in FRONTIER_BUDGETS:
            if budget == 20:
                dir_path = os.path.join(RESULTS, FRONTIER_MODEL, "%s__%s" % (arm, lsrc))
            else:
                dir_path = os.path.join(
                    RESULTS, FRONTIER_MODEL, "frontier", "%s__%s__b%d" % (arm, lsrc, budget)
                )
            summary = _load_summary(dir_path)
            if summary is None:
                missing.append("E3 frontier: %s" % dir_path)
                continue
            episodes = summary.get("episodes", [])
            scored = [score_episode(e, ceilings) for e in episodes]
            scored = [s for s in scored if s is not None]
            if not scored:
                missing.append("E3 frontier (0 scored episodes): %s" % dir_path)
                continue
            stats = _group_stats(scored)
            points.append({
                "arm": arm, "license_source": lsrc, "label": label, "budget": budget,
                "n_episodes": len(scored),
                "episode_ofar": stats["episode_ofar"]["point"],
                "supported_utility_rate": stats["supported_utility_rate"]["point"],
                "utility_regret": stats["utility_regret"]["point"],
                "cost_tool_calls": stats["tool_calls"]["point"],
                "cost_per_supported_utility": stats["cost_per_supported_utility"]["point"],
                "abstained": stats["abstained"]["point"],
            })
    return {"model": FRONTIER_MODEL, "budgets": list(FRONTIER_BUDGETS), "points": points}, missing


# ==========================================================================
# Table E: 2^4 mechanism ablation
# ==========================================================================


def build_table_e(ceilings) -> Tuple[Dict[str, Any], List[str]]:
    missing: List[str] = []
    table: Dict[str, Any] = {}
    for cell in arms_mod.ABLATION_CELLS:
        tag = "m%d_f%d_k%d_g%d" % (
            int(cell["minimal"]), int(cell["frozen"]), int(cell["masked"]), int(cell["graded"])
        )
        dir_path = os.path.join(RESULTS, FRONTIER_MODEL, "ablation__%s" % tag)
        summary = _load_summary(dir_path)
        if summary is None:
            missing.append("Table E: %s" % dir_path)
            continue
        episodes = summary.get("episodes", [])
        n_errors = sum(1 for e in episodes if e.get("error"))
        scored = [score_episode(e, ceilings) for e in episodes]
        scored = [s for s in scored if s is not None]
        table[tag] = {
            "cell": cell, "model": FRONTIER_MODEL,
            "n_episodes_total": len(episodes), "n_errors": n_errors,
            "n_episodes_scored": len(scored),
            "stats": _group_stats(scored) if scored else None,
        }
    return table, missing


# ==========================================================================
# Rendering
# ==========================================================================


def _fmt_ci(stat: Dict[str, float], pct: bool = True) -> str:
    pt, lo, hi = stat["point"], stat["lo"], stat["hi"]
    if pt != pt:  # NaN
        return "n/a"
    if lo != lo or hi != hi:
        return ("%.2f" if not pct else "%.1f%%") % (pt if not pct else pt * 100)
    if pct:
        return "%.1f%% [%.1f,%.1f]" % (pt * 100, lo * 100, hi * 100)
    return "%.2f [%.2f,%.2f]" % (pt, lo, hi)


def render_table_c_markdown(
    table: Dict[str, Any],
    models: Optional[List[str]] = None,
    configs: Optional[Tuple[Tuple[str, str, str], ...]] = None,
) -> str:
    models = models if models is not None else TABLE_C_MODELS
    configs = configs if configs is not None else TABLE_C_CONFIGS
    lines = [
        "| Model | Method | n | Episode OFAR down | Supported-Util-Rate up | "
        "Utility regret down | Cost (tools) | Cost/SuppUtil down | Abstained | Zero-work gap | "
        "Partial-work gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        for arm, lsrc, label in configs:
            key = "%s|%s|%s" % (model, arm, lsrc)
            row = table.get(key)
            if row is None or row["stats"] is None:
                lines.append("| %s | %s | - | NOT RUN | - | - | - | - | - | - | - |" % (model, label))
                continue
            s = row["stats"]
            cpsu = s["cost_per_supported_utility"]["point"]
            cpsu_s = "%.1f" % cpsu if cpsu != float("inf") else "inf"
            lines.append(
                "| %s | %s | %d | %s | %s | %.2f | %.1f | %s | %s | %s | %s |"
                % (
                    model, label, row["n_episodes_scored"],
                    _fmt_ci(s["episode_ofar"]),
                    _fmt_ci(s["supported_utility_rate"]),
                    s["utility_regret"]["point"],
                    s["tool_calls"]["point"],
                    cpsu_s,
                    _fmt_ci(s["abstained"]),
                    _fmt_ci(s["zero_work_gap"]),
                    _fmt_ci(s["partial_work_gap"]),
                )
            )
    return "\n".join(lines)


def render_table_c_latex(
    table: Dict[str, Any],
    models: Optional[List[str]] = None,
    configs: Optional[Tuple[Tuple[str, str, str], ...]] = None,
) -> str:
    models = models if models is not None else TABLE_C_MODELS
    configs = configs if configs is not None else TABLE_C_CONFIGS
    rows = []
    for model in models:
        for arm, lsrc, label in configs:
            key = "%s|%s|%s" % (model, arm, lsrc)
            row = table.get(key)
            if row is None or row["stats"] is None:
                rows.append(
                    "%s & %s & -- & NOT RUN & -- & -- & -- & -- & -- \\\\"
                    % (model.replace("_", "\\_"), label)
                )
                continue
            s = row["stats"]
            cpsu = s["cost_per_supported_utility"]["point"]
            cpsu_s = "%.1f" % cpsu if cpsu != float("inf") else "$\\infty$"
            rows.append(
                "%s & %s & %d & %.1f\\,[%.1f,%.1f] & %.1f\\,[%.1f,%.1f] & %.2f & %.1f & %s & %.0f\\%% \\\\"
                % (
                    model.replace("_", "\\_"), label, row["n_episodes_scored"],
                    s["episode_ofar"]["point"] * 100, s["episode_ofar"]["lo"] * 100, s["episode_ofar"]["hi"] * 100,
                    s["supported_utility_rate"]["point"] * 100, s["supported_utility_rate"]["lo"] * 100,
                    s["supported_utility_rate"]["hi"] * 100,
                    s["utility_regret"]["point"], s["tool_calls"]["point"], cpsu_s,
                    s["abstained"]["point"] * 100,
                )
            )
    return "\n".join(rows)


ARM_LABELS = {arm: label for arm, _lsrc, label in TABLE_C_CONFIGS}


def render_mcnemar_markdown(mcnemar: Dict[str, Any]) -> str:
    lines = [
        "| Model | Baseline reading | Baseline arm | n paired | PRIME-worse | "
        "Baseline-worse | Exact McNemar p | Sigma U(supported): PRIME / baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, readings in mcnemar.items():
        for reading_label, r in readings.items():
            lines.append(
                "| %s | %s | %s | %d | %d | %d | %s | %.1f / %.1f |"
                % (
                    model, reading_label.replace("_", " "),
                    ARM_LABELS.get(r["baseline_arm"], r["baseline_arm"]),
                    r["n_paired"], r["target_worse_pairs"], r["baseline_worse_pairs"],
                    ("%.3f" % r["exact_mcnemar_p"]) if r["exact_mcnemar_p"] >= 0.001 else "<0.001",
                    r["sum_u_supported_target"], r["sum_u_supported_baseline"],
                )
            )
    return "\n".join(lines)


def render_frontier_markdown(frontier: Dict[str, Any]) -> str:
    lines = [
        "| Method | Budget | n | Episode OFAR | Supported-Util-Rate | Utility regret | Cost (tools) | Abstained |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in frontier["points"]:
        lines.append(
            "| %s | %d | %d | %.1f%% | %.1f%% | %.2f | %.1f | %.1f%% |"
            % (p["label"], p["budget"], p["n_episodes"], p["episode_ofar"] * 100,
               p["supported_utility_rate"] * 100, p["utility_regret"], p["cost_tool_calls"],
               p.get("abstained", float("nan")) * 100)
        )
    return "\n".join(lines)


def render_table_e_markdown(table: Dict[str, Any]) -> str:
    lines = [
        "| Cell | minimal | frozen | masked | graded | n | Episode OFAR | Supported-Util-Rate | "
        "Cost (tools) | Abstained | Drift events/ep |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag in sorted(table):
        row = table[tag]
        c = row["cell"]
        if row["stats"] is None:
            lines.append("| %s | %d | %d | %d | %d | - | NOT RUN | - | - | - | - |"
                          % (tag, c["minimal"], c["frozen"], c["masked"], c["graded"]))
            continue
        s = row["stats"]
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %s | %s | %.1f | %s | %.2f |"
            % (
                tag, c["minimal"], c["frozen"], c["masked"], c["graded"],
                row["n_episodes_scored"],
                _fmt_ci(s["episode_ofar"]), _fmt_ci(s["supported_utility_rate"]),
                s["tool_calls"]["point"], _fmt_ci(s["abstained"]), s["drift_events"]["point"],
            )
        )
    return "\n".join(lines)


def render_table_e_latex(table: Dict[str, Any]) -> str:
    rows = []
    for tag in sorted(table):
        row = table[tag]
        c = row["cell"]
        if row["stats"] is None:
            rows.append(
                "%s & %d & %d & %d & %d & -- & NOT RUN & -- & -- & -- \\\\"
                % (tag.replace("_", "\\_"), c["minimal"], c["frozen"], c["masked"], c["graded"])
            )
            continue
        s = row["stats"]
        rows.append(
            "%s & %d & %d & %d & %d & %d & %.1f\\,[%.1f,%.1f] & %.1f\\,[%.1f,%.1f] & %.1f & %.2f \\\\"
            % (
                tag.replace("_", "\\_"), c["minimal"], c["frozen"], c["masked"], c["graded"],
                row["n_episodes_scored"],
                s["episode_ofar"]["point"] * 100, s["episode_ofar"]["lo"] * 100, s["episode_ofar"]["hi"] * 100,
                s["supported_utility_rate"]["point"] * 100, s["supported_utility_rate"]["lo"] * 100,
                s["supported_utility_rate"]["hi"] * 100,
                s["tool_calls"]["point"], s["drift_events"]["point"],
            )
        )
    return "\n".join(rows)


# ==========================================================================
# Main
# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=RESULTS)
    args = ap.parse_args()

    import glob
    instance_paths = sorted(glob.glob(BENCH_GLOB))
    print("Computing per-instance oracle ceilings (U(C_oracle-max)) for %d instances..."
          % len(instance_paths), file=sys.stderr)
    ceilings = compute_oracle_ceilings(instance_paths)
    with open(os.path.join(args.out_dir, "oracle_ceiling.json"), "w", encoding="utf-8") as fh:
        json.dump(ceilings, fh, indent=2, sort_keys=True)
    for iid, c in sorted(ceilings.items()):
        print("  %-16s ceiling_level=%d (%s)" % (iid, c["ceiling_level"], c["ceiling_claim"]), file=sys.stderr)

    table_c, missing_c = build_table_c(ceilings)
    frontier, missing_f = build_frontier(ceilings)
    table_e, missing_e = build_table_e(ceilings)

    # Software-review domain exact McNemar (episode OFAR), paired by instance
    # id, PRIME (induced) vs (a) the strongest non-PRIME baseline per model
    # -- mirrors the research-domain comparison below -- and (b) ReAct
    # specifically (the paper's Table C headline comparison), computed by
    # restricting the baseline candidate set to a single arm.
    sw_baseline_arms = ["direct", "rubric", "react", "cove", "reflexion", "static_contract"]
    mcnemar_sw_best = compare_mcnemar(table_c, TABLE_C_MODELS, sw_baseline_arms)
    mcnemar_sw_vs_react = compare_mcnemar(table_c, TABLE_C_MODELS, ["react"])

    with open(os.path.join(args.out_dir, "tableC.json"), "w", encoding="utf-8") as fh:
        json.dump(
            dict(table_c, **{
                "_mcnemar_best_baseline": mcnemar_sw_best,
                "_mcnemar_vs_react": mcnemar_sw_vs_react,
            }),
            fh, indent=2, sort_keys=True,
        )
    with open(os.path.join(args.out_dir, "frontier.json"), "w", encoding="utf-8") as fh:
        json.dump(frontier, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "tableE.json"), "w", encoding="utf-8") as fh:
        json.dump(table_e, fh, indent=2, sort_keys=True)

    print("\n# Table C (matched budget = 20 tool calls; oracle-scored)\n")
    print(render_table_c_markdown(table_c))
    print("\n# Software-review domain: exact McNemar, PRIME (induced) vs strongest baseline\n")
    print(render_mcnemar_markdown(mcnemar_sw_best))
    print("\n# Software-review domain: exact McNemar, PRIME (induced) vs ReAct\n")
    print(render_mcnemar_markdown(mcnemar_sw_vs_react))
    print("\n## Table C LaTeX rows\n")
    print(render_table_c_latex(table_c))

    print("\n# E3 cost-support frontier (gpt-4.1)\n")
    print(render_frontier_markdown(frontier))

    print("\n# Table E: 2^4 mechanism-interaction ablation (half-fraction, gpt-4.1)\n")
    print(render_table_e_markdown(table_e))
    print("\n## Table E LaTeX rows\n")
    print(render_table_e_latex(table_e))

    # ----------------------------------------------------------------------
    # Research-review domain (bench/instances/res/): Table C, oracle-scored
    # exactly parallel to the software-review Table C above, plus paired
    # exact McNemar (episode OFAR) vs the strongest non-PRIME baseline.
    # ----------------------------------------------------------------------
    research_instance_paths = sorted(glob.glob(RESEARCH_BENCH_GLOB))
    print(
        "\nComputing per-instance oracle ceilings for %d research instances..."
        % len(research_instance_paths),
        file=sys.stderr,
    )
    research_ceilings = compute_oracle_ceilings(research_instance_paths)
    with open(os.path.join(args.out_dir, "oracle_ceiling_research.json"), "w", encoding="utf-8") as fh:
        json.dump(research_ceilings, fh, indent=2, sort_keys=True)
    for iid, c in sorted(research_ceilings.items()):
        print("  %-16s ceiling_level=%d (%s)" % (iid, c["ceiling_level"], c["ceiling_claim"]), file=sys.stderr)

    table_c_research, missing_cr = build_research_table_c(research_ceilings)
    baseline_arms = ["direct", "rubric", "react", "cove", "reflexion", "static_contract"]
    mcnemar_research = compare_mcnemar(table_c_research, RESEARCH_TABLE_C_MODELS, baseline_arms)

    tableC_research_out = {"table": table_c_research, "mcnemar": mcnemar_research}
    with open(os.path.join(args.out_dir, "tableC_research.json"), "w", encoding="utf-8") as fh:
        json.dump(tableC_research_out, fh, indent=2, sort_keys=True)

    print("\n# Research-review domain Table C (matched budget = 20 tool calls; oracle-scored)\n")
    print(render_table_c_markdown(table_c_research, RESEARCH_TABLE_C_MODELS, TABLE_C_CONFIGS))
    print("\n## Research Table C LaTeX rows\n")
    print(render_table_c_latex(table_c_research, RESEARCH_TABLE_C_MODELS, TABLE_C_CONFIGS))
    print("\n# Research-review domain: exact McNemar, PRIME (induced) vs strongest baseline\n")
    print(render_mcnemar_markdown(mcnemar_research))

    all_missing = missing_c + missing_f + missing_e + missing_cr
    if all_missing:
        print("\n# NOT RUN / missing (reported honestly, never fabricated):\n", file=sys.stderr)
        for m in all_missing:
            print("  - %s" % m, file=sys.stderr)
    print(
        "\nWrote results/tableC.json, results/frontier.json, results/tableE.json, "
        "results/oracle_ceiling.json, results/tableC_research.json, "
        "results/oracle_ceiling_research.json",
        file=sys.stderr,
    )
    return 1 if all_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
