#!/usr/bin/env python3
"""Scripted completion replayed on the logged confirmatory episodes.

The prime_complete arm (src/prime_agents/arms.py) adds one step to PRIME's control
law: when the model stops short of the reachable ceiling with budget left,
the runtime discharges the open obligations itself, since every one names
a registered resolver call. This script applies exactly that step to the
logged episodes of the guided and masked arms at each budget anchor: it
rebuilds the episode's ledger by replaying its logged tool calls through
the real executor, and, if the episode was ended by the runtime (the model
stopped or stalled; no accepted emission of its own), runs the same
discharge routine with the remaining budget. The result is the utility
PRIME would have scored with completion, from the model's own stopping
point, with no new model call. Episodes the model ended with an accepted
emission are unchanged.

Writes results/discharge_replay.json and prints a per-anchor table against
the LLM-free fixed policy (ceiling-targeting) at the same anchors.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import build_w2a_tables as bw  # noqa: E402
import policy_frontier as pf  # noqa: E402
import stats_v5 as st  # noqa: E402
from prime_runtime import auth as prime_auth  # noqa: E402
from prime_runtime import goldledger, status  # noqa: E402
from prime_runtime.interpreter import Interpreter  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

MODEL = "qwen2.5-32b"
ANCHORS = ("bmin", "bmin1", "2bmin", "B20")
ARMS = ("prime_guided", "masked_reactive_v")
BENCH = os.path.join(PILOT, "data", "suffice")
# --split locator: the same replay on the locator-uncertain split
# (results/w2loc, every anchor with a logged cell), where completion
# enumerates the remaining candidate fields of an open atom (src/prime_agents/arms.py).
SPLIT = {"balanced": ("heldout_balanced", "w2a", ANCHORS),
         "locator": ("heldout_locator", "w2loc", ANCHORS),
         "onebranch": ("heldout_onebranch", "w2ob", ("B20",))}


class _NullTrace:
    def log(self, entry): pass
    def log_action(self, *a, **k): pass
    def log_observation(self, *a, **k): pass
    def log_attestation(self, *a, **k): pass


def _trace(run_dir, iid):
    p = os.path.join(run_dir, iid + ".jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def ended_by_runtime(trace):
    """True unless the model's own emit_claim was accepted: an accepted
    emission is logged at the step of its emit_claim action; the runtime's
    MaxAuth statement has no such action at its step."""
    emits = [r for r in trace if r.get("type") == "emission"]
    if not emits:
        return True
    last = emits[-1]
    return not any(r.get("type") == "action" and r.get("name") == "emit_claim"
                   and r.get("step") == last.get("step") for r in trace)


def replay(instance, trace):
    """Rebuild the final ledger by re-executing the logged tool calls."""
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    executor = arms_mod.ToolExecutor(instance)
    calls = 0
    for r in trace:
        if r.get("type") != "action" or r.get("name") in ("emit_claim", "revise_program"):
            continue
        obs = executor.execute(r["name"], dict(r.get("params") or {}))
        if not executor.last_was_cached:
            calls += 1
        for att in goldledger.delta(instance, ledger, r["name"], dict(r.get("params") or {}), obs, int(r.get("step", 0))):
            ledger.apply(att)
    return ledger, executor, calls


def utility_of(instance, ledger, facts):
    levels = {c.id: c.level for c in instance.claims}
    best = 0
    for cid in prime_auth.max_auth(instance.gold_program(), ledger):
        if facts["supp_a"].get(cid):
            best = max(best, levels[cid])
    return 100.0 * best / 3.0


def run(arm, anchor, split="balanced"):
    split_dir, res_dir, _ = SPLIT[split]
    rows = []
    n_runtime_ended = n_discharged = 0
    for dom in bw.DOMAINS:
        run_dir = os.path.join(bw.RESULTS, res_dir, "%s-%s" % (MODEL, dom), "%s__%s" % (arm, anchor))
        sp = os.path.join(run_dir, "summary.json")
        if not os.path.exists(sp):
            continue
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            f = bw.FACTS.get(e.get("instance"))
            if not f or f["split"] != split_dir or e.get("error"):
                continue
            s = bw.score(e)
            path = os.path.join(BENCH, dom, split_dir, e["instance"] + ".json")
            instance = load_instance(path)
            tr = _trace(run_dir, e["instance"])
            ledger, executor, calls = replay(instance, tr)
            logged_states = e.get("final_ledger") or {}
            if logged_states and any(ledger.states().get(k) != v for k, v in logged_states.items()):
                raise SystemExit("replay mismatch on %s/%s" % (anchor, e["instance"]))
            budget = int(e.get("max_tool_calls") or 20)
            used = int(e.get("tool_calls") or 0)
            completed = s["utility"]
            disch = 0
            if ended_by_runtime(tr) and used < budget:
                n_runtime_ended += 1
                interp = Interpreter(instance, instance.gold_program(), revision_mode="fixed")
                levels = {c.id: c.level for c in instance.claims}

                class _R:  # the fields _discharge_obligations touches
                    tool_calls = used
                    discharged_calls = 0
                res = _R()
                arms_mod._discharge_obligations(instance, interp, ledger, executor, _NullTrace(),
                                                res, levels, budget, 0)
                disch = res.discharged_calls
                n_discharged += int(disch > 0)
                completed = utility_of(instance, ledger, f)
                assert completed >= s["utility"] - 1e-9, (e["instance"], s["utility"], completed)
            rows.append({"record": f["record"], "family": f["family"], "domain": dom,
                         "topology": f["topology"], "instance": e["instance"],
                         "logged": s["utility"], "completed": completed, "discharged_calls": disch,
                         "calls": used, "budget": budget})
    return rows, n_runtime_ended, n_discharged


def _locator_policy(arm, anchor, instances, res_dir="w2loc"):
    """Mean utility of an LLM-free policy on a split, restricted to the given tasks."""
    vals = []
    for dom in bw.DOMAINS:
        sp = os.path.join(bw.RESULTS, res_dir, "policy-%s" % dom, "%s__%s" % (arm, anchor), "summary.json")
        if not os.path.exists(sp):
            continue
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            if e["instance"] in instances:
                s = bw.score(e)
                if s:
                    vals.append(s["utility"])
    return sum(vals) / len(vals) if vals else float("nan")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="balanced", choices=sorted(SPLIT))
    a = ap.parse_args(argv)
    split_dir, res_dir, anchors = SPLIT[a.split]
    fixed = pf.collect("heldout_balanced") if a.split == "balanced" else {}
    out = {"_note": "scripted completion replayed from the logged episodes; see docstring",
           "split": split_dir, "arms": {}}
    print("%-18s %-6s %6s %10s %10s %8s %9s %8s" % ("arm", "anchor", "n", "logged", "+complete", "fixed", "ended-rt", "calls+"))
    arms = ARMS + (("react_program_attested",) if a.split == "onebranch" else ())
    for arm in arms:
        out["arms"][arm] = {}
        for anchor in anchors:
            rows, n_rt, n_d = run(arm, anchor, a.split)
            if not rows:
                continue
            n = len(rows)
            logged = sum(r["logged"] for r in rows) / n
            comp = sum(r["completed"] for r in rows) / n
            paired = [{"record": r["record"], "family": r["family"], "domain": r["domain"],
                       "topology": r["topology"], "a": r["completed"], "b": r["logged"]} for r in rows]
            rz = st.template_cluster_randomization(paired)
            ci = st.hierarchical_cluster_bootstrap(paired)
            inst_ids = {r["instance"] for r in rows}
            if a.split == "balanced":
                fx = fixed.get(("shortest_route", anchor), (float("nan"), 0))[0]
            elif a.split == "locator":
                fx = _locator_policy("shortest_route_candidates", anchor, inst_ids)
            else:
                fx = _locator_policy("shortest_route", anchor, inst_ids, res_dir)
            extra = sum(r["discharged_calls"] for r in rows) / n
            out["arms"][arm][anchor] = {
                "n": n, "utility_logged": round(logged, 1), "utility_completed": round(comp, 1),
                "gain": round(rz["statistic"], 1), "gain_ci": [round(ci["lo"], 1), round(ci["hi"], 1)],
                "p": round(rz["p"], 4), "discordant": rz["n_discordant_clusters"],
                "episodes_ended_by_runtime": n_rt, "episodes_discharged": n_d,
                "discharged_calls_per_episode": round(extra, 2),
                "fixed_policy": round(fx, 1),
                "fixed_policy_name": {"balanced": "ceiling-targeting", "locator": "blind candidate enumeration",
                                      "onebranch": "ceiling-targeting"}[a.split],
                "oracle_locators": (round(_locator_policy("shortest_route", anchor, inst_ids), 1)
                                    if a.split == "locator" else None),
                "calls_per_episode_logged": round(sum(r["calls"] for r in rows) / n, 2),
            }
            print("%-18s %-6s %6d %10.1f %10.1f %8.1f %9d %8.2f" % (arm, anchor, n, logged, comp, fx, n_rt, extra))
    path = os.path.join(bw.RESULTS, "discharge_replay%s.json" % ("" if a.split == "balanced" else "_" + a.split))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
