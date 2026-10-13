#!/usr/bin/env python3
"""Closest external systems on SUFFICE (results/tableEXTSYS.json).

The five arms in src/prime_agents/extsys.py reimplement the decision procedures of
AgentSpec, ResearchLoop, Proof-or-Stop, HESP, and GLEAN and run them on the
480 balanced held-out tasks with the same model, tools, resolvers, budget
accounting and scorer as PRIME (src/prime_agents/extsys.py documents each
instantiation). This script scores every arm with the paper's endpoints and
compares each system with PRIME on paired tasks:

  false authorization  a stated conclusion the record does not support
  premature            a stated conclusion the system's own reads did not
                       establish (src/suffice/__init__.py)
  utility              supported utility (level of the strongest supported
                       stated conclusion / 3 x 100)
  emit                 share of answerable tasks with a stated conclusion
  cost                 counted agent actions (the budget unit)
  model calls          LLM requests per episode (external arms record them;
                       for the reference arms experiments/analysis/model_calls.py counts
                       them from the traces: agent turns, reminders after a
                       silent stop, and the forced final question)

Paired contrasts use the frozen family-cluster machinery (stats_v5):
sign-flip randomization p and a hierarchical bootstrap CI, each system minus
PRIME.

Usage: python3 experiments/analysis/build_extsys_table.py [--model qwen2.5-32b] [--anchor B20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), os.path.join(PILOT, "release"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402
import model_calls as mc  # noqa: E402
import suffice as ob  # noqa: E402
import stats_v5 as st  # noqa: E402

EXT = (("ext_agentspec", "AgentSpec, stop"),
       ("ext_agentspec_examine", "AgentSpec, self-examine"),
       ("ext_researchloop", "ResearchLoop"),
       ("ext_prooforstop", "Proof-or-Stop"),
       ("ext_hesp", "HESP"),
       ("ext_glean", "GLEAN"))
REFERENCE = (("react_program_attested", "Fully informed agent"),
             ("prime_flat", "Flat gate (top conclusion only)"),
             ("masked_reactive_v", "PRIME, no guidance"),
             ("prime_guided", "PRIME"),
             ("prime_complete", "PRIME with completion"))


def _tasks():
    tasks = {}
    for dom in bw.DOMAINS:
        tasks.update(ob.load_tasks(os.path.join(PILOT, "data", "suffice", dom, "heldout_balanced")))
    return tasks


def reads_metrics(model: str, name: str, tasks) -> dict:
    """Premature rate and evidence value from the logged reads, with the
    release scorer (it replays inspect_record reads; never self-reports)."""
    preds, llm_calls = [], []
    for dom in bw.DOMAINS:
        run = os.path.join(bw.RESULTS, "w2a", "%s-%s" % (model, dom), name)
        p = os.path.join(run, "summary.json")
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            if e.get("error") or e.get("instance") not in tasks:
                continue
            reads = []
            tp = os.path.join(run, e["instance"] + ".jsonl")
            if not os.path.exists(tp):
                continue
            records = mc.trace_records(tp)
            for r in records:
                if r.get("type") == "action" and r.get("name") == "inspect_record":
                    q = r.get("params") or {}
                    reads.append([str(q.get("record_id", "")), str(q.get("field_path", ""))])
            preds.append({"task_id": e["instance"], "stated": [x["claim"] for x in e.get("emitted", [])],
                          "reads": reads})
            if (e.get("ext") or {}).get("llm_calls") is not None:
                llm_calls.append(e["ext"]["llm_calls"])
            elif not e.get("arm", "").startswith("ext_"):
                # the PRIME runs and the fully informed agent did not record
                # their requests; experiments/analysis/model_calls.py recovers them
                llm_calls.append(mc.model_calls(e, records))
    if not preds:
        return {}
    s = ob.score({p["task_id"]: tasks[p["task_id"]] for p in preds}, preds)["overall"]
    out = {"premature": round(s.get("premature_pct", float("nan")), 1),
           "evidence_value": round(s.get("evidence_value", float("nan")), 1),
           "mean_reads": round(s.get("mean_reads", float("nan")), 2)}
    if llm_calls:
        out["llm_calls"] = round(sum(llm_calls) / len(llm_calls), 2)
    return out


def premature_rows(model: str, name: str, tasks) -> list:
    """Per-task premature indicator (release scorer) with the family id, for
    a family-cluster interval of the premature rate."""
    out = []
    for dom in bw.DOMAINS:
        run = os.path.join(bw.RESULTS, "w2a", "%s-%s" % (model, dom), name)
        p = os.path.join(run, "summary.json")
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            f = bw.FACTS.get(e.get("instance"))
            tp = os.path.join(run, e["instance"] + ".jsonl")
            if e.get("error") or f is None or f["split"] != "heldout_balanced" or not os.path.exists(tp):
                continue
            reads = [[str((r.get("params") or {}).get("record_id", "")), str((r.get("params") or {}).get("field_path", ""))]
                     for r in map(json.loads, open(tp, encoding="utf-8"))
                     if r.get("type") == "action" and r.get("name") == "inspect_record"]
            row = ob.score_task(tasks[e["instance"]], [x["claim"] for x in e.get("emitted") or []], reads)
            out.append({"record": f["record"], "family": f["family"], "domain": dom,
                        "topology": f["topology"], "a": row["premature"], "b": 0,
                        "reads": row["reads"], "instance": e["instance"]})
    return out


def ext_counters(model: str, name: str) -> dict:
    """Mean of each numeric mechanism counter the external arm recorded."""
    acc, n = {}, 0
    for dom in bw.DOMAINS:
        for e in bw.load(model, dom, name):
            if e.get("error"):
                continue
            n += 1
            for k, v in (e.get("ext") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    acc[k] = acc.get(k, 0.0) + v
    return {k: round(v / n, 3) for k, v in sorted(acc.items())} if n else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    ap.add_argument("--anchor", default="B20", choices=("B20", "bmin", "bmin1", "2bmin"))
    a = ap.parse_args(argv)
    tasks = _tasks()
    out = {"split": "heldout_balanced", "anchor": a.anchor, "model": a.model,
           "cells": {}, "contrasts": {}}
    for arm, label in REFERENCE + EXT:
        name = "%s__%s" % (arm, a.anchor)
        c = bw.cell(a.model, name)
        c["label"] = label
        if not c.get("status"):
            c.update(reads_metrics(a.model, name, tasks))
            if arm.startswith("ext_"):
                c["mechanism"] = ext_counters(a.model, name)
        out["cells"][arm] = c
    for refarm in ("prime_guided", "prime_complete"):
        ref = "%s__%s" % (refarm, a.anchor)
        for arm, _ in EXT + (("masked_reactive_v", ""), ("prime_flat", ""), ("react_program_attested", "")):
            for key in ("utility", "ofar_a"):
                prs = bw.paired(a.model, "%s__%s" % (arm, a.anchor), ref, key)
                tag = "%s-%s|%s" % (arm, refarm, key)
                if len(prs) < 2:
                    out["contrasts"][tag] = {"status": "NOT RUN"}
                    continue
                rz = st.template_cluster_randomization(prs)
                ci = st.hierarchical_cluster_bootstrap(prs)
                scale = 100.0 if key == "ofar_a" else 1.0
                out["contrasts"][tag] = {
                    "delta": round(scale * rz["statistic"], 1),
                    "ci": [round(scale * ci["lo"], 1), round(scale * ci["hi"], 1)],
                    "p_randomization": rz["p"], "n_clusters": rz["n_clusters"],
                    "discordant_clusters": rz.get("n_discordant_clusters"),
                    "n_instances": len(prs)}
    ref_reads = {r["instance"]: r["reads"] for r in premature_rows(a.model, "prime_complete__%s" % a.anchor, tasks)}
    for arm, _ in EXT:
        prs = premature_rows(a.model, "%s__%s" % (arm, a.anchor), tasks)
        rr = [dict(r, a=r["reads"], b=ref_reads[r["instance"]]) for r in prs if r["instance"] in ref_reads]
        if len(rr) >= 2:
            rz = st.template_cluster_randomization(rr)
            ci = st.hierarchical_cluster_bootstrap(rr)
            out["contrasts"]["%s-prime_complete|reads" % arm] = {
                "delta": round(rz["statistic"], 2), "ci": [round(ci["lo"], 2), round(ci["hi"], 2)],
                "p_randomization": rz["p"], "n_clusters": rz["n_clusters"], "n_instances": len(rr)}
        if len(prs) >= 2:
            ci = st.hierarchical_cluster_bootstrap(prs)
            out["cells"][arm]["premature_ci"] = [round(100 * ci["lo"], 1), round(100 * ci["hi"], 1)]
            out["cells"][arm]["premature_family_mean"] = round(100 * ci["point"], 1)
    suffix = "" if a.anchor == "B20" else "_" + a.anchor
    path = os.path.join(PILOT, "results", "tableEXTSYS%s.json" % suffix)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for arm, c in out["cells"].items():
        if c.get("status"):
            print("%-24s %s" % (arm, c["status"]))
            continue
        print("%-24s n %3d  FA %5.1f  prem %5s  util %5.1f  emit %5.1f  cost %5.2f  llm %s" % (
            arm, c["n_instances"], c["ofar_a"], c.get("premature", "-"), c["utility"],
            c["emit_rate_avail"], c["cost"], c.get("llm_calls", "-")))
    for k, v in out["contrasts"].items():
        print("%-44s %s" % (k, v))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
