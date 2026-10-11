#!/usr/bin/env python3
"""PRIME's schedule versus HESP's controller (results/tableCONTROLLER.json).

Both are code controllers over the same probes, resolvers, budgets, and
scorer (no model in the loop): PRIME's runtime schedule is the ceiling-
targeting policy (check an open atom of the cheapest live route to the
strongest conclusion that still has one; state the strongest established
conclusion), and HESP's controller picks probes by expected information gain
and concludes when a hypothesis reaches posterior 0.8. Paired family-cluster
contrasts (stats_v5) of PRIME's schedule minus HESP's controller on utility,
premature conclusions, and reads, at the four budget anchors.

Usage: python3 analysis/build_controller_table.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, os.path.join(PILOT, "runner"), os.path.join(PILOT, "release"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402
import obligationbench as ob  # noqa: E402
import stats_v5 as st  # noqa: E402

ARMS = (("shortest_route", "PRIME schedule (runtime)"),
        ("hesp_controller", "HESP controller"),
        ("cheapest_literal", "Cheapest open check"),
        ("random_masked", "Random order"))
ANCHORS = ("bmin", "bmin1", "2bmin", "B20")


def per_task(name: str, tasks) -> dict:
    out = {}
    for dom in bw.DOMAINS:
        run = os.path.join(bw.RESULTS, "w2a", "policy-%s" % dom, name)
        p = os.path.join(run, "summary.json")
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            f = bw.FACTS.get(e.get("instance"))
            if e.get("error") or f is None or f["split"] != "heldout_balanced":
                continue
            tp = os.path.join(run, e["instance"] + ".jsonl")
            if os.path.exists(tp):
                reads = [[str((r.get("params") or {}).get("record_id", "")), str((r.get("params") or {}).get("field_path", ""))]
                         for r in map(json.loads, open(tp, encoding="utf-8"))
                         if r.get("type") == "action" and r.get("name") == "inspect_record"]
                n_reads = None
            else:
                # scripted policies (runner/policies.py) keep no trace: their
                # reads are the fields of the atoms their final ledger
                # resolved, and their read count is their tool-call count
                atoms = {a["id"]: a for a in tasks[e["instance"]]["atoms"]}
                reads = []
                for aid, v in (e.get("final_states") or {}).items():
                    r = (atoms.get(aid) or {}).get("resolver") or {}
                    prm = r.get("params") or {}
                    if v in ("T", "F") and prm.get("field_path"):
                        reads.append([str(prm.get("record_id", "")), str(prm.get("field_path", ""))])
                n_reads = float(e.get("tool_calls", 0))
            stated = [x["claim"] for x in e.get("emitted") or []]
            row = ob.score_task(tasks[e["instance"]], stated, reads)
            if n_reads is not None:
                row["reads"] = n_reads
            out[e["instance"]] = {"record": f["record"], "family": f["family"], "domain": dom,
                                  "topology": f["topology"], "utility": row["utility"],
                                  "premature": 100.0 * row["premature"], "reads": float(row["reads"]),
                                  "fa": 100.0 * row["false_authorization"], "abstain": 100.0 * (1 - row["stated"])}
    return out


def main() -> int:
    tasks = {}
    for dom in bw.DOMAINS:
        tasks.update(ob.load_tasks(os.path.join(PILOT, "bench", "instances_v2", dom, "heldout_balanced")))
    out = {"cells": {}, "contrasts": {}}
    for anchor in ANCHORS:
        data = {arm: per_task("%s__%s" % (arm, anchor), tasks) for arm, _ in ARMS}
        for arm, label in ARMS:
            d = data[arm]
            if not d:
                continue
            n = len(d)
            out["cells"]["%s|%s" % (arm, anchor)] = {
                "label": label, "n": n,
                **{k: round(sum(r[k] for r in d.values()) / n, 2) for k in ("utility", "premature", "reads", "fa", "abstain")}}
        a, b = data["shortest_route"], data["hesp_controller"]
        common = sorted(set(a) & set(b))
        for key in ("utility", "premature", "reads", "abstain"):
            rows = [{"record": a[i]["record"], "family": a[i]["family"], "domain": a[i]["domain"],
                     "topology": a[i]["topology"], "a": a[i][key], "b": b[i][key]} for i in common]
            if len(rows) < 2:
                continue
            rz = st.template_cluster_randomization(rows)
            ci = st.hierarchical_cluster_bootstrap(rows)
            out["contrasts"]["prime_schedule-hesp|%s|%s" % (key, anchor)] = {
                "delta": round(rz["statistic"], 2), "ci": [round(ci["lo"], 2), round(ci["hi"], 2)],
                "p_randomization": rz["p"], "n_clusters": rz["n_clusters"],
                "discordant_clusters": rz.get("n_discordant_clusters"), "n_instances": len(rows),
                "tasks_a_better": sum(1 for r in rows if r["a"] > r["b"]),
                "tasks_b_better": sum(1 for r in rows if r["b"] > r["a"])}
    path = os.path.join(PILOT, "results", "tableCONTROLLER.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for k, c in out["cells"].items():
        print("%-26s n %d util %5.1f  prem %5.1f  FA %4.1f  abst %5.1f  reads %.2f" % (
            k, c["n"], c["utility"], c["premature"], c["fa"], c["abstain"], c["reads"]))
    for k, c in out["contrasts"].items():
        print("%-40s %s" % (k, c))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
