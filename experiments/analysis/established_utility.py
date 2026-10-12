#!/usr/bin/env python3
"""Utility crediting only established conclusions (results/established_utility_hesp.json).

Supported utility credits a stated conclusion whenever the record supports
it. PRIME's premise is stricter: a conclusion should be stated only when the
system's own recorded checks establish it. This credits a stated conclusion
only if it is supported AND established by the system's own reads (replayed
with the release scorer, never self-reported), and compares PRIME with
completion with HESP on paired tasks with the frozen family-cluster
statistics.

Usage: python3 experiments/analysis/established_utility.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), os.path.join(PILOT, "release"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402
import obligationbench as ob  # noqa: E402
import stats_v5 as st  # noqa: E402


def per_task(name: str, tasks) -> dict:
    out = {}
    for dom in bw.DOMAINS:
        run = os.path.join(bw.RESULTS, "w2a", "qwen2.5-32b-%s" % dom, name)
        for e in json.load(open(os.path.join(run, "summary.json"), encoding="utf-8"))["episodes"]:
            f = bw.FACTS.get(e["instance"])
            if f is None or f["split"] != "heldout_balanced" or e.get("error"):
                continue
            tp = os.path.join(run, e["instance"] + ".jsonl")
            reads = [[str(r["params"].get("record_id", "")), str(r["params"].get("field_path", ""))]
                     for r in map(json.loads, open(tp, encoding="utf-8"))
                     if r.get("type") == "action" and r.get("name") == "inspect_record"]
            t = tasks[e["instance"]]
            lv = {c["id"]: c["level"] for c in t["claims"]}
            env = ob.RecordEnv(t)
            for rid, fp in reads:
                env.inspect_record(rid, fp)
            est, supp = ob.established(t, env.ledger), ob.supported(t)
            best = max([lv[c] for c in (x["claim"] for x in e.get("emitted") or [])
                        if supp.get(c) and c in est] + [0])
            out[e["instance"]] = {"u": 100.0 * best / 3, "facts": f}
    return out


def main() -> int:
    tasks = {}
    for dom in bw.DOMAINS:
        tasks.update(ob.load_tasks(os.path.join(PILOT, "data", "obligationbench", dom, "heldout_balanced")))
    p, h = per_task("prime_complete__B20", tasks), per_task("ext_hesp__B20", tasks)
    rows = [{"record": p[i]["facts"]["record"], "family": p[i]["facts"]["family"],
             "domain": p[i]["facts"]["domain"], "topology": p[i]["facts"]["topology"],
             "a": p[i]["u"], "b": h[i]["u"]} for i in sorted(set(p) & set(h))]
    rz = st.template_cluster_randomization(rows)
    ci = st.hierarchical_cluster_bootstrap(rows)
    out = {"prime_complete": round(sum(r["a"] for r in rows) / len(rows), 2),
           "hesp": round(sum(r["b"] for r in rows) / len(rows), 2),
           "delta_family_mean": round(rz["statistic"], 2), "ci": [round(ci["lo"], 2), round(ci["hi"], 2)],
           "p_randomization": rz["p"], "n_instances": len(rows),
           "tasks_prime_better": sum(1 for r in rows if r["a"] > r["b"]),
           "tasks_hesp_better": sum(1 for r in rows if r["b"] > r["a"])}
    with open(os.path.join(PILOT, "results", "established_utility_hesp.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
