#!/usr/bin/env python3
"""Proof-or-Stop and ResearchLoop under PRIME's stall cap (results/stall_cap_replay.json).

PRIME's runtime answers at most three turns without an action per episode;
the fourth ends the episode and the runtime states the strongest authorized
conclusion. The Proof-or-Stop arm instead ends the loop after more than
three CONSECUTIVE such turns (the counter resets after any real action), and
the ResearchLoop arm allows two failed attempts per task. This replays every
logged episode of the two arms, truncated at its fourth turn without an
action, and states what each system would have reported at that point (the
highest stage its gate admits / the strongest admitted claim, both computed
from the attestations recorded up to that step), so the systems can be
compared with PRIME under PRIME's own cap. No model calls.

Usage: python3 experiments/analysis/stall_cap_replay.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_w2a_tables as bw  # noqa: E402
import stats_v5 as st  # noqa: E402

CAP = 3


def replay(name: str):
    out = {}
    for dom in bw.DOMAINS:
        run = os.path.join(bw.RESULTS, "w2a", "qwen2.5-32b-%s" % dom, name)
        p = os.path.join(run, "summary.json")
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            f = bw.FACTS.get(e.get("instance"))
            tp = os.path.join(run, e["instance"] + ".jsonl")
            if e.get("error") or f is None or not os.path.exists(tp):
                continue
            rows = [json.loads(l) for l in open(tp, encoding="utf-8") if l.strip()]
            stalls = [r["step"] for r in rows if r.get("type") == "agent_text"]
            cut = stalls[CAP] if len(stalls) > CAP else None
            vals = {}
            for r in rows:
                if r.get("type") == "attestation" and (cut is None or r["step"] <= cut):
                    a = r["attestation"]
                    vals[a["atom"]] = a["value"]
            inst = json.load(open(f_path(e["instance"]), encoding="utf-8"))
            lv = {c["id"]: c["level"] for c in inst["claims"]}
            held = [c for c, rs in inst["gold_licenses"].items()
                    if any(all((vals.get(x[1:]) == "F") if x.startswith("!") else (vals.get(x) == "T") for x in r) for r in rs)]
            # order-coherent programs: the strongest held claim is the report
            stated = [max(held, key=lambda c: lv[c])] if held else []
            s = bw.score(dict(e, emitted=[{"claim": c} for c in stated]))
            s_orig = bw.score(e)
            out[e["instance"]] = {"capped": s["utility"], "orig": s_orig["utility"], "fa": s["ofar_a"],
                                  "truncated": int(cut is not None), "facts": f}
    return out


_PATHS = {}


def f_path(iid):
    if not _PATHS:
        import glob
        for p in glob.glob(os.path.join(PILOT, "data", "obligationbench", "*", "*", "*.json")):
            _PATHS[os.path.basename(p)[:-5]] = p
    return _PATHS[iid]


def main() -> int:
    res = {}
    prime = {}
    for dom in bw.DOMAINS:
        for e in bw.load("qwen2.5-32b", dom, "prime_guided__B20"):
            s = bw.score(e)
            if s:
                prime[e["instance"]] = s["utility"]
    for arm in ("ext_prooforstop", "ext_researchloop"):
        d = replay("%s__B20" % arm)
        if not d:
            res[arm] = {"status": "NOT RUN"}
            continue
        n = len(d)
        rows = [{"record": v["facts"]["record"], "family": v["facts"]["family"], "domain": v["facts"]["domain"],
                 "topology": v["facts"]["topology"], "a": v["capped"], "b": prime[i]} for i, v in d.items() if i in prime]
        rz = st.template_cluster_randomization(rows)
        ci = st.hierarchical_cluster_bootstrap(rows)
        res[arm] = {"n": n, "utility_logged": round(sum(v["orig"] for v in d.values()) / n, 1),
                    "utility_capped": round(sum(v["capped"] for v in d.values()) / n, 1),
                    "fa_capped": round(100 * sum(v["fa"] for v in d.values()) / n, 1),
                    "episodes_truncated": sum(v["truncated"] for v in d.values()),
                    "capped_minus_prime": {"delta": round(rz["statistic"], 1), "ci": [round(ci["lo"], 1), round(ci["hi"], 1)],
                                           "p_randomization": rz["p"], "n_instances": len(rows)}}
        print(arm, res[arm])
    with open(os.path.join(PILOT, "results", "stall_cap_replay.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
