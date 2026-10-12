#!/usr/bin/env python3
"""Contrasts and read statistics for the locator-uncertain split
(results/tableLOC_extra.json), the numbers Experiment C quotes beyond the
cell means of tableLOC.json.

Family-mean paired contrasts (sign-flip randomization, hierarchical
bootstrap CI, discordant families) between PRIME, PRIME with replayed
completion, the mask alone, and the LLM-free policies (blind enumeration of
the candidates, the same policy with the true locators, a locator-blind
random open check); and, at b_min, where PRIME's reads went (true field,
listed decoy, elsewhere) and how often its first read per atom hit the true
field, against the 1/3 a blind first try gets.

Usage: python3 experiments/analysis/build_locator_extra.py
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

import build_w2a_tables as bw  # noqa: E402
import discharge_replay as dr  # noqa: E402
import stats_v5 as st  # noqa: E402

SPLIT = "heldout_locator"
MODEL = "qwen2.5-32b"


def llm_cell(arm, anchor):
    out = {}
    for dom in bw.DOMAINS:
        sp = os.path.join(bw.RESULTS, "w2loc", "%s-%s" % (MODEL, dom), "%s__%s" % (arm, anchor), "summary.json")
        if not os.path.exists(sp):
            continue
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            s = bw.score(e)
            if s and s["facts"]["split"] == SPLIT:
                out[e["instance"]] = s
    return out


def policy_cell(arm, anchor):
    out = {}
    for dom in bw.DOMAINS:
        sp = os.path.join(bw.RESULTS, "w2loc", "policy-%s" % dom, "%s__%s" % (arm, anchor), "summary.json")
        if not os.path.exists(sp):
            continue
        for e in json.load(open(sp, encoding="utf-8"))["episodes"]:
            s = bw.score(e)
            if s and s["facts"]["split"] == SPLIT:
                out[e["instance"]] = s
    return out


def contrast(A, B):
    prs = []
    for iid in sorted(set(A) & set(B)):
        f = A[iid]["facts"]
        prs.append({"record": f["record"], "family": f["family"], "domain": f["domain"],
                    "topology": f["topology"], "a": A[iid]["utility"], "b": B[iid]["utility"]})
    if len(prs) < 2:
        return {"status": "NOT RUN"}
    rz = st.template_cluster_randomization(prs)
    ci = st.hierarchical_cluster_bootstrap(prs)
    return {"delta": round(rz["statistic"], 1), "ci": [round(ci["lo"], 1), round(ci["hi"], 1)],
            "p": round(rz["p"], 4), "discordant": rz["n_discordant_clusters"],
            "n_clusters": rz["n_clusters"], "n_instances": len(prs)}


def completion_cell(anchor):
    rows, _, _ = dr.run("prime_guided", anchor, "locator")
    base = llm_cell("prime_guided", anchor)
    out = {}
    for r in rows:
        s = dict(base[r["instance"]])
        s["utility"] = r["completed"]
        out[r["instance"]] = s
    return out


def reads_at(anchor):
    """Where PRIME's inspect_record reads went and the first read per atom."""
    true_reads = decoy_reads = other_reads = 0
    first_total = first_true = 0
    for dom in bw.DOMAINS:
        run_dir = os.path.join(bw.RESULTS, "w2loc", "%s-%s" % (MODEL, dom), "prime_guided__%s" % anchor)
        for tp in glob.glob(os.path.join(run_dir, "*.jsonl")):
            iid = os.path.basename(tp)[:-6]
            inst = json.load(open(os.path.join(bw.BENCH, dom, SPLIT, iid + ".json"), encoding="utf-8"))
            true_of = {a["id"]: ((a.get("resolver") or {}).get("params") or {}).get("field_path") for a in inst["atoms"]}
            cands = {a["id"]: list((a.get("spec_hint") or {}).get("candidates") or []) for a in inst["atoms"]}
            first_seen = {}
            for line in open(tp, encoding="utf-8"):
                r = json.loads(line)
                if r.get("type") != "action" or r.get("name") != "inspect_record":
                    continue
                path = (r.get("params") or {}).get("field_path")
                owners = [aid for aid, t in true_of.items() if t == path]
                if owners:
                    true_reads += 1
                else:
                    owners = [aid for aid, c in cands.items() if path in c]
                    if owners:
                        decoy_reads += 1
                    else:
                        other_reads += 1
                for aid in owners:
                    if aid not in first_seen:
                        first_seen[aid] = (path == true_of[aid])
            first_total += len(first_seen)
            first_true += sum(1 for v in first_seen.values() if v)
    return {"true": true_reads, "decoy": decoy_reads, "other": other_reads,
            "first_reads": first_total,
            "first_read_true_pct": round(100.0 * first_true / first_total, 1) if first_total else None}


def main() -> int:
    out = {"split": SPLIT, "contrasts": {}}
    prime = {a: llm_cell("prime_guided", a) for a in ("bmin", "2bmin", "B20")}
    mask = {a: llm_cell("masked_reactive_v", a) for a in ("bmin", "2bmin")}
    blind = {a: policy_cell("shortest_route_candidates", a) for a in ("bmin", "2bmin", "B20")}
    true_loc = {a: policy_cell("shortest_route", a) for a in ("bmin", "2bmin", "B20")}
    rnd_blind = {a: policy_cell("random_masked_candidates", a) for a in ("bmin", "2bmin", "B20")}
    comp = {a: completion_cell(a) for a in ("bmin", "2bmin", "B20")}
    C = out["contrasts"]
    for a in ("bmin", "2bmin", "B20"):
        C["%s|prime-vs-blind" % a] = contrast(prime[a], blind[a])
        C["%s|prime-vs-true_locators" % a] = contrast(prime[a], true_loc[a])
        C["%s|prime-vs-random_blind" % a] = contrast(prime[a], rnd_blind[a])
        C["%s|completion-vs-blind" % a] = contrast(comp[a], blind[a])
        C["%s|completion-vs-true_locators" % a] = contrast(comp[a], true_loc[a])
        C["%s|completion-vs-prime" % a] = contrast(comp[a], prime[a])
        if a in mask:
            C["%s|prime-vs-mask" % a] = contrast(prime[a], mask[a])
    out["cells"] = {
        "random_blind": {a: round(sum(s["utility"] for s in rnd_blind[a].values()) / len(rnd_blind[a]), 1)
                         for a in rnd_blind if rnd_blind[a]},
        "completion": {a: round(sum(s["utility"] for s in comp[a].values()) / len(comp[a]), 1) for a in comp},
    }
    out["reads_bmin"] = reads_at("bmin")
    path = os.path.join(bw.RESULTS, "tableLOC_extra.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for k, v in sorted(C.items()):
        print("%-36s %s" % (k, v))
    print("cells", out["cells"])
    print("reads at b_min", out["reads_bmin"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
