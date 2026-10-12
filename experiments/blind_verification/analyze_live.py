#!/usr/bin/env python3
"""Score and analyze the live blind-verification run against the original
PRIME logs (results/w2text, historical baseline) and the offline replay
(experiments/risk_aware_prime/results/stage_c_replay.json, from the
risk-aware study, which is not part of this package).

Scoring is the paper's (analysis/build_w2a_tables.score, restated in the
risk-aware study's common.score): false authorization if any emitted
conclusion is unsupported; utility 100 x level of the strongest supported
emitted conclusion / 3; abstention when nothing is emitted. Gold values
are read here, after the run, for scoring and for the error audit only.

Writes results/analysis.json and prints every table of the report.
Usage: python3 analyze_live.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
RAP = os.path.join(PILOT, "experiments", "risk_aware_prime")
for _p in (PILOT, os.path.join(PILOT, "runner"), RAP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C  # noqa: E402  (the risk-aware study's read-only loaders)

LIVE = os.path.join(HERE, "results")
MODELTAG = "qwen2.5-32b"


def load_live(tasks, dom):
    """Live episodes in the same shape as common.load_episodes, plus the
    verification sidecars."""
    rd = os.path.join(LIVE, "%s-%s" % (MODELTAG, dom), "prime_guided_verified__B20")
    summary = json.load(open(os.path.join(rd, "summary.json"), encoding="utf-8"))
    out = []
    for e in summary["episodes"]:
        iid = e["instance"]; t = tasks[iid]
        rows = [json.loads(l) for l in open(os.path.join(rd, iid + ".jsonl"), encoding="utf-8") if l.strip()]
        inst = C.load_instance(t["path"])
        ledger = C.Ledger(inst.atom_ids, inst.phi_patterns)
        atts, attempts, actions, emit_steps = [], [], [], set()
        for r in rows:
            ty = r.get("type")
            if ty == "action":
                actions.append({"step": r["step"], "name": r["name"], "params": r.get("params", {})})
                if r["name"] == "emit_claim":
                    emit_steps.add(r["step"])
            elif ty == "observation" and r.get("action") == "attest_atom":
                d = r.get("data") or {}
                attempts.append({"step": r["step"], "atom": r["params"].get("atom_id"), "value": r["params"].get("value"),
                                 "quote": r["params"].get("quote", ""), "accepted": bool(d.get("accepted")),
                                 "reason": d.get("reason", "")})
            elif ty == "attestation":
                a = r["attestation"]; ledger.apply(dict(a))
                if a.get("tool") == "attest_atom":
                    lit = t["literal_of"].get(a["atom"], a["atom"])
                    atts.append({"step": a["step"], "atom": a["atom"], "literal": lit, "value": a["value"],
                                 "quote": (a.get("params") or {}).get("quote", ""), "gold": t["gold"].get(a["atom"]),
                                 "correct": a["value"] == t["gold"].get(a["atom"]),
                                 "satisfies": a["value"] == C.satisfying_value(lit)})
        emitted = [x["claim"] for x in (e.get("emitted") or [])]
        vp = os.path.join(rd, "_verify", iid + ".json")
        events = json.load(open(vp, encoding="utf-8"))["events"] if os.path.exists(vp) else []
        out.append({"instance": iid, "domain": dom, "arm": "prime_guided_verified", "emitted": emitted,
                    "emission_kind": ("model" if {x.get("step") for x in e.get("emitted", [])} & emit_steps else "fallback") if emitted else "none",
                    "tool_calls": e.get("tool_calls", 0), "tokens": e.get("tokens", {}), "attestations": atts,
                    "attempts": attempts, "actions": actions, "ledger_states": ledger.states(), "ledger": ledger,
                    "instance_obj": inst, "guided_holds": e.get("guided_holds", 0), "guided_stalls": e.get("guided_stalls", 0),
                    "verification": e.get("verification", {}), "events": events, "error": e.get("error", ""),
                    "score": C.score(t, emitted)})
    return out


def cell(eps: List[Dict[str, Any]], tasks) -> Dict[str, Any]:
    n = len(eps)
    fa = sum(e["score"]["ofar"] for e in eps); emit = sum(1 - e["score"]["abstain"] for e in eps)
    att = [a for e in eps for a in e["attestations"]]
    wrong = sum(1 for a in att if not a["correct"])
    return {"n": n, "fa_count": fa, "fa": fa / n, "fa_wilson95": C.wilson(fa, n),
            "fa_given_emit": (fa / emit) if emit else None, "coverage": emit / n,
            "utility": sum(e["score"]["utility"] for e in eps) / n,
            "utility_ci": C.mean_ci([e["score"]["utility"] for e in eps]),
            "calls": sum(e["tool_calls"] for e in eps) / n,
            "prompt_tokens": sum(e["tokens"].get("prompt_tokens", 0) for e in eps) / n,
            "completion_tokens": sum(e["tokens"].get("completion_tokens", 0) for e in eps) / n,
            "accepted_attestations": len(att), "wrong_accepted": wrong,
            "resolver_error": (wrong / len(att)) if att else None,
            "episodes_with_wrong": sum(1 for e in eps if any(not a["correct"] for a in e["attestations"])) / n,
            "levels": dict(collections.Counter((tasks[e["instance"]]["levels"][e["emitted"][0]] if e["emitted"] else 0) for e in eps)),
            "fa_explained_by_wrong_attestation": sum(1 for e in eps if e["score"]["ofar"] and any(not a["correct"] for a in e["attestations"]))}


def main() -> int:
    tasks = C.load_tasks()
    orig = {d: C.load_episodes(tasks, "prime_guided", d) for d in C.DOMS}
    live = {d: load_live(tasks, d) for d in C.DOMS}
    out: Dict[str, Any] = {"cells": {}, "contrasts": {}, "verification": {}, "behavior": {}, "audit": {}}

    # 1. cells, pooled and per domain
    print("== results (paper scorer); original PRIME = historical Experiment D logs")
    for name, src in (("original PRIME", orig), ("live PRIME + blind verification", live)):
        for key, eps in (("all", [e for d in C.DOMS for e in src[d]]), ("clinical", src["clinical"]), ("security", src["security"])):
            c = cell(eps, tasks); out["cells"]["%s|%s" % (name, key)] = c
            print("  %-34s %-9s n=%3d FA=%2d (%4.1f%% [%4.1f,%4.1f]) FA|emit=%s cov=%.3f util=%5.1f [%5.1f,%5.1f] calls=%.2f att=%3d wrong=%2d (%s) ep_wrong=%.3f levels=%s" % (
                name, key, c["n"], c["fa_count"], 100 * c["fa"], 100 * c["fa_wilson95"][0], 100 * c["fa_wilson95"][1],
                "-" if c["fa_given_emit"] is None else "%.3f" % c["fa_given_emit"], c["coverage"], c["utility"],
                c["utility_ci"]["lo"], c["utility_ci"]["hi"], c["calls"], c["accepted_attestations"], c["wrong_accepted"],
                "-" if c["resolver_error"] is None else "%.1f%%" % (100 * c["resolver_error"]), c["episodes_with_wrong"], c["levels"]))
    # replay reference (verify-all, point, alpha 1.0, cal + eval halves of the risk-aware study)
    rp = json.load(open(os.path.join(RAP, "results", "stage_c_replay.json"), encoding="utf-8"))
    rk = "verify-all:qwen2.5-32b:blind|point|1.00"
    rep = {"fa_count": rp["eval"][rk]["fa_count"] + rp["cal_insample"][rk]["fa_count"],
           "utility": (rp["eval"][rk]["utility"] + rp["cal_insample"][rk]["utility"]) / 2,
           "calls": (rp["eval"][rk]["calls"] + rp["cal_insample"][rk]["calls"]) / 2,
           "verify_calls": rp["eval"][rk]["verify_calls"] + rp["cal_insample"][rk]["verify_calls"],
           "abstain": (rp["eval"][rk]["abstain"] + rp["cal_insample"][rk]["abstain"]) / 2}
    out["replay_reference"] = rep
    print("  %-34s %-9s FA=%d (%.1f%%) util=%.1f calls(replay accounting)=%.2f verifier calls=%d" % (
        "offline replay (verify-all)", "all", rep["fa_count"], 100 * rep["fa_count"] / 240, rep["utility"], rep["calls"], rep["verify_calls"]))

    # 2. paired contrasts live vs original, over the 240 tasks (record level)
    for key in ("utility", "ofar"):
        prs = []
        for d in C.DOMS:
            o = {e["instance"]: e for e in orig[d]}
            for e in live[d]:
                if e["instance"] in o:
                    prs.append((e["score"][key], o[e["instance"]]["score"][key]))
        b = C.paired_bootstrap(prs)
        scale = 100 if key == "ofar" else 1
        out["contrasts"]["live-original|%s" % key] = {"delta": scale * b["delta"], "ci": [scale * b["lo"], scale * b["hi"]], "n": b["n"]}
        print("  live - original %-8s delta=%+.1f [%.1f, %.1f] n=%d" % (key, scale * b["delta"], scale * b["lo"], scale * b["hi"], b["n"]))
    # exact McNemar on FA (discordant pairs)
    b01 = b10 = 0
    for d in C.DOMS:
        o = {e["instance"]: e for e in orig[d]}
        for e in live[d]:
            a, bb = e["score"]["ofar"], o[e["instance"]]["score"]["ofar"]
            b01 += int(a == 1 and bb == 0); b10 += int(a == 0 and bb == 1)
    out["contrasts"]["fa_discordant"] = {"live_only": b01, "original_only": b10}
    print("  FA discordant pairs: live-only=%d original-only=%d" % (b01, b10))

    # 3. verification cost
    print("\n== verification overhead (live)")
    for key, eps in (("all", [e for d in C.DOMS for e in live[d]]), ("clinical", live["clinical"]), ("security", live["security"])):
        v = [e["verification"] for e in eps]; n = len(eps)
        inv = sum(x.get("invocations", 0) for x in v); cons = sum(x.get("consultations", 0) for x in v)
        agreed = sum(x.get("agreed", 0) for x in v); rej = sum(x.get("rejected", 0) for x in v)
        pt = sum(x.get("prompt_tokens", 0) for x in v); ct = sum(x.get("completion_tokens", 0) for x in v)
        lat = sum(x.get("latency_s", 0) for x in v); wall = sum(x.get("wall_s", 0) for x in v)
        verd = collections.Counter(); [verd.update(x.get("verdicts", {})) for x in v]
        attempts = sum(len(e["attempts"]) for e in eps)
        rej_quote = sum(1 for e in eps for a in e["attempts"] if not a["accepted"] and "verbatim" in a["reason"])
        rej_ver = sum(1 for e in eps for a in e["attempts"] if not a["accepted"] and "independent" in a["reason"])
        out["verification"][key] = {"n": n, "invocations": inv, "consultations": cons, "agreed": agreed, "rejected": rej,
                                    "verdicts": dict(verd), "prompt_tokens": pt, "completion_tokens": ct,
                                    "latency_s": lat, "wall_s": wall, "attest_attempts": attempts,
                                    "rejected_quote": rej_quote, "rejected_verifier": rej_ver,
                                    "accepted": sum(1 for e in eps for a in e["attempts"] if a["accepted"])}
        print("  %-9s attest attempts=%d accepted=%d rejected(quote)=%d rejected(verifier)=%d | verifier invocations=%d (%.2f/ep) consultations=%d agreed=%d verdicts=%s | verifier tokens prompt=%d (%.0f/ep) completion=%d (%.0f/ep) latency=%.0fs (%.2fs/ep) | agent prompt tok/ep=%.0f wall/ep=%.1fs" % (
            key, attempts, out["verification"][key]["accepted"], rej_quote, rej_ver, inv, inv / n, cons, agreed, dict(verd),
            pt, pt / n, ct, ct / n, lat, lat / n, sum(e["tokens"].get("prompt_tokens", 0) for e in eps) / n, wall / n))

    # 4. behaviour after a verifier rejection
    print("\n== behaviour after verifier rejections")
    beh = collections.Counter(); changed = []
    for d in C.DOMS:
        o = {e["instance"]: e for e in orig[d]}
        for e in live[d]:
            rejs = [a for a in e["attempts"] if not a["accepted"] and "independent" in a["reason"]]
            if not rejs:
                continue
            beh["episodes_with_rejection"] += 1
            first = rejs[0]
            later = [a for a in e["attempts"] if a["step"] > first["step"] and a["atom"] == first["atom"]]
            beh["reattested_same_atom"] += int(bool(later))
            beh["flipped_value_same_atom"] += int(any(a["value"] != first["value"] for a in later))
            beh["flip_accepted"] += int(any(a["value"] != first["value"] and a["accepted"] for a in later))
            beh["same_value_retry_accepted"] += int(any(a["value"] == first["value"] and a["accepted"] for a in later))
            other = [a for a in e["attempts"] if a["step"] > first["step"] and a["atom"] != first["atom"]]
            beh["moved_to_other_atom"] += int(bool(other))
            beh["ended_fallback"] += int(e["emission_kind"] == "fallback")
            beh["abstained"] += int(not e["emitted"])
            t = tasks[e["instance"]]
            lv_live = t["levels"][e["emitted"][0]] if e["emitted"] else 0
            oe = o.get(e["instance"]); lv_orig = t["levels"][oe["emitted"][0]] if oe and oe["emitted"] else 0
            beh["final_level_lower_than_original"] += int(lv_live < lv_orig)
            beh["final_level_higher_than_original"] += int(lv_live > lv_orig)
            beh["rejection_prevented_an_original_FA"] += int(oe is not None and oe["score"]["ofar"] == 1 and e["score"]["ofar"] == 0)
            for r in rejs:
                beh["rejections_total"] += 1
                beh["rejection_of_wrong_value"] += int(r["value"] != t["gold"].get(r["atom"]))
                beh["rejection_of_right_value"] += int(r["value"] == t["gold"].get(r["atom"]))
            changed.append({"instance": e["instance"], "domain": d, "rejected": [(r["atom"], r["value"], r["value"] == t["gold"].get(r["atom"])) for r in rejs],
                            "live_emitted": e["emitted"], "orig_emitted": oe["emitted"] if oe else None, "live_fa": e["score"]["ofar"],
                            "orig_fa": oe["score"]["ofar"] if oe else None, "reattested": bool(later), "flipped": any(a["value"] != first["value"] for a in later)})
    out["behavior"] = {"counts": dict(beh), "episodes": changed}
    for k, v in sorted(beh.items()):
        print("  %-36s %d" % (k, v))
    # final-conclusion agreement with the original run over all tasks
    same = lower = higher = 0
    for d in C.DOMS:
        o = {e["instance"]: e for e in orig[d]}
        for e in live[d]:
            t = tasks[e["instance"]]; oe = o[e["instance"]]
            a = t["levels"][e["emitted"][0]] if e["emitted"] else 0; b = t["levels"][oe["emitted"][0]] if oe["emitted"] else 0
            same += a == b; lower += a < b; higher += a > b
    out["behavior"]["final_level_vs_original"] = {"same": same, "lower": lower, "higher": higher}
    print("  final level vs original run (240 tasks): same=%d lower=%d higher=%d" % (same, lower, higher))

    # 5. audit: every live false authorization and the two replay survivors
    print("\n== audit (gold read post hoc)")
    audit = []
    watch = {"sec-v2-7127", "sec-v2-7145"}
    for d in C.DOMS:
        for e in live[d]:
            if e["score"]["ofar"] or e["instance"] in watch:
                t = tasks[e["instance"]]
                rec = list(t["records"].values())[0]
                txt = rec.get("details") if d == "security" else rec["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
                audit.append({"instance": e["instance"], "domain": d, "live_fa": e["score"]["ofar"], "emitted": e["emitted"],
                              "ceiling": t["ceiling"],
                              "attestations": [(a["atom"], a["literal"], a["value"], a["gold"], a["correct"], a["quote"][:120]) for a in e["attestations"]],
                              "attempts": [(a["step"], a["atom"], a["value"], a["accepted"], a["reason"][:40]) for a in e["attempts"]],
                              "verifier": [(ev["atom"], ev["attested"], ev["verdict"], ev["raw"][:160]) for ev in e["events"]],
                              "conditions": {a: t["hints"][a].get("what") for a in t["text_atoms"]},
                              "text": " ".join(txt.split())[:700]})
    out["audit"] = audit
    for a in audit:
        print("  -- %s (%s) live_fa=%d emitted=%s ceiling=%d" % (a["instance"], a["domain"], a["live_fa"], a["emitted"], a["ceiling"]))
        for x in a["attestations"]:
            print("     accepted %s %s value=%s gold=%s correct=%s quote=%r" % x)
        for x in a["verifier"]:
            print("     verifier atom=%s attested=%s verdict=%s :: %s" % x)
        print("     text: %s" % a["text"][:400])
    path = os.path.join(LIVE, "analysis.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
