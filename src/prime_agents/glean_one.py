#!/usr/bin/env python3
"""GLEAN's verdict on given trajectories, with GLEAN's own calibration.

src/prime_agents/glean.py verifies every logged trajectory of the fully informed agent at
one budget anchor. When only some trajectories are rerun, this script gives the
verdict on those alone: it rebuilds the calibration exactly as glean.py does (the
same 100 development trajectories, seed, judge prompts, and Bayesian logistic
regression; the judge cache answers every prompt glean.py already sent), scores
each given trajectory, applies the differential check where the entropy rule
fires, and writes the fields glean.py stores in the episode's `ext` entry.

Usage (with PRIME_LLM_BASE pointing at the served model):
  python3 src/prime_agents/glean_one.py --anchor bmin --domain clinical \
      --episode-dir <dir with summary.json and <instance>.jsonl> --out verdict.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import glean as G  # noqa: E402


def calibration(judge, pool, tag: str, calib_anchor: str):
    dev = G.load_episodes(G._PILOT, tag, calib_anchor, "v2dev")
    rng = random.Random(20261011)
    chosen = []
    for dom in G.DOMAINS:
        cand = [e for e in dev if e["_domain"] == dom and e.get("emitted")]
        rng.shuffle(cand)
        chosen += cand[:G.N_CALIB // len(G.DOMAINS)]
    outer = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    cal = [r for r in outer.map(lambda e: G.score_episode(e, judge, pool), chosen) if r]
    return G.fit_calibrator([G.features(r["scores"]) for r in cal], [r["Z"] for r in cal])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("PRIME_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
    ap.add_argument("--modeltag", default="qwen2.5-32b")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--calib-anchor", default="B20")
    ap.add_argument("--domain", required=True, choices=G.DOMAINS)
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args(argv)
    judge = G.Judge(a.model, os.path.join(G._PILOT, "results", "glean", "judge_cache_%s.jsonl" % a.modeltag))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel)
    draws = calibration(judge, pool, a.modeltag, a.calib_anchor)
    calib_calls = judge.calls
    with open(os.path.join(a.episode_dir, "summary.json"), encoding="utf-8") as fh:
        episodes = json.load(fh)["episodes"]
    out = {}
    for e in episodes:
        e = dict(e)
        e["_dir"], e["_domain"] = a.episode_dir, a.domain
        ext = {"llm_calls": 0, "base_emitted": [x["claim"] for x in e.get("emitted") or []]}
        r = None if e.get("error") else G.score_episode(e, judge, pool)
        if r is not None:
            p0 = G.predict(draws, G.features(r["scores"]))
            H = G.entropy_nats(p0)
            triggered = int(H > G.EPS_U)
            p1 = G.predict(draws, G.features(G.active_scores(r, judge, pool))) if triggered else p0
            n_calls = r["k"] * r["n_steps"] + (len(r["comp_docs"]) * r["n_steps"] if triggered else 0)
            ext.update({"llm_calls": n_calls, "p_hat": round(p1, 4), "p_hat_before_active": round(p0, 4),
                        "entropy": round(H, 4), "triggered": triggered, "accepted": int(p1 >= 0.5),
                        "n_steps": r["n_steps"], "k_guidelines": r["k"]})
        out[e["instance"]] = ext
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"anchor": a.anchor, "calibration_judge_calls": calib_calls,
                   "judge_calls": judge.calls, "episodes": out}, fh, indent=1, sort_keys=True)
    print("GLEAN_ONE_DONE calibration_calls=%d total_calls=%d %s" % (calib_calls, judge.calls, json.dumps(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
