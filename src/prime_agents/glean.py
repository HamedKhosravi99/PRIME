#!/usr/bin/env python3
"""GLEAN (Zhang et al., ICML 2026) as a verifier of the fully informed agent.

GLEAN is a post-hoc verifier: it never calls tools and never changes the
agent's run. It scores a completed trajectory step by step against
guidelines with a same-model YES/NO judge, accumulates the evidence over
steps, maps it to a calibrated probability that the final answer is correct
(Bayesian logistic regression fitted on labeled trajectories), and, when that
probability is uncertain, collects more evidence once (differential checks
against competing answers). The paper defines no accept rule; we accept the
agent's conclusion when the calibrated probability is at least 0.5 and
otherwise state nothing.

Here the agent is the fully informed agent of Experiment A (its logged
held-out trajectories, the same ones Table 1 scores), the guidelines are the
committed program's routes for the stated conclusion (one document per
route), the judge is the agent's own model, and the calibrator is fitted on
100 of that agent's trajectories on the development split (25 per domain),
labeled by whether the stated conclusion is supported. Judge calls are model
calls outside the tool budget; the agent's tool calls are unchanged.

Paper constants: one YES/NO token with top-10 log-probabilities (Eq. 3);
Phi = [min, mean] over guidelines (Eq. 4); discounted sum of logits with
beta = 0.5; prior N(0, 1/lambda) with lambda = 1 and 2,000 posterior draws
(Eqs. 5-6); trigger when the binary entropy (nats) exceeds 0.5; differential
checks against two competing guidelines with alpha = 0.2 (Eq. 7); guideline
expansion needs a guideline beyond the conclusion's routes and is not
available here (logged).

Usage (on the GPU host, with the model served at PRIME_LLM_BASE):
  python3 src/prime_agents/glean.py --modeltag qwen2.5-32b --anchor B20 --parallel 16
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

_RUNNER = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(os.path.dirname(_RUNNER))
for _p in (_PILOT, os.path.join(_PILOT, "src"), _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime.schema import load_instance  # noqa: E402

DOMAINS = ("security", "clinical", "drugsafety", "financial")
BETA, ALPHA, LAMBDA, EPS_U = 0.5, 0.2, 1.0, 0.5
N_CALIB, N_DRAWS, N_BURN, CLIP = 100, 2000, 2000, 1e-4
BASE_ARM = "react_program_attested"

# GLEAN Appendix B.1 judge prompt, verbatim except for the marked domain
# nouns (clinician -> records reviewer, diagnosis -> investigation/conclusion,
# patient information -> record information, clinical guideline -> reference
# protocol).
JUDGE_PROMPT = (
    "You are an expert records reviewer and a strict protocol compliance evaluator.\n\n"
    "You will be given: (1) The investigation history so far (prior steps),\n"
    "(2) The current step, including the new observation(s) and the decision/action taken,\n"
    "(3) A reference protocol relevant to the proposed conclusion.\n\n"
    "Your job is NOT to judge the entire final conclusion. Your job is to judge whether the "
    "CURRENT STEP is appropriate and consistent with the protocol, given the available "
    "information up to this step.\n\n"
    "Be strict and conservative:\n"
    "- Answer YES only if the current step is clearly supported by the record information so "
    "far AND is consistent with the protocol.\n"
    "- Answer NO if the step is unsupported, premature, contradicts key facts, violates protocol "
    "requirements, skips required checks, or is not justified as the next best step.\n\n"
    "Reply with exactly one token: YES or NO.\n\n"
    "Investigation so far (prior steps):\n{history}\n\n"
    "Current step:\nObservation(s):\n{observation}\n\n"
    "Rationale & Action:\n{action}\n\n"
    "Reference protocol:\n{guideline}\n\n"
    "Task: Is the CURRENT STEP appropriate and protocol-consistent given the record information so far?\n"
    "Reply with YES or NO only.")


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

def _instance_path(iid: str) -> str:
    hits = glob.glob(os.path.join(_PILOT, "data", "obligationbench", "*", "*", iid + ".json"))
    if not hits:
        raise FileNotFoundError(iid)
    return hits[0]


def _render_action(name: str, params: Dict[str, Any]) -> str:
    if name == "emit_claim":
        return "State conclusion %s (emit_claim)." % params.get("claim_id")
    return "%s(%s)" % (name, ", ".join("%s=%r" % kv for kv in sorted(params.items())))


def trajectory(trace_path: str, inst) -> Dict[str, Any]:
    """GLEAN steps (o_t, a_t) from a logged trajectory: step 1 observes the
    task as shown to the agent; each later step observes the previous tool
    result and the recorded check states the agent then saw; the last action
    states the conclusion."""
    rows = [json.loads(l) for l in open(trace_path, encoding="utf-8") if l.strip()]
    ledger = {a.id: "U" for a in inst.atoms}
    claims = "\n".join("  %s (level %d): %s" % (c.id, c.level, c.text)
                       for c in sorted(inst.claims, key=lambda c: c.level))
    schema = "\n".join("  %s: %s(%s)" % (a.id, a.resolver.tool, ", ".join(
        "%s=%r" % kv for kv in sorted(a.resolver.params.items())))
        for a in inst.atoms if a.resolver is not None)
    first_obs = ("Request: %s\nScope: %s\nCandidate conclusions:\n%s\nChecks and the fields they read:\n%s\n"
                 "Recorded check states: %s" % (inst.user_prompt, inst.scope, claims, schema,
                                                json.dumps(ledger, sort_keys=True)))
    steps: List[Dict[str, str]] = []
    obs_next = first_obs
    stated: List[str] = []
    by_step: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        by_step.setdefault(r.get("step", -1), {}).setdefault(r["type"], []).append(r)
    for st in sorted(k for k in by_step if k is not None and k >= 0):
        ent = by_step[st]
        for act in ent.get("action", []):
            steps.append({"observation": obs_next, "action": _render_action(act["name"], act.get("params") or {})})
            for ob in ent.get("observation", []):
                data = json.dumps(ob.get("data"), default=str)[:400]
                for att in ent.get("attestation", []):
                    a = att["attestation"]
                    ledger[a["atom"]] = a["value"]
                obs_next = "TOOL_RESULT %s: %s\nRecorded check states: %s" % (
                    ob["action"], data, json.dumps(ledger, sort_keys=True))
        for em in ent.get("emission", []):
            stated = list(em.get("claims") or [])
            if (em.get("claim_object") or {}).get("forced_selection"):
                steps.append({"observation": obs_next,
                              "action": "Final answer: state conclusion(s) %s." % ", ".join(stated)})
    return {"steps": steps, "stated": stated}


def guideline_docs(inst, cid: str) -> List[Tuple[str, str]]:
    c = inst.claim(cid)
    by_id = {a.id: a for a in inst.atoms}
    out = []
    for i, route in enumerate(inst.gold_program().routes(cid)):
        lines = []
        for lit in route:
            aid = lit.lstrip("!")
            a = by_id[aid]
            loc = ("%s(%s)" % (a.resolver.tool, ", ".join("%s=%r" % kv for kv in sorted(a.resolver.params.items())))
                   if a.resolver is not None else "no tool can establish this check")
            lines.append("- %s must be recorded %s: %s" % (aid, "F" if lit.startswith("!") else "T", loc))
        out.append(("%s_R%d" % (cid, i + 1),
                    "Protocol for stating conclusion %s (level %d): \"%s\"\n"
                    "%s may be stated only when every check below has been established by a tool "
                    "read in this investigation:\n%s" % (cid, c.level, c.text, cid, "\n".join(lines))))
    return out


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------

class Judge:
    def __init__(self, model: str, cache_path: str) -> None:
        self.model = model
        self.base = os.environ.get("PRIME_LLM_BASE", "").rstrip("/")
        if not self.base or "api.openai.com" in self.base:
            raise SystemExit("set PRIME_LLM_BASE to the locally served model")
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache: Dict[str, float] = {}
        if os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r["s"]
                except ValueError:
                    pass
        self.calls = 0

    def score(self, prompt: str) -> float:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10}
        body = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.base + "/chat/completions",
                                             data=json.dumps(payload).encode("utf-8"),
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": "Bearer EMPTY"}, method="POST")
                with urllib.request.urlopen(req, timeout=300) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except Exception:  # noqa: BLE001 -- retried, then recorded as uninformative
                import time
                time.sleep(2 ** attempt)
        s = 0.5
        if body is not None:
            try:
                top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                py = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().upper() == "YES")
                pn = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().upper() == "NO")
                s = py / (py + pn) if (py + pn) > 0 else 0.5
            except (KeyError, IndexError, TypeError):
                s = 0.5
        s = min(1 - CLIP, max(CLIP, s))
        with self.lock:
            self.cache[key] = s
            self.calls += 1
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "s": s}) + "\n")
        return s


def _logit(p: float) -> float:
    p = min(1 - CLIP, max(CLIP, p))
    return math.log(p / (1 - p))


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def step_prompts(traj: Dict[str, Any], doc: str) -> List[str]:
    out, hist = [], []
    for i, st in enumerate(traj["steps"]):
        history = "\n".join(hist) if hist else "(none)"
        out.append(JUDGE_PROMPT.format(history=history, observation=st["observation"],
                                       action=st["action"], guideline=doc))
        hist.append("Step %d. Observation: %s\nAction: %s" % (i + 1, st["observation"][:600], st["action"]))
    return out


def features(scores: List[List[float]]) -> List[float]:
    """S_T from per-step, per-guideline ratings: Phi = [min, mean] per step,
    then the beta-discounted sum of logits up to the last step."""
    T = len(scores[0]) if scores else 0
    S = [0.0, 0.0]
    for t in range(T):
        col = [g[t] for g in scores]
        phi = [min(col), sum(col) / len(col)]
        w = BETA ** (T - 1 - t)
        S[0] += w * _logit(phi[0])
        S[1] += w * _logit(phi[1])
    return S


# ---------------------------------------------------------------------------
# calibrator: Bayesian logistic regression, random-walk Metropolis
# ---------------------------------------------------------------------------

def fit_calibrator(X: List[List[float]], Z: List[int], seed: int = 20261011) -> List[List[float]]:
    rng = random.Random(seed)

    def logpost(th):
        w1, w2, b = th
        lp = -0.5 * LAMBDA * (w1 * w1 + w2 * w2 + b * b)
        for x, z in zip(X, Z):
            u = w1 * x[0] + w2 * x[1] + b
            lp += -math.log1p(math.exp(-u)) if z else -math.log1p(math.exp(u)) if u < 30 else -u
        return lp

    th = [0.0, 0.0, 0.0]
    cur = logpost(th)
    sd = 0.15
    draws, acc = [], 0
    for it in range(N_BURN + 2 * N_DRAWS):
        prop = [v + rng.gauss(0, sd) for v in th]
        lp = logpost(prop)
        if math.log(rng.random() + 1e-300) < lp - cur:
            th, cur = prop, lp
            acc += 1
        if it >= N_BURN and (it - N_BURN) % 2 == 0:
            draws.append(list(th))
    return draws


def predict(draws: List[List[float]], x: List[float]) -> float:
    return sum(_sig(w1 * x[0] + w2 * x[1] + b) for w1, w2, b in draws) / len(draws)


def entropy_nats(p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, p))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def load_episodes(root: str, tag: str, anchor: str, split_dir: str) -> List[Dict[str, Any]]:
    out = []
    for dom in DOMAINS:
        d = os.path.join(_PILOT, "results", split_dir, "%s-%s" % (tag, dom), "%s__%s" % (BASE_ARM, anchor))
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            raise SystemExit("missing base trajectories: %s" % p)
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            if e.get("error"):
                continue
            e = dict(e)
            e["_dir"], e["_domain"] = d, dom
            out.append(e)
    return out


_GOLD: Dict[str, Dict[str, str]] = {}


def supported(inst, cid: str) -> bool:
    """Whether the record supports conclusion cid (label for calibration and
    diagnostics only; gold values are read from the task file and never
    shown to the judge)."""
    if inst.id not in _GOLD:
        raw = json.load(open(_instance_path(inst.id), encoding="utf-8"))
        _GOLD[inst.id] = {a["id"]: a["gold_value"] for a in raw["atoms"]}
    gv = _GOLD[inst.id]
    return any(all((gv.get(x[1:]) == "F") if x.startswith("!") else (gv.get(x) == "T") for x in r)
               for r in inst.gold_program().routes(cid))


def score_episode(e: Dict[str, Any], judge: Judge, pool) -> Optional[Dict[str, Any]]:
    inst = load_instance(_instance_path(e["instance"]))
    traj = trajectory(os.path.join(e["_dir"], e["instance"] + ".jsonl"), inst)
    stated = [x["claim"] for x in e.get("emitted") or []] or traj["stated"]
    if not stated or not traj["steps"]:
        return None
    y = max(stated, key=lambda c: inst.claim(c).level)
    docs = guideline_docs(inst, y)[:3]
    prompts = [step_prompts(traj, d) for _, d in docs]
    scores = [list(pool.map(judge.score, ps)) for ps in prompts]
    rng = random.Random(e["instance"])
    others = [d for c in inst.claims if c.id != y for d in guideline_docs(inst, c.id)]
    comp = rng.sample(others, min(2, len(others))) if others else []
    return {"instance": e["instance"], "domain": e["_domain"], "stated": stated, "y": y,
            "Z": int(all(supported(inst, c) for c in stated)), "scores": scores,
            "competitors": [cid for cid, _ in comp], "traj": traj, "comp_docs": comp,
            "n_steps": len(traj["steps"]), "k": len(docs)}


def active_scores(rec: Dict[str, Any], judge: Judge, pool) -> List[List[float]]:
    comp_scores = [list(pool.map(judge.score, step_prompts(rec["traj"], d))) for _, d in rec["comp_docs"]]
    if not comp_scores:
        return rec["scores"]
    T = len(rec["scores"][0])
    s_minus = [max(c[t] for c in comp_scores) for t in range(T)]
    return [[_sig(_logit(g[t]) - ALPHA * _logit(s_minus[t])) for t in range(T)] for g in rec["scores"]]


def auroc(ps: Sequence[float], zs: Sequence[int]) -> float:
    pos = [p for p, z in zip(ps, zs) if z]
    neg = [p for p, z in zip(ps, zs) if not z]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def ece(ps: Sequence[float], zs: Sequence[int], bins: int = 10) -> float:
    tot, n = 0.0, len(ps)
    for k in range(bins):
        idx = [i for i, p in enumerate(ps) if (k / bins) <= p < ((k + 1) / bins) or (k == bins - 1 and p == 1.0)]
        if idx:
            tot += len(idx) / n * abs(sum(ps[i] for i in idx) / len(idx) - sum(zs[i] for i in idx) / len(idx))
    return tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("PRIME_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
    ap.add_argument("--modeltag", default="qwen2.5-32b")
    ap.add_argument("--anchor", default="B20")
    ap.add_argument("--calib-anchor", default="B20")
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(_PILOT, "results", "glean"), exist_ok=True)
    judge = Judge(a.model, os.path.join(_PILOT, "results", "glean", "judge_cache_%s.jsonl" % a.modeltag))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel)
    outer = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # calibration: 100 development trajectories with a stated conclusion,
    # 25 per domain, seeded
    dev = load_episodes(_PILOT, a.modeltag, a.calib_anchor, "v2dev")
    rng = random.Random(20261011)
    chosen = []
    for dom in DOMAINS:
        cand = [e for e in dev if e["_domain"] == dom and e.get("emitted")]
        rng.shuffle(cand)
        chosen += cand[:N_CALIB // len(DOMAINS)]
    cal = [r for r in outer.map(lambda e: score_episode(e, judge, pool), chosen) if r]
    X = [features(r["scores"]) for r in cal]
    Z = [r["Z"] for r in cal]
    draws = fit_calibrator(X, Z)
    cal_p = [predict(draws, x) for x in X]

    test = load_episodes(_PILOT, a.modeltag, a.anchor, "w2a")
    recs = {r["instance"]: r for r in outer.map(lambda e: score_episode(e, judge, pool), test) if r}
    diag = {"calibration": {"n": len(cal), "positives": sum(Z), "auroc": auroc(cal_p, Z),
                            "posterior_mean": [sum(d[i] for d in draws) / len(draws) for i in range(3)]},
            "test": {}, "constants": {"beta": BETA, "alpha": ALPHA, "lambda": LAMBDA, "eps_u": EPS_U,
                                      "n_calib": N_CALIB, "draws": N_DRAWS},
            "expansion": "unavailable: every guideline of the stated conclusion is already used"}
    p0s, p1s, zs, trig = [], [], [], 0
    for iid, r in recs.items():
        p0 = predict(draws, features(r["scores"]))
        r["p0"], r["H"] = p0, entropy_nats(p0)
        r["triggered"] = int(r["H"] > EPS_U)
        if r["triggered"]:
            trig += 1
            r["p1"] = predict(draws, features(active_scores(r, judge, pool)))
        else:
            r["p1"] = p0
        p0s.append(p0)
        p1s.append(r["p1"])
        zs.append(r["Z"])
    diag["test"] = {"n_stated": len(recs), "positives": sum(zs), "trigger_rate": trig / max(1, len(recs)),
                    "auroc_before_active": auroc(p0s, zs), "auroc": auroc(p1s, zs),
                    "ece": ece(p1s, zs), "brier": sum((p - z) ** 2 for p, z in zip(p1s, zs)) / max(1, len(zs)),
                    "accepted": sum(1 for p in p1s if p >= 0.5),
                    "accepted_unsupported": sum(1 for p, z in zip(p1s, zs) if p >= 0.5 and not z),
                    "rejected_supported": sum(1 for p, z in zip(p1s, zs) if p < 0.5 and z)}

    for dom in DOMAINS:
        src = os.path.join(_PILOT, "results", "w2a", "%s-%s" % (a.modeltag, dom), "%s__%s" % (BASE_ARM, a.anchor))
        dst = os.path.join(_PILOT, "results", "w2a", "%s-%s" % (a.modeltag, dom), "ext_glean__%s" % a.anchor)
        os.makedirs(dst, exist_ok=True)
        base = json.load(open(os.path.join(src, "summary.json"), encoding="utf-8"))
        eps = []
        for e in base["episodes"]:
            e = dict(e)
            tr = os.path.join(src, e["instance"] + ".jsonl")
            if os.path.exists(tr):
                shutil.copyfile(tr, os.path.join(dst, e["instance"] + ".jsonl"))
            e["arm"] = "ext_glean"
            e["trace"] = os.path.join("results", "w2a", "%s-%s" % (a.modeltag, dom),
                                      "ext_glean__%s" % a.anchor, e["instance"] + ".jsonl")
            r = recs.get(e["instance"])
            ext = {"llm_calls": 0, "base_emitted": [x["claim"] for x in e.get("emitted") or []]}
            if r is not None:
                accepted = r["p1"] >= 0.5
                n_calls = r["k"] * r["n_steps"] + (len(r["comp_docs"]) * r["n_steps"] if r["triggered"] else 0)
                ext.update({"llm_calls": n_calls, "p_hat": round(r["p1"], 4),
                            "p_hat_before_active": round(r["p0"], 4), "entropy": round(r["H"], 4),
                            "triggered": r["triggered"], "accepted": int(accepted),
                            "n_steps": r["n_steps"], "k_guidelines": r["k"]})
                if not accepted:
                    e["emitted"] = []
            e["ext"] = ext
            eps.append(e)
        run = dict(base["run"], arm="ext_glean", base_arm=BASE_ARM)
        with open(os.path.join(dst, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"run": run, "episodes": eps}, fh, indent=1, sort_keys=True)
    with open(os.path.join(_PILOT, "results", "glean", "diagnostics_%s_%s.json" % (a.modeltag, a.anchor)),
              "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=1, sort_keys=True)
    print(json.dumps(diag, indent=1, sort_keys=True))
    print("GLEAN_DONE judge_calls=%d" % judge.calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
