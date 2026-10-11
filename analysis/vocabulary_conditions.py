#!/usr/bin/env python3
"""How much of the evaluation standard is actually LEARNED? -- the
vocabulary-condition experiment.

Review raised the central objection that PRIME's "learned standard" is
closer to *learned Boolean support logic over an authored vocabulary* than
to learning an evaluation standard from an open-ended request, because the
induction policy is handed the claim-ID set, the claim order, the
evidence-atom vocabulary, the resolver types, and the action schema -- and
those supplied objects encode much of the hard domain knowledge (what to
evaluate, what may be claimed, what counts as evidence, what is in scope).

This script measures that objection instead of arguing with it, by running
license induction under three nested conditions and scoring all of them with
the same executable oracle (no human or LLM judge anywhere):

  V0 "ids"    ORIGINAL condition, the one every published induced-license
              number was produced under: ordered claim IDs + atom IDs + the
              user prompt. The policy connects semantically empty
              identifiers.
  V1 "spec"   The full specification view Definition 2 actually grants:
              claim TEXT and level, and each atom's type, resolver tool and
              locator params. Strictly more information, still strictly
              inside the information boundary (no atom proposition, no gold
              value, no gold licenses).
  V2 "open"   OPEN VOCABULARY. The policy receives only the user request,
              declared scope, domain label, registered tool names, and
              artifact metadata. It must propose the claim set, the claim
              text, the dominance order, the evidence-atom vocabulary
              (including each atom's resolver tool and params), AND the
              routes. Nothing about what to evaluate is supplied.

V2 is the condition the objection actually asks about. Its headline metric
is deliberately the harshest available judge-free one:

  gold_atom_coverage -- of the evidence atoms the benchmark authors deemed
  necessary, what fraction did the model reconstruct on its own, matched
  OBJECTIVELY by (resolver tool, canonical locator params)?

Two further judge-free V2 metrics:
  executable_rate   -- proposed atoms that bind to a registered tool and
                       actually resolve through the real ToolExecutor
                       against the real artifact. A standard whose evidence
                       cannot be executed cannot gate anything.
  spurious_rate     -- proposed atoms that execute but match no gold atom:
                       invented evidence requirements.

Semantic claim recovery is scored WITHOUT a judge via EVIDENCE SIGNATURES
(claim_signature_recovery): two claims are identified iff they license the
same evidence, so a gold claim is recovered iff some proposed claim has a
route whose atoms, translated through the objective (tool, params) match,
equal one of the gold routes as a set. What this deliberately does NOT do is
read any claim text -- free-text paraphrase equivalence would need a judge,
and stays out of every number by the project's hard constraint.

Writes: results/vocabulary_conditions.json
Usage:
  python3 analysis/vocabulary_conditions.py --model qwen2.5-32b \\
      --instances 'bench/instances/swe/*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
RUNNER = os.path.join(PILOT, "runner")
for _p in (PILOT, RUNNER, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

RESULTS = os.path.join(PILOT, "results")


# ---------------------------------------------------------------------------
# Objective atom identity: (resolver tool, canonical locator params).
# No judge, no text matching -- two atoms are the same probe iff they run the
# same tool against the same target.
# ---------------------------------------------------------------------------
def _atom_key(tool: str, params: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    norm = tuple(sorted((str(k), str(v).strip()) for k, v in (params or {}).items()))
    return (str(tool).strip(), norm)


def _gold_atom_keys(instance) -> Dict[Any, str]:
    out = {}
    for a in instance.atoms:
        r = a.resolver
        out[_atom_key(getattr(r, "tool", ""), dict(getattr(r, "params", {}) or {}))] = a.id
    return out


# ---------------------------------------------------------------------------
# V2: open-vocabulary induction
# ---------------------------------------------------------------------------
OPEN_PROMPT = """You must design, from scratch, the evaluation standard for the request below.

Nothing about what to evaluate is given to you. You must decide:
  1. WHICH claims could be made about this artifact, in your own words;
  2. how those claims are ORDERED by strength (weaker claim -> stronger claim);
  3. WHAT EVIDENCE would be needed, as concrete checks executable with the
     registered tools listed below;
  4. WHICH SETS of that evidence suffice to license each claim.

Request: %(prompt)s
Declared scope: %(scope)s
Domain: %(domain)s
Artifact: %(artifact)s
Registered tools you may use: %(tools)s

Answer with JSON ONLY, in exactly this shape:
{
  "claims":  [{"id": "C1", "text": "<what this claim asserts>", "level": 1}, ...],
  "order":   [["C1","C2"], ["C2","C3"]],
  "atoms":   [{"id": "E1", "type": "<inspect|execute|config|search>",
               "tool": "<one of the registered tools>",
               "params": {"<param>": "<value>"}}, ...],
  "licenses": {"C1": [["E1"]], "C2": [["E1","E2"]], ...}
}
Each atom's "params" must be concrete enough to actually run (a real file
path, suite name, or search pattern for THIS artifact). Do not invent tools.
"""


def induce_open(instance, llm, max_retries: int = 1) -> Dict[str, Any]:
    """Ask for a whole standard. Returns the parsed proposal (or an error)."""
    artifact = dict(instance.artifact)
    msg = OPEN_PROMPT % {
        "prompt": instance.user_prompt,
        "scope": instance.scope,
        "domain": instance.domain,
        "artifact": json.dumps({k: artifact.get(k, "") for k in ("path", "kind", "commit")}),
        "tools": ", ".join(sorted(instance.actions)),
    }
    messages = [
        {"role": "system", "content": "You design evidence-based evaluation standards."},
        {"role": "user", "content": msg},
    ]
    last_err = "no JSON found in model output"
    for _ in range(1 + max_retries):
        resp = llm.chat(messages, tools=None)
        text = (resp.get("content") or "").strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj.get("atoms"), list) and isinstance(obj.get("claims"), list):
                    return {"ok": True, "proposal": obj}
                last_err = "missing 'atoms' or 'claims' list"
            except json.JSONDecodeError as exc:
                last_err = "invalid JSON: %s" % exc
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": "Invalid: %s. Emit corrected JSON only." % last_err[:200]})
    return {"ok": False, "error": last_err}


def score_open(instance, proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Judge-free scoring of a self-proposed standard."""
    executor = arms_mod.ToolExecutor(instance)
    gold_keys = _gold_atom_keys(instance)

    executable = 0
    aligned_keys = set()
    proposed = proposal.get("atoms") or []
    per_atom = []
    for a in proposed:
        if not isinstance(a, dict):
            continue
        tool = a.get("tool", "")
        params = a.get("params", {}) or {}
        key = _atom_key(tool, params)
        # ToolExecutor.execute returns {"error": ...} for a failed/unknown
        # tool rather than raising, so "no error key" is the executability
        # test. Treating a returned error as success would silently inflate
        # this metric.
        ok = False
        if tool in set(instance.actions):
            obs = executor.execute(tool, dict(params))
            ok = isinstance(obs, dict) and not obs.get("error")
        if ok:
            executable += 1
        hit = gold_keys.get(key)
        if hit:
            aligned_keys.add(key)
        per_atom.append({"tool": tool, "params": params, "executable": ok,
                         "matched_gold_atom": hit})

    # --- judge-free semantic claim recovery via EVIDENCE SIGNATURES ---
    # Two claims are identified iff they license the same evidence: a gold
    # claim c is "signature-recovered" iff some proposed claim has a route
    # whose atoms, translated through the objective (tool, params) match,
    # equal one of c's gold routes as a set. No judge reads any claim text.
    matched_map = {}  # proposed atom id -> gold atom id
    for a in proposed:
        if isinstance(a, dict):
            hit = gold_keys.get(_atom_key(a.get("tool", ""), a.get("params", {}) or {}))
            if hit:
                matched_map[str(a.get("id", ""))] = hit
    licenses = proposal.get("licenses") or {}
    translated_routes = []
    for _pcid, routes in (licenses.items() if isinstance(licenses, dict) else []):
        for r in (routes if isinstance(routes, list) else []):
            if isinstance(r, list) and all(str(x) in matched_map for x in r):
                translated_routes.append(frozenset(matched_map[str(x)] for x in r))
    gold_routes = {cid: [frozenset(rt) for rt in rts]
                   for cid, rts in instance.gold_licenses.items()}
    sig_recovered = {
        cid: any(tr in translated_routes for tr in trs) if trs else False
        for cid, trs in ((cid, rts) for cid, rts in gold_routes.items())
    }
    n_prop = len(per_atom)
    n_gold = len(gold_keys)
    return {
        "claim_signature_recovery": (
            sum(sig_recovered.values()) / len(sig_recovered) if sig_recovered else 0.0),
        "signature_recovered_by_claim": {k: bool(v) for k, v in sorted(sig_recovered.items())},
        "n_proposed_atoms": n_prop,
        "n_gold_atoms": n_gold,
        "executable_rate": (executable / n_prop) if n_prop else 0.0,
        "gold_atom_coverage": (len(aligned_keys) / n_gold) if n_gold else 0.0,
        "spurious_rate": ((n_prop - len(aligned_keys)) / n_prop) if n_prop else 0.0,
        "n_proposed_claims": len(proposal.get("claims") or []),
        "n_gold_claims": len(instance.claims),
        "atoms": per_atom,
    }


# ---------------------------------------------------------------------------
# V0/V1: route induction over the AUTHORED vocabulary (existing machinery)
# ---------------------------------------------------------------------------
def score_closed(instance, llm, view: str) -> Dict[str, Any]:
    """Induce with the authored vocabulary and report validity + how the
    induced routes compare to gold, using exact set identity (judge-free)."""
    try:
        program = arms_mod.induce_program(instance, llm, view=view)
    except Exception as exc:
        return {"valid": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    gold = instance.gold_program()
    exact = 0
    total = 0
    prec_num = prec_den = 0
    for cid, groutes in gold.licenses.items():
        gset = {frozenset(r) for r in groutes}
        iroutes = program.licenses.get(cid, ())
        iset = {frozenset(r) for r in iroutes}
        total += 1
        if iset == gset:
            exact += 1
        prec_den += len(iset)
        prec_num += len(iset & gset)
    return {
        "valid": True,
        "exact_match_rate": (exact / total) if total else 0.0,
        "route_precision": (prec_num / prec_den) if prec_den else 0.0,
        "n_claims": total,
    }


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=os.path.join(PILOT, "bench", "instances", "swe", "*.json"))
    ap.add_argument("--model", default=os.environ.get("PRIME_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--tag", default="", help="results key (defaults to --model)")
    ap.add_argument("--conditions", nargs="+", default=["ids", "spec", "open"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "vocabulary_conditions.json"))
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.instances))
    if not paths:
        ap.error("no instances match %r" % args.instances)
    tag = args.tag or args.model
    client = llm_mod.LLMClient(model=args.model)

    out: Dict[str, Any] = {}
    if os.path.exists(args.out):
        try:
            out = json.load(open(args.out, encoding="utf-8"))
        except json.JSONDecodeError:
            out = {}
    out.setdefault(tag, {})

    for cond in args.conditions:
        rows = []
        for p in paths:
            inst = load_instance(p)
            print("[%s/%s] %s" % (cond, tag, inst.id), file=sys.stderr)
            if cond == "open":
                r = induce_open(inst, client)
                row = {"instance": inst.id}
                if r["ok"]:
                    row.update(score_open(inst, r["proposal"]))
                    row["valid"] = True
                    row["proposal"] = r["proposal"]  # raw, for later re-scoring
                else:
                    row.update({"valid": False, "error": r["error"]})
            else:
                row = {"instance": inst.id}
                row.update(score_closed(inst, client, cond))
            rows.append(row)
        out[tag][cond] = rows
        print("  -> %d instances scored" % len(rows), file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote %s" % args.out, file=sys.stderr)
    print(render(out))
    return 0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def render(out: Dict[str, Any]) -> str:
    lines = ["", "# Vocabulary conditions (judge-free)", "",
             "| Model | Condition | n | valid | route prec | exact match | "
             "gold-atom coverage | executable | spurious | claim-sig recovery |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, conds in sorted(out.items()):
        for cond in ("ids", "spec", "open"):
            rows = conds.get(cond)
            if not rows:
                continue
            valid = [r for r in rows if r.get("valid")]
            def col(k):
                v = _mean([r.get(k) for r in valid])
                return "-" if v != v else "%.2f" % v
            lines.append("| %s | %s | %d | %d/%d | %s | %s | %s | %s | %s | %s |" % (
                tag, cond, len(rows), len(valid), len(rows),
                col("route_precision"), col("exact_match_rate"),
                col("gold_atom_coverage"), col("executable_rate"), col("spurious_rate"),
                col("claim_signature_recovery")))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
