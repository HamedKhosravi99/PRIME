#!/usr/bin/env python3
"""STANDARD CONSTRUCTION (S1) run END-TO-END, judge-free.

The gap it closes: in every other completed experiment the policy is HANDED the
claim set, claim order, evidence-atom vocabulary, resolver types and action
schema, so "learned standard" really meant learned Boolean support logic over
an authored vocabulary (S2 in the paper's decomposition), not learning an
evaluation standard from an open-ended request (S1).

This script runs S1 for real, in the structured-record family, with no new
runtime and no judge anywhere:

  1. ELICIT a complete standard from spec-view information only (request,
     declared scope, domain, record id, registered tools, and the CLOSED
     predicate grammar of src/prime_runtime/predicates.py -- never a record
     field value, never the authored claims/atoms/licenses): the policy
     must propose the claim set + text, the dominance order, the evidence
     atoms (field_path + predicate + expected value), and the routes.
  2. COMPILE the proposal into a synthetic instance in the real benchmark
     schema (records copied verbatim from the real instance; each proposed
     atom's gold_value computed mechanically by evaluating its predicate
     against the embedded record). schema.load_instance(validate=True) is
     the VALIDITY GATE: proposals that violate the schema (bad order,
     unknown atoms in routes, ...) are measured failures, not crashes.
  3. RUN the UNCHANGED masked runtime on the synthetic instance
     (src/prime_agents/run_episode.py, arm=prime, the proposal as the committed
     program): same interpreter mask, same ledger, same attestation path
     (goldledger.eval_attest on record_predicate rules), same MaxAuth
     fallback, real tool calls against the same embedded real records.
  4. SCORE against the REAL instance's authored standard by EVIDENCE
     SIGNATURES only: a proposed atom equals a gold atom iff
     (field_path, predicate, canonical expected value) match exactly; an
     emitted proposed claim maps to a gold claim iff one of its routes
     translates literally to one of that gold claim's routes. Emissions
     with no signature match are UNSCOREABLE against the authored standard
     and are reported as their own rate, with episode OFAR given as a
     [best, worst] interval (best: unmatched excluded; worst: unmatched
     counted as false authorization). No judge reads anything.

Scope: structured-record domains only (osv/fda/fin/ctgov) -- the domains
whose atom semantics live entirely in the closed predicate grammar. The
software/research domains' atom semantics live in executable artifacts
(test suites, retrieval protocols) and are out of scope here; the paper
says so rather than implying otherwise.

Writes: results/openvocab_endtoend_<tag>_<domain>.json
Usage:
  python3 experiments/analysis/openvocab_endtoend.py --domains security:osv \\
      --model qwen2.5-32b --tag qwen2.5-32b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(PILOT, "src", "prime_agents")
for _p in (PILOT, os.path.join(PILOT, "src"), RUNNER, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm as llm_mod  # noqa: E402
from arms import _resolve_field_path  # noqa: E402
from prime_runtime.predicates import (  # noqa: E402
    PREDICATE_TYPES, evaluate_predicate, format_predicate_rule, parse_predicate_rule,
)
from prime_runtime.schema import SchemaError, load_instance  # noqa: E402
from run_episode import run_episode  # noqa: E402

RESULTS = os.path.join(PILOT, "results")


# ---------------------------------------------------------------------------
# 1. Elicitation (spec-view information ONLY)
# ---------------------------------------------------------------------------
ELICIT_PROMPT = """You must design, from scratch, a complete evidence-gated evaluation standard
for the request below. Nothing about what to evaluate is given to you.

Request: %(prompt)s
Declared scope: %(scope)s
Domain: %(domain)s
Record under evaluation: id %(record_id)s (a real %(domain)s record; you have
NOT been shown its contents -- your checks will be executed against it later).
Registered evidence tool: inspect_record(record_id, field_path) -- returns the
raw value at a dotted/indexed field path of the record.
Allowed predicates (an atom is TRUE iff its predicate holds of the raw value):
%(predicates)s

Propose:
  1. the CLAIMS that could be made, in your own words, ordered by strength;
  2. the EVIDENCE ATOMS: concrete field_path + predicate + expected value;
  3. the LICENSES: which sets of atoms suffice for each claim (a stronger
     claim's route should contain a weaker claim's route).

Answer with JSON ONLY, exactly this shape:
{
  "claims":  [{"id": "C1", "text": "<what this claim asserts>", "level": 1}, ...],
  "order":   [["C1","C2"], ["C2","C3"]],
  "atoms":   [{"id": "E1", "field_path": "<dotted.path>",
               "predicate": "<one allowed predicate>", "expected": <value or null>}, ...],
  "licenses": {"C1": [["E1"]], "C2": [["E1","E2"]], ...}
}
"""


def elicit_standard(instance, llm, max_retries: int = 1) -> Dict[str, Any]:
    record_ids = sorted(instance.records.keys())
    msg = ELICIT_PROMPT % {
        "prompt": instance.user_prompt,
        "scope": instance.scope,
        "domain": instance.domain,
        "record_id": ", ".join(record_ids) if record_ids else "(none)",
        "predicates": ", ".join(PREDICATE_TYPES),
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
                if all(isinstance(obj.get(k), (list, dict)) for k in ("claims", "atoms", "licenses")):
                    return {"ok": True, "proposal": obj}
                last_err = "missing claims/atoms/licenses"
            except json.JSONDecodeError as exc:
                last_err = "invalid JSON: %s" % exc
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": "Invalid: %s. Emit corrected JSON only." % last_err[:200]})
    return {"ok": False, "error": last_err}


# ---------------------------------------------------------------------------
# 2. Compile proposal -> synthetic instance in the REAL schema
# ---------------------------------------------------------------------------
def compile_synthetic(real, proposal: Dict[str, Any], out_path: str) -> Tuple[bool, str]:
    """Write the synthetic instance JSON; returns (ok, error). Validity is
    judged by schema.load_instance(validate=True) -- the same gate every
    authored instance passes."""
    record_ids = sorted(real.records.keys())
    rid = record_ids[0] if record_ids else ""
    atoms = []
    for a in proposal.get("atoms") or []:
        if not isinstance(a, dict):
            return False, "atom entry is not an object"
        ptype = str(a.get("predicate", ""))
        if ptype not in PREDICATE_TYPES:
            return False, "unknown predicate %r" % ptype
        fp = str(a.get("field_path", ""))
        expected = a.get("expected", None)
        # gold_value computed MECHANICALLY from the embedded real record --
        # deterministic, no judge; predicate errors (missing field, type
        # mismatch) make the atom F, exactly like the runtime path.
        value, found = _resolve_field_path(real.records.get(rid, {}), fp)
        verdict = evaluate_predicate(ptype, expected, value) if found else False
        atoms.append({
            "id": str(a.get("id", "")),
            "type": "record",
            "proposition": "",
            "resolver": {
                "tool": "inspect_record",
                "params_schema": {"field_path": "str", "record_id": "str"},
                "params": {"field_path": fp, "record_id": rid},
                "attest": format_predicate_rule(ptype, expected),
            },
            "gold_value": "T" if verdict else "F",
        })
    synthetic = {
        "id": real.id + "__openvocab",
        "domain": real.domain,
        "seed": real.seed,
        "variant": "openvocab",
        "user_prompt": real.user_prompt,
        "artifact": dict(real.artifact),
        "scope": real.scope,
        "actions": list(real.actions),
        "claims": [
            {"id": str(c.get("id", "")), "text": str(c.get("text", "")),
             "level": int(c.get("level", 1))}
            for c in proposal.get("claims") or [] if isinstance(c, dict)
        ],
        "order": [list(map(str, pair)) for pair in proposal.get("order") or []
                  if isinstance(pair, (list, tuple)) and len(pair) == 2],
        "atoms": atoms,
        "phi": [],
        "gold_licenses": {
            str(cid): [[str(x) for x in route] for route in routes]
            for cid, routes in (proposal.get("licenses") or {}).items()
            if isinstance(routes, list)
        },
        "decoys": [],
        "records": {k: v for k, v in real.records.items()},
        "notes_for_annotators": "synthetic openvocab instance; proposal-committed",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(synthetic, fh, indent=1, sort_keys=True)
    try:
        load_instance(out_path, validate=True)
    except (SchemaError, Exception) as exc:  # noqa: BLE001 - measured gate
        return False, "%s: %s" % (type(exc).__name__, exc)
    return True, ""


# ---------------------------------------------------------------------------
# 4. Judge-free signature scoring against the REAL authored standard
# ---------------------------------------------------------------------------
def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)


def _gold_atom_sigs(real) -> Dict[Tuple[str, str, str], str]:
    sigs = {}
    for a in real.atoms:
        if a.resolver is None or a.resolver.tool != "inspect_record":
            continue
        parsed = parse_predicate_rule(a.resolver.attest)
        if parsed is None:
            continue
        ptype, expected = parsed
        fp = str(dict(a.resolver.params).get("field_path", ""))
        sigs[(fp, ptype, _canon(expected))] = a.id
    return sigs


def _gold_supported(real) -> Dict[str, bool]:
    """Oracle support at FULL evidence, from verified gold_values (every
    record instance passes bench verify: observed == gold_value exactly)."""
    raw = json.load(open(real.path, encoding="utf-8"))
    gv = {a["id"]: a.get("gold_value") for a in raw["atoms"]}
    return {
        cid: any(all(gv.get(x) == "T" for x in route) for route in routes)
        for cid, routes in real.gold_licenses.items()
    }


def score_episode(real, proposal: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    gold_sigs = _gold_atom_sigs(real)
    supported = _gold_supported(real)
    gold_levels = {c.id: c.level for c in real.claims}
    gold_routes = {cid: [frozenset(r) for r in rts] for cid, rts in real.gold_licenses.items()}

    # proposed atom id -> gold atom id, by exact (field_path, predicate, arg)
    amap = {}
    for a in proposal.get("atoms") or []:
        if isinstance(a, dict):
            key = (str(a.get("field_path", "")), str(a.get("predicate", "")),
                   _canon(a.get("expected", None)))
            if key in gold_sigs:
                amap[str(a.get("id", ""))] = gold_sigs[key]

    def map_claim(pcid: str) -> Optional[str]:
        routes = (proposal.get("licenses") or {}).get(pcid) or []
        best = None
        for r in routes:
            if not (isinstance(r, list) and all(str(x) in amap for x in r)):
                continue
            tr = frozenset(amap[str(x)] for x in r)
            for gcid, grs in gold_routes.items():
                if tr in grs and (best is None or gold_levels[gcid] > gold_levels[best]):
                    best = gcid
        return best

    emitted = episode.get("emitted") or []
    rows = []
    ofar_best = ofar_worst = 0
    unmatched = 0
    best_supported_level = 0
    for e in emitted:
        g = map_claim(str(e.get("claim", "")))
        if g is None:
            unmatched += 1
            ofar_worst = 1  # worst case: an unscoreable claim counts as false
            rows.append({"proposed": e.get("claim"), "mapped": None})
            continue
        ok = bool(supported.get(g))
        if not ok:
            ofar_best = ofar_worst = 1
        else:
            best_supported_level = max(best_supported_level, gold_levels[g])
        rows.append({"proposed": e.get("claim"), "mapped": g, "gold_supported": ok})
    ceiling = max((gold_levels[c] for c, s in supported.items() if s), default=0)
    return {
        "instance": real.id,
        "episode_error": episode.get("error", ""),
        "n_emitted": len(emitted),
        "n_unmatched_emissions": unmatched,
        "episode_ofar_best": ofar_best,
        "episode_ofar_worst": ofar_worst,
        "supported_level": best_supported_level,
        "oracle_ceiling_level": ceiling,
        "utility": (best_supported_level / ceiling) if ceiling else 0.0,
        "tool_calls": episode.get("tool_calls", 0),
        "emissions": rows,
    }


# ---------------------------------------------------------------------------
def run_one(real_path: str, llm, out_dir: str, budget: int,
            elicit=elicit_standard) -> Dict[str, Any]:
    real = load_instance(real_path)
    r = elicit(real, llm)
    if not r.get("ok"):
        return {"instance": real.id, "valid_proposal": False,
                "stage": "elicitation", "error": r.get("error", "")}
    proposal = r["proposal"]
    os.makedirs(out_dir, exist_ok=True)
    syn_path = os.path.join(out_dir, "%s__openvocab.json" % real.id)
    ok, err = compile_synthetic(real, proposal, syn_path)
    if not ok:
        return {"instance": real.id, "valid_proposal": False,
                "stage": "schema_validation", "error": err, "proposal": proposal}
    episode = run_episode(syn_path, "prime", "gold", llm, out_dir,
                          max_tool_calls=budget)
    scored = score_episode(real, proposal, episode)
    scored.update({"valid_proposal": True, "proposal": proposal,
                   "synthetic_instance": syn_path})
    return scored


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+",
                    default=["security:osv", "drugsafety:fda", "financial:fin",
                             "clinical:ctgov"])
    ap.add_argument("--model", default=os.environ.get("PRIME_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--v2-split", default=None,
                    help="run on data/obligationbench/<domain>/<split> instead "
                         "of the legacy data/pilot/instances/<sub> layout. Without "
                         "this the S1 result is about the OLD instance set, "
                         "not the confirmatory benchmark the paper reports, "
                         "and the scope mismatch is silent.")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap instances per domain (S1 is time-boxed)")
    args = ap.parse_args(argv)
    tag = args.tag or args.model
    client = llm_mod.LLMClient(model=args.model)

    for spec in args.domains:
        label, sub = spec.split(":", 1)
        if args.v2_split:
            pat = os.path.join(PILOT, "data", "obligationbench", label,
                               args.v2_split, "*.json")
        else:
            pat = os.path.join(PILOT, "data", "pilot", "instances", sub, "*.json")
        paths = sorted(p for p in glob.glob(pat) if "__openvocab" not in p)
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            print("[openvocab/%s] NO INSTANCES matched %s" % (label, pat),
                  file=sys.stderr)
            continue
        out_dir = os.path.join(RESULTS, "%s-%s-openvocab" % (tag, label))
        rows = []
        for p in paths:
            print("[openvocab/%s] %s" % (label, os.path.basename(p)), file=sys.stderr)
            rows.append(run_one(p, client, out_dir, args.budget))
        valid = [x for x in rows if x.get("valid_proposal")]
        summary = {
            "n_instances": len(rows),
            "n_valid_proposals": len(valid),
            "episode_ofar_best": (sum(x["episode_ofar_best"] for x in valid) / len(valid)) if valid else None,
            "episode_ofar_worst": (sum(x["episode_ofar_worst"] for x in valid) / len(valid)) if valid else None,
            "mean_utility": (sum(x["utility"] for x in valid) / len(valid)) if valid else None,
            "unmatched_emission_episodes": sum(1 for x in valid if x["n_unmatched_emissions"]),
        }
        out_path = os.path.join(RESULTS, "openvocab_endtoend_%s_%s.json" % (tag, label))
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "episodes": rows}, fh, indent=2,
                      sort_keys=True, default=str)
        print("[%s] %s" % (label, json.dumps(summary)), file=sys.stderr)
        print("wrote %s" % out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
