#!/usr/bin/env python3
"""W3 conservative induction stack. Five components, all judge-free:

  (a) STATE-LABEL-FIRST induction. For |E| small (record domains: 6-9
      atoms), enumerate the realizable evidence states, ask the model
      whether each claim is supported in each state from a SPEC-ONLY
      description, aggregate labels conservatively, then compile the
      accepted support relation to minimal DNF routes mechanically
      (prime implicants). Contrast: one-shot route generation asks the
      model for the routes directly.
  (b) CONSERVATIVE AGGREGATION policies over draws: unanimous, k-of-n,
      majority -- yielding a precision/coverage frontier rather than one
      hidden threshold.
  (c) COUNTERFACTUAL ROUTE VALIDATION as a pre-commitment gate:
      literal-necessity and missing-literal probes, answered by the same
      model over constructed states (never by the authored oracle).
  (d) TYPED ROUTE LINTER: pure-code structural + binding + predicate-type
      lints (no LLM, no gold).
  (e) SAFE MERGE + ELIGIBILITY GATE: route-union across candidate
      programs; guidance is eligible only for claims whose routes survive
      (c) and (d).

Nothing here reads gold_value or gold_licenses: those are used only by
the SCORING side (experiments/analysis/*_tables.py) after commitment.

Usage:
  python3 experiments/analysis/induction_v5.py --instances 'data/obligationbench/security/dev/*.json' \
      --model qwen2.5-32b --method state_first_unanimous --out results/ind_v5/...
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime.predicates import parse_predicate_rule  # noqa: E402
from prime_runtime.schema import Instance, load_instance  # noqa: E402

MAX_EXHAUSTIVE_ATOMS = 12


# ---------------------------------------------------------------------------
# (a) state enumeration + spec-only description
# ---------------------------------------------------------------------------
def realizable_states(instance: Instance) -> List[FrozenSet[str]]:
    """All truth assignments over the atom set, as the frozenset of atoms
    assigned TRUE. Phi forbid-patterns (if any) prune the space. Atoms with
    resolver=None are authoring-time unavailable: they can never be TRUE."""
    atoms = list(instance.atom_ids)
    if len(atoms) > MAX_EXHAUSTIVE_ATOMS:
        raise ValueError("atom set too large for exhaustive enumeration: %d" % len(atoms))
    impossible = {a.id for a in instance.atoms if a.resolver is None}
    free = [a for a in atoms if a not in impossible]
    out: List[FrozenSet[str]] = []
    for bits in itertools.product((False, True), repeat=len(free)):
        true_set = frozenset(a for a, b in zip(free, bits) if b)
        assign = {a: ("T" if a in true_set else "F") for a in atoms}
        if any(pat and all(assign.get(k) == v for k, v in dict(pat).items())
               for pat in (instance.phi_patterns or ())):
            continue
        out.append(true_set)
    return out


def describe_state(instance: Instance, true_set: FrozenSet[str]) -> str:
    """Spec-only rendering of a hypothetical evidence state: what each check
    interrogates and whether it came back established or not. No record
    values, no gold labels."""
    lines = []
    for a in sorted(instance.atoms, key=lambda x: x.id):
        r = a.resolver
        if r is None:
            lines.append("  %s: UNAVAILABLE (no resolver for this artifact)" % a.id)
            continue
        params = ", ".join("%s=%r" % kv for kv in sorted(dict(r.params or {}).items()))
        state = "ESTABLISHED" if a.id in true_set else "NOT ESTABLISHED"
        lines.append("  %s (%s via %s(%s)): %s" % (a.id, a.type, r.tool, params, state))
    return "\n".join(lines)


LABEL_PROMPT = """You are defining the evidence standard for this request.

Request: %(prompt)s
Scope: %(scope)s

Candidate claim %(cid)s (level %(level)d): %(text)s

Hypothetical evidence state (what has and has not been established):
%(state)s

Given ONLY this evidence state, is claim %(cid)s fully supported?
Answer with JSON only: {"supported": true|false}"""


def label_states(instance: Instance, llm, states: Sequence[FrozenSet[str]],
                 draws: int = 1) -> Dict[str, Dict[FrozenSet[str], List[bool]]]:
    """model labels: claim -> state -> [bool per draw]."""
    out: Dict[str, Dict[FrozenSet[str], List[bool]]] = {c.id: {} for c in instance.claims}
    for c in instance.claims:
        for st in states:
            votes: List[bool] = []
            msg = LABEL_PROMPT % {"prompt": instance.user_prompt, "scope": instance.scope,
                                  "cid": c.id, "level": c.level, "text": c.text,
                                  "state": describe_state(instance, st)}
            for _ in range(draws):
                try:
                    resp = llm.chat([{"role": "system",
                                      "content": "You design evidence standards."},
                                     {"role": "user", "content": msg}], tools=None)
                    m = re.search(r"\{.*\}", resp.get("content", "") or "", re.DOTALL)
                    votes.append(bool(json.loads(m.group(0)).get("supported")) if m else False)
                except Exception:
                    votes.append(False)
            out[c.id][st] = votes
    return out


# ---------------------------------------------------------------------------
# (b) conservative aggregation
# ---------------------------------------------------------------------------
AGGREGATORS = ("unanimous", "majority", "k_of_n")


def aggregate(votes: List[bool], policy: str, k: int = 3) -> bool:
    if not votes:
        return False
    if policy == "unanimous":
        return all(votes)
    if policy == "majority":
        return sum(votes) * 2 > len(votes)
    if policy == "k_of_n":
        return sum(votes) >= min(k, len(votes))
    raise ValueError(policy)


# ---------------------------------------------------------------------------
# (a-cont) mechanical DNF compilation: minimal (prime) implicants
# ---------------------------------------------------------------------------
# Above this vocabulary size the subset search stops being free and the cap
# becomes a real engineering tradeoff; at or below it, capping is pure loss.
_UNCAPPED_VOCAB = 16


def compile_routes_exact(accepted: Set[FrozenSet[str]],
                         all_states: Sequence[FrozenSet[str]],
                         atoms: Sequence[str],
                         max_width: Optional[int] = None) -> List[List[str]]:
    """Sound minimal routes: r admitted iff EVERY enumerated state containing
    r is accepted, and r is minimal under inclusion.

    max_width defaults to the full vocabulary rather than to a fixed 6. The
    old default silently truncated: 189 gold routes in ObligationBench are
    7 atoms wide, so under a cap of 6 those claims compiled to NOTHING and
    the induced program simply could not license them. That would have
    surfaced in W2b as an induction-quality result when it was a
    configuration cap, so the cap is now opt-in and its effect is
    observable via routes_truncated().
    """
    acc = set(accepted)
    routes: List[FrozenSet[str]] = []
    if max_width is None:
        max_width = len(atoms) if len(atoms) <= _UNCAPPED_VOCAB else 6
    for width in range(1, max_width + 1):
        for cand in itertools.combinations(sorted(atoms), width):
            cs = frozenset(cand)
            if any(r <= cs for r in routes):
                continue
            sound = all((st in acc) for st in all_states if cs <= st)
            witnessed = any(cs <= st for st in acc)
            if sound and witnessed:
                routes.append(cs)
    return [sorted(r) for r in routes]


# ---------------------------------------------------------------------------
# (d) typed route linter (pure code)
# ---------------------------------------------------------------------------
def lint_program(instance: Instance, licenses: Dict[str, List[List[str]]]) -> List[Dict[str, str]]:
    """Objective structural / binding / predicate-type lints. Returns a list
    of findings; an empty list is a clean program."""
    findings: List[Dict[str, str]] = []
    known = set(instance.atom_ids)
    by_id = {a.id: a for a in instance.atoms}
    levels = {c.id: c.level for c in instance.claims}
    order = {(w, s) for w, s in instance.order}

    for cid, routes in licenses.items():
        if cid not in levels:
            findings.append({"lint": "unknown_claim", "claim": cid})
            continue
        if not routes:
            findings.append({"lint": "routeless_claim", "claim": cid})
            continue
        sets = [frozenset(r) for r in routes]
        for r in routes:
            if not r:
                findings.append({"lint": "empty_route", "claim": cid})
            for lit in r:
                a = lit.lstrip("!")
                if a not in known:
                    findings.append({"lint": "unknown_atom", "claim": cid, "atom": a})
                    continue
                atom = by_id[a]
                if atom.resolver is None:
                    findings.append({"lint": "unresolvable_atom_in_route",
                                     "claim": cid, "atom": a})
                    continue
                params = dict(atom.resolver.params or {})
                if atom.type == "record" and not ("record_id" in params and "field_path" in params):
                    findings.append({"lint": "retrieval_identity_incomplete",
                                     "claim": cid, "atom": a})
                parsed = parse_predicate_rule(atom.resolver.attest)
                if parsed:
                    ptype, expected = parsed
                    if ptype in ("numeric_eq", "numeric_ge", "numeric_le") and \
                            not isinstance(expected, (int, float)):
                        findings.append({"lint": "ordered_comparison_bad_operand",
                                         "claim": cid, "atom": a, "predicate": ptype})
                    if ptype in ("date_before", "date_after") and \
                            not (isinstance(expected, str) and re.match(r"^\d{4}-\d{2}-\d{2}", expected)):
                        findings.append({"lint": "date_comparison_bad_bound",
                                         "claim": cid, "atom": a, "predicate": ptype})
        for i, a in enumerate(sets):
            for j, b in enumerate(sets):
                if i != j and b < a:
                    findings.append({"lint": "dominated_route", "claim": cid,
                                     "route": ",".join(sorted(a))})
    # order coherence: a stronger claim's every route must contain some route
    # of each weaker comparable claim
    for w, s in order:
        for rs in licenses.get(s, []):
            weak = licenses.get(w, [])
            if weak and not any(set(q) <= set(rs) for q in weak):
                findings.append({"lint": "order_incoherent", "weak": w, "strong": s,
                                 "route": ",".join(sorted(rs))})
    used = {x.lstrip("!") for rs in licenses.values() for r in rs for x in r}
    for a in known - used:
        findings.append({"lint": "unused_vocabulary_atom", "atom": a})
    return findings


# ---------------------------------------------------------------------------
# (c) counterfactual validation (model-answered, gold-free)
# ---------------------------------------------------------------------------
NECESSITY_PROMPT = """Claim %(cid)s: %(text)s

Evidence state:
%(state)s

Given ONLY this state, is claim %(cid)s fully supported?
Answer with JSON only: {"supported": true|false}"""


def counterfactual_validate(instance: Instance, licenses: Dict[str, List[List[str]]],
                            llm, draws: int = 1) -> Dict[str, Any]:
    """Literal-necessity: for each route literal, build the state where the
    rest of the route holds and that literal does NOT; the claim must NOT be
    supported. Missing-literal: for atoms outside the route, the route state
    plus that atom must still be supported (else the route is incomplete).
    Findings are model-derived, never oracle-derived."""
    report: Dict[str, Any] = {"necessity_failures": [], "completeness_failures": [],
                              "n_probes": 0}

    def ask(cid: str, text: str, true_set: FrozenSet[str]) -> bool:
        msg = NECESSITY_PROMPT % {"cid": cid, "text": text,
                                  "state": describe_state(instance, true_set)}
        votes = []
        for _ in range(draws):
            try:
                resp = llm.chat([{"role": "system", "content": "You design evidence standards."},
                                 {"role": "user", "content": msg}], tools=None)
                m = re.search(r"\{.*\}", resp.get("content", "") or "", re.DOTALL)
                votes.append(bool(json.loads(m.group(0)).get("supported")) if m else False)
            except Exception:
                votes.append(False)
        report["n_probes"] += 1
        return aggregate(votes, "majority")

    texts = {c.id: c.text for c in instance.claims}
    for cid, routes in licenses.items():
        if cid not in texts:
            continue
        for r in routes:
            rs = frozenset(x.lstrip("!") for x in r)
            for lit in rs:
                if ask(cid, texts[cid], rs - {lit}):
                    report["necessity_failures"].append(
                        {"claim": cid, "route": sorted(rs), "literal": lit})
            spare = set(instance.atom_ids) - rs
            if spare:
                extra = sorted(spare)[0]
                if not ask(cid, texts[cid], rs | {extra}):
                    report["completeness_failures"].append(
                        {"claim": cid, "route": sorted(rs), "added": extra})
    return report


# ---------------------------------------------------------------------------
# (e) safe merge + eligibility gate
# ---------------------------------------------------------------------------
def safe_merge(candidates: Sequence[Dict[str, List[List[str]]]]) -> Dict[str, List[List[str]]]:
    """Route-union merge: for each claim, take the UNION of literals of
    semantically corresponding routes (favoring over-specification over
    dangerous missing conjuncts). Correspondence = route rank
    after sorting by (width, atoms)."""
    out: Dict[str, List[List[str]]] = {}
    claims = sorted({c for cand in candidates for c in cand})
    for cid in claims:
        ranked = [sorted((sorted(r) for r in cand.get(cid, [])), key=lambda r: (len(r), r))
                  for cand in candidates if cand.get(cid)]
        if not ranked:
            continue
        depth = max(len(x) for x in ranked)
        merged = []
        for i in range(depth):
            union: Set[str] = set()
            for lst in ranked:
                if i < len(lst):
                    union |= set(lst[i])
            if union:
                merged.append(sorted(union))
        # drop dominated duplicates
        keep: List[List[str]] = []
        for r in sorted(merged, key=len):
            if not any(set(k) <= set(r) for k in keep):
                keep.append(r)
        out[cid] = keep
    return out


def eligibility(instance: Instance, licenses: Dict[str, List[List[str]]],
                lint: List[Dict[str, str]], cf: Dict[str, Any]) -> Dict[str, bool]:
    """A claim is guidance-eligible iff no lint finding and no counterfactual
    failure names it (prime_guided_gated guides only toward eligible
    claims)."""
    bad = {f.get("claim") for f in lint if f.get("claim")}
    bad |= {f.get("strong") for f in lint if f.get("lint") == "order_incoherent"}
    bad |= {f["claim"] for f in cf.get("necessity_failures", [])}
    bad |= {f["claim"] for f in cf.get("completeness_failures", [])}
    return {cid: (cid not in bad) for cid in licenses}


# ---------------------------------------------------------------------------
def induce_state_first(instance: Instance, llm, policy: str = "unanimous",
                       draws: int = 3) -> Dict[str, List[List[str]]]:
    states = realizable_states(instance)
    labels = label_states(instance, llm, states, draws=draws)
    licenses: Dict[str, List[List[str]]] = {}
    for c in instance.claims:
        accepted = {st for st, votes in labels[c.id].items()
                    if aggregate(votes, policy)}
        # explicit: never inherit a width cap that silently drops wide routes
        licenses[c.id] = compile_routes_exact(
            accepted, states, list(instance.atom_ids),
            max_width=len(instance.atom_ids))
    return licenses


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--model", default=os.environ.get("PRIME_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--method", default="state_first_unanimous",
                    choices=("state_first_unanimous", "state_first_majority",
                             "state_first_k_of_n"))
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--counterfactual", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import llm as llm_mod
    client = llm_mod.LLMClient(model=args.model)
    policy = args.method.replace("state_first_", "")
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(args.instances)):
        inst = load_instance(path)
        lic = induce_state_first(inst, client, policy=policy, draws=args.draws)
        lint = lint_program(inst, lic)
        cf = counterfactual_validate(inst, lic, client) if args.counterfactual else {}
        elig = eligibility(inst, lic, lint, cf)
        rows.append({"instance": inst.id, "licenses": lic, "lint": lint,
                     "counterfactual": cf, "eligible": elig})
        print("[%s] claims=%d lint=%d eligible=%d" %
              (inst.id, len(lic), len(lint), sum(elig.values())), file=sys.stderr)
    with open(os.path.join(args.out, "induced.json"), "w", encoding="utf-8") as fh:
        json.dump({"method": args.method, "draws": args.draws, "model": args.model,
                   "rows": rows}, fh, indent=1, sort_keys=True)
    print("wrote %s/induced.json (%d instances)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
