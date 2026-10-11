#!/usr/bin/env python3
"""Table B IMPROVEMENT experiment (Task B): two judge-free methods that try
to raise route precision / lower AuthFP beyond the de-confounded ceiling
measured by ``analysis/induction_table_b.py``.

HARD CONSTRAINT (unchanged from Table B): no human or LLM judge anywhere.
The only oracle is the executable ``prime_runtime.oracle`` (a deterministic
Boolean support predicate F* -- see that module's docstring). Every number
here comes from a completed run against real instances; nothing is
fabricated.

Method 1 -- oracle-guided iterative refinement (Task B #3)
------------------------------------------------------------
Induce once (``runner.arms.induce_program``, unmodified); canonicalize the
result (memo §3.2, reusing ``induction_table_b.canonicalize_program``);
evaluate it against F* with ``prime_runtime.oracle.auth_fp``/``auth_fn``
(exact enumeration over Omega_Phi -- the SAME oracle Table B scores every
program against); turn every flagged evidence state into one concrete,
deterministic counterexample line ("your program authorizes/misses claim C
at evidence {...}, but the independent oracle disagrees") and feed the
whole critique back as the next user turn; re-induce a full REVISED
program; repeat for up to ``--rounds`` (K, default 3) rounds or until the
oracle finds zero disagreements. This is exactly CEGIS
(counterexample-guided inductive synthesis) applied to license induction --
the critique text is a pure function of ``oracle.auth_fp``/``auth_fn``'s
``flagged_states`` output, never of any model's or human's opinion of the
program, and the model never sees the gold license family itself, only
oracle disagreements at specific evidence states (the same kind of
"compiler-error-style feedback" ``induce_program`` already uses for schema
validity -- this generalizes that idea to *semantic* validity against F*).

Method 2 -- self-consistency majority vote (Task B #4)
----------------------------------------------------------
Reuses the N (>=5) independent temp=0 draws already collected by
``induction_table_b.py --draws N`` (pass that run's JSON via
``--self-consistency-from``) -- no extra API calls, no extra assumptions.
For each (model, instance) and each claim, keep a route iff it appears, as
an EXACT canonical route (identical literal set), in >= ceil(N/2) of the N
draws (a deterministic majority vote -- an invalid draw contributes no
votes for any route, matching the abstention convention used everywhere
else in Table B); build the resulting majority-vote program and score it
against the oracle with the exact same ``induction_table_b.score_instance``
Table B itself uses.

Both methods report route precision / prime-rate / coverage / AuthFP /
exact-match BEFORE vs AFTER, aggregated with
``induction_table_b.aggregate`` and ``induction_table_b.bootstrap_table_b_cis``
(the identical functions Table B uses -- not a re-implementation, so the
two reports are guaranteed metric-consistent) -- a 95% instance-cluster
bootstrap CI, same convention as Table B / ``analysis/aggregate.py``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, PILOT)

from prime_runtime.schema import SchemaError, load_instance  # noqa: E402
from runner.llm import LLMClient  # noqa: E402
from runner.arms import InducedProgramInvalid, _build_validated_program, induce_program  # noqa: E402
from analysis import induction_table_b as itb  # noqa: E402

MAX_EXAMPLES_PER_CLAIM = 2  # counterexamples shown per claim per critique
CanonicalFamily = Mapping[str, Sequence[FrozenSet[str]]]


# ---------------------------------------------------------------------------
# Method 1: oracle-guided iterative refinement (CEGIS-style, judge-free)
# ---------------------------------------------------------------------------


def _facts_str(facts: Mapping[str, str], atom_order: Sequence[str]) -> str:
    return "{" + ", ".join("%s=%s" % (a, facts.get(a, "U")) for a in atom_order) + "}"


def build_oracle_critique(
    inst, oracle, ind_canonical: CanonicalFamily, max_examples: int = MAX_EXAMPLES_PER_CLAIM
) -> Tuple[str, int, int]:
    """Deterministic, judge-free critique (Task B #3): every line is derived
    directly from ``prime_runtime.oracle.auth_fp``/``auth_fn``'s
    ``flagged_states`` -- no LLM and no human ever reads the induced program
    to produce this text. Returns ``(critique_text, n_over, n_under)``;
    ``n_over == n_under == 0`` means the program already agrees with the
    oracle everywhere in Omega_Phi (the refinement loop's stopping rule)."""
    eff_program = itb._make_effective_program(inst, ind_canonical)
    _, _, fp_detail = oracle.auth_fp(eff_program)
    _, _, fn_detail = oracle.auth_fn(eff_program)

    over_by_claim: Dict[str, List[Mapping[str, str]]] = {}
    for st in fp_detail["flagged_states"]:
        for c in st["claims"]:
            over_by_claim.setdefault(c, []).append(st["facts"])
    under_by_claim: Dict[str, List[Mapping[str, str]]] = {}
    for st in fn_detail["flagged_states"]:
        for c in st["claims"]:
            under_by_claim.setdefault(c, []).append(st["facts"])

    lines: List[str] = []
    n_over = n_under = 0
    for cid in inst.claim_ids:
        overs = over_by_claim.get(cid, [])
        unders = under_by_claim.get(cid, [])
        if not overs and not unders:
            continue
        current = [sorted(r) for r in ind_canonical.get(cid, ())]
        lines.append("Claim %s (your current routes: %s):" % (cid, current))
        for facts in overs[:max_examples]:
            lines.append(
                "  - OVER-AUTHORIZES at evidence %s: at this exact evidence "
                "combination, at least one of your routes for %s is fully "
                "confirmed, but the independent executable oracle does NOT "
                "support %s there. Revise %s so that none of its routes are "
                "satisfied by this evidence." % (_facts_str(facts, inst.atom_ids), cid, cid, cid)
            )
            n_over += 1
        for facts in unders[:max_examples]:
            lines.append(
                "  - MISSED at evidence %s: the independent executable "
                "oracle DOES support %s at this exact evidence combination, "
                "but none of your current routes for %s are satisfied "
                "there. Add or adjust a route for %s to cover this "
                "evidence." % (_facts_str(facts, inst.atom_ids), cid, cid, cid)
            )
            n_under += 1
        hidden_over = max(0, len(overs) - max_examples)
        hidden_under = max(0, len(unders) - max_examples)
        if hidden_over or hidden_under:
            lines.append(
                "  - (%d more over-authorizing and %d more missed evidence "
                "combination(s) for %s not shown)" % (hidden_over, hidden_under, cid)
            )
    header = (
        "The executable oracle found %d disagreement(s) with your last "
        "program (%d over-authorizing, %d missed):" % (n_over + n_under, n_over, n_under)
    )
    return header + "\n" + "\n".join(lines), n_over, n_under


def _propose_revision(
    inst, llm, current_canonical: CanonicalFamily, critique: str, max_retries: int = 1
) -> Optional[Dict[str, Any]]:
    """One oracle-guided revision round: send the critique + the current
    program, ask for a full replacement program in the SAME JSON format
    ``induce_program`` uses. Returns the parsed ``licenses`` mapping, or
    None if the model never produced parseable JSON after retrying."""
    current_licenses = {cid: [sorted(r) for r in fam] for cid, fam in current_canonical.items()}
    prompt = (
        "%s\n\n"
        "Your last proposed program was:\n%s\n\n"
        "Propose a REVISED claim-license program as JSON with one key "
        "'licenses' mapping each claim id to a LIST OF ROUTES (same format "
        "as before: each route is a list of evidence-atom ids that together "
        "suffice for the claim; a single-atom route is still a list). Fix "
        "every disagreement listed above; keep every other route unchanged "
        "unless it also needs to change to fix a disagreement. Do not add "
        "atoms beyond what is needed to resolve the disagreements.\n"
        "Claims (weak to strong): %s. Available atoms: %s. Use ONLY these "
        "exact ids. Answer with JSON only."
        % (critique, json.dumps({"licenses": current_licenses}), list(inst.claim_ids), list(inst.atom_ids))
    )
    messages = [
        {"role": "system", "content": "You design evidence license programs."},
        {"role": "user", "content": inst.user_prompt + "\n" + prompt},
    ]
    for _attempt in range(1 + max_retries):
        resp = llm.chat(messages, tools=None)
        text = resp.get("content", "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)).get("licenses", {})
            except (ValueError, TypeError):
                pass
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {"role": "user", "content": "Invalid JSON. Respond with corrected JSON only, in the exact format described."}
        )
    return None


def refine_instance(inst, llm, oracle, K: int = 3, max_examples: int = MAX_EXAMPLES_PER_CLAIM) -> Dict[str, Any]:
    """One (model, instance) oracle-guided refinement run (Task B #3):
    induce -> {evaluate against F*, critique, re-induce} x up to K rounds.
    Returns before/after ``induction_table_b.score_instance``-shaped
    records plus round bookkeeping."""
    try:
        program = induce_program(inst, llm)
    except InducedProgramInvalid as exc:
        return {"valid_start": False, "error": str(exc), "rounds_used": 0, "before": None, "after": None,
                "stopped_early": False, "revision_failed": False}

    before = itb.score_instance(inst, oracle, program, True)
    current_program = program
    rounds_used = 0
    stopped_early = False
    revision_failed = False
    for round_idx in range(1, K + 1):
        ind_canonical = itb.canonicalize_program(current_program, oracle.omega)
        critique, n_over, n_under = build_oracle_critique(inst, oracle, ind_canonical, max_examples)
        if n_over == 0 and n_under == 0:
            stopped_early = True
            break
        rounds_used = round_idx
        new_licenses = _propose_revision(inst, llm, ind_canonical, critique)
        if new_licenses is None:
            revision_failed = True
            break
        try:
            current_program = _build_validated_program(inst, new_licenses)
        except SchemaError:
            revision_failed = True
            break
    after = itb.score_instance(inst, oracle, current_program, True)
    return {
        "valid_start": True, "rounds_used": rounds_used, "stopped_early": stopped_early,
        "revision_failed": revision_failed, "before": before, "after": after,
    }


def run_refinement(model: str, paths: Sequence[str], args) -> Dict[str, Any]:
    base, key = itb._pick_base_key(model, args)
    llm = LLMClient(model=model, base=base, key=key, temperature=0.0)
    before_recs: List[Dict[str, Any]] = []
    after_recs: List[Dict[str, Any]] = []
    per_instance_meta: List[Dict[str, Any]] = []
    n = valid = 0
    for p in paths:
        inst = load_instance(p)
        oracle = itb._ORACLE_CACHE.setdefault(p, itb.InstanceOracle(inst))
        n += 1
        result = refine_instance(inst, llm, oracle, K=args.rounds, max_examples=args.max_examples)
        if result["valid_start"]:
            valid += 1
            b, a = dict(result["before"]), dict(result["after"])
            b["instance"] = a["instance"] = inst.id
            b["valid"] = a["valid"] = True
            b["error"] = a["error"] = ""
        else:
            # Induction never produced a schema-valid starting program: score
            # as abstention (§3.9), identically for "before" and "after" --
            # there is nothing to refine. Mirrors Table B's own invalid-draw path.
            empty = itb.score_instance(inst, oracle, None, False)
            b, a = dict(empty), dict(empty)
            b["instance"] = a["instance"] = inst.id
            b["valid"] = a["valid"] = False
            b["error"] = a["error"] = result.get("error", "")
        before_recs.append(b)
        after_recs.append(a)
        per_instance_meta.append({
            "instance": inst.id,
            "valid_start": result["valid_start"],
            "rounds_used": result.get("rounds_used", 0),
            "stopped_early": result.get("stopped_early", False),
            "revision_failed": result.get("revision_failed", False),
            "error": result.get("error", ""),
        })

    before_mr = {"model": model, "n": n, "valid": valid, "per_instance": before_recs}
    after_mr = {"model": model, "n": n, "valid": valid, "per_instance": after_recs}
    return {
        "model": model, "n": n, "valid_start": valid,
        "before_aggregate": itb.aggregate(before_mr),
        "after_aggregate": itb.aggregate(after_mr),
        "before_ci": itb.bootstrap_table_b_cis(before_mr, iters=args.bootstrap_iters),
        "after_ci": itb.bootstrap_table_b_cis(after_mr, iters=args.bootstrap_iters),
        "per_instance_meta": per_instance_meta,
        "before_raw": before_recs,
        "after_raw": after_recs,
    }


# ---------------------------------------------------------------------------
# Method 2: self-consistency majority vote (deterministic, judge-free)
# ---------------------------------------------------------------------------


def majority_vote_canonical(
    per_draw_licenses: Sequence[Optional[Mapping[str, Sequence[Sequence[str]]]]],
    claim_ids: Sequence[str],
) -> Tuple[Dict[str, Tuple[FrozenSet[str], ...]], int]:
    """Task B #4: keep a claim's route iff it appears, as an EXACT canonical
    route, in >= ceil(N/2) of the N draws. ``per_draw_licenses[i]`` is None
    (or falsy) for an invalid draw -- contributes zero votes for any route,
    the same abstention convention used throughout Table B."""
    n = len(per_draw_licenses)
    threshold = math.ceil(n / 2) if n else 0
    counts: Dict[Tuple[str, FrozenSet[str]], int] = {}
    for lic in per_draw_licenses:
        if not lic:
            continue
        for cid, fam in lic.items():
            for route in {frozenset(r) for r in fam}:
                key = (cid, route)
                counts[key] = counts.get(key, 0) + 1
    voted: Dict[str, Tuple[FrozenSet[str], ...]] = {cid: () for cid in claim_ids}
    for (cid, route), cnt in counts.items():
        if cnt >= threshold:
            voted[cid] = voted.get(cid, ()) + (route,)
    return voted, threshold


def run_self_consistency(
    temp0_json_path: str, paths: Sequence[str], models: Sequence[str], bootstrap_iters: int
) -> Dict[str, Any]:
    with open(temp0_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    raw_by_model = {r["model"]: r for r in data["raw"]}
    out: Dict[str, Any] = {}
    for model in models:
        if model not in raw_by_model:
            print("  (skipping %s: not present in %s)" % (model, temp0_json_path), file=sys.stderr)
            continue
        mr = raw_by_model[model]
        if "induced_licenses" not in (mr["per_instance"][0] if mr["per_instance"] else {}):
            raise RuntimeError(
                "%s has no per-draw 'induced_licenses' -- re-run induction_table_b.py "
                "(this repo's current version) to regenerate it before self-consistency voting" % temp0_json_path
            )
        by_instance: Dict[str, List[dict]] = {}
        for rec in mr["per_instance"]:
            by_instance.setdefault(rec["instance"], []).append(rec)
        voted_recs: List[Dict[str, Any]] = []
        meta: List[Dict[str, Any]] = []
        for p in paths:
            inst = load_instance(p)
            if inst.id not in by_instance:
                continue
            oracle = itb._ORACLE_CACHE.setdefault(p, itb.InstanceOracle(inst))
            draws = by_instance[inst.id]
            licenses_list = [d["induced_licenses"] if d["valid"] else None for d in draws]
            voted, threshold = majority_vote_canonical(licenses_list, inst.claim_ids)
            program = itb._make_effective_program(inst, voted)
            rec = itb.score_instance(inst, oracle, program, True)
            rec["instance"] = inst.id
            rec["valid"] = True
            rec["error"] = ""
            voted_recs.append(rec)
            meta.append({
                "instance": inst.id, "n_draws": len(draws), "threshold": threshold,
                "n_valid_draws": sum(1 for d in draws if d["valid"]),
            })
        voted_mr = {"model": model, "n": len(voted_recs), "valid": len(voted_recs), "per_instance": voted_recs}
        out[model] = {
            "aggregate": itb.aggregate(voted_mr),
            "ci": itb.bootstrap_table_b_cis(voted_mr, iters=bootstrap_iters),
            "meta": meta,
            "raw": voted_recs,
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(itb.DEFAULT_MODELS))
    ap.add_argument("--instances", default=os.path.join(PILOT, "bench/instances/swe/*.json"))
    ap.add_argument("--rounds", type=int, default=3, help="K: max oracle-guided refinement rounds")
    ap.add_argument("--max-examples", type=int, default=MAX_EXAMPLES_PER_CLAIM,
                     help="counterexamples shown per claim per critique round")
    ap.add_argument("--bootstrap-iters", type=int, default=5000)
    ap.add_argument("--skip-refinement", action="store_true")
    ap.add_argument(
        "--self-consistency-from", default=None,
        help="path to a Table B --draws N (N>=2) temp=0 JSON to run self-consistency "
        "majority voting against (Task B #4); omit to skip that method",
    )
    ap.add_argument("--openai-base", default=os.environ.get("PRIME_OPENAI_BASE", itb.OPENAI_BASE))
    ap.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--local-base", default=os.environ.get("PRIME_LOCAL_BASE", itb.LOCAL_BASE))
    ap.add_argument("--local-key", default=os.environ.get("PRIME_LOCAL_KEY", itb.LOCAL_KEY))
    ap.add_argument("--out", default=os.path.join(PILOT, "results", "induction_refinement.json"))
    args = ap.parse_args()

    paths = sorted(glob.glob(args.instances))
    if not paths:
        print("no instances matched %r" % args.instances, file=sys.stderr)
        return 1

    print("F* source: %s" % itb.F_STAR_SOURCE)
    print("instances: %s" % ", ".join(os.path.basename(p) for p in paths))

    refinement_results: Dict[str, Any] = {}
    if not args.skip_refinement:
        print("\n### Method 1: oracle-guided iterative refinement (K=%d) ###" % args.rounds)
        for model in args.models:
            print("-- %s --" % model)
            res = run_refinement(model, paths, args)
            refinement_results[model] = res
            ba, aa = res["before_aggregate"], res["after_aggregate"]
            print(
                "  before: precision=%.2f AuthFP_ep=%.2f AuthFP_real=%.2f prime=%.2f exact=%.2f"
                % (ba["route_precision_semantic"], ba["AuthFP_episode"], ba["AuthFP_real"], ba["prime_rate"], ba["exact_match"])
            )
            print(
                "  after:  precision=%.2f AuthFP_ep=%.2f AuthFP_real=%.2f prime=%.2f exact=%.2f"
                % (aa["route_precision_semantic"], aa["AuthFP_episode"], aa["AuthFP_real"], aa["prime_rate"], aa["exact_match"])
            )

    self_consistency_results: Dict[str, Any] = {}
    if args.self_consistency_from:
        print("\n### Method 2: self-consistency majority vote (from %s) ###" % args.self_consistency_from)
        self_consistency_results = run_self_consistency(
            args.self_consistency_from, paths, args.models, args.bootstrap_iters
        )
        for model, res in self_consistency_results.items():
            a = res["aggregate"]
            print(
                "  %s: voted precision=%.2f AuthFP_ep=%.2f AuthFP_real=%.2f prime=%.2f exact=%.2f"
                % (model, a["route_precision_semantic"], a["AuthFP_episode"], a["AuthFP_real"], a["prime_rate"], a["exact_match"])
            )

    out = {
        "F_STAR_SOURCE": itb.F_STAR_SOURCE,
        "instances": [os.path.basename(p) for p in paths],
        "K_rounds": args.rounds,
        "max_examples_per_claim": args.max_examples,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_cluster_unit": "instance",
        "refinement": refinement_results,
        "self_consistency": self_consistency_results,
        "self_consistency_source": args.self_consistency_from,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
