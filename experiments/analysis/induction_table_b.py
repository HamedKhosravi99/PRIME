#!/usr/bin/env python3
"""Table B: license-INDUCTION QUALITY against the executable oracle.

Measures induction quality (rather than leaving it under-specified) by
scoring the induced program against an executable support oracle F*, not
merely against schema-validity. Covered here:
  - program AST + canonicalization
  - the metric families: claim/order/atom/route/minimality
  - the direct induction metrics table
  - UsableEpisodeRate
  - invalid programs counted in the denominator
  - AuthFP / AuthFN against F*

Oracle
------
This script imports ``prime_runtime.oracle`` (built independently of this
script) and uses it as F* wherever it defines the relevant function:
``enumerate_complete_states`` for Q, ``oracle_authorization`` for F*(c, omega),
``derive_prime_routes`` for the canonical gold family (prime routes are one
derived representation of F*, never the benchmark truth itself), and
``auth_fp``/``auth_fn`` for the AuthFP/AuthFN metrics. ``prime_runtime.oracle``
performs its own independent Boolean evaluation of the gold-route DNF (it does
not call into ``auth.py``/``closure.py`` to decide what F* says -- see its
module docstring), so PRIME's own route-matching machinery never defines
benchmark truth. If ``prime_runtime.oracle`` is ever unavailable (e.g. this
script is re-run in an older checkout), we fall back to a substitute: the
instance's gold route family evaluated under the runtime's own
Auth/downward-closure semantics (``prime_runtime.auth``) -- semantically
identical for these SWE instances (tests/test_oracle.py checks
that derived and authored gold routes coincide on every instance under
data/pilot/instances/), but NOT independent of PRIME's own code the way
``prime_runtime.oracle`` is, so a run in fallback mode is flagged as such
(``F_STAR_SOURCE`` in the JSON output) and should be re-run once the oracle
module is available.

What is and is not elicited from the model
-------------------------------------------
``runner.arms.induce_program`` (used here UNCHANGED) only
elicits the ROUTE component of the program: claim ids, the claim order, and
the atom vocabulary are all supplied to the model as fixed context and are
hard-coded into the built ``Program`` by ``arms._build_validated_program``.
Consequently:
  - "claim-id set vs gold" and "order-edge F1" are TRIVIALLY 1.0 for every
    schema-valid induced program in the CURRENT elicitation format -- there
    is no freedom for the model to add/drop a claim or an order edge.  We
    report this honestly (see the ``*_note`` fields in the JSON) instead of
    quietly presenting a proxy under the official name.
  - "Resolver exact-match" is likewise trivial: an atom id maps to exactly
    one registered resolver already declared by the instance; the model
    cannot propose a resolver, only cite an atom id from the given list.
  - We ADDITIONALLY report non-trivial PROXIES that use the one thing the
    model actually predicts (which atoms it decides are evidentially
    necessary, and for which claims it supplies a nonempty license family):
    ``claim_license_recall/precision/f1`` and ``atom_use_recall/precision/f1``.
    These are clearly named distinctly from the "official" (trivial) fields.

Canonicalization -- applies to the INDUCED program only
-------------------------------------------------------
Before comparison, the induced route family is canonicalized: (1) drop
self-contradictory routes (both a literal and its negation), (2) Phi-reduce
-- iteratively drop a literal if every Phi-realizable completion of the
remaining literals already forces it (a pure Phi-based redundancy,
independent of the oracle), (3) drop routes that are Phi-unrealizable,
(4) dedup identical routes, (5) drop dominated routes (a strict superset of
a sibling in the same family), (6) sort for a canonical lexical
representative. Phi-reduction and dominance-pruning are exact-set-preserving
over Omega_Phi (removing a Phi-redundant literal, or a dominated route,
never changes which realizable states the family accepts), so the SAME
canonical induced program is used for every downstream metric (route
precision/coverage/prime-rate, AuthFP/AuthFN, exact-match). The gold side
uses ``oracle.derive_prime_routes`` directly (already the Phi-aware prime-
implicant family of F*) rather than a second,
redundant canonicalization pass.

Metrics (per model, over all SWE instances; single draw per (model,
instance) unless --draws > 1 -- see ``draws_per_cell`` in the output)
-----------------------------------------------------------------------
  valid_rate               schema-valid / attempted  (denominator = ALL attempts)
  route_recall              gold canonical routes recovered exactly, macro over claims
  route_precision (semantic) forall realizable omega: p(omega)=1 => F*(c,omega)=1
  route_coverage (semantic)  fraction of F*-supported states covered by >=1 induced route
  prime_rate                 induced routes that are F*-sufficient AND F*-irredundant
  exact_match                canonical induced family == canonical gold family, ALL claims
  AuthFP / AuthFN             exact enumeration over Omega_Phi
  AuthFP_real / AuthFN_real   the same, at the instance's actual (honest) evidence state
  UsableEpisodeRate           valid program AND nontrivial AND oracle-sound output
                               at the real evidence state; ALL attempts in the denominator
                               (an invalid program is scored as "abstains" -- it can never
                               AuthFP, but it always AuthFNs whenever F* is nonempty)
  claim_license_R/P/F1        proxy (see above)
  atom_use_R/P/F1              proxy (see above)
  failure modes                omitted / unsound_under_specified / over_specified /
                               incomplete_coverage / exact / exact_alt, per claim
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PILOT)
sys.path.insert(0, os.path.join(PILOT, "src"))

from prime_runtime import auth as _auth  # noqa: E402
from prime_runtime import closure as _closure  # noqa: E402
from prime_runtime.schema import (  # noqa: E402
    Program,
    load_instance,
    make_program,
    parse_literal,
)
from prime_agents.llm import LLMClient  # noqa: E402
from prime_agents.arms import InducedProgramInvalid, induce_program  # noqa: E402
from bootstrap_utils import (  # noqa: E402
    cluster_bootstrap_mean,
    cluster_bootstrap_ratio,
)

F_STAR_SOURCE = "gold_route_family_via_auth_closure (fallback: prime_runtime.oracle not importable)"
try:
    import prime_runtime.oracle as _oracle_mod  # type: ignore

    F_STAR_SOURCE = "prime_runtime.oracle (support/oracle_authorization/derive_prime_routes/auth_fp/auth_fn)"
except ImportError:  # pragma: no cover - exercised only in a checkout without oracle.py
    _oracle_mod = None

DEFAULT_MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-5",
]
LOCAL_BASE = "http://localhost:11434/v1"
LOCAL_KEY = "local"
OPENAI_BASE = "https://api.openai.com/v1"

Facts = Dict[str, str]  # atom -> 'T'|'F' (complete states) or 'T'|'F'|'U' (real states)
OmegaList = List[Facts]


# ---------------------------------------------------------------------------
# Literal/route satisfaction over a facts map (shared convention with
# prime_runtime.oracle's ``_sat_literal``: a literal is confirmed iff its
# atom is determined to the literal's polarity; 'U'/'C'/absent confirm
# neither polarity -- fail-closed, matching closure.entails).
# ---------------------------------------------------------------------------


def _confirmed(literal: str, facts: Mapping[str, str]) -> bool:
    atom, pos = parse_literal(literal)
    return facts.get(atom, "U") == ("T" if pos else "F")


def _sat(route: Sequence[str], facts: Mapping[str, str]) -> bool:
    return all(_confirmed(lit, facts) for lit in route)


def _real_facts(inst) -> Facts:
    """The instance's actual (honest) evidence resolution: gold_value per
    atom, or 'U' when the atom has no resolver / no declared gold_value (the
    __noev variants' deliberately-unresolvable atoms)."""
    with open(inst.path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {a["id"]: (a.get("gold_value") if a.get("gold_value") in ("T", "F") else "U") for a in raw["atoms"]}


# ---------------------------------------------------------------------------
# Canonicalization -- applies to the INDUCED program
# ---------------------------------------------------------------------------


def _phi_reduce(route: FrozenSet[str], omega_list: OmegaList) -> FrozenSet[str]:
    """Drop a literal iff every Phi-realizable completion of the remainder
    already forces it (pure-Phi redundancy, oracle-independent)."""
    route = set(route)
    changed = True
    while changed:
        changed = False
        for lit in sorted(route):
            rest = route - {lit}
            if not rest:
                continue  # never reduce a route to empty (different semantics)
            atom, pos = parse_literal(lit)
            want = "T" if pos else "F"
            sub = [w for w in omega_list if _sat(rest, w)]
            if sub and all(w.get(atom) == want for w in sub):
                route = rest
                changed = True
                break
    return frozenset(route)


def canonicalize_family(
    routes: Sequence[Sequence[str]], omega_list: OmegaList
) -> Tuple[FrozenSet[str], ...]:
    """One claim's route family -> canonical tuple of frozensets. See module
    docstring "Canonicalization" for the six-step procedure."""
    reduced = set()
    for route in routes:
        lits = set(route)
        atoms_seen: Dict[str, bool] = {}
        contradictory = False
        for lit in lits:
            atom, pos = parse_literal(lit)
            if atom in atoms_seen and atoms_seen[atom] != pos:
                contradictory = True
                break
            atoms_seen[atom] = pos
        if contradictory:
            continue
        r = _phi_reduce(frozenset(lits), omega_list)
        if not any(_sat(r, w) for w in omega_list):
            continue  # Phi-unrealizable even after reduction
        reduced.add(r)
    kept = [r for r in reduced if not any(r > other for other in reduced if other != r)]
    kept.sort(key=lambda s: sorted(s))
    return tuple(kept)


def canonicalize_program(program: Program, omega_list: OmegaList) -> Dict[str, Tuple[FrozenSet[str], ...]]:
    return {cid: canonicalize_family(program.routes(cid), omega_list) for cid in program.claims}


def _make_effective_program(inst, canonical: Mapping[str, Sequence[FrozenSet[str]]]) -> Program:
    return make_program(
        claims=inst.claim_ids, order=inst.order,
        licenses={c: [list(r) for r in fam] for c, fam in canonical.items()},
        scope=inst.scope, atoms=inst.atom_ids,
    )


def _full_auth_matrix(program: Program, omega_list: OmegaList) -> Dict[str, List[bool]]:
    """Per claim, whether it is in Auth_program(facts) for every state
    (index-aligned with omega_list) -- used only by the fallback backend."""
    licensed = {d: [any(_sat(r, w) for r in program.routes(d)) for w in omega_list] for d in program.claims}
    dominators = {c: [d for d in program.claims if _auth.preceq(c, d, program)] for c in program.claims}
    return {c: [any(licensed[d][i] for d in dominators[c]) for i in range(len(omega_list))] for c in program.claims}


# ---------------------------------------------------------------------------
# Oracle backends (model-independent, cached once per instance)
# ---------------------------------------------------------------------------


class _OracleBackedInstanceOracle:
    """F* via ``prime_runtime.oracle`` -- the genuinely independent
    executable oracle."""

    def __init__(self, inst) -> None:
        self.inst = inst
        self.omega: OmegaList = list(_oracle_mod.enumerate_complete_states(inst))
        self.levels = {c.id: c.level for c in inst.claims}
        self.gold_canonical = {
            cid: tuple(sorted((frozenset(r) for r in _oracle_mod.derive_prime_routes(inst, cid)), key=lambda s: sorted(s)))
            for cid in inst.claim_ids
        }
        self._fstar = {
            cid: [_oracle_mod.oracle_authorization(inst, cid, facts) for facts in self.omega]
            for cid in inst.claim_ids
        }

    def fstar(self, cid: str, i: int) -> bool:
        return self._fstar[cid][i]

    def route_precise(self, route: FrozenSet[str], cid: str) -> bool:
        return all(self.fstar(cid, i) for i, w in enumerate(self.omega) if _sat(route, w))

    def route_irredundant(self, route: FrozenSet[str], cid: str) -> bool:
        if len(route) <= 1:
            return True
        for lit in route:
            rest = route - {lit}
            if self.route_precise(rest, cid):
                return False
        return True

    def supported_idxs(self, cid: str) -> List[int]:
        return [i for i in range(len(self.omega)) if self.fstar(cid, i)]

    def auth_fp(self, program: Program) -> Tuple[float, bool, Dict[str, Any]]:
        res = _oracle_mod.auth_fp(program, self.inst, self.omega)
        return res["rate"], res["n_flagged"] > 0, res

    def auth_fn(self, program: Program) -> Tuple[float, bool, Dict[str, Any]]:
        res = _oracle_mod.auth_fn(program, self.inst, self.omega)
        return res["rate"], res["n_flagged"] > 0, res

    def real_auth_star(self) -> Optional[set]:
        facts = _real_facts(self.inst)
        return {c for c in self.inst.claim_ids if _oracle_mod.oracle_authorization(self.inst, c, facts)}

    def real_auth_hat(self, program: Program) -> Optional[set]:
        facts = _real_facts(self.inst)
        try:
            led = _oracle_mod.facts_to_ledger(self.inst, facts)
        except Exception:
            return None
        return set(_auth.auth_ids(program, led))


class _FallbackInstanceOracle:
    """Fallback F* (used only when ``prime_runtime.oracle`` cannot be
    imported): the instance's gold route family, evaluated under the SAME
    Auth/downward-closure semantics the runtime itself uses. NOT independent
    of ``auth.py``/``closure.py`` the way ``prime_runtime.oracle`` is
    -- re-run against the oracle module once available."""

    def __init__(self, inst) -> None:
        self.inst = inst
        bool_omega = _closure.realizable_assignments(inst.atom_ids, inst.phi_patterns)
        self.omega: OmegaList = [{a: ("T" if v else "F") for a, v in w.items()} for w in bool_omega]
        self.levels = {c.id: c.level for c in inst.claims}
        self.gold_canonical = canonicalize_program(inst.gold_program(), self.omega)
        gold_canon_prog = _make_effective_program(inst, self.gold_canonical)
        self._fstar = _full_auth_matrix(gold_canon_prog, self.omega)

    def fstar(self, cid: str, i: int) -> bool:
        return self._fstar[cid][i]

    def route_precise(self, route: FrozenSet[str], cid: str) -> bool:
        return all(self.fstar(cid, i) for i, w in enumerate(self.omega) if _sat(route, w))

    def route_irredundant(self, route: FrozenSet[str], cid: str) -> bool:
        if len(route) <= 1:
            return True
        for lit in route:
            rest = route - {lit}
            if self.route_precise(rest, cid):
                return False
        return True

    def supported_idxs(self, cid: str) -> List[int]:
        return [i for i in range(len(self.omega)) if self.fstar(cid, i)]

    def auth_fp(self, program: Program) -> Tuple[float, bool, Dict[str, Any]]:
        ind_auth = _full_auth_matrix(program, self.omega)
        n = len(self.omega)
        flagged = []
        for i in range(n):
            bad = [c for c in self.inst.claim_ids if ind_auth[c][i] and not self.fstar(c, i)]
            if bad:
                flagged.append({"facts": self.omega[i], "claims": bad})
        rate = len(flagged) / n if n else 0.0
        return rate, len(flagged) > 0, {"flagged_states": flagged, "rate": rate, "n_flagged": len(flagged), "n_states": n}

    def auth_fn(self, program: Program) -> Tuple[float, bool, Dict[str, Any]]:
        ind_auth = _full_auth_matrix(program, self.omega)
        n = len(self.omega)
        flagged = []
        for i in range(n):
            bad = [c for c in self.inst.claim_ids if self.fstar(c, i) and not ind_auth[c][i]]
            if bad:
                flagged.append({"facts": self.omega[i], "claims": bad})
        rate = len(flagged) / n if n else 0.0
        return rate, len(flagged) > 0, {"flagged_states": flagged, "rate": rate, "n_flagged": len(flagged), "n_states": n}

    def _real_closure(self, facts: Facts):
        cmap = _closure.cl_states(self.inst.atom_ids, self.inst.phi_patterns, facts)
        return None if _closure.is_inconsistent(cmap) else cmap

    def real_auth_star(self) -> Optional[set]:
        cmap = self._real_closure(_real_facts(self.inst))
        if cmap is None:
            return None
        gold_prog = self.inst.gold_program()
        licensed = {d: any(_closure.satisfies(cmap, r) for r in gold_prog.routes(d)) for d in gold_prog.claims}
        return {c for c in gold_prog.claims if any(_auth.preceq(c, d, gold_prog) for d in gold_prog.claims if licensed[d])}

    def real_auth_hat(self, program: Program) -> Optional[set]:
        cmap = self._real_closure(_real_facts(self.inst))
        if cmap is None:
            return None
        licensed = {d: any(_closure.satisfies(cmap, r) for r in program.routes(d)) for d in program.claims}
        return {c for c in program.claims if any(_auth.preceq(c, d, program) for d in program.claims if licensed[d])}


InstanceOracle = _OracleBackedInstanceOracle if _oracle_mod is not None else _FallbackInstanceOracle
_ORACLE_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Per-(model, instance) scoring
# ---------------------------------------------------------------------------


def classify_claim(cid, ind_fam, gold_fam, oracle) -> str:
    if not ind_fam:
        return "omitted"
    flags = [(oracle.route_precise(r, cid), oracle.route_irredundant(r, cid)) for r in ind_fam]
    if any(not p for p, _ in flags):
        return "unsound_under_specified"
    if set(ind_fam) == set(gold_fam):
        return "exact"
    if any(not irr for _, irr in flags):
        return "over_specified"
    supported = oracle.supported_idxs(cid)
    if not supported:
        return "vacuous_claim"
    covered = [i for i in supported if any(_sat(r, oracle.omega[i]) for r in ind_fam)]
    if len(covered) < len(supported):
        return "incomplete_coverage"
    return "exact_alt"


def score_instance(inst, oracle, induced_program: Optional[Program], valid: bool) -> Dict[str, Any]:
    """Score one (model, instance) draw. ``induced_program`` is None when
    induction was invalid; we still score it (invalid programs count in the
    denominator), as the runtime's documented abstention behavior (an empty
    license family everywhere)."""
    if induced_program is None:
        ind_canonical: Dict[str, Tuple[FrozenSet[str], ...]] = {cid: () for cid in inst.claim_ids}
    else:
        ind_canonical = canonicalize_program(induced_program, oracle.omega)
    eff_program = _make_effective_program(inst, ind_canonical)

    authfp_state_frac, authfp_episode, fp_detail = oracle.auth_fp(eff_program)
    authfn_state_frac, authfn_episode, fn_detail = oracle.auth_fn(eff_program)

    util_num = sum(oracle.levels.get(c, 1) for st in fn_detail["flagged_states"] for c in st["claims"])
    util_den = sum(
        oracle.levels.get(c, 1)
        for i in range(len(oracle.omega))
        for c in inst.claim_ids
        if oracle.fstar(c, i)
    )

    route_records = []
    for cid in inst.claim_ids:
        for r in ind_canonical.get(cid, ()):
            precise = oracle.route_precise(r, cid)
            irred = oracle.route_irredundant(r, cid)
            route_records.append({"claim": cid, "precise": precise, "prime": precise and irred})

    rec_num = rec_den = 0
    for cid in inst.claim_ids:
        g = set(oracle.gold_canonical.get(cid, ()))
        i_fam = set(ind_canonical.get(cid, ()))
        rec_den += len(g)
        rec_num += sum(1 for gr in g if gr in i_fam)

    coverages = []
    for cid in inst.claim_ids:
        supported = oracle.supported_idxs(cid)
        if not supported:
            continue
        fam = ind_canonical.get(cid, ())
        covered = [i for i in supported if any(_sat(r, oracle.omega[i]) for r in fam)]
        coverages.append(len(covered) / len(supported))

    exact_program = all(
        set(ind_canonical.get(cid, ())) == set(oracle.gold_canonical.get(cid, ()))
        for cid in inst.claim_ids
    )

    gold_claims_with_routes = {cid for cid in inst.claim_ids if oracle.gold_canonical.get(cid)}
    ind_claims_with_routes = {cid for cid in inst.claim_ids if ind_canonical.get(cid)}
    gold_atoms_used = {parse_literal(l)[0] for fam in oracle.gold_canonical.values() for r in fam for l in r}
    ind_atoms_used = {parse_literal(l)[0] for fam in ind_canonical.values() for r in fam for l in r}

    failure_modes = {
        cid: classify_claim(cid, ind_canonical.get(cid, ()), oracle.gold_canonical.get(cid, ()), oracle)
        for cid in inst.claim_ids
    }

    real_auth_star = oracle.real_auth_star()
    real_auth_hat = oracle.real_auth_hat(eff_program) if valid else set()
    if real_auth_star is None or real_auth_hat is None:
        real_authfp = real_authfn = usable = None
    else:
        real_authfp = bool(real_auth_hat - real_auth_star)
        real_authfn = bool(real_auth_star - real_auth_hat)
        usable = bool(valid and real_auth_hat and not real_authfp)

    return {
        "route_recall_num": rec_num, "route_recall_den": rec_den,
        "route_records": route_records,
        # Canonical induced route family (self-consistency voting and
        # oracle-guided refinement both need the actual induced routes, not
        # just precise/prime booleans, so every draw carries its own
        # canonicalized program here -- an empty family per claim for an
        # invalid draw, which correctly contributes zero votes downstream).
        "induced_licenses": {cid: [sorted(r) for r in fam] for cid, fam in ind_canonical.items()},
        "coverages": coverages,
        "exact_program": exact_program,
        "authfp_state_frac": authfp_state_frac, "authfn_state_frac": authfn_state_frac,
        "authfp_episode": authfp_episode, "authfn_episode": authfn_episode,
        "util_num": util_num, "util_den": util_den,
        "gold_claims_with_routes": len(gold_claims_with_routes),
        "claim_license_hit": len(gold_claims_with_routes & ind_claims_with_routes),
        "ind_claims_with_routes": len(ind_claims_with_routes),
        "gold_atoms_used": len(gold_atoms_used),
        "atom_use_hit": len(gold_atoms_used & ind_atoms_used),
        "ind_atoms_used": len(ind_atoms_used),
        "failure_modes": failure_modes,
        "real_authfp": real_authfp, "real_authfn": real_authfn, "usable": usable,
    }


# ---------------------------------------------------------------------------
# Aggregation / driver
# ---------------------------------------------------------------------------


def _pick_base_key(model: str, args) -> Tuple[str, str]:
    if model.startswith("qwen"):
        return args.local_base, args.local_key
    return args.openai_base, args.openai_key


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def run_model(model: str, paths: Sequence[str], args) -> Dict[str, Any]:
    base, key = _pick_base_key(model, args)
    temperature = getattr(args, "temperature", 0.0)
    llm = LLMClient(model=model, base=base, key=key, temperature=temperature)
    n = valid = 0
    per_instance = []
    for p in paths:
        inst = load_instance(p)
        oracle = _ORACLE_CACHE.setdefault(p, InstanceOracle(inst))
        for draw_idx in range(args.draws):
            n += 1
            program = None
            is_valid = False
            err = ""
            for _attempt in range(2):  # transient-network resilience only
                try:
                    program = induce_program(inst, llm)
                    is_valid = True
                    break
                except InducedProgramInvalid as exc:
                    err = str(exc)
                    break
                except Exception as exc:  # noqa: BLE001 - transient network error
                    err = "%s: %s" % (type(exc).__name__, exc)
                    time.sleep(1.5)
            if is_valid:
                valid += 1
            record = score_instance(inst, oracle, program, is_valid)
            record["instance"] = inst.id
            record["draw"] = draw_idx
            record["valid"] = is_valid
            record["error"] = err
            per_instance.append(record)
    return {
        "model": model, "n": n, "valid": valid, "per_instance": per_instance,
        "temperature_requested": temperature,
        "temperature_effective": ("~1.0 (API default; gpt-5/o-series reject a custom "
                                   "value so the request omits the parameter)"
                                   if llm._restricted_temp() else temperature),
    }


def aggregate(model_result: Dict[str, Any]) -> Dict[str, Any]:
    recs = model_result["per_instance"]
    n = model_result["n"]
    valid = model_result["valid"]
    valid_recs = [r for r in recs if r["valid"]]

    def macro(key_num, key_den):
        tot_num = sum(r[key_num] for r in valid_recs)
        tot_den = sum(r[key_den] for r in valid_recs)
        return tot_num / tot_den if tot_den else 0.0

    route_recall = macro("route_recall_num", "route_recall_den")

    all_routes = [rr for r in valid_recs for rr in r["route_records"]]
    route_precision = sum(1 for rr in all_routes if rr["precise"]) / len(all_routes) if all_routes else 0.0
    prime_rate = sum(1 for rr in all_routes if rr["prime"]) / len(all_routes) if all_routes else 0.0

    all_cov = [c for r in valid_recs for c in r["coverages"]]
    route_coverage = sum(all_cov) / len(all_cov) if all_cov else 0.0

    exact_match = sum(1 for r in valid_recs if r["exact_program"]) / len(valid_recs) if valid_recs else 0.0

    authfp_episode = sum(1 for r in valid_recs if r["authfp_episode"]) / len(valid_recs) if valid_recs else 0.0
    authfn_episode = sum(1 for r in valid_recs if r["authfn_episode"]) / len(valid_recs) if valid_recs else 0.0
    authfp_state = sum(r["authfp_state_frac"] for r in valid_recs) / len(valid_recs) if valid_recs else 0.0
    authfn_state = sum(r["authfn_state_frac"] for r in valid_recs) / len(valid_recs) if valid_recs else 0.0

    util_num = sum(r["util_num"] for r in valid_recs)
    util_den = sum(r["util_den"] for r in valid_recs)
    util_weighted_authfn = util_num / util_den if util_den else 0.0

    claim_lic_recall = macro("claim_license_hit", "gold_claims_with_routes")
    claim_lic_precision = macro("claim_license_hit", "ind_claims_with_routes")
    atom_use_recall = macro("atom_use_hit", "gold_atoms_used")
    atom_use_precision = macro("atom_use_hit", "ind_atoms_used")

    # ALL attempts (valid + invalid) in the denominator. score_instance
    # scores an invalid draw as abstention: it can never AuthFP but AuthFNs
    # whenever F* authorizes anything at the real state; usable is always
    # False for an invalid draw.
    known_fp = [r for r in recs if r["real_authfp"] is not None]
    known_fn = [r for r in recs if r["real_authfn"] is not None]
    authfp_real = sum(1 for r in known_fp if r["real_authfp"]) / len(known_fp) if known_fp else 0.0
    authfn_real = sum(1 for r in known_fn if r["real_authfn"]) / len(known_fn) if known_fn else 0.0
    usable_rate = sum(1 for r in recs if r.get("usable")) / n if n else 0.0

    failure_counts: Dict[str, int] = {}
    total_claims = 0
    for r in valid_recs:
        for _cid, bucket in r["failure_modes"].items():
            failure_counts[bucket] = failure_counts.get(bucket, 0) + 1
            total_claims += 1
    failure_frac = {k: v / total_claims for k, v in failure_counts.items()} if total_claims else {}

    return {
        "model": model_result["model"],
        "n": n, "valid": valid, "valid_rate": valid / n if n else 0.0,
        "route_recall": route_recall,
        "route_precision_semantic": route_precision,
        "route_coverage_semantic": route_coverage,
        "prime_rate": prime_rate,
        "exact_match": exact_match,
        "order_edge_f1": 1.0,
        "order_edge_f1_note": "trivial: order is harness-fixed, not elicited by induce_program",
        "resolver_exact_match": 1.0,
        "resolver_exact_match_note": "trivial: atom->resolver binding is harness-fixed, not elicited",
        "schema_miss_rate": 0.0,
        "schema_miss_rate_note": "trivial: claim/atom schema is closed and supplied to the model, so an out-of-schema request cannot occur in this elicitation",
        "AuthFP_episode": authfp_episode,
        "AuthFP_state_mean": authfp_state,
        "AuthFP_real": authfp_real,
        "AuthFN_episode": authfn_episode,
        "AuthFN_state_mean": authfn_state,
        "AuthFN_real": authfn_real,
        "utility_weighted_AuthFN": util_weighted_authfn,
        "UsableEpisodeRate": usable_rate,
        "claim_license_recall_proxy": claim_lic_recall,
        "claim_license_precision_proxy": claim_lic_precision,
        "claim_license_f1_proxy": _f1(claim_lic_precision, claim_lic_recall),
        "atom_use_recall_proxy": atom_use_recall,
        "atom_use_precision_proxy": atom_use_precision,
        "atom_use_f1_proxy": _f1(atom_use_precision, atom_use_recall),
        "failure_modes_frac": failure_frac,
        "failure_modes_count": failure_counts,
        "n_claims_scored": total_claims,
    }


CI_METRICS = (
    "valid_rate", "route_recall", "route_precision_semantic", "route_coverage_semantic",
    "prime_rate", "exact_match", "AuthFP_episode", "AuthFP_real",
)


def bootstrap_table_b_cis(model_result: Dict[str, Any], iters: int = 5000, seed: int = 7) -> Dict[str, Any]:
    """A proper 95% CI for the headline metrics (N=9 instances with a single
    draw each is too few for point estimates alone), via an INSTANCE-level
    cluster bootstrap (see ``experiments/analysis/bootstrap_utils.py`` -- the same
    resampling convention ``experiments/analysis/aggregate.py`` already uses for its
    seed-clustered CIs, here generalized to draws x instances: a bootstrap
    replicate resamples instances with replacement and pools ALL of a resampled
    instance's draws together, never resampling within an instance).

    Returns ``{metric: {"point", "lo", "hi", "n_clusters"}}``. The point
    estimate here is recomputed from the same pooled data ``aggregate()``
    uses (not a separate estimator), so it matches ``aggregate()``'s value
    up to floating point; the CI is the new information.
    """
    recs = model_result["per_instance"]
    valid_recs = [r for r in recs if r["valid"]]

    def _group(records: Sequence[Mapping[str, Any]]) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for r in records:
            out.setdefault(r["instance"], []).append(r)
        return out

    valid_by_instance = _group(valid_recs)

    valid_flags: Dict[str, List[float]] = {}
    for r in recs:
        valid_flags.setdefault(r["instance"], []).append(1.0 if r["valid"] else 0.0)

    precision_vals: Dict[str, List[float]] = {}
    prime_vals: Dict[str, List[float]] = {}
    for inst, rs in valid_by_instance.items():
        for r in rs:
            for rr in r["route_records"]:
                precision_vals.setdefault(inst, []).append(1.0 if rr["precise"] else 0.0)
                prime_vals.setdefault(inst, []).append(1.0 if rr["prime"] else 0.0)

    coverage_vals: Dict[str, List[float]] = {}
    for inst, rs in valid_by_instance.items():
        for r in rs:
            coverage_vals.setdefault(inst, []).extend(r["coverages"])

    exact_vals: Dict[str, List[float]] = {}
    authfp_ep_vals: Dict[str, List[float]] = {}
    for inst, rs in valid_by_instance.items():
        for r in rs:
            exact_vals.setdefault(inst, []).append(1.0 if r["exact_program"] else 0.0)
            authfp_ep_vals.setdefault(inst, []).append(1.0 if r["authfp_episode"] else 0.0)

    authfp_real_vals: Dict[str, List[float]] = {}
    for r in recs:  # all attempts, matching AuthFP_real's own denominator
        if r["real_authfp"] is not None:
            authfp_real_vals.setdefault(r["instance"], []).append(1.0 if r["real_authfp"] else 0.0)

    recall_pairs: Dict[str, List[Tuple[float, float]]] = {}
    for inst, rs in valid_by_instance.items():
        for r in rs:
            recall_pairs.setdefault(inst, []).append((r["route_recall_num"], r["route_recall_den"]))

    out: Dict[str, Any] = {}
    for name, data in (
        ("valid_rate", valid_flags),
        ("route_precision_semantic", precision_vals),
        ("route_coverage_semantic", coverage_vals),
        ("prime_rate", prime_vals),
        ("exact_match", exact_vals),
        ("AuthFP_episode", authfp_ep_vals),
        ("AuthFP_real", authfp_real_vals),
    ):
        pt, lo, hi = cluster_bootstrap_mean(data, iters=iters, seed=seed)
        out[name] = {"point": pt, "lo": lo, "hi": hi, "n_clusters": len(data)}
    pt, lo, hi = cluster_bootstrap_ratio(recall_pairs, iters=iters, seed=seed)
    out["route_recall"] = {"point": pt, "lo": lo, "hi": hi, "n_clusters": len(recall_pairs)}
    return out


MD_HEADER = (
    "| model | valid | route R | route P (sem.) | coverage | prime | exact | "
    "AuthFP (ep.) | AuthFN (ep.) | Usable | unsound-under% | over-spec% |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def md_row(a: Dict[str, Any]) -> str:
    fm = a["failure_modes_frac"]
    return "| %s | %d/%d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f%% | %.1f%% |" % (
        a["model"], a["valid"], a["n"],
        a["route_recall"], a["route_precision_semantic"], a["route_coverage_semantic"],
        a["prime_rate"], a["exact_match"],
        a["AuthFP_episode"], a["AuthFN_episode"], a["UsableEpisodeRate"],
        100 * fm.get("unsound_under_specified", 0.0), 100 * fm.get("over_specified", 0.0),
    )


def latex_rows(aggs: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for a in aggs:
        name = a["model"].replace("_", r"\_")
        lines.append(
            "%s & %d/%d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\"
            % (
                name, a["valid"], a["n"],
                a["route_recall"], a["route_precision_semantic"], a["route_coverage_semantic"],
                a["prime_rate"], a["exact_match"],
                a["AuthFP_episode"], a["AuthFN_episode"], a["UsableEpisodeRate"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--instances", default=os.path.join(PILOT, "data", "pilot", "instances", "swe", "*.json"))
    ap.add_argument("--draws", type=int, default=1, help="induction draws per (model, instance)")
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="sampling temperature requested for every temp-0-capable model "
        "(gpt-5/o-series always ignore this -- see LLMClient._restricted_temp). "
        "Use e.g. 1.0 to run the matched-temperature panel.",
    )
    ap.add_argument("--bootstrap-iters", type=int, default=5000, help="cluster-bootstrap resamples per CI")
    ap.add_argument("--openai-base", default=os.environ.get("PRIME_OPENAI_BASE", OPENAI_BASE))
    ap.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--local-base", default=os.environ.get("PRIME_LOCAL_BASE", LOCAL_BASE))
    ap.add_argument("--local-key", default=os.environ.get("PRIME_LOCAL_KEY", LOCAL_KEY))
    ap.add_argument("--out", default=os.path.join(PILOT, "results", "induction_table_b.json"))
    args = ap.parse_args()

    paths = sorted(glob.glob(args.instances))
    if not paths:
        print("no instances matched %r" % args.instances, file=sys.stderr)
        return 1

    print("F* source: %s" % F_STAR_SOURCE)
    print("instances: %s" % ", ".join(os.path.basename(p) for p in paths))
    print("draws per (model, instance): %d%s" % (
        args.draws, "  (SINGLE-DRAW cells -- report accordingly)" if args.draws == 1 else ""))
    print()
    print(MD_HEADER)

    all_aggs = []
    all_raw = []
    all_cis = []
    for model in args.models:
        res = run_model(model, paths, args)
        agg = aggregate(res)
        ci = bootstrap_table_b_cis(res, iters=args.bootstrap_iters)
        all_aggs.append(agg)
        all_raw.append(res)
        all_cis.append(ci)
        print(md_row(agg))

    print()
    print("route precision / AuthFP_real with 95%% CI (instance-cluster bootstrap, %d resamples):"
          % args.bootstrap_iters)
    for model, ci in zip(args.models, all_cis):
        p, fp = ci["route_precision_semantic"], ci["AuthFP_real"]
        print("  %-30s route_precision=%.2f [%.2f,%.2f]  AuthFP_real=%.2f [%.2f,%.2f]  (n_instances=%d)"
              % (model, p["point"], p["lo"], p["hi"], fp["point"], fp["lo"], fp["hi"], p["n_clusters"]))

    print()
    print("LaTeX table body:")
    print(latex_rows(all_aggs))

    out = {
        "F_STAR_SOURCE": F_STAR_SOURCE,
        "draws_per_cell": args.draws,
        "single_draw": args.draws == 1,
        "temperature_requested": args.temperature,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_cluster_unit": "instance",
        "instances": [os.path.basename(p) for p in paths],
        "aggregate": all_aggs,
        "ci": [dict(model=m, **{"metrics": c}) for m, c in zip(args.models, all_cis)],
        "raw": all_raw,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
