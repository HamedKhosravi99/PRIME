"""Exact robust route absorption checker.

Certifies trusted between-episode patches and grounds the revision-freedom
ablation arms — it is NOT a policy action (full in-episode freeze).

Contract: a revised route p' attached to claim c preserves robust committed
refinement (P' ⊑_rob P_0) iff some route q in the absorption family

    B_0(c) = union of route families of all committed claims d with
             gamma(c) <=* gamma(d)  (including c itself)

is syntactically contained in pi(p') (p' projected to P_0's atom
vocabulary).  Syntactic containment — not complete-assignment entailment and
not one-closure checking — is the deployed check; both shortcuts are falsified
by the recorded counterexamples (kept as regression tests R1/R2/R3):

    R1  Phi = a∨b, routes {{a},{b}}: the empty route passes ontic entailment
        yet authorizes at the all-U ledger (satisfaction is epistemic).
    R2  Phi = a→b, committed {{b}}, revised {a}: closure of {a} contains {b},
        but closure is not monotone when conflicts mask derived literals.
    R3  Phi = (a→b)∧(a→c), routes {{a,b},{a,c}}, revised {a}: reachable-safe
        under some Deltas, rejected by the robust (all conflict masks)
        contract.

Runtime enforcement rules: new claims / order extensions are
freeze violations; unrealizable revised routes are rejected as ill-formed;
route deletion is always accepted; literal addition is auto-accepted (the
original route witnesses absorption); literal removal or route addition must
pass containment.  Polynomial in the number and width of committed routes.
"""

from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Union

from . import auth as _auth
from . import closure as _closure
from .schema import Program, atom_of, make_program

ProgramLike = Union[Program, Mapping[str, Any]]


def _as_program(p: ProgramLike) -> Program:
    if isinstance(p, Program):
        return p
    claims = [c["id"] if isinstance(c, Mapping) else c for c in p.get("claims", ())]
    return make_program(
        claims=claims,
        order=p.get("order", ()),
        licenses=p.get("licenses", {}),
        scope=str(p.get("scope", "")),
        atoms=p.get("atoms", ()),
    )


def absorption_family(claim_id: str, program: Program) -> List[FrozenSet[str]]:
    """B_0(c): routes of c and of every committed claim dominating c."""
    fam: List[FrozenSet[str]] = []
    for d in program.claims:
        if _auth.preceq(claim_id, d, program):
            fam.extend(frozenset(r) for r in program.routes(d))
    return fam


def check_patch(
    p0: ProgramLike,
    p_new: ProgramLike,
    phi: Sequence[Mapping[str, str]] = (),
) -> bool:
    """Robust-contract containment check: True iff P_new ⊑_rob P_0.

    ``phi`` (optional): forbidden determined-value patterns over P_0's atom
    vocabulary; when supplied, unrealizable revised routes are rejected as
    ill-formed.
    """
    P0 = _as_program(p0)
    Pn = _as_program(p_new)
    committed_claims = set(P0.claims)
    # Freeze: no new claims, no order pairs beyond P_0's reflexive-transitive
    # order (adding dominance edges could expand downward-closed Auth).
    for cid in Pn.claims:
        if cid not in committed_claims:
            return False
    for w, s in Pn.order:
        if not _auth.preceq(w, s, P0):
            return False
    p0_atoms = set(P0.atoms)
    for cid, routes in Pn.licenses.items():
        if cid not in committed_claims:
            return False
        family = absorption_family(cid, P0)
        for route in routes:
            if phi and not _closure.route_realizable(route, P0.atoms, phi):
                return False  # ill-formed (unrealizable) revised route
            projected: Set[str] = {
                lit for lit in route if atom_of(lit) in p0_atoms
            }
            if not any(q <= projected for q in family):
                return False
    return True  # route deletion is always accepted (absent routes pass)
