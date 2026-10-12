"""Obligation-guided completion: reachable-ceiling and minimal-route
computation for the ``prime_guided`` arm (the paper states the
utility-optimality theorem for this mechanism;
tests/test_guided_optimality.py checks it).

This module computes GUIDANCE ONLY: which atoms to gather next, and whether
termination is currently permitted. It never authorizes anything -- emission
is, and remains, gated exclusively by ``auth.auth_ids`` / ``interpreter.emit``
against the COMMITTED program. Nothing here mutates
the ledger, the program, or bypasses the mask; it is a pure read of the
program + ledger + resolver table already used by status.py/auth.py.

Design
------
At a ledger state z with remaining tool-call budget B (status.py's typed
availability + auth.py's MaxAuth, reused, not re-derived):

1. For every claim c in the committed program, its OWN-ROUTE cost is the
   size of the smallest set of still-OPEN atoms among c's non-DEAD routes
   (status.route_status / status.avail, unchanged). A claim already
   authorized has cost 0. A claim with no non-dead route is unreachable
   (cost = None).

   Restricting to a claim's OWN routes (rather than also crediting it via a
   dominating claim's route) is safe given the benchmark's ORDER-COHERENCE
   invariant (schema.validate_program, checked at instance/program load
   time): every route of a stronger claim contains some route of each
   weaker comparable claim. Consequently if a stronger claim's route is
   reachable (no atom in it is dead), the weaker claim's contained route is
   automatically reachable too -- so scanning claims independently already
   finds the true ceiling.

2. The REACHABLE CEILING is the claim with the highest ``level`` whose cost
   is 0 (already authorized) or <= B (status.py's R_t(c) reused as the per-route
   residual; this module only adds the budget comparison and the argmax).

3. The ACTIVE OBLIGATION is the still-open literals of the ceiling's own
   minimal route (empty once the ceiling is authorized).

4. ``must_continue`` is the termination-rule predicate: true iff the ceiling
   is STRICTLY stronger than whatever is authorized right now and its
   obligation is non-empty -- the caller (src/prime_agents/arms.py's ``prime_guided``
   arm) refuses a premature emit_claim / silent stop while this holds.

Only OPEN atoms are surfaced as obligations (never REPAIRABLE): the current
benchmark family exposes no repair/retraction tool to the agent, so a
repair-blocked route is not actionable by ordinary tool calls and is treated
as unreachable by this mechanism (a documented limitation, not silently
ignored).
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import status as _status
from .schema import Program, atom_of


def _route_open_atoms(
    route: Sequence[str], ledger, resolvers: Mapping[str, Mapping[str, Any]]
) -> Optional[List[str]]:
    """Still-OPEN literals of ``route``, or None if the route is unreachable
    by ordinary tool calls (DEAD, or blocked on a conflict with no exposed
    repair tool -- REPAIR_BLOCKED)."""
    st = _status.route_status(route, ledger, resolvers)
    if st in (_status.DEAD, _status.REPAIR_BLOCKED):
        return None
    return [
        lit for lit in route
        if _status.avail(lit, ledger, resolvers) == _status.OPEN
    ]


def _call_class(atom_id: str, resolvers: Mapping[str, Mapping[str, Any]]):
    """Call-class identity of an atom: (tool, canonical params) when the
    resolver table carries it, else the atom id itself (each atom its own
    class -- the older atom-count semantics, preserved for tables that do
    not declare call classes)."""
    spec = resolvers.get(atom_id, {}) or {}
    return spec.get("call_class", atom_id)


def _open_cost(open_atoms, resolvers) -> int:
    """Cost of a set of open literals = number of DISTINCT call classes."""
    from .schema import atom_of as _ao
    return len({_call_class(_ao(l), resolvers) for l in open_atoms})


def classify_topology(program: Program) -> str:
    """Deterministic topology classifier (docs/EXECUTION_PLAN.md, Part R,
    P5): "ladder" iff every claim has exactly one route AND all routes are
    totally ordered by inclusion (a single inclusion chain); else
    "branching". Tested in test_guidance_v5.py; prefix-first ordering
    activates only on ladders."""
    routes = []
    for cid in program.claims:
        rs = list(program.routes(cid))
        if len(rs) != 1:
            return "branching"
        routes.append(frozenset(rs[0]))
    routes.sort(key=len)
    for a, b in zip(routes, routes[1:]):
        if not a <= b:
            return "branching"
    return "ladder"


def _claim_cost(
    claim_id: str, program: Program, ledger, resolvers: Mapping[str, Mapping[str, Any]]
) -> Optional[Tuple[int, Tuple[str, ...], Tuple[str, ...]]]:
    """(cost, minimal_route, open_atoms) minimizing COST = distinct open
    call classes (v5 semantics) over claim_id's OWN non-dead routes; None if
    none exists."""
    best: Optional[Tuple[int, Tuple[str, ...], Tuple[str, ...]]] = None
    for route in program.routes(claim_id):
        open_atoms = _route_open_atoms(route, ledger, resolvers)
        if open_atoms is None:
            continue
        cand = (_open_cost(open_atoms, resolvers), tuple(route), tuple(open_atoms))
        if (
            best is None
            or cand[0] < best[0]
            or (cand[0] == best[0] and len(cand[1]) < len(best[1]))
        ):
            best = cand
    return best


def reachable_ceiling(
    program: Program,
    ledger,
    resolvers: Mapping[str, Mapping[str, Any]],
    levels: Mapping[str, int],
    remaining_budget: int,
) -> Dict[str, Any]:
    """The reachable ceiling + minimal route + termination predicate at one
    ledger state, computed purely from the COMMITTED program (never gold/
    oracle when the program is induced -- the caller passes whatever program
    is actually committed for this episode)."""
    from . import auth as _auth  # local import: avoid any import-order surprise

    authorized = set(_auth.auth_ids(program, ledger))
    per_claim: Dict[str, Optional[Tuple[int, Tuple[str, ...], Tuple[str, ...]]]] = {
        cid: _claim_cost(cid, program, ledger, resolvers) for cid in program.claims
    }

    def _lvl(cid: Optional[str]) -> int:
        return int(levels.get(cid, 0)) if cid is not None else 0

    auth_in_program = [c for c in authorized if c in program.claims]
    authorized_max = (
        sorted(auth_in_program, key=lambda c: (-_lvl(c), c))[0] if auth_in_program else None
    )

    candidates: List[Tuple[str, int, Tuple[str, ...], Tuple[str, ...]]] = []
    for cid in program.claims:
        if cid in authorized:
            candidates.append((cid, 0, (), ()))
            continue
        cost = per_claim[cid]
        if cost is None:
            continue
        n_open, route, open_atoms = cost
        if n_open <= remaining_budget:
            candidates.append((cid, n_open, route, open_atoms))

    if candidates:
        candidates.sort(key=lambda t: (-_lvl(t[0]), t[1], t[0]))
        ceiling_claim, _cost, ceiling_route, ceiling_open = candidates[0]
    else:
        ceiling_claim, ceiling_route, ceiling_open = authorized_max, (), ()

    blocking_atoms = list(ceiling_open) if ceiling_claim not in authorized else []
    if blocking_atoms and classify_topology(program) == "ladder":
        # Prefix-first tie-break WITHIN the ceiling's minimal route
        # (docs/EXECUTION_PLAN.md, Part R, P5): resolve shared lower-prefix
        # literals before claim-specific extensions. Depth of a literal =
        # number of program routes containing it (on a ladder, prefix atoms
        # appear in more routes). Deterministic: depth desc, then atom id.
        # Never changes WHICH claim is the ceiling -- ordering only.
        def _depth(lit):
            a = atom_of(lit)
            return sum(1 for cid in program.claims
                       for r in program.routes(cid)
                       if any(atom_of(x) == a for x in r))
        blocking_atoms.sort(key=lambda l: (-_depth(l), atom_of(l)))
    tools = {
        atom_of(lit): (resolvers.get(atom_of(lit), {}) or {}).get("tool")
        for lit in blocking_atoms
    }

    must_continue = (
        ceiling_claim is not None
        and bool(blocking_atoms)
        and _lvl(ceiling_claim) > _lvl(authorized_max)
    )

    return {
        "ceiling_claim": ceiling_claim,
        "ceiling_level": _lvl(ceiling_claim),
        "authorized_max_claim": authorized_max,
        "authorized_max_level": _lvl(authorized_max),
        "minimal_route": list(ceiling_route),
        "blocking_atoms": blocking_atoms,
        "tools": tools,
        "must_continue": must_continue,
        "remaining_budget": remaining_budget,
        "per_claim": {
            cid: (
                {"cost": c[0], "route": list(c[1]), "open": list(c[2])}
                if c is not None
                else {"cost": None, "route": [], "open": []}
            )
            for cid, c in per_claim.items()
        },
    }


def claim_levels(instance) -> Dict[str, int]:
    """{claim_id: level} lookup for one Instance -- the small piece of
    context ``reachable_ceiling`` needs beyond the Program (Program.claims
    is just ids; level is an Instance-level annotation)."""
    return {c.id: c.level for c in instance.claims}
