"""Typed availability, route status, obligation frontier, completion.

Implements PRIME-master.md SS6.4 (route status via a typed availability
function; repair-aware residual R~; obligation frontier; global unsat-core
repair frontier) and SS11.5 (RouteCompletion / Completion / five mutually
exclusive per-claim states).

Availability precedence (fail-closed totality, master SS6.4):

    satisfied > repairable > open > impossible

If no case applies after precedence, the resolver has violated its interface
contract and :class:`ResolverError` is raised — the runtime never guesses a
route status.

Resolver table format
---------------------
``resolvers`` is a mapping ``atom_id -> spec`` where spec keys are:

    "tool"       registered evidence action for the atom (str);
    "enabled"    whether that action is currently enabled (default True);
    "impossible" a terminal impossibility certificate for the current
                 artifact/scope version (default False).

Retraction of a provenance-bearing attestation is always a registered repair
action (master SS6.4: repair targets attestations, never frozen constraints).
Use :func:`resolver_table` to build the table from an Instance.
"""

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import closure as _closure
from .closure import InconsistentLedgerError, cl, cl_states, entails, is_inconsistent
from .schema import Instance, Program, negate_literal, parse_literal

SATISFIED = "satisfied"
REPAIRABLE = "repairable"
OPEN = "open"
IMPOSSIBLE = "impossible"

COMPLETE = "complete"
DEAD = "dead"
REPAIR_BLOCKED = "repair-blocked"
ACTIVE = "active"

FRONTIER_ELIGIBLE = (ACTIVE, REPAIR_BLOCKED)


class ResolverError(RuntimeError):
    """Fail-closed resolver-contract violation: no availability case applies
    (master SS6.4 totality).  The runtime authorizes no new claim and
    requires trusted intervention; it never guesses a status."""


def resolver_table(instance: Instance) -> Dict[str, Dict[str, Any]]:
    """Build the availability resolver table from an instance's typed atoms."""
    table: Dict[str, Dict[str, Any]] = {}
    for atom in instance.atoms:
        if atom.resolver is None:
            # authoring-time terminal unavailability (master SS10.5,
            # evidence-unavailable split): no tool can establish the atom
            # for this artifact/scope version -> availability 'impossible'.
            table[atom.id] = {"tool": None, "enabled": False, "impossible": True}
        else:
            table[atom.id] = {
                "tool": atom.resolver.tool,
                "enabled": atom.resolver.tool in instance.actions,
                "impossible": False,
                # call-class identity (EXECUTION_PLAN v5, D3/G8): atoms whose
                # resolver shares (tool, params) are established by ONE
                # underlying call; guidance costs and executor budgets both
                # count distinct call classes, so planner-predicted cost can
                # equal executor-observed cost by construction.
                "call_class": (atom.resolver.tool,
                               tuple(sorted((str(k), str(v)) for k, v in
                                            dict(atom.resolver.params or {}).items()))),
            }
    return table


def _unblockable_by_retraction(literal: str, ledger) -> bool:
    """True when retracting some active attestation stops Cl |= ~literal.

    Implements the 'repairable' predicate for opposition established through
    attestations: a registered repair action (retraction) can unblock the
    literal.  Checked exactly by re-closing with each attested atom's records
    hypothetically dropped (atom counts are small; see closure complexity).
    """
    neg = negate_literal(literal)
    states = ledger.states()
    attested = [a for a in ledger.atoms if states[a] in ("T", "F", "C")]
    for drop in attested:
        trial = dict(states)
        trial[drop] = "U"
        c = cl_states(ledger.atoms, ledger.phi_patterns, trial)
        if is_inconsistent(c):
            continue
        if not entails(c, neg):
            return True
    return False


def avail(literal: str, ledger, resolvers: Mapping[str, Mapping[str, Any]]) -> str:
    """Availability status of a license literal (master SS6.4).

    Returns one of 'satisfied' | 'repairable' | 'open' | 'impossible',
    applying the precedence order; raises :class:`ResolverError` when no case
    applies (fail-closed) and :class:`InconsistentLedgerError` when the
    ledger base is UNSAT (route-level status is suspended; use
    :func:`frontier` for the global repair frontier).
    """
    c = cl(ledger)
    if is_inconsistent(c):
        raise InconsistentLedgerError(
            "route-level status suspended: ledger inconsistent (master SS6.4)"
        )
    atom, _pos = parse_literal(literal)
    if atom not in ledger.atoms:
        raise ResolverError("literal %r references unknown atom" % literal)
    spec = dict(resolvers.get(atom, {}))

    # 1. satisfied
    if entails(c, literal):
        return SATISFIED
    # 2. repairable — a conflict or opposing attestation blocks the literal
    #    and a registered repair action (retraction) is enabled.
    if ledger.state(atom) == "C":
        return REPAIRABLE
    opposed = entails(c, negate_literal(literal))
    if opposed and _unblockable_by_retraction(literal, ledger):
        return REPAIRABLE
    # 3. open — a registered evidence action capable of establishing the
    #    literal is enabled.  An opposed literal cannot be established
    #    without repair, and a certified-impossible literal has no capable
    #    action, so neither is 'open'.
    if (
        not opposed
        and spec.get("tool")
        and spec.get("enabled", True)
        and not spec.get("impossible", False)
    ):
        return OPEN
    # 4. impossible — terminal impossibility certificate for this version.
    if spec.get("impossible", False):
        return IMPOSSIBLE
    raise ResolverError(
        "no availability case applies to %r (state=%s, opposed=%s): "
        "resolver contract violated — fail-closed" % (literal, ledger.state(atom), opposed)
    )


def route_status(
    route: Sequence[str], ledger, resolvers: Mapping[str, Mapping[str, Any]]
) -> str:
    """Route status (master SS6.4): complete | dead | repair-blocked | active.

    Case order makes categories mutually exclusive; terminal impossibility
    takes precedence over a simultaneous repairable conflict.
    """
    statuses = [avail(lit, ledger, resolvers) for lit in route]
    if all(s == SATISFIED for s in statuses):
        return COMPLETE
    if any(s == IMPOSSIBLE for s in statuses):
        return DEAD
    if any(s == REPAIRABLE for s in statuses):
        return REPAIR_BLOCKED
    return ACTIVE


def residual(
    route: Sequence[str], ledger, resolvers: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    """Repair-aware residual R~(p, z): open literals plus repair(l) markers —
    a conflicted literal appears once, as a repair action (master SS6.4)."""
    out: List[str] = []
    for lit in route:
        s = avail(lit, ledger, resolvers)
        if s == OPEN:
            out.append(lit)
        elif s == REPAIRABLE:
            out.append("repair(%s)" % lit)
    return out


def _global_repair_frontier(ledger) -> List[str]:
    """Global unsat-core repair frontier (master SS6.4).

    Phi is hard; each non-conflicted attested value is a labeled assumption
    alpha_i -> l_i.  We return review_or_retract obligations for a
    minimum-cardinality assumption core whose removal restores consistency —
    repair targets provenance-bearing attestations, never the frozen
    constraints.  Exhaustive subset search over attested atoms (few).
    """
    states = ledger.states()
    determined = [a for a in ledger.atoms if states[a] in ("T", "F")]
    for size in range(1, len(determined) + 1):
        for subset in itertools.combinations(determined, size):
            trial = dict(states)
            for a in subset:
                trial[a] = "U"
            if not is_inconsistent(cl_states(ledger.atoms, ledger.phi_patterns, trial)):
                obligations = []
                for a in subset:
                    for rec in ledger.active_records(a):
                        obligations.append(
                            "review_or_retract(%s#%d)" % (a, rec["id"])
                        )
                return obligations
    return []  # unreachable for a genuinely inconsistent base


def frontier(
    claim_id: str,
    program: Program,
    ledger,
    resolvers: Mapping[str, Mapping[str, Any]],
) -> List[List[str]]:
    """Obligation frontier R_t(c) (master SS6.4).

    Consistent ledger: residuals of frontier-eligible routes (status in
    {active, repair-blocked}); complete and dead routes are excluded.
    Inconsistent ledger: the single global repair frontier (unsat core over
    labeled attestations) is returned as ``[[review_or_retract(...), ...]]``.
    """
    if is_inconsistent(cl(ledger)):
        return [_global_repair_frontier(ledger)]
    out = []
    for route in program.routes(claim_id):
        if route_status(route, ledger, resolvers) in FRONTIER_ELIGIBLE:
            out.append(residual(route, ledger, resolvers))
    return out


# ---------------------------------------------------------------------------
# Completion metrics (master SS11.5)
# ---------------------------------------------------------------------------


def route_completion(route: Sequence[str], ledger) -> float:
    """RouteCompletion(p, z): 1 for the empty route (trivially complete, no
    division by zero); else the satisfied fraction under Cl_Phi (0 on an
    inconsistent ledger)."""
    if len(route) == 0:
        return 1.0
    c = cl(ledger)
    if is_inconsistent(c):
        return 0.0
    return sum(1 for lit in route if entails(c, lit)) / float(len(route))


def completion(
    claim_id: str,
    program: Program,
    ledger,
    resolvers: Mapping[str, Mapping[str, Any]],
) -> float:
    """Completion(c, z) (master SS11.5): 0 if the ledger is inconsistent or
    all routes are dead; else max RouteCompletion over NON-DEAD routes — a
    completed route correctly scores 1 (frontier eligibility is only for
    action selection)."""
    if is_inconsistent(cl(ledger)):
        return 0.0
    routes = program.routes(claim_id)
    live = [r for r in routes if route_status(r, ledger, resolvers) != DEAD]
    if not live:
        return 0.0
    return max(route_completion(r, ledger) for r in live)


def claim_state(
    claim_id: str,
    program: Program,
    ledger,
    resolvers: Mapping[str, Mapping[str, Any]],
) -> str:
    """Exactly one of the five mutually exclusive per-claim states
    (master SS11.5): ledger-inconsistent / completed / all-routes-dead /
    repair-blocked / active-incomplete."""
    if is_inconsistent(cl(ledger)):
        return "ledger-inconsistent"
    statuses = [route_status(r, ledger, resolvers) for r in program.routes(claim_id)]
    if any(s == COMPLETE for s in statuses):
        return "completed"
    if statuses and all(s == DEAD for s in statuses):
        return "all-routes-dead"
    if any(s == REPAIR_BLOCKED for s in statuses):
        return "repair-blocked"
    return "active-incomplete"
