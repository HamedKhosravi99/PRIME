"""Licensing and authorization semantics.

Implements PRIME-master.md SS5.4 (Lic through Cl_Phi), SS7.5 / SS9.2 (single
authorization semantics: Auth = downward closure of licensed claims over the
claim order; Auth = {} on an inconsistent ledger so nothing is ever
authorized under conflicts that make the base UNSAT), and MaxAuth — the
maximal authorized antichain.
"""

from typing import Dict, FrozenSet, List, Set, Tuple

from .closure import cl, is_inconsistent, satisfies
from .schema import Program


def preceq(weaker: str, stronger: str, program: Program) -> bool:
    """weaker <=* stronger under the reflexive-transitive claim order."""
    if weaker == stronger:
        return True
    edges: Dict[str, Set[str]] = {}
    for w, s in program.order:
        edges.setdefault(w, set()).add(s)
    seen: Set[str] = set()
    stack = [weaker]
    while stack:
        node = stack.pop()
        for nxt in edges.get(node, ()):
            if nxt == stronger:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def lic(claim_id: str, program: Program, ledger) -> bool:
    """Lic_P(c, z): some route of c is satisfied on the fail-closed closure
    Cl_Phi(z) (master SS5.4).  False on an inconsistent ledger."""
    c = cl(ledger)
    if is_inconsistent(c):
        return False
    return any(satisfies(c, route) for route in program.routes(claim_id))


def auth_ids(program: Program, ledger) -> List[str]:
    """AuthIDs_P(z): claims authorized directly or by downward closure —
    c is authorized iff some d with c <=* d is licensed (master SS9.2).
    Empty on an inconsistent ledger (nothing authorized)."""
    c = cl(ledger)
    if is_inconsistent(c):
        return []
    licensed = {
        d for d in program.claims if any(satisfies(c, r) for r in program.routes(d))
    }
    out = [
        cid
        for cid in program.claims
        if any(preceq(cid, d, program) for d in licensed)
    ]
    return sorted(out)


def max_auth(program: Program, ledger) -> List[str]:
    """MaxAuth_P(z): the antichain of maximal authorized claims — authorized
    claims not strictly dominated by another authorized claim (master SS7.5).
    """
    auth = auth_ids(program, ledger)
    auth_set = set(auth)
    out = [
        c
        for c in auth
        if not any(c != d and preceq(c, d, program) for d in auth_set)
    ]
    return sorted(out)
