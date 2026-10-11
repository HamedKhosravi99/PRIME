"""Fail-closed solver closure Cl_Phi (PRIME-master.md SS5.4).

Semantics (master SS5.4, round-9 "fail-closed solver closure"):

1. Conflicted atoms are EXCLUDED from the classical fact set facts°(z) and
   remain 'C' in the closure.
2. If Phi ∧ facts°(z) is unsatisfiable, the ledger state is inconsistent and
   NOTHING is authorized (returned as an :class:`InconsistentLedger` marker,
   never a classical explosion).
3. For a consistent base, an atom closes to 'T' (resp. 'F') only when exactly
   one polarity is entailed; otherwise 'U'.

Entailment procedure
--------------------
Phi is a conjunction of forbidden determined-value patterns; a complete
Boolean assignment omega is *realizable* iff it matches no pattern.  With
atom counts <= 20 (schema.MAX_ATOMS) we implement Phi-entailment by exact
exhaustive enumeration over complete assignments restricted to realizable
ones — no external solver, exactly correct.

Complexity: one closure call enumerates all 2^m complete assignments and
filters each against |Phi| patterns, then scans the consistent extensions
once per atom: O(2^m * (|Phi| + m)) time.  Realizable-assignment sets are
memoized per (atom vocabulary, Phi) for m <= 16; larger vocabularies are
enumerated lazily on each call to bound memory.
"""

import itertools
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .schema import parse_literal

_CACHE_MAX_ATOMS = 16
_realizable_cache: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[Tuple[str, str], ...], ...]], Tuple[Tuple[bool, ...], ...]] = {}


class InconsistentLedger:
    """Marker: Phi ∧ facts°(z) is UNSAT — ledger inconsistent, nothing
    authorized until an explicit retraction resolves it (master SS5.4)."""

    def __init__(self, facts: Optional[Mapping[str, bool]] = None) -> None:
        self.facts: Dict[str, bool] = dict(facts or {})

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "InconsistentLedger(facts=%r)" % (self.facts,)


class InconsistentLedgerError(RuntimeError):
    """Raised by status functions that cannot operate on an inconsistent
    ledger (route-level status is suspended, master SS6.4)."""


def is_inconsistent(closure_result: object) -> bool:
    return isinstance(closure_result, InconsistentLedger)


def _phi_key(phi: Sequence[Mapping[str, str]]) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    return tuple(sorted(tuple(sorted(p.items())) for p in phi))


def _matches(pattern: Mapping[str, str], omega: Mapping[str, bool]) -> bool:
    return all(omega[a] == (v == "T") for a, v in pattern.items())


def _iter_assignments(atoms: Sequence[str]) -> Iterator[Dict[str, bool]]:
    for bits in itertools.product((False, True), repeat=len(atoms)):
        yield dict(zip(atoms, bits))


def realizable_assignments(
    atoms: Sequence[str], phi: Sequence[Mapping[str, str]]
) -> List[Dict[str, bool]]:
    """All complete assignments omega in Omega_Phi (forbidden-pattern filter).

    Memoized for small vocabularies; see module docstring for complexity.
    """
    atoms_t = tuple(atoms)
    key = (atoms_t, _phi_key(phi))
    if len(atoms_t) <= _CACHE_MAX_ATOMS and key in _realizable_cache:
        rows = _realizable_cache[key]
        return [dict(zip(atoms_t, row)) for row in rows]
    out = [
        omega
        for omega in _iter_assignments(atoms_t)
        if not any(_matches(p, omega) for p in phi)
    ]
    if len(atoms_t) <= _CACHE_MAX_ATOMS:
        _realizable_cache[key] = tuple(tuple(o[a] for a in atoms_t) for o in out)
    return out


def route_realizable(
    route: Iterable[str], atoms: Sequence[str], phi: Sequence[Mapping[str, str]]
) -> bool:
    """Exists omega in Omega_Phi satisfying every literal of the route."""
    want: Dict[str, bool] = {}
    for lit in route:
        atom, pos = parse_literal(lit)
        if atom in want and want[atom] != pos:
            return False  # p contains l and !l
        want[atom] = pos
    for omega in realizable_assignments(atoms, phi):
        if all(omega[a] == v for a, v in want.items()):
            return True
    return False


def cl_states(
    atoms: Sequence[str],
    phi: Sequence[Mapping[str, str]],
    states: Mapping[str, str],
):
    """Closure over a plain state map; returns dict[atom,state] or an
    :class:`InconsistentLedger` marker.  See :func:`cl`."""
    atoms_t = tuple(atoms)
    facts = {a: (states[a] == "T") for a in atoms_t if states[a] in ("T", "F")}
    consistent = [
        omega
        for omega in realizable_assignments(atoms_t, phi)
        if all(omega[a] == v for a, v in facts.items())
    ]
    if not consistent:
        return InconsistentLedger(facts)
    out: Dict[str, str] = {}
    for a in atoms_t:
        if states[a] == "C":
            out[a] = "C"  # conflicted atoms stay C (fail-closed)
            continue
        seen = {omega[a] for omega in consistent}
        if seen == {True}:
            out[a] = "T"
        elif seen == {False}:
            out[a] = "F"
        else:
            out[a] = "U"
    return out


def cl(ledger):
    """Fail-closed closure Cl_Phi(z) of a :class:`~prime_runtime.ledger.Ledger`.

    Returns ``dict[atom, 'U'|'T'|'F'|'C']`` for a consistent base, or an
    :class:`InconsistentLedger` marker when Phi ∧ facts°(z) is UNSAT
    (master SS5.4; SPEC.md Runtime API).
    """
    return cl_states(ledger.atoms, ledger.phi_patterns, ledger.states())


def entails(closure_map: Mapping[str, str], literal: str) -> bool:
    """Cl_Phi(z) |= literal — the literal's atom closed to the literal's
    polarity.  Conflicted ('C') and unknown ('U') atoms satisfy neither
    polarity (master SS5.4)."""
    atom, pos = parse_literal(literal)
    return closure_map.get(atom) == ("T" if pos else "F")


def satisfies(closure_map: Mapping[str, str], route: Iterable[str]) -> bool:
    """Cl_Phi(z) |= p for a conjunctive route p (empty route: trivially)."""
    return all(entails(closure_map, lit) for lit in route)
