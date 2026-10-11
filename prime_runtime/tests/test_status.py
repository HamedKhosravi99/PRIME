"""Unit tests for availability precedence, route status, frontier (SS6.4)."""

import pytest

from prime_runtime import closure, status
from prime_runtime.ledger import Ledger
from prime_runtime.schema import make_program


def _att(atom, value, step=1):
    return {"atom": atom, "value": value, "tool": "t", "params": {}, "output_digest": "", "step": step}


def _resolvers(atoms, **overrides):
    table = {a: {"tool": "t", "enabled": True, "impossible": False} for a in atoms}
    for atom, spec in overrides.items():
        table[atom].update(spec)
    return table


def test_precedence_satisfied_over_open():
    led = Ledger(("a",), [])
    res = _resolvers(("a",))
    led.apply(_att("a", "T"))
    assert status.avail("a", led, res) == status.SATISFIED


def test_precedence_repairable_over_open_on_conflict():
    led = Ledger(("a",), [])
    res = _resolvers(("a",))
    led.apply(_att("a", "T", 1))
    led.apply(_att("a", "F", 2))
    assert status.avail("a", led, res) == status.REPAIRABLE


def test_open_when_undetermined_and_action_enabled():
    led = Ledger(("a",), [])
    assert status.avail("a", led, _resolvers(("a",))) == status.OPEN


def test_impossible_with_certificate():
    led = Ledger(("a",), [])
    res = _resolvers(("a",), a={"impossible": True})
    assert status.avail("a", led, res) == status.IMPOSSIBLE


def test_derived_opposition_is_repairable_via_retraction():
    """Phi = ¬(a∧b); attesting a=T closes b to F.  The opposing value is
    established through a provenance-bearing attestation, so retraction is a
    registered repair action: b is repairable, not resolver-error."""
    phi = [{"a": "T", "b": "T"}]
    led = Ledger(("a", "b"), phi)
    led.apply(_att("a", "T"))
    res = _resolvers(("a", "b"))
    assert status.avail("b", led, res) == status.REPAIRABLE


def test_resolver_error_fail_closed():
    """Phi forbids b=T outright: ¬b is entailed with no attestation to
    retract, no enabled action can establish b, and no impossibility
    certificate was issued — the resolver contract is violated and the
    runtime must fail closed (never guess a status)."""
    phi = [{"b": "T"}]
    led = Ledger(("a", "b"), phi)
    res = _resolvers(("a", "b"))
    with pytest.raises(status.ResolverError):
        status.avail("b", led, res)


def test_avail_suspended_on_inconsistent_ledger():
    phi = [{"a": "T", "b": "F"}]
    led = Ledger(("a", "b"), phi)
    led.apply(_att("a", "T", 1))
    led.apply(_att("b", "F", 2))
    with pytest.raises(closure.InconsistentLedgerError):
        status.avail("a", led, _resolvers(("a", "b")))


def test_route_status_all_four_values_and_precedence():
    atoms = ("a", "b", "c", "d")
    led = Ledger(atoms, [])
    led.apply(_att("a", "T", 1))
    led.apply(_att("b", "T", 2))
    led.apply(_att("b", "F", 3))  # conflict on b
    res = _resolvers(atoms, d={"impossible": True})
    assert status.route_status(["a"], led, res) == status.COMPLETE
    assert status.route_status(["a", "c"], led, res) == status.ACTIVE
    assert status.route_status(["a", "b", "c"], led, res) == status.REPAIR_BLOCKED
    # dead takes precedence over a simultaneous repairable conflict:
    assert status.route_status(["b", "d"], led, res) == status.DEAD


def test_residual_repair_marker_appears_once():
    atoms = ("a", "b", "c")
    led = Ledger(atoms, [])
    led.apply(_att("a", "T", 1))
    led.apply(_att("b", "T", 2))
    led.apply(_att("b", "F", 3))
    res = _resolvers(atoms)
    # a satisfied (absent), b conflicted -> repair(b) once, c open -> raw literal.
    assert status.residual(["a", "b", "c"], led, res) == ["repair(b)", "c"]


def test_frontier_eligibility_and_dead_exclusion():
    atoms = ("a", "b")
    prog = make_program(("C",), (), {"C": [["a"], ["b"]]}, "s", atoms)
    led = Ledger(atoms, [])
    res = _resolvers(atoms, b={"impossible": True})
    fr = status.frontier("C", prog, led, res)
    assert fr == [["a"]]  # dead route {b} excluded, open literal a remains


def test_global_repair_frontier_on_inconsistency():
    """Inconsistent ledger: the frontier is the unsat-core repair frontier —
    review_or_retract obligations over provenance-bearing attestations,
    never the frozen constraints (master SS6.4)."""
    phi = [{"a": "T", "b": "F"}]
    atoms = ("a", "b")
    prog = make_program(("C",), (), {"C": [["a"]]}, "s", atoms)
    led = Ledger(atoms, phi)
    led.apply(_att("a", "T", 1))
    led.apply(_att("b", "F", 2))
    res = _resolvers(atoms)
    fr = status.frontier("C", prog, led, res)
    assert len(fr) == 1
    assert fr[0]  # non-empty core
    assert all(ob.startswith("review_or_retract(") for ob in fr[0])
    # Minimum-cardinality core: retracting ONE of the two suffices.
    assert len(fr[0]) == 1
