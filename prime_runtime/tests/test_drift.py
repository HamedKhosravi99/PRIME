"""Revision-freedom ablation: acceptance rules and exact drift detection
(in-episode freeze, robust route absorption).

Toy program: C1={E1}, C2={E1,E2}, C3={E1,E2,E3,E4}, order C1<C2<C3.
Absorption family B_0(c) = routes of c and every STRONGER claim, so
  B_0(C1) = {{E1},{E1,E2},{E1,E2,E3,E4}}   B_0(C3) = {{E1,E2,E3,E4}}.
A new/edited route p' for c is authorization-expanding (a DRIFT event) iff no
route of B_0(c) is a subset of p'.
"""

import pytest

from prime_runtime.interpreter import FrozenProgramError, Interpreter


def _interp(inst, mode):
    return Interpreter(inst, revision_mode=mode)


def test_fixed_mode_forbids_revision(toy_instance):
    """The deployed path: revision is not a policy action at all."""
    interp = _interp(toy_instance, "fixed")
    with pytest.raises(FrozenProgramError):
        interp.propose_revision("add_literal", {"claim": "C1", "route_index": 0, "literal": "E3"})


def test_kind_gating(toy_instance):
    """strengthen_only permits only add_literal / delete_route."""
    interp = _interp(toy_instance, "strengthen_only")
    r = interp.propose_revision("add_route", {"claim": "C1", "route": ["E3"]})
    assert not r["accepted"] and "not permitted" in r["reason"]


def test_strengthening_is_not_drift(toy_instance):
    """Adding a literal shrinks a route's Auth-set: never authorization-
    expanding, hence never a drift event."""
    interp = _interp(toy_instance, "strengthen_only")
    r = interp.propose_revision(
        "add_literal", {"claim": "C1", "route_index": 0, "literal": "E3"}
    )
    assert r["accepted"] and r["auth_expanding"] is False
    assert interp.drift_events == 0


def test_absorbed_new_route_is_not_drift(toy_instance):
    """A new route for C1 that still contains a committed route ({E1} subset of
    {E1,E3}) adds no authorization: accepted but not drift."""
    interp = _interp(toy_instance, "arbitrary")
    r = interp.propose_revision("add_route", {"claim": "C1", "route": ["E1", "E3"]})
    assert r["accepted"] and r["auth_expanding"] is False
    assert interp.drift_events == 0


def test_weakening_route_is_drift(toy_instance):
    """Removing E4 from C3's only route lets C3 be authorized on less evidence
    than the committed standard ever allowed: an authorization-expanding
    revision = a drift event."""
    interp = _interp(toy_instance, "arbitrary")
    r = interp.propose_revision(
        "remove_literal", {"claim": "C3", "route_index": 0, "literal": "E4"}
    )
    assert r["accepted"] and r["auth_expanding"] is True
    assert interp.drift_events == 1


def test_new_cheap_route_is_drift(toy_instance):
    """Adding an unrelated cheap route {E3} to C1 (no committed route is a
    subset) manufactures authorization: drift."""
    interp = _interp(toy_instance, "arbitrary")
    r = interp.propose_revision("add_route", {"claim": "C1", "route": ["E3"]})
    assert r["accepted"] and r["auth_expanding"] is True
    assert interp.drift_events == 1


def test_posthoc_replace_program(toy_instance):
    """Post-hoc mode: the model supplies a whole program; a program that
    licenses C3 with a single cheap atom is accepted (well-formed) and flagged
    as drift against the committed B_0."""
    interp = _interp(toy_instance, "posthoc")
    r = interp.propose_revision(
        "replace_program", {"licenses": {"C1": [["E1"]], "C2": [["E1", "E2"]], "C3": [["E3"]]}}
    )
    assert r["accepted"] and r["auth_expanding"] is True


def test_drift_reference_is_original_program(toy_instance):
    """strengthen-then-weaken must not leak: after a strengthening the drift
    check still compares against the ORIGINAL committed program, so a later
    weakening back below the committed standard is caught."""
    interp = _interp(toy_instance, "arbitrary")
    interp.propose_revision("add_literal", {"claim": "C3", "route_index": 0, "literal": "E1"})
    r = interp.propose_revision("add_route", {"claim": "C3", "route": ["E3"]})
    assert r["accepted"] and r["auth_expanding"] is True  # vs committed B_0, not the strengthened interim
