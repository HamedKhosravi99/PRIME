"""Absorption checker unit tests beyond the pinned R1-R3 regressions
(test_regressions.py)."""

import pytest

from prime_runtime import absorption
from prime_runtime.renderer import RendererError, render
from prime_runtime.schema import make_program


def _prog(licenses, order=(), claims=("W", "S"), atoms=("a", "b", "c")):
    return make_program(claims, order, licenses, "s", atoms)


def test_absorption_family_includes_dominating_claims():
    """Round-9 counterexample: with P(S)={{a}}, P(W)={{a,b}} and W <= S,
    revising W's route to {a} adds no authorization (S already authorizes W
    downward) — per-claim-family checking would be too strict."""
    p0 = _prog({"S": [["a"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    p_new = _prog({"S": [["a"]], "W": [["a"]]}, order=(("W", "S"),))
    assert absorption.check_patch(p0, p_new) is True


def test_literal_addition_auto_accepted():
    p0 = _prog({"S": [["a"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    p_new = _prog({"S": [["a", "c"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    assert absorption.check_patch(p0, p_new) is True


def test_route_deletion_always_accepted():
    p0 = _prog({"S": [["a"], ["b"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    p_new = _prog({"S": [["a"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    assert absorption.check_patch(p0, p_new) is True


def test_uncontained_route_addition_rejected():
    p0 = _prog({"S": [["a"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    p_new = _prog({"S": [["a"], ["c"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    assert absorption.check_patch(p0, p_new) is False


def test_new_claim_or_new_order_edge_rejected():
    p0 = _prog({"S": [["a"]]}, claims=("S",))
    p_new_claim = _prog({"S": [["a"]], "Z": [["a"]]}, claims=("S", "Z"))
    assert absorption.check_patch(p0, p_new_claim) is False
    p0_two = _prog({"S": [["a"]], "W": [["a", "b"]]})  # no order edges
    p_new_edge = _prog({"S": [["a"]], "W": [["a", "b"]]}, order=(("W", "S"),))
    assert absorption.check_patch(p0_two, p_new_edge) is False


def test_unrealizable_revised_route_rejected_as_ill_formed():
    phi = [{"a": "T", "b": "T"}]  # forbid a ∧ b
    p0 = _prog({"S": [["a"]], "W": [["a", "c"]]}, order=(("W", "S"),))
    p_new = _prog({"S": [["a", "b"]], "W": [["a", "c"]]}, order=(("W", "S"),))
    assert absorption.check_patch(p0, p_new, phi) is False


def test_new_atoms_projected_out():
    """Literals over atoms absent from P0's vocabulary only strengthen a
    route; containment is checked on the projection pi(p')."""
    p0 = _prog({"S": [["a"]]}, claims=("S",), atoms=("a",))
    p_new = make_program(("S",), (), {"S": [["a", "n1"]]}, "s", ("a", "n1"))
    assert absorption.check_patch(p0, p_new) is True


def test_renderer_refuses_unregistered_claim_content():
    with pytest.raises(RendererError):
        render({"authorized_claims": ["C9"]}, {"C1": "readable"})
