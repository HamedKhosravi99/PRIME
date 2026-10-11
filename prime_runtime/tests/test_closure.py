"""Unit tests for the fail-closed solver closure Cl_Phi (master SS5.4)."""

import pytest

from prime_runtime import closure
from prime_runtime.ledger import Ledger


def _att(atom, value, step=1):
    return {"atom": atom, "value": value, "tool": "t", "params": {}, "output_digest": "", "step": step}


def test_resolution_example_unit_propagation_insufficient():
    """Phi = (¬a∨b∨c) ∧ (¬a∨¬b∨c): attesting a entails c by resolution —
    unit propagation derives nothing, exhaustive closure must find c=T
    (master SS6.2 recorded counterexample), with b,c unconflicted."""
    phi = [
        {"a": "T", "b": "F", "c": "F"},  # forbid = clause ¬a∨b∨c violated
        {"a": "T", "b": "T", "c": "F"},  # forbid = clause ¬a∨¬b∨c violated
    ]
    led = Ledger(("a", "b", "c"), phi)
    led.apply(_att("a", "T"))
    c = closure.cl(led)
    assert not closure.is_inconsistent(c)
    assert c["a"] == "T"
    assert c["c"] == "T"  # closed by resolution
    assert c["b"] == "U"  # not determined


def test_closure_exactly_one_polarity():
    """A literal closes to T/F only when exactly one polarity is entailed."""
    phi = [{"a": "T", "b": "F"}]  # a→b
    led = Ledger(("a", "b"), phi)
    c0 = closure.cl(led)
    assert c0 == {"a": "U", "b": "U"}
    led.apply(_att("a", "T"))
    c1 = closure.cl(led)
    assert c1 == {"a": "T", "b": "T"}  # b entailed by a→b
    led2 = Ledger(("a", "b"), phi)
    led2.apply(_att("b", "F"))
    c2 = closure.cl(led2)
    assert c2 == {"a": "F", "b": "F"}  # contrapositive


def test_conflicted_atoms_excluded_and_stay_c():
    """Conflicted atoms leave the fact set and remain C; they mask derived
    literals (fail-closed, non-explosive)."""
    phi = [{"a": "T", "b": "F"}]  # a→b
    led = Ledger(("a", "b"), phi)
    led.apply(_att("a", "T", 1))
    led.apply(_att("a", "F", 2))  # conflict on a
    assert led.state("a") == "C"
    c = closure.cl(led)
    assert not closure.is_inconsistent(c)
    assert c["a"] == "C"
    assert c["b"] == "U"  # NOT derived from the conflicted a


def test_unsat_base_marks_inconsistent():
    phi = [{"a": "T", "b": "F"}]
    led = Ledger(("a", "b"), phi)
    led.apply(_att("a", "T", 1))
    led.apply(_att("b", "F", 2))
    c = closure.cl(led)
    assert closure.is_inconsistent(c)
    assert isinstance(c, closure.InconsistentLedger)
    assert c.facts == {"a": True, "b": False}


def test_route_realizable_and_entails_helpers():
    phi = [{"a": "F", "b": "F"}]  # a∨b
    assert closure.route_realizable(["a"], ("a", "b"), phi)
    assert closure.route_realizable(["!a", "b"], ("a", "b"), phi)
    assert not closure.route_realizable(["!a", "!b"], ("a", "b"), phi)
    assert not closure.route_realizable(["a", "!a"], ("a", "b"), phi)
    led = Ledger(("a", "b"), phi)
    led.apply(_att("a", "F"))
    c = closure.cl(led)
    assert c["b"] == "T"  # a∨b with a=F forces b
    assert closure.entails(c, "b") and closure.entails(c, "!a")
    assert closure.satisfies(c, ["!a", "b"])
    assert not closure.satisfies(c, ["a"])
