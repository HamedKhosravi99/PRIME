"""Unit tests for the attested ledger and its integrity rule."""

import pytest

from prime_runtime.ledger import Ledger, LedgerError


def _att(atom, value, step=1, tool="t"):
    return {"atom": atom, "value": value, "tool": tool, "params": {}, "output_digest": "d", "step": step}


def test_states_and_versioning():
    led = Ledger(("a", "b"), [])
    assert led.states() == {"a": "U", "b": "U"}
    assert led.version == 0
    led.apply(_att("a", "T"))
    assert led.state("a") == "T"
    assert led.version == 1


def test_conflict_is_permanent_until_retraction():
    led = Ledger(("a",), [])
    r1 = led.apply(_att("a", "T", 1))
    r2 = led.apply(_att("a", "F", 2))
    assert led.state("a") == "C"
    # Agreeing further evidence does NOT clear the conflict.
    led.apply(_att("a", "T", 3))
    assert led.state("a") == "C"
    # Explicit retraction of the F record resolves it.
    led.apply({"kind": "retract", "references": r2, "tool": "adjudicator", "step": 4})
    assert led.state("a") == "T"
    assert led.version == 4


def test_retraction_validates_reference():
    led = Ledger(("a",), [])
    with pytest.raises(LedgerError):
        led.apply({"kind": "retract", "references": 99, "tool": "t", "step": 1})
    rid = led.apply(_att("a", "T"))
    led.apply({"kind": "retract", "references": rid, "tool": "t", "step": 2})
    assert led.state("a") == "U"
    with pytest.raises(LedgerError):  # cannot retract twice
        led.apply({"kind": "retract", "references": rid, "tool": "t", "step": 3})


def test_apply_is_validated_and_only_mutation_path():
    led = Ledger(("a",), [])
    with pytest.raises(LedgerError):
        led.apply({"atom": "zz", "value": "T", "tool": "t", "step": 1})
    with pytest.raises(LedgerError):
        led.apply({"atom": "a", "value": "yes", "tool": "t", "step": 1})
    with pytest.raises(LedgerError):
        led.apply({"atom": "a", "value": "T"})  # missing provenance keys
    # No public setter exists for states.
    assert not hasattr(led, "set_state")
    # records/states are copies: mutating them does not touch the ledger.
    led.apply(_att("a", "T"))
    recs = led.records
    recs[0]["value"] = "F"
    assert led.state("a") == "T"
