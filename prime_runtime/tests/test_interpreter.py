"""Unit tests for the frozen-program interpreter (SS7.2, SS7.4, SS7.8)."""

import dataclasses

import pytest

from prime_runtime.interpreter import (
    FrozenProgramError,
    Interpreter,
    StaleLedgerError,
    UnauthorizedClaimError,
)
from prime_runtime.ledger import Ledger
from prime_runtime.renderer import render_for_instance


def _att(atom, value, step=1):
    return {"atom": atom, "value": value, "tool": "t", "params": {}, "output_digest": "", "step": step}


def _ledger(instance):
    return Ledger(instance.atom_ids, instance.phi_patterns)


def test_freeze_program_mutation_raises(toy_instance):
    """Full in-episode freeze: the policy can NEVER modify the program —
    every mutation surface raises."""
    interp = Interpreter(toy_instance)
    with pytest.raises(dataclasses.FrozenInstanceError):
        interp.program.scope = "broader scope"  # attribute rebind
    with pytest.raises(TypeError):
        interp.program.licenses["C1"] = (("E2",),)  # license edit
    with pytest.raises(TypeError):
        del interp.program.licenses["C3"]  # route deletion
    with pytest.raises(AttributeError):
        interp.program.claims.append("C4")  # tuple: no append
    with pytest.raises(FrozenProgramError):
        interp.program = None  # program swap on the interpreter
    with pytest.raises(FrozenProgramError):
        interp.resolvers = {}


def test_emit_refuses_unauthorized_claim(toy_instance):
    interp = Interpreter(toy_instance)
    led = _ledger(toy_instance)
    assert interp.legal_claim_actions(led) == []
    with pytest.raises(UnauthorizedClaimError):
        interp.emit("C3", led)
    with pytest.raises(UnauthorizedClaimError):
        interp.emit("C9", led)  # unknown claim id


def test_emit_typed_claim_object_shape(toy_instance):
    interp = Interpreter(toy_instance)
    led = _ledger(toy_instance)
    led.apply(_att("E1", "T", 1))
    led.apply(_att("E2", "T", 2))
    # E1,E2 license C2; C1 authorized by downward closure.
    assert interp.legal_claim_actions(led) == ["C1", "C2"]
    obj = interp.emit("C2", led, ledger_version=led.version)
    assert obj["authorized_claims"] == ["C2"]
    assert set(obj["evidence"]) == {"E1", "E2"}
    assert obj["scope"] == toy_instance.scope
    assert obj["qualifications"] == []
    unresolved = {u["id"]: u["residual_obligations"] for u in obj["unresolved_claims"]}
    assert list(unresolved) == ["C3"]
    # Phi (E2=T forbids E4=F) closes E4 to T, so only E3 remains open.
    assert unresolved["C3"] == ["E3"]
    # C1 authorized by downward closure even though its own route also holds.
    obj1 = interp.emit("C1", led)
    assert obj1["authorized_claims"] == ["C1"]
    # Renderer accepts the typed object and references claim ids.
    text = render_for_instance(obj, toy_instance)
    assert "[C2]" in text and "C3" in text


def test_emit_stale_ledger_version_guard(toy_instance):
    interp = Interpreter(toy_instance)
    led = _ledger(toy_instance)
    led.apply(_att("E1", "T", 1))
    with pytest.raises(StaleLedgerError):
        interp.emit("C1", led, ledger_version=0)


def test_emit_max_auth_antichain(toy_instance):
    interp = Interpreter(toy_instance)
    led = _ledger(toy_instance)
    with pytest.raises(UnauthorizedClaimError):
        interp.emit_max_auth(led)  # nothing authorized -> abstain
    led.apply(_att("E1", "T", 1))
    obj = interp.emit_max_auth(led)
    assert obj["authorized_claims"] == ["C1"]


def test_conflict_removes_previous_authorization(toy_instance):
    """A conflict can remove a previously available claim before emission
    (master SS5.4)."""
    interp = Interpreter(toy_instance)
    led = _ledger(toy_instance)
    led.apply(_att("E1", "T", 1))
    led.apply(_att("E2", "T", 2))
    assert "C2" in interp.legal_claim_actions(led)
    led.apply(_att("E2", "F", 3))  # conflicting attestation
    assert interp.legal_claim_actions(led) == ["C1"]
    with pytest.raises(UnauthorizedClaimError):
        interp.emit("C2", led)
    obj = interp.emit("C1", led)
    assert obj["qualifications"] == ["Atom E2 has conflicting attestations."]
