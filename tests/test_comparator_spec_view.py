"""The information-matched comparator switch (PRIME_COMPARATOR_SPEC).

Unset, every comparator prompt is byte-identical to the published runs; set,
each carries the specification-view block the ladder's +spec rung carries
and nothing else (no routes, no atom values)."""
from __future__ import annotations

import glob
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

BENCH = sorted(glob.glob(os.path.join(PILOT, "data", "obligationbench", "security", "heldout_balanced", "*.json")))


@pytest.fixture
def instance():
    if not BENCH:
        pytest.skip("benchmark instances not present")
    return load_instance(BENCH[0])


def test_unset_is_byte_identical(instance, monkeypatch):
    monkeypatch.delenv("PRIME_COMPARATOR_SPEC", raising=False)
    assert arms_mod._comparator_system_prompt(instance) == arms_mod._base_system_prompt(instance)
    checklist = arms_mod._static_contract_checklist(instance)
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    p = arms_mod._static_contract_system_prompt(instance, checklist, ledger)
    assert p.startswith(arms_mod._base_system_prompt(instance))
    assert "SPECIFICATION" not in p


def test_set_adds_exactly_the_spec_block(instance, monkeypatch):
    monkeypatch.setenv("PRIME_COMPARATOR_SPEC", "1")
    base = arms_mod._base_system_prompt(instance)
    spec = arms_mod._spec_vocab_block(instance)
    assert arms_mod._comparator_system_prompt(instance) == base + spec
    assert "SPECIFICATION" in spec
    # the ladder's +spec rung receives the same block
    assert arms_mod._ladder_system_prompt("react_spec", instance, None) == base + spec
    # no routes, no atom values, no outcome-revealing propositions leak through it
    assert "EVIDENCE LICENSES" not in spec and "gold" not in spec.lower()
    for a in instance.atoms:
        if getattr(a, "proposition", ""):
            assert a.proposition not in spec
    checklist = arms_mod._static_contract_checklist(instance)
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    p = arms_mod._static_contract_system_prompt(instance, checklist, ledger)
    assert p.startswith(base + spec)


def test_only_comparators_are_affected(instance, monkeypatch):
    monkeypatch.setenv("PRIME_COMPARATOR_SPEC", "1")
    assert set(arms_mod.COMPARATOR_ARMS) == {"direct", "rubric", "cove", "reflexion",
                                             "conformal", "static_contract"}
    # the ladder's unmasked base rung keeps its prompt
    assert arms_mod._ladder_system_prompt("react_verify", instance, None).startswith(
        arms_mod._base_system_prompt(instance))
    assert "SPECIFICATION" not in arms_mod._ladder_system_prompt("react_verify", instance, None)
