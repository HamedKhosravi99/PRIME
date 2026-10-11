"""Tests for the ``masked_reactive`` arm -- the strong formal-contract
baseline requested by review: same gold DNF routes, same graded claim
lattice, same enforced claim mask, but NO obligation-directed action
selection and NO minimal-route planning.

The whole scientific value of this arm rests on it differing from ``prime``
in exactly one respect, so these tests pin both halves of that: what it
KEEPS (gate, program, lattice, mask) and what it DROPS (frontier).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(os.path.dirname(_TESTS))
_RUNNER = os.path.join(_P, "runner")
for _p in (_P, _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
from prime_runtime.interpreter import Interpreter  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

FIXTURE = os.path.join(_P, "prime_runtime", "tests", "fixtures", "instance_toy.json")
REAL = os.path.join(_P, "bench", "instances", "swe", "swe-0001.json")


def _setup(path=FIXTURE):
    instance = load_instance(path)
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    interp = Interpreter(instance, instance.gold_program(), revision_mode="fixed")
    return instance, ledger, interp


def test_masked_reactive_is_registered_as_prime_like():
    """It must take the committed-program path (gold program + Interpreter +
    mask), not a baseline path."""
    import re
    src = open(os.path.join(_RUNNER, "arms.py"), encoding="utf-8").read()
    assert '"masked_reactive"' in src
    # the tuple spans several lines, so match the whole expression rather
    # than the first line of it
    m = re.search(r"is_prime_like = arm in \(([^)]*)\)", src, re.S)
    assert m, "is_prime_like assignment not found"
    names = set(re.findall(r'"(\w+)"', m.group(1)))
    assert "masked_reactive" in names
    assert "masked_reactive_v" in names
    assert "prime_guided_gated" in names, \
        "the induced-condition primary must take the committed-program path"


def test_prompt_keeps_the_mask_line():
    """The gate must be as visible as it is to ``prime``: the agent is told
    exactly which claims emit_claim will accept."""
    instance, ledger, interp = _setup()
    p = arms_mod._masked_reactive_system_prompt(instance, interp, ledger)
    assert "AUTHORIZED CLAIMS" in p
    assert "emit_claim will accept" in p


def test_prompt_drops_the_obligation_frontier():
    """The control law must be absent: no frontier, no minimal routes, no
    open-work listing. This is the ONLY intended difference from prime."""
    instance, ledger, interp = _setup()
    p = arms_mod._masked_reactive_system_prompt(instance, interp, ledger)
    assert "OBLIGATION FRONTIER" not in p
    assert "minimal routes" not in p
    assert "Discharge obligations" not in p


def test_difference_from_prime_prompt_is_exactly_the_frontier():
    """Diff the two prompts: every line unique to prime's must belong to the
    frontier/control-law block, and masked_reactive must add no capability
    of its own beyond rephrasing the closing instruction."""
    instance, ledger, interp = _setup()
    prime_p = arms_mod._prime_system_prompt(instance, interp, ledger)
    mr_p = arms_mod._masked_reactive_system_prompt(instance, interp, ledger)

    prime_only = set(prime_p.splitlines()) - set(mr_p.splitlines())
    # Everything prime says and masked_reactive does not must be about the
    # frontier / obligations / the license-state header.
    for line in prime_only:
        low = line.lower()
        assert ("obligation" in low or "frontier" in low or "license state" in low
                or "discharge" in low or line.strip().startswith("{")
                or line.strip() == ""), "unexpected prime-only line: %r" % line

    # And masked_reactive must not introduce any evidence-seeking hint.
    mr_only = set(mr_p.splitlines()) - set(prime_p.splitlines())
    for line in mr_only:
        low = line.lower()
        assert "route" not in low and "obligation" not in low, (
            "masked_reactive leaked control-law info: %r" % line)


def test_uses_the_same_gold_dnf_program_not_a_conjunctive_collapse():
    """It must keep the disjunctive alternatives that static_contract
    destroys. For every claim, masked_reactive's program is the gold
    program itself."""
    instance, _, interp = _setup(REAL)
    gold = instance.gold_program()
    assert interp.program == gold
    # static_contract, by contrast, unions all literals into one conjunction.
    checklist = arms_mod._static_contract_checklist(instance)
    for cid, routes in gold.licenses.items():
        if len(routes) > 1:
            union = set()
            for r in routes:
                union |= set(r)
            assert set(checklist[cid]) == union, (
                "static_contract should be the conjunctive collapse")
            assert not any(set(r) == union for r in routes), (
                "gold routes should be strictly smaller than the collapse")


def test_mask_still_refuses_an_unauthorized_claim():
    """Soundness is unchanged: the interpreter mask is enforced exactly as
    for the prime arm."""
    from prime_runtime.interpreter import UnauthorizedClaimError

    instance, ledger, interp = _setup()
    top = max(instance.claims, key=lambda c: (c.level, c.id))
    with pytest.raises(UnauthorizedClaimError):
        interp.emit(top.id, ledger)  # empty ledger authorizes nothing


def test_graded_lattice_available():
    """The full claim lattice is offered (no answer/abstain collapse), so the
    MaxAuth fallback can emit a weaker authorized claim."""
    instance, ledger, interp = _setup()
    assert len(instance.claims) > 1
    legal = interp.legal_claim_actions(ledger)
    assert isinstance(legal, (list, tuple))
