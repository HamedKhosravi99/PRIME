"""The five regression tests pinned by SPEC.md SSRuntime API.

R1-R3 are the recorded counterexamples to the failed absorption shortcuts
(PRIME-master.md SS9.3); plus the fail-closed explosion test and the
completion-metric-scores-1-on-complete test.
"""

import pytest

from prime_runtime import absorption, auth, closure, status
from prime_runtime.ledger import Ledger
from prime_runtime.schema import make_program


def _program(routes, order=(), claims=("C",), atoms=("a", "b", "c")):
    return make_program(claims=claims, order=order, licenses=routes, scope="s", atoms=atoms)


def test_r1_empty_route_rejected_under_disjunctive_phi():
    """R1: Phi = a∨b, routes {{a},{b}} — the empty route passes ontic
    (complete-assignment) entailment but must be rejected: satisfaction is
    epistemic; at the all-U ledger the committed program authorizes nothing."""
    phi = [{"a": "F", "b": "F"}]  # forbid a=F ∧ b=F  <=>  a∨b
    p0 = _program({"C": [["a"], ["b"]]}, atoms=("a", "b"))
    p_new = _program({"C": [[]]}, atoms=("a", "b"))
    assert absorption.check_patch(p0, p_new, phi) is False
    # Sanity: the empty route would indeed authorize at the all-U ledger.
    led = Ledger(("a", "b"), phi)
    assert auth.auth_ids(p_new, led) == ["C"]
    assert auth.auth_ids(p0, led) == []


def test_r2_conflict_masking_route_a_vs_committed_b():
    """R2: Phi = a→b, committed {{b}}, revised {a}.  One-closure checking
    accepts ({a} closes to {a,b} ⊇ {b}) but closure is not monotone when
    conflicts mask derived literals — the robust checker must reject."""
    phi = [{"a": "T", "b": "F"}]  # forbid a=T ∧ b=F  <=>  a→b
    p0 = _program({"C": [["b"]]}, atoms=("a", "b"))
    p_new = _program({"C": [["a"]]}, atoms=("a", "b"))
    assert absorption.check_patch(p0, p_new, phi) is False
    # Witness of the expansion at a conflict state: z(a)=T, z(b)=C.
    led = Ledger(("a", "b"), phi)
    led.apply({"atom": "a", "value": "T", "tool": "t", "params": {}, "output_digest": "", "step": 1})
    led.apply({"atom": "b", "value": "T", "tool": "t", "params": {}, "output_digest": "", "step": 2})
    led.apply({"atom": "b", "value": "F", "tool": "t", "params": {}, "output_digest": "", "step": 3})
    assert led.state("b") == "C"
    assert auth.auth_ids(p_new, led) == ["C"]  # revised program authorizes
    assert auth.auth_ids(p0, led) == []  # committed one does not


def test_r3_joint_conflict_route_a_rejected():
    """R3: Phi = (a→b)∧(a→c), routes {{a,b},{a,c}}, revised {a} — reachable-
    safe when b,c never conflict simultaneously, but the robust contract
    quantifies over ALL conflict masks and must reject."""
    phi = [{"a": "T", "b": "F"}, {"a": "T", "c": "F"}]
    p0 = _program({"C": [["a", "b"], ["a", "c"]]})
    p_new = _program({"C": [["a"]]})
    assert absorption.check_patch(p0, p_new, phi) is False


def test_fail_closed_explosion_authorizes_nothing():
    """Fail-closed explosion test: a=T, b=F under a→b makes Phi ∧ facts°
    UNSAT — the ledger is inconsistent and NOTHING is authorized (a classical
    closure would entail everything and satisfy every license)."""
    phi = [{"a": "T", "b": "F"}]
    p = _program({"C": [["a"]]}, atoms=("a", "b"))
    led = Ledger(("a", "b"), phi)
    led.apply({"atom": "a", "value": "T", "tool": "t", "params": {}, "output_digest": "", "step": 1})
    led.apply({"atom": "b", "value": "F", "tool": "t", "params": {}, "output_digest": "", "step": 2})
    c = closure.cl(led)
    assert closure.is_inconsistent(c)
    assert auth.lic("C", p, led) is False
    assert auth.auth_ids(p, led) == []
    assert auth.max_auth(p, led) == []


def test_completion_scores_one_on_complete_route():
    """Completion-metric test (master SS11.5): a completed route scores
    exactly 1 — frontier eligibility is only for action selection, and the
    empty route is trivially complete (no division by zero)."""
    p = _program({"C": [["a", "b"]]}, atoms=("a", "b", "c"))
    led = Ledger(("a", "b", "c"), [])
    resolvers = {a: {"tool": "t", "enabled": True} for a in ("a", "b", "c")}
    for i, atom in enumerate(("a", "b")):
        led.apply({"atom": atom, "value": "T", "tool": "t", "params": {}, "output_digest": "", "step": i})
    assert status.route_completion(["a", "b"], led) == 1.0
    assert status.completion("C", p, led, resolvers) == 1.0
    assert status.claim_state("C", p, led, resolvers) == "completed"
    assert status.route_completion([], led) == 1.0
    # And the completed route is NOT on the frontier (no further work).
    assert status.frontier("C", p, led, resolvers) == []
