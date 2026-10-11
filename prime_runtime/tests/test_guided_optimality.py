"""Machine-checked record for the CORRECTED guided utility-optimality
theorem (paper Theorem `thm:guided`; prime_runtime/GUIDED_COMPLETION_REPORT.md
SS4 + Correction).

Review established that the theorem's original statement,

    level(MaxAuth_P(z_T)) = max{ level(c) : cost_P(c, z_0) <= B_0 },

under compliance alone, is FALSE. This file pins both directions of the
failure as concrete counterexamples against the real ``guidance.py`` (kept
permanently, in the same spirit as the falsified-checker-shortcut regression
tests), and then verifies the corrected statement -- the same equality under
two additional hypotheses --

    (H2) each compliant call resolves exactly the one targeted literal;
    (H3) no targeted literal resolves against its route's needed polarity

-- by exhaustive enumeration over programs x truth assignments x budgets,
with a faithful simulation of the compliant guided loop. No LLM anywhere.

The simulator mirrors runner/arms.py's ``prime_guided`` termination rule:
while ``must_continue``, a compliant agent resolves the first surfaced
blocking atom; when ``must_continue`` goes false (ceiling authorized, ceiling
recomputed downward, or budget exhausted) the episode emits the strongest
currently-authorized claim (the unchanged MaxAuth fallback).
"""
from __future__ import annotations

import itertools
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(os.path.dirname(_TESTS))
for _p in (_PILOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime import guidance  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import make_program  # noqa: E402


def _att(atom, value, step):
    return {"atom": atom, "value": value, "tool": "t", "params": {},
            "output_digest": "", "step": step}


def _resolvers(atoms):
    return {a: {"tool": "t", "enabled": True, "impossible": False} for a in atoms}


def run_guided(program, levels, truth, budget, resolve_groups=None):
    """Faithful compliant simulation of the prime_guided loop.

    ``truth``: atom -> "T"/"F" (the instance's real evidence).
    ``resolve_groups``: optional atom -> group key; ONE call resolves every
    atom sharing the called atom's group (models goldledger.delta firing all
    atoms whose resolver params match one executed call -- the financial
    domain's measured behavior, i.e. an H2 violation).

    Returns (emitted_level, h3_clean, calls_used):
      emitted_level  level of the strongest authorized claim at termination
                     (0 = abstention);
      h3_clean       True iff no targeted literal resolved against its
                     route's needed polarity (H3 held on this episode).
    """
    res = _resolvers(program.atoms)
    ledger = Ledger(program.atoms, ())
    B = budget
    step = 0
    h3_clean = True
    while True:
        g = guidance.reachable_ceiling(program, ledger, res, levels, B)
        if not g["must_continue"] or B <= 0:
            final = guidance.reachable_ceiling(program, ledger, res, levels, 0)
            return final["authorized_max_level"], h3_clean, step
        target = g["blocking_atoms"][0]
        step += 1
        B -= 1
        to_resolve = [target]
        if resolve_groups is not None:
            gk = resolve_groups.get(target)
            if gk is not None:
                to_resolve = [a for a in program.atoms
                              if resolve_groups.get(a) == gk
                              and ledger.state(a) == "U"]  # ledger states are U/T/F/C
        for atom in to_resolve:
            ledger.apply(_att(atom, truth[atom], step))
        if truth[target] != "T":  # targeted literal resolved against route
            h3_clean = False


def rhs_original(program, levels, budget):
    """The theorem's right-hand side: max level over claims whose own-route
    atom-count cost AT THE EMPTY LEDGER is within the initial budget."""
    res = _resolvers(program.atoms)
    g = guidance.reachable_ceiling(program, Ledger(program.atoms, ()), res, levels, budget)
    best = 0
    for cid, info in g["per_claim"].items():
        if info["cost"] is not None and info["cost"] <= budget:
            best = max(best, levels[cid])
    return best


# ---------------------------------------------------------------------------
# The two recorded counterexamples (necessity of H3 and H2)
# ---------------------------------------------------------------------------

def _branching_program():
    """C1 has routes [a] and [c]; C2 needs [c,d]; C3 needs [a,b].
    Order-coherent: C2's route contains C1's route [c]; C3's contains [a]."""
    return make_program(
        claims=("C1", "C2", "C3"),
        order=(("C1", "C2"), ("C1", "C3")),
        licenses={"C1": [["a"], ["c"]], "C2": [["c", "d"]], "C3": [["a", "b"]]},
        scope="unit",
        atoms=("a", "b", "c", "d"),
    )


LEVELS3 = {"C1": 1, "C2": 2, "C3": 3}


def test_counterexample_H3_refuted_route_breaks_the_original_equality():
    """b is FALSE. cost(C3, z0)=2 <= B0=2, so the original RHS is level 3;
    the evidence-consistent variant (C2's route c,d is all-true, cost 2)
    would still be level 2. A compliant episode targets C3, spends the whole
    budget discovering b=F, and terminates at C1 -- level 1. Budget spent on
    a route later refuted is not refunded, so BOTH candidate equalities fail
    without H3."""
    program = _branching_program()
    truth = {"a": "T", "b": "F", "c": "T", "d": "T"}
    emitted, h3_clean, _ = run_guided(program, LEVELS3, truth, budget=2)
    assert not h3_clean
    assert rhs_original(program, LEVELS3, 2) == 3
    assert emitted == 1  # < 3 (original RHS) and < 2 (evidence-consistent max)


def test_counterexample_H3_chain_version_route_multiplicity():
    """The PAPER's Remark counterexample (review round 2): a full CHAIN
    C1 <= C2 <= C3, order-coherent along the chain, where the H3-free
    equality still fails -- and fails specifically because of route
    MULTIPLICITY within a level (C2's second route {c,e} survives but the
    budget burned on the refuted {a,b,f} branch is not refunded). This is
    the exact T4b/T4c boundary: single-inclusion-chain ladders are inside
    T4b; route multiplicity is outside."""
    prog = make_program(
        claims=("C1", "C2", "C3"),
        order=(("C1", "C2"), ("C2", "C3")),
        licenses={"C1": [["a"], ["c"]], "C2": [["a", "b"], ["c", "e"]],
                  "C3": [["a", "b", "f"]]},
        scope="unit", atoms=("a", "b", "c", "e", "f"))
    levels = {"C1": 1, "C2": 2, "C3": 3}
    truth = {"a": "T", "b": "F", "c": "T", "e": "T", "f": "T"}
    emitted, h3_clean, _ = run_guided(prog, levels, truth, budget=3)
    assert not h3_clean
    assert rhs_original(prog, levels, 3) == 3
    # evidence-consistent max is C2 (all-true route {c,e}, cost 2 <= 3)
    assert emitted == 1  # < 2 < 3: equality fails against both baselines
    # order-coherence along the FULL chain, so the example is on-manifold
    for strong, weaks in (("C2", ["C1"]), ("C3", ["C2", "C1"])):
        for r in prog.licenses[strong]:
            for w in weaks:
                assert any(set(q) <= set(r) for q in prog.licenses[w])


def test_counterexample_H2_multiatom_resolution_exceeds_the_original_equality():
    """a and b share a resolver group: one call resolves both. All atoms
    true, B0=1. cost(C3, z0)=2 > 1, so the original RHS is level 1 -- but the
    single call authorizes C3 and the episode emits level 3 > RHS. Without
    H2 the equality fails in the OTHER direction (atom-count cost
    over-estimates true tool-call cost; the financial domain's shared-locator
    atoms realize exactly this)."""
    program = make_program(
        claims=("C1", "C3"),
        order=(("C1", "C3"),),
        licenses={"C1": [["a"]], "C3": [["a", "b"]]},
        scope="unit",
        atoms=("a", "b"),
    )
    levels = {"C1": 1, "C3": 3}
    truth = {"a": "T", "b": "T"}
    groups = {"a": "shared", "b": "shared"}
    emitted, h3_clean, calls = run_guided(program, levels, truth, budget=1,
                                          resolve_groups=groups)
    assert h3_clean and calls == 1
    assert rhs_original(program, levels, 1) == 1
    assert emitted == 3  # strictly exceeds the original RHS


# ---------------------------------------------------------------------------
# The corrected theorem, verified by exhaustive enumeration
# ---------------------------------------------------------------------------

def _programs():
    nested = make_program(
        claims=("C1", "C2", "C3"),
        order=(("C1", "C2"), ("C2", "C3")),
        licenses={"C1": [["E1"]], "C2": [["E1", "E2"]], "C3": [["E1", "E2", "E3"]]},
        scope="unit",
        atoms=("E1", "E2", "E3"),
    )
    return [(nested, {"C1": 1, "C2": 2, "C3": 3}), (_branching_program(), LEVELS3)]


def test_corrected_theorem_exhaustive_enumeration():
    """Under H1 (compliance, by construction of the simulator) + H2
    (single-atom resolution) + H3 (episode had no refuted targeted literal):

        emitted level == max{ level(c) : cost_P(c, z_0) <= B_0 }   (equality)

    and on every H3-VIOLATING episode the emitted level is <= that max
    (guidance stays sound and never overshoots; only optimality degrades).
    Enumerates every truth assignment x every budget for two order-coherent
    programs: 2 programs x (2^3 + 2^4) truth assignments x budgets 0..5."""
    checked_clean = checked_dirty = 0
    for program, levels in _programs():
        atoms = program.atoms
        for values in itertools.product("TF", repeat=len(atoms)):
            truth = dict(zip(atoms, values))
            for budget in range(0, len(atoms) + 2):
                emitted, h3_clean, _ = run_guided(program, levels, truth, budget)
                rhs = rhs_original(program, levels, budget)
                if h3_clean:
                    assert emitted == rhs, (
                        "H1-H3 equality failed: program=%s truth=%s B=%d "
                        "emitted=%d rhs=%d" % (program.claims, truth, budget, emitted, rhs))
                    checked_clean += 1
                else:
                    assert emitted <= rhs, (
                        "overshoot on H3-violating episode: truth=%s B=%d "
                        "emitted=%d rhs=%d" % (truth, budget, emitted, rhs))
                    checked_dirty += 1
    # both regimes must actually be exercised for this test to mean anything
    assert checked_clean >= 50
    assert checked_dirty >= 20
