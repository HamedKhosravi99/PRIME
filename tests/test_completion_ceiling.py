"""Machine-checked record for the completion-ceiling theorem (paper Theorem
`thm:ceiling`).

Statement checked here. Correct program (the intended standard is the one the
committed program encodes), exact resolvers with known fields, conclusions
forming a chain. Let c_dag be the strongest supported conclusion and E_dag
the atoms on the routes of c_dag and of every stronger conclusion. Let
N(z) be the number of call classes that still contain an unsettled atom of
E_dag. If agent checking ends without an accepted emission at a ledger z
with remaining budget b >= N(z), scripted completion followed by the final
MaxAuth statement reports c_dag.

The checker uses the real ``guidance.reachable_ceiling`` and mirrors
src/prime_agents/arms.py's ``_discharge_obligations`` (known fields: one call per atom,
first untried blocking atom of the target's cheapest live route). The agent
prefix is an arbitrary sequence of checks, on or off the routes, so the
statement is checked for every agent behaviour before completion. It also
records that the bound is tight: b = N(z) - 1 fails on some instances.
"""
from __future__ import annotations

import itertools
import os
import random
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(_TESTS)
for _p in (_PILOT, os.path.join(_PILOT, "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime import guidance  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import atom_of, make_program  # noqa: E402

CLAIMS = ("C1", "C2", "C3")
LEVELS = {"C1": 1, "C2": 2, "C3": 3}


def _resolvers(atoms, classes):
    return {a: {"tool": "t", "enabled": True, "impossible": False,
                "call_class": classes[a]} for a in atoms}


def _apply_class(ledger, atoms, classes, truth, cls, step):
    """One call of class ``cls`` settles every atom of the class (as
    goldledger.delta does for atoms that share a resolver call)."""
    for a in atoms:
        if classes[a] == cls and ledger.state(a) == "U":
            ledger.apply({"atom": a, "value": truth[a], "tool": "t", "params": {},
                          "output_digest": "", "step": step})


def _holds(route, truth):
    for lit in route:
        a = atom_of(lit)
        if (truth[a] == "T") != (not lit.startswith("!")):
            return False
    return True


def ceiling(licenses, truth):
    """Strongest conclusion with a route true under the complete assignment
    (chain: a stronger route also supports every weaker conclusion)."""
    best = 0
    for c in CLAIMS:
        if any(_holds(r, truth) for r in licenses[c]):
            best = max(best, LEVELS[c])
    return best


def e_dag(licenses, level):
    """Atoms on routes of conclusions at least as strong as the ceiling
    (all route atoms when nothing is supported)."""
    return {atom_of(l) for c in CLAIMS if LEVELS[c] >= max(level, 1)
            for r in licenses[c] for l in r}


def n_open(ledger, edag, classes):
    return len({classes[a] for a in edag if ledger.state(a) == "U"})


def complete(program, ledger, atoms, classes, truth, budget):
    """Mirror of src/prime_agents/arms.py::_discharge_obligations with known fields."""
    res = _resolvers(atoms, classes)
    tried, step = set(), 0
    while budget > 0:
        g = guidance.reachable_ceiling(program, ledger, res, LEVELS, budget)
        if not g["must_continue"]:
            break
        nxt = next((atom_of(l) for l in g["blocking_atoms"]
                    if atom_of(l) not in tried), None)
        if nxt is None:
            break
        tried.add(nxt)
        step += 1
        budget -= 1
        _apply_class(ledger, atoms, classes, truth, classes[nxt], step)
    return guidance.reachable_ceiling(program, ledger, res, LEVELS, 0)["authorized_max_level"]


def _random_instance(rng):
    m = rng.randint(2, 6)
    atoms = ["a%d" % i for i in range(m)]
    k = rng.randint(1, m)  # number of call classes
    classes = {a: "k%d" % rng.randrange(k) for a in atoms}
    licenses = {}
    for c in CLAIMS:
        routes = []
        for _ in range(rng.randint(1, 2)):
            size = rng.randint(1, min(4, m))
            lits = [a if rng.random() < 0.8 else "!" + a for a in rng.sample(atoms, size)]
            routes.append(lits)
        licenses[c] = routes
    program = make_program(claims=CLAIMS, order=(("C1", "C2"), ("C2", "C3")),
                           licenses=licenses, scope="unit", atoms=tuple(atoms))
    return program, licenses, atoms, classes


def test_completion_reaches_ceiling_whenever_budget_covers_open_route_checks():
    rng = random.Random(20261012)
    checked = tight = 0
    for _ in range(500):
        program, licenses, atoms, classes = _random_instance(rng)
        for values in itertools.product("TF", repeat=len(atoms)):
            truth = dict(zip(atoms, values))
            lvl = ceiling(licenses, truth)
            edag = e_dag(licenses, lvl)
            # an arbitrary agent prefix: checks on any classes, any order
            all_classes = sorted(set(classes.values()))
            prefix = rng.sample(all_classes, rng.randint(0, len(all_classes)))
            ledger = Ledger(tuple(atoms), ())
            for i, cls in enumerate(prefix):
                _apply_class(ledger, atoms, classes, truth, cls, i + 1)
            need = n_open(ledger, edag, classes)
            for b in (need, need + 1, need + 3):
                lg = Ledger(tuple(atoms), ())
                for i, cls in enumerate(prefix):
                    _apply_class(lg, atoms, classes, truth, cls, i + 1)
                got = complete(program, lg, atoms, classes, truth, b)
                assert got == lvl, (licenses, truth, prefix, b, got, lvl)
                checked += 1
            if need > 0:
                lg = Ledger(tuple(atoms), ())
                for i, cls in enumerate(prefix):
                    _apply_class(lg, atoms, classes, truth, cls, i + 1)
                if complete(program, lg, atoms, classes, truth, need - 1) < lvl:
                    tight += 1
    assert checked > 15000
    # the bound is not loose everywhere: one call fewer fails somewhere
    assert tight > 0


def test_bound_is_tight_on_the_recorded_chain_counterexample():
    """The tightness example of Section 4 (e1..e5 = a, b, c, e, f): C2's route
    {c,e} is true, but the rule first targets C3 and checks its route in the
    order a, f, b, spending three checks before b fails. E_dag = {a,b,c,e,f},
    so N(z0) = 5: budget 5 reaches C2, while budgets 3 (= b_min) and 4
    (= N(z0) - 1) end at C1."""
    licenses = {"C1": [["a"], ["c"]], "C2": [["a", "b"], ["c", "e"]],
                "C3": [["a", "f", "b"]]}
    atoms = ["a", "b", "c", "e", "f"]
    classes = {a: a for a in atoms}
    program = make_program(claims=CLAIMS, order=(("C1", "C2"), ("C2", "C3")),
                           licenses=licenses, scope="unit", atoms=tuple(atoms))
    truth = {"a": "T", "b": "F", "c": "T", "e": "T", "f": "T"}
    assert ceiling(licenses, truth) == 2
    assert len(e_dag(licenses, 2)) == 5
    assert complete(program, Ledger(tuple(atoms), ()), atoms, classes, truth, 5) == 2
    assert complete(program, Ledger(tuple(atoms), ()), atoms, classes, truth, 4) == 1
    assert complete(program, Ledger(tuple(atoms), ()), atoms, classes, truth, 3) == 1
