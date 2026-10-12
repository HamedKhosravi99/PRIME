"""The appendix theorem that guidance reaches the strongest affordable
conclusion, checked without the shared-checks-first condition.

The compliant guided run is simulated exactly as the theorem defines it:
guidance targets t(D) = max R(D), the largest live conclusion whose open call
classes fit the remaining budget, and the agent calls ANY open class of the
target route. Every such choice sequence is explored, for every nested
program with up to three conclusions, every grouping of atoms into call
classes, every truth assignment, and every budget in range. The stated level
must equal k*, the largest level whose route holds and costs at most B0.

The three counterexamples stated in the appendix (conclusions not a chain, two
routes per conclusion, routes not nested) are checked as well.
"""
from __future__ import annotations

import itertools


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def _stated_levels(routes, cls_of, cost, truth, budget):
    """All levels a compliant run can end with (over every call order)."""
    n = len(routes)
    classes = [frozenset(cls_of[a] for a in r) for r in routes]
    out = set()

    def live(k, called):
        return all(truth[a] for a in routes[k] if cls_of[a] in called)

    def run(called, spent):
        assert spent <= budget  # guidance never overspends the budget
        reach = [k for k in range(n) if live(k, called)
                 and sum(cost[c] for c in classes[k] - called) <= budget - spent]
        if reach:
            open_t = classes[max(reach)] - called
            if open_t:
                for c in sorted(open_t):
                    run(called | {c}, spent + cost[c])
                return
        est = [k for k in range(n) if live(k, called) and classes[k] <= called]
        out.add(max(est) + 1 if est else 0)

    run(frozenset(), 0)
    return out


def _k_star(routes, cls_of, cost, truth, budget):
    best = 0
    for k, r in enumerate(routes):
        if all(truth[a] for a in r) and sum(cost[c] for c in {cls_of[a] for a in r}) <= budget:
            best = k + 1
    return best


def _enumerate(m, cost_choices, budgets):
    atoms = list(range(m))
    cases = failures = 0
    for n in range(1, 4):
        for entry in itertools.product(range(1, n + 2), repeat=m):
            routes = [frozenset(a for a in atoms if entry[a] <= k) for k in range(1, n + 1)]
            for part in _partitions(atoms):
                cls_of = {a: i for i, block in enumerate(part) for a in block}
                for costs in itertools.product(cost_choices, repeat=len(part)):
                    cost = dict(enumerate(costs))
                    for tv in itertools.product([False, True], repeat=m):
                        truth = dict(enumerate(tv))
                        for budget in budgets:
                            cases += 1
                            if _stated_levels(routes, cls_of, cost, truth, budget) != {
                                    _k_star(routes, cls_of, cost, truth, budget)}:
                                failures += 1
    return cases, failures


def test_guidance_reaches_k_star_in_every_call_order():
    three = _enumerate(3, (1, 2), range(0, 8))
    four = _enumerate(4, (1,), range(0, 6))
    assert three == (139392, 0)
    assert four == (508320, 0)  # 647,712 cases in all, as the appendix states


def _general_run(levels, order, routes, truth, budget, priority):
    """Guidance on a general program: target the strongest reachable
    conclusion through its cheapest live route; the agent checks that route's
    open atoms in `priority` order. Unit costs, one class per atom."""
    known, spent = {}, 0

    def live(r):
        return all(truth[a] for a in r if a in known)

    while True:
        target = None
        for c in sorted(routes, key=lambda c: -levels[c]):
            cands = [r for r in routes[c] if live(r)
                     and len([a for a in r if a not in known]) <= budget - spent]
            if cands:
                target = min(cands, key=lambda r: len([a for a in r if a not in known]))
                break
        if target is None:
            break
        todo = [a for a in priority if a in target and a not in known]
        if not todo:
            break
        known[todo[0]] = truth[todo[0]]
        spent += 1
    licensed = {c for c, rs in routes.items() if any(live(r) and all(a in known for a in r) for r in rs)}
    auth, grew = set(licensed), True
    while grew:
        grew = False
        for weak, strong in order:
            if strong in auth and weak not in auth:
                auth.add(weak)
                grew = True
    return max((levels[c] for c in auth), default=0)


def test_dropped_conditions_break_the_equality():
    # (1) conclusions not a chain: stated c1, although c2's route holds and fits
    assert _general_run({"c1": 1, "c2": 2, "c3": 3}, [("c1", "c2"), ("c1", "c3")],
                        {"c1": [{"a"}], "c2": [{"a", "b"}], "c3": [{"a", "e"}]},
                        {"a": 1, "b": 1, "e": 0}, 2, "abe") == 1
    # (2) two routes per conclusion, a checked before b: stated c1, although {g,h} fits
    assert _general_run({"c1": 1, "c2": 2, "c3": 3}, [("c1", "c2"), ("c2", "c3")],
                        {"c1": [{"a"}, {"g"}], "c2": [{"a", "b"}, {"g", "h"}],
                         "c3": [{"a", "b", "f"}]},
                        {"a": 1, "b": 0, "g": 1, "h": 1, "f": 1}, 3, "abfgh") == 1
    # (3) routes not nested: abstains, although c1's route holds and fits
    assert _general_run({"c1": 1, "c2": 2}, [("c1", "c2")],
                        {"c1": [{"a"}], "c2": [{"b"}]}, {"a": 1, "b": 0}, 1, "ab") == 0
