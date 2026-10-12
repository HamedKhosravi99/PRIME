"""The LLM-free policy frontier (experiments/analysis/policy_frontier.py).

These arms are scripted or computed -- real ToolExecutor calls and the real
attested ledger, but deterministic action selection -- so their numbers are
reproducible without a GPU and are pinned here. They carry two claims the
confirmatory LLM runs cannot make on their own:

  * There is NO planning headroom above a simple deployable heuristic:
    ceiling-targeting already matches the nondeployable clairvoyant ceiling.
    So an E2 null is a property of the benchmark's route structure, not a
    failure to measure -- and no control law can beat that bound.
  * Ordering and direction separate (docs/EXECUTION_PLAN.md, Part R4, Q4):
    prefix-first ordering alone leaves a gap that ceiling-targeting closes,
    and the gap exists ONLY under budget pressure.
"""
from __future__ import annotations

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "experiments", "analysis"), os.path.join(_P, "src", "prime_agents")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import policy_frontier as pf  # noqa: E402

import pytest  # noqa: E402

_M = pf.collect()
_B = pf.collect(topology="branching")
needs_runs = pytest.mark.skipif(not _M, reason="policy runs not present")


@needs_runs
def test_every_frontier_arm_ran_at_every_anchor():
    for arm in pf.FRONTIER:
        for k in pf.ANCHORS:
            assert (arm, k) in _M, "%s missing at %s" % (arm, k)
            assert _M[(arm, k)][1] == 480, (arm, k, _M[(arm, k)][1])


@needs_runs
def test_deployable_heuristic_matches_the_clairvoyant_ceiling():
    """If this ever fails there IS planning headroom and the control-law
    null would need re-interpreting."""
    for k in pf.ANCHORS:
        sr = _M[("shortest_route", k)][0]
        cc = _M[("clairvoyant_ceiling", k)][0]
        assert abs(sr - cc) < 1e-6, (k, sr, cc)


@needs_runs
def test_frontier_is_monotone_in_policy_strength():
    for k in pf.ANCHORS:
        vals = [_M[(a, k)][0] for a in pf.FRONTIER]
        assert vals == sorted(vals), (k, vals)


@needs_runs
def test_direction_beats_ordering_only_under_budget_pressure():
    """Ordering vs direction (docs/EXECUTION_PLAN.md, Part R4, Q4), answered
    with no model in the loop."""
    for k in ("bmin", "bmin1"):
        gap = _M[("shortest_route", k)][0] - _M[("masked_reactive_prefix", k)][0]
        assert gap > 4.0, (k, gap)
    for k in ("2bmin", "B20"):
        gap = _M[("shortest_route", k)][0] - _M[("masked_reactive_prefix", k)][0]
        assert abs(gap) < 1e-6, (k, gap)


@needs_runs
def test_direction_effect_is_larger_in_the_branching_stratum():
    """K1: the control-law effect is pre-registered as branching-only, and
    the nested stratum is the predicted-null control."""
    pooled = _M[("shortest_route", "bmin")][0] - _M[("masked_reactive_prefix", "bmin")][0]
    branch = _B[("shortest_route", "bmin")][0] - _B[("masked_reactive_prefix", "bmin")][0]
    assert branch > pooled > 0, (branch, pooled)


@needs_runs
def test_random_baseline_is_the_weakest_and_improves_with_budget():
    r = [_M[("random_masked", k)][0] for k in pf.ANCHORS]
    assert r == sorted(r)
    assert r[0] < _M[("shortest_route", "bmin")][0]
