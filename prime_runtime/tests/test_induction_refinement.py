"""Tests for analysis/induction_refinement.py (oracle-guided iterative
refinement + self-consistency majority voting -- the judge-free attempt to
improve induction).

No network / no ollama: ``build_oracle_critique`` and ``majority_vote_canonical``
are pure functions of the executable oracle / of plain data, and
``refine_instance`` is exercised against a scripted fake LLM (mirroring
``runner.llm.ScriptedLLM``'s spirit) so the control flow (round counting,
early stopping, invalid-revision handling) is tested deterministically and
fast, exactly like the rest of this suite.
"""

import json
import os

import pytest

from analysis import induction_refinement as refine
from analysis import induction_table_b as itb
from prime_runtime.schema import load_instance

_TESTS = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.join(_TESTS, "fixtures", "instance_tiny_refine.json")


@pytest.fixture
def tiny_inst():
    return load_instance(_FIXTURE)


@pytest.fixture
def tiny_oracle(tiny_inst):
    return itb.InstanceOracle(tiny_inst)


# ---------------------------------------------------------------------------
# build_oracle_critique: deterministic, judge-free, oracle-derived
# ---------------------------------------------------------------------------


def test_critique_flags_over_authorization_for_under_specified_route(tiny_inst, tiny_oracle):
    """Gold route is [E1, E2]; a route citing only E1 is satisfied whenever
    E1=T (including E1=T,E2=F, where the oracle does NOT support C1) -- a
    pure over-authorization, zero misses."""
    canonical = {"C1": (frozenset({"E1"}),)}
    critique, n_over, n_under = refine.build_oracle_critique(tiny_inst, tiny_oracle, canonical)
    assert n_over >= 1
    assert n_under == 0
    assert "OVER-AUTHORIZES" in critique
    assert "C1" in critique


def test_critique_flags_under_authorization_for_empty_program(tiny_inst, tiny_oracle):
    """An abstaining program (no routes at all) never over-authorizes but
    misses the one oracle-supported state (E1=T,E2=T)."""
    canonical = {"C1": ()}
    critique, n_over, n_under = refine.build_oracle_critique(tiny_inst, tiny_oracle, canonical)
    assert n_over == 0
    assert n_under >= 1
    assert "MISSED" in critique


def test_critique_is_empty_for_the_exact_gold_route(tiny_inst, tiny_oracle):
    canonical = {"C1": (frozenset({"E1", "E2"}),)}
    critique, n_over, n_under = refine.build_oracle_critique(tiny_inst, tiny_oracle, canonical)
    assert (n_over, n_under) == (0, 0)
    assert "found 0 disagreement(s)" in critique


# ---------------------------------------------------------------------------
# majority_vote_canonical: deterministic majority vote
# ---------------------------------------------------------------------------


def test_majority_vote_keeps_routes_at_or_above_ceil_n_over_2():
    # N=5 draws; a route appearing in 3/5 (>= ceil(5/2)=3) is kept, one
    # appearing in 2/5 is dropped.
    draws = [
        {"C1": [["E1", "E2"]]},
        {"C1": [["E1", "E2"]]},
        {"C1": [["E1", "E2"]]},
        {"C1": [["E1"]]},
        {"C1": [["E1"]]},
    ]
    voted, threshold = refine.majority_vote_canonical(draws, ["C1"])
    assert threshold == 3
    assert voted["C1"] == (frozenset({"E1", "E2"}),)


def test_majority_vote_invalid_draws_contribute_no_votes():
    draws = [
        {"C1": [["E1", "E2"]]},
        None,
        None,
        None,
        None,
    ]
    voted, threshold = refine.majority_vote_canonical(draws, ["C1"])
    assert threshold == 3
    assert voted["C1"] == ()  # 1/5 < threshold 3: dropped


def test_majority_vote_unanimous_route_always_kept():
    draws = [{"C1": [["E1"]]}] * 5
    voted, _ = refine.majority_vote_canonical(draws, ["C1"])
    assert voted["C1"] == (frozenset({"E1"}),)


def test_majority_vote_claim_with_zero_votes_gets_empty_family():
    voted, _ = refine.majority_vote_canonical([None, None], ["C1", "C2"])
    assert voted == {"C1": (), "C2": ()}


# ---------------------------------------------------------------------------
# refine_instance: control flow against a scripted fake LLM
# ---------------------------------------------------------------------------


class _QueueLLM:
    """Returns one canned chat-completion 'content' string per call, in
    order -- enough to drive ``induce_program``/``_propose_revision``
    deterministically without any network or ollama dependency."""

    def __init__(self, contents):
        self._queue = list(contents)
        self.n_calls = 0

    def chat(self, messages, tools=None):
        self.n_calls += 1
        content = self._queue.pop(0) if self._queue else "{}"
        return {"content": content, "tool_calls": [], "usage": {}}


def _licenses_payload(routes_by_claim):
    return json.dumps({"licenses": routes_by_claim})


def test_refine_instance_fixes_an_under_specified_route_in_one_round(tiny_inst, tiny_oracle):
    llm = _QueueLLM([
        _licenses_payload({"C1": [["E1"]]}),           # initial induction: under-specified
        _licenses_payload({"C1": [["E1", "E2"]]}),      # round-1 revision: exact gold route
    ])
    result = refine.refine_instance(tiny_inst, llm, tiny_oracle, K=3)
    assert result["valid_start"] is True
    assert result["rounds_used"] == 1
    assert result["stopped_early"] is True
    assert result["revision_failed"] is False
    before_precise = [rr["precise"] for rr in result["before"]["route_records"]]
    after_precise = [rr["precise"] for rr in result["after"]["route_records"]]
    assert not all(before_precise)  # the under-specified starting route was imprecise
    assert all(after_precise)       # refinement produced an oracle-precise program
    assert llm.n_calls == 2


def test_refine_instance_stops_immediately_if_already_oracle_perfect(tiny_inst, tiny_oracle):
    llm = _QueueLLM([_licenses_payload({"C1": [["E1", "E2"]]})])
    result = refine.refine_instance(tiny_inst, llm, tiny_oracle, K=3)
    assert result["rounds_used"] == 0
    assert result["stopped_early"] is True
    assert result["before"]["exact_program"] is True
    assert result["after"]["exact_program"] is True
    assert llm.n_calls == 1  # only the initial induction call; no revision round needed


def test_refine_instance_caps_at_k_rounds_when_model_never_improves(tiny_inst, tiny_oracle):
    # The model proposes the SAME under-specified route every time: K=2
    # rounds should both fire (never reaching zero disagreements) and the
    # program should end up unchanged (still imprecise), not crash.
    llm = _QueueLLM([_licenses_payload({"C1": [["E1"]]})] * 3)
    result = refine.refine_instance(tiny_inst, llm, tiny_oracle, K=2)
    assert result["rounds_used"] == 2
    assert result["stopped_early"] is False
    assert result["before"]["route_records"] == result["after"]["route_records"]


def test_refine_instance_invalid_starting_induction_is_reported_not_crashed(tiny_inst, tiny_oracle):
    llm = _QueueLLM(["not json at all"] * 6)
    result = refine.refine_instance(tiny_inst, llm, tiny_oracle, K=3)
    assert result["valid_start"] is False
    assert result["before"] is None and result["after"] is None
    assert result["error"]


def test_refine_instance_unparseable_revision_stops_without_crashing(tiny_inst, tiny_oracle):
    llm = _QueueLLM([
        _licenses_payload({"C1": [["E1"]]}),  # initial: under-specified
        "not json",                            # round-1 revision attempt 1: fails
        "still not json",                      # round-1 revision retry: fails
    ])
    result = refine.refine_instance(tiny_inst, llm, tiny_oracle, K=3)
    assert result["valid_start"] is True
    assert result["revision_failed"] is True
    assert result["rounds_used"] == 1
    # Program is unchanged from the (imprecise) starting point.
    assert result["before"]["route_records"] == result["after"]["route_records"]


# ---------------------------------------------------------------------------
# run_self_consistency: end-to-end against a small synthetic Table-B JSON
# ---------------------------------------------------------------------------


def test_run_self_consistency_end_to_end(tmp_path, tiny_inst):
    """Builds a tiny fake induction_table_b.py-shaped JSON (5 draws, 3
    voting for the gold route, 2 for an under-specified one) and checks the
    voted program matches the majority."""
    draws = []
    for routes in (
        [["E1", "E2"]], [["E1", "E2"]], [["E1", "E2"]], [["E1"]], [["E1"]],
    ):
        draws.append({
            "instance": tiny_inst.id, "valid": True, "error": "",
            "induced_licenses": {"C1": routes},
        })
    fake = {
        "F_STAR_SOURCE": "test",
        "raw": [{"model": "fake-model", "n": 5, "valid": 5, "per_instance": draws}],
    }
    src = tmp_path / "fake_table_b.json"
    src.write_text(json.dumps(fake))

    out = refine.run_self_consistency(str(src), [_FIXTURE], ["fake-model"], bootstrap_iters=200)
    assert "fake-model" in out
    agg = out["fake-model"]["aggregate"]
    assert agg["exact_match"] == 1.0  # majority route == exact gold route
    assert agg["route_precision_semantic"] == 1.0
    meta = out["fake-model"]["meta"][0]
    assert meta["n_draws"] == 5 and meta["threshold"] == 3 and meta["n_valid_draws"] == 5


def test_run_self_consistency_missing_induced_licenses_raises(tmp_path):
    fake = {
        "raw": [{"model": "old-model", "n": 1, "valid": 1, "per_instance": [
            {"instance": "x", "valid": True, "error": ""},
        ]}],
    }
    src = tmp_path / "old_table_b.json"
    src.write_text(json.dumps(fake))
    with pytest.raises(RuntimeError):
        refine.run_self_consistency(str(src), [_FIXTURE], ["old-model"], bootstrap_iters=10)
