"""Tests for the uncertainty additions to experiments/analysis/induction_table_b.py:
the instance-cluster bootstrap CI (``bootstrap_table_b_cis``), the per-draw
``induced_licenses`` capture (needed by self-consistency voting), and the
``--temperature`` threading through ``LLMClient`` (matched-temperature panel).

No network / no ollama: everything here is exercised on synthetic
``model_result`` dicts shaped exactly like ``run_model``'s real output, so
these tests are as fast and hermetic as the rest of the suite.
"""

import math
import os

import pytest

import induction_table_b as itb
from prime_runtime.schema import load_instance
from prime_agents.llm import LLMClient

_TESTS = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.join(_TESTS, "fixtures", "instance_tiny_refine.json")


def _rec(instance, valid, precise_flags, exact, authfp_episode, real_authfp, num=1, den=1):
    return {
        "instance": instance, "valid": valid, "error": "",
        "route_recall_num": num, "route_recall_den": den,
        "route_records": [{"claim": "C1", "precise": p, "prime": p} for p in precise_flags],
        "coverages": [1.0] if precise_flags else [],
        "exact_program": exact,
        "authfp_state_frac": 0.1, "authfn_state_frac": 0.0,
        "authfp_episode": authfp_episode, "authfn_episode": False,
        "util_num": 0, "util_den": 1,
        "gold_claims_with_routes": 1, "claim_license_hit": 1, "ind_claims_with_routes": 1,
        "gold_atoms_used": 2, "atom_use_hit": 2, "ind_atoms_used": 2,
        "failure_modes": {"C1": "exact" if exact else "unsound_under_specified"},
        "real_authfp": real_authfp, "real_authfn": False, "usable": valid and not real_authfp,
        "induced_licenses": {"C1": [["E1", "E2"]] if exact else [["E1"]]},
    }


def _synthetic_model_result():
    # Two instances, two draws each: i1 is perfect on both draws; i2 is
    # imprecise on both draws (over-authorizes) -- a clean, hand-checkable
    # point estimate for every metric.
    per_instance = [
        _rec("i1", True, [True], True, False, False),
        _rec("i1", True, [True], True, False, False),
        _rec("i2", True, [False], False, True, True),
        _rec("i2", True, [False], False, True, True),
    ]
    return {"model": "synthetic", "n": 4, "valid": 4, "per_instance": per_instance}


def test_bootstrap_cis_point_estimates_match_aggregate():
    mr = _synthetic_model_result()
    agg = itb.aggregate(mr)
    ci = itb.bootstrap_table_b_cis(mr, iters=1000, seed=1)
    for name in itb.CI_METRICS:
        assert ci[name]["point"] == pytest.approx(agg[name])


def test_bootstrap_cis_bounds_are_sane():
    mr = _synthetic_model_result()
    ci = itb.bootstrap_table_b_cis(mr, iters=1000, seed=1)
    for name, entry in ci.items():
        assert entry["lo"] <= entry["point"] + 1e-9
        assert entry["point"] <= entry["hi"] + 1e-9
        assert entry["n_clusters"] == 2  # two instances = two clusters


def test_bootstrap_cis_ci_reflects_between_instance_disagreement():
    """i1 is always precise and i2 is always imprecise: resampling instances
    with replacement can produce (i1,i1) -> precision 1.0 or (i2,i2) ->
    precision 0.0, so the CI must span most of [0,1] -- with N=9 instances
    and a single draw each, a metric that is a coin flip across instances
    must SHOW a wide CI, not a falsely tight one."""
    mr = _synthetic_model_result()
    ci = itb.bootstrap_table_b_cis(mr, iters=3000, seed=1)
    prec = ci["route_precision_semantic"]
    assert prec["lo"] < 0.5 < prec["hi"]


def test_bootstrap_cis_handles_invalid_draws_in_denominator():
    """An invalid draw must still count in AuthFP_real's denominator when
    real_authfp is known, and never counts toward route-precision (no route
    records for an invalid draw) -- mirrors score_instance's own
    all-attempts denominator convention, not a separate rule invented here."""
    per_instance = [
        _rec("i1", True, [True], True, False, False),
        {
            "instance": "i1", "valid": False, "error": "boom",
            "route_recall_num": 0, "route_recall_den": 1, "route_records": [], "coverages": [],
            "exact_program": False, "authfp_state_frac": 0.0, "authfn_state_frac": 0.5,
            "authfp_episode": False, "authfn_episode": True, "util_num": 1, "util_den": 1,
            "gold_claims_with_routes": 1, "claim_license_hit": 0, "ind_claims_with_routes": 0,
            "gold_atoms_used": 2, "atom_use_hit": 0, "ind_atoms_used": 0,
            "failure_modes": {}, "real_authfp": False, "real_authfn": True, "usable": False,
            "induced_licenses": {"C1": []},
        },
    ]
    mr = {"model": "synthetic", "n": 2, "valid": 1, "per_instance": per_instance}
    ci = itb.bootstrap_table_b_cis(mr, iters=200, seed=1)
    # Both draws (valid + invalid) contribute to AuthFP_real's denominator.
    assert ci["AuthFP_real"]["n_clusters"] == 1  # one instance cluster ("i1")
    assert ci["valid_rate"]["point"] == 0.5
    # Route precision is computed over valid draws' route_records only.
    assert ci["route_precision_semantic"]["point"] == 1.0


def test_bootstrap_cis_empty_valid_recs_do_not_crash():
    per_instance = [
        {
            "instance": "i1", "valid": False, "error": "boom",
            "route_recall_num": 0, "route_recall_den": 1, "route_records": [], "coverages": [],
            "exact_program": False, "authfp_state_frac": 0.0, "authfn_state_frac": 0.5,
            "authfp_episode": False, "authfn_episode": True, "util_num": 1, "util_den": 1,
            "gold_claims_with_routes": 1, "claim_license_hit": 0, "ind_claims_with_routes": 0,
            "gold_atoms_used": 2, "atom_use_hit": 0, "ind_atoms_used": 0,
            "failure_modes": {}, "real_authfp": None, "real_authfn": None, "usable": False,
            "induced_licenses": {"C1": []},
        },
    ]
    mr = {"model": "synthetic", "n": 1, "valid": 0, "per_instance": per_instance}
    ci = itb.bootstrap_table_b_cis(mr, iters=100)
    assert math.isnan(ci["route_precision_semantic"]["point"])
    assert math.isnan(ci["AuthFP_real"]["point"])  # real_authfp is None everywhere


# ---------------------------------------------------------------------------
# induced_licenses capture (self-consistency needs the real routes, not
# just precise/prime booleans)
# ---------------------------------------------------------------------------


def test_score_instance_records_induced_licenses():
    inst = load_instance(_FIXTURE)
    oracle = itb.InstanceOracle(inst)
    program = itb._make_effective_program(inst, {"C1": (frozenset({"E1"}),)})
    rec = itb.score_instance(inst, oracle, program, True)
    assert rec["induced_licenses"] == {"C1": [["E1"]]}


def test_score_instance_induced_licenses_empty_for_invalid_draw():
    inst = load_instance(_FIXTURE)
    oracle = itb.InstanceOracle(inst)
    rec = itb.score_instance(inst, oracle, None, False)
    assert rec["induced_licenses"] == {"C1": []}


# ---------------------------------------------------------------------------
# --temperature threading (LLMClient._restricted_temp already existed;
# run_model now passes args.temperature through and records what happened)
# ---------------------------------------------------------------------------


def test_llmclient_honors_requested_temperature_for_ordinary_models():
    # base is explicit: these assert temperature handling, not the
    # endpoint, and the default base is a paid host (guarded)
    llm = LLMClient(model="gpt-4.1", temperature=1.0,
                    base="http://localhost:8000/v1")
    assert llm.temperature == 1.0
    assert llm._restricted_temp() is False


def test_llmclient_restricted_temp_models_ignore_requested_temperature():
    llm = LLMClient(model="gpt-5", temperature=0.0,
                    base="http://localhost:8000/v1")
    assert llm._restricted_temp() is True
    # The requested value is still stored (for bookkeeping/logging) even
    # though _chat_openai/_chat_urllib both skip sending it for gpt-5.
    assert llm.temperature == 0.0
