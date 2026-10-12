"""Calibration tests for the frozen statistical machinery
(experiments/analysis/stats_v5.py). Each test pins a property the paper's inference
depends on; all are deterministic (fixed seeds)."""
from __future__ import annotations

import os
import random
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "experiments", "analysis")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import stats_v5 as st  # noqa: E402


def mkrows(per_family, domain="d", topology="branching"):
    """per_family: {family: [(a, b), ...]}"""
    rows = []
    for fam, pairs in per_family.items():
        for i, (a, b) in enumerate(pairs):
            rows.append({"record": "%s-%d" % (fam, i), "family": fam,
                         "domain": domain, "topology": topology, "a": a, "b": b})
    return rows


# ---------------------------------------------------------------------------
# randomization test
# ---------------------------------------------------------------------------

def test_randomization_null_when_no_effect():
    rows = mkrows({("f%d" % f): [(1, 1)] * 5 for f in range(20)})
    r = st.template_cluster_randomization(rows, n_perm=2000)
    assert r["p"] == 1.0 or r["p"] > 0.9
    assert r["n_discordant_clusters"] == 0
    assert r["n_clusters"] == 20


def test_randomization_detects_consistent_cluster_effect():
    rows = mkrows({("f%d" % f): [(1, 0)] * 4 for f in range(20)})
    r = st.template_cluster_randomization(rows, n_perm=2000)
    assert r["p"] < 0.01
    assert r["statistic"] == 1.0


def test_randomization_respects_cluster_not_record_count():
    """Many records inside ONE family cannot manufacture significance --
    this is the overclaim the two-way clustering exists to prevent."""
    rows = mkrows({"only": [(1, 0)] * 500})
    r = st.template_cluster_randomization(rows, n_perm=2000)
    assert r["n_clusters"] == 1
    assert r["p"] > 0.4  # a single cluster can never reject


def test_randomization_is_deterministic():
    rows = mkrows({("f%d" % f): [(f % 3, 0)] for f in range(12)})
    a = st.template_cluster_randomization(rows, n_perm=1000)
    b = st.template_cluster_randomization(rows, n_perm=1000)
    assert a["p"] == b["p"]


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_ci_covers_point_and_excludes_zero_for_real_effect():
    rows = mkrows({("f%d" % f): [(1, 0)] * 4 for f in range(20)})
    ci = st.hierarchical_cluster_bootstrap(rows, n_boot=800)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["lo"] > 0


def test_bootstrap_ci_includes_zero_for_null():
    rng = random.Random(7)
    rows = mkrows({("f%d" % f): [(rng.gauss(0, 1), 0) for _ in range(5)]
                   for f in range(25)})
    ci = st.hierarchical_cluster_bootstrap(rows, n_boot=800)
    assert ci["lo"] < 0 < ci["hi"]


# ---------------------------------------------------------------------------
# TOST
# ---------------------------------------------------------------------------

def test_tost_declares_equivalence_for_tight_null():
    rng = random.Random(3)
    rows = mkrows({("f%d" % f): [(rng.gauss(0, 0.3), 0) for _ in range(5)]
                   for f in range(30)})
    r = st.tost(rows, delta=5.0)
    assert r["equivalent"] is True
    assert abs(r["mean"]) < 5.0


def test_tost_refuses_equivalence_when_effect_exceeds_delta():
    rows = mkrows({("f%d" % f): [(10.0, 0)] * 4 for f in range(20)})
    r = st.tost(rows, delta=5.0)
    assert r["equivalent"] is False


def test_tost_refuses_equivalence_when_underpowered():
    """Wide dispersion, few clusters: neither difference nor equivalence --
    the 'inconclusive' branch the paper pre-commits to reporting."""
    rng = random.Random(11)
    rows = mkrows({("f%d" % f): [(rng.gauss(0, 40), 0) for _ in range(3)]
                   for f in range(6)})
    r = st.tost(rows, delta=5.0)
    assert r["equivalent"] is False
    assert st.power_tost(rows, 5.0) < 0.8


def test_power_tost_high_when_tight():
    rng = random.Random(5)
    rows = mkrows({("f%d" % f): [(rng.gauss(0, 0.5), 0) for _ in range(5)]
                   for f in range(40)})
    assert st.power_tost(rows, 5.0) > 0.9


# ---------------------------------------------------------------------------
# mcnemar + holm
# ---------------------------------------------------------------------------

def test_exact_mcnemar_matches_known_case():
    rows = mkrows({"f": [(1, 0)] * 10 + [(0, 1)] * 0})
    r = st.exact_mcnemar(rows)
    assert r["b"] == 10 and r["c"] == 0
    assert r["p"] < 0.01


def test_holm_orders_and_adjusts():
    out = st.holm({"x": 0.001, "y": 0.02, "z": 0.5})
    assert out["x"]["p_holm"] == 0.003
    assert out["z"]["p_holm"] == 0.5
    assert out["x"]["reject_05"] and not out["z"]["reject_05"]
    # monotone
    assert out["x"]["p_holm"] <= out["y"]["p_holm"] <= out["z"]["p_holm"]
