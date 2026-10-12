"""Tests for analysis/bootstrap_utils.py (the instance-cluster bootstrap CI
shared by induction_table_b.py and induction_refinement.py)."""

import math

from analysis.bootstrap_utils import cluster_bootstrap_mean, cluster_bootstrap_ratio


# ---------------------------------------------------------------------------
# cluster_bootstrap_mean
# ---------------------------------------------------------------------------


def test_mean_point_estimate_is_the_pooled_mean():
    data = {"i1": [1.0, 1.0, 0.0], "i2": [1.0], "i3": [0.0, 0.0]}
    pooled = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    pt, lo, hi = cluster_bootstrap_mean(data, iters=2000, seed=1)
    assert pt == sum(pooled) / len(pooled)
    assert 0.0 <= lo <= pt <= hi <= 1.0


def test_mean_zero_variance_when_every_cluster_identical():
    data = {"i1": [1.0, 1.0], "i2": [1.0], "i3": [1.0, 1.0, 1.0]}
    pt, lo, hi = cluster_bootstrap_mean(data, iters=1000, seed=1)
    assert pt == lo == hi == 1.0


def test_mean_single_cluster_ci_collapses_to_point():
    """With only one cluster, every bootstrap resample IS that cluster
    (resampling with replacement from a size-1 population always returns
    the same element), so lo == hi == point exactly."""
    data = {"only": [1.0, 0.0, 1.0, 0.0]}
    pt, lo, hi = cluster_bootstrap_mean(data, iters=500, seed=3)
    assert pt == 0.5
    assert lo == hi == pt


def test_mean_empty_input_returns_nan():
    pt, lo, hi = cluster_bootstrap_mean({}, iters=100)
    assert math.isnan(pt) and math.isnan(lo) and math.isnan(hi)
    pt, lo, hi = cluster_bootstrap_mean({"i1": []}, iters=100)
    assert math.isnan(pt)


def test_mean_is_deterministic_given_a_seed():
    data = {"i1": [1.0, 0.0, 1.0], "i2": [0.0, 0.0], "i3": [1.0]}
    r1 = cluster_bootstrap_mean(data, iters=800, seed=42)
    r2 = cluster_bootstrap_mean(data, iters=800, seed=42)
    assert r1 == r2


def test_mean_more_clusters_of_mixed_data_narrows_or_keeps_ci():
    """More independent, non-degenerate clusters should not make the CI
    wider than a single cluster with the same per-cluster values repeated
    (sanity check on the resampling direction, not a tight bound)."""
    one = {"i1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]}
    many = {"i%d" % k: [1.0, 0.0] for k in range(8)}
    _, lo1, hi1 = cluster_bootstrap_mean(one, iters=3000, seed=5)
    _, lo2, hi2 = cluster_bootstrap_mean(many, iters=3000, seed=5)
    assert (hi2 - lo2) <= (hi1 - lo1) + 1e-9


# ---------------------------------------------------------------------------
# cluster_bootstrap_ratio
# ---------------------------------------------------------------------------


def test_ratio_point_estimate_is_ratio_of_pooled_sums():
    data = {"i1": [(2.0, 4.0), (1.0, 1.0)], "i2": [(0.0, 3.0)]}
    pt, lo, hi = cluster_bootstrap_ratio(data, iters=2000, seed=2)
    assert pt == (2.0 + 1.0 + 0.0) / (4.0 + 1.0 + 3.0)
    assert 0.0 <= lo <= pt <= hi <= 1.0


def test_ratio_zero_denominator_pairs_do_not_crash():
    data = {"i1": [(0.0, 0.0)], "i2": [(1.0, 2.0)]}
    pt, lo, hi = cluster_bootstrap_ratio(data, iters=500, seed=1)
    assert pt == 1.0 / 2.0


def test_ratio_empty_input_returns_nan():
    pt, lo, hi = cluster_bootstrap_ratio({}, iters=50)
    assert math.isnan(pt)


def test_ratio_single_cluster_collapses_to_point():
    data = {"only": [(3.0, 4.0), (1.0, 4.0)]}
    pt, lo, hi = cluster_bootstrap_ratio(data, iters=300, seed=9)
    assert pt == (3.0 + 1.0) / (4.0 + 4.0)
    assert lo == hi == pt
