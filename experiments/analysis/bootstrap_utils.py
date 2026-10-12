#!/usr/bin/env python3
"""Cluster-bootstrap helpers shared by the induction-quality (Table B) and
induction-refinement scripts (N=9 instances, single draw -> N draws x
instances with a proper CI).

Resampling unit: the INSTANCE. This mirrors ``experiments/analysis/aggregate.py``'s
existing seed-level cluster bootstrap (that script's docstring: "episodes
sharing a seed are dependent (controlled variants), so all CIs are cluster
bootstrap over seeds" -- its own ``episode_metrics`` even falls back to the
instance id when no explicit seed is present, ``ep.get("seed",
ep.get("instance", "?"))``). Here the natural cluster is the INSTANCE: every
draw of a given (model, instance) shares that instance's atoms/gold
program/oracle and is therefore dependent, while different instances are
independent draws of the benchmark. A bootstrap replicate resamples
INSTANCES with replacement and pools ALL of a resampled instance's draws
together (never resampling within an instance) -- exactly
``aggregate.cluster_bootstrap``'s convention, generalized here to also cover
ratio-of-sums metrics (route recall) via :func:`cluster_bootstrap_ratio`.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

Number = float


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def cluster_bootstrap_mean(
    values_by_cluster: Dict[str, List[float]], iters: int = 5000, seed: int = 7
) -> Tuple[float, float, float]:
    """Point estimate + 95% CI for a plain mean of per-cluster atomic values.

    ``values_by_cluster[instance_id]`` is the list of every atomic
    observation (one route record, one draw's 0/1 outcome, one coverage
    fraction, ...) belonging to that instance, pooled over every draw.
    """
    clusters = sorted(c for c, vs in values_by_cluster.items() if vs)
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = _mean([v for c in clusters for v in values_by_cluster[c]])
    stats = []
    for _ in range(iters):
        sample = [rng.choice(clusters) for _ in clusters]
        vals = [v for c in sample for v in values_by_cluster[c]]
        stats.append(_mean(vals))
    stats.sort()
    lo = stats[int(0.025 * iters)]
    hi = stats[min(int(0.975 * iters), iters - 1)]
    return point, lo, hi


def cluster_bootstrap_ratio(
    pairs_by_cluster: Dict[str, List[Tuple[float, float]]], iters: int = 5000, seed: int = 7
) -> Tuple[float, float, float]:
    """Point estimate + 95% CI for a ratio-of-sums metric (e.g. route recall
    = sum(recovered) / sum(gold routes), macro over claims/draws but
    resampled at the instance-cluster level).

    ``pairs_by_cluster[instance_id]`` is a list of (numerator, denominator)
    pairs -- one per draw (or per claim, per the caller's atomic-unit
    choice) belonging to that instance.
    """
    clusters = sorted(c for c, ps in pairs_by_cluster.items() if ps)
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)

    def _ratio(cluster_list: Sequence[str]) -> float:
        num = sum(n for c in cluster_list for n, d in pairs_by_cluster[c])
        den = sum(d for c in cluster_list for n, d in pairs_by_cluster[c])
        return num / den if den else 0.0

    point = _ratio(clusters)
    stats = []
    for _ in range(iters):
        sample = [rng.choice(clusters) for _ in clusters]
        stats.append(_ratio(sample))
    stats.sort()
    lo = stats[int(0.025 * iters)]
    hi = stats[min(int(0.975 * iters), iters - 1)]
    return point, lo, hi
