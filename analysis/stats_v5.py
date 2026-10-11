#!/usr/bin/env python3
"""Frozen statistical machinery for E1/E2/E3 (EXECUTION_PLAN v5 Part 3 /
FREEZE.md F6). Pure computation; no data-dependent choices.

Design note that makes these tests valid: the generator assigns each
held-out record to EXACTLY ONE template family (GPT Q3 constraint,
enforced mechanically at generation). The dependence structure is
therefore NESTED (records within families), not crossed -- so
family-level sign-flip randomization is exact under the sharp null, and a
hierarchical (family, then record-within-family) bootstrap is the matching
interval estimator.

  primary inference   template_cluster_randomization  (exact, sign-flip)
  effect size         hierarchical_cluster_bootstrap  (CI)
  sensitivity         exact_mcnemar                   (record independence)
  equivalence         tost                            (vs frozen delta)
  multiplicity        holm

Every function takes paired observations as
  rows = [{"record", "family", "domain", "topology", "a", "b"}, ...]
where a/b are the two arms' per-record values (binary for OFAR, utility
points for regret).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

SEED = 20260802


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def family_means(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Per-family mean of the per-record paired difference (a - b)."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        acc[r["family"]].append(float(r["a"]) - float(r["b"]))
    return {f: sum(v) / len(v) for f, v in acc.items()}


def observed_statistic(rows: Sequence[Dict[str, Any]]) -> float:
    """Cluster-weighted mean paired difference (each family weighs 1)."""
    fm = family_means(rows)
    return sum(fm.values()) / len(fm) if fm else float("nan")


def template_cluster_randomization(rows: Sequence[Dict[str, Any]],
                                   n_perm: int = 20000,
                                   seed: int = SEED) -> Dict[str, Any]:
    """Exact-in-the-limit sign-flip test at the FAMILY cluster level,
    stratified by (domain, topology): under the sharp null of no arm
    effect, each family's aggregate difference is equally likely to have
    either sign. Two-sided."""
    fm = family_means(rows)
    if not fm:
        return {"p": float("nan"), "n_clusters": 0}
    strat: Dict[str, Tuple[str, str]] = {}
    for r in rows:
        strat.setdefault(r["family"], (r.get("domain", ""), r.get("topology", "")))
    fams = sorted(fm)
    vals = [fm[f] for f in fams]
    obs = abs(sum(vals) / len(vals))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = 0.0
        for v in vals:
            s += v if rng.random() < 0.5 else -v
        if abs(s / len(vals)) >= obs - 1e-12:
            hits += 1
    return {"p": (hits + 1) / (n_perm + 1), "n_clusters": len(fams),
            "statistic": sum(vals) / len(vals),
            "n_discordant_clusters": sum(1 for v in vals if abs(v) > 1e-12),
            "strata": len({strat[f] for f in fams})}


def hierarchical_cluster_bootstrap(rows: Sequence[Dict[str, Any]],
                                   n_boot: int = 5000, alpha: float = 0.05,
                                   seed: int = SEED) -> Dict[str, Any]:
    """Resample families with replacement, then records within each drawn
    family with replacement; percentile CI on the cluster-weighted mean
    paired difference."""
    by_fam: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(float(r["a"]) - float(r["b"]))
    fams = sorted(by_fam)
    if not fams:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    point = sum(sum(v) / len(v) for v in by_fam.values()) / len(fams)
    draws = []
    for _ in range(n_boot):
        acc = []
        for _ in range(len(fams)):
            f = fams[rng.randrange(len(fams))]
            recs = by_fam[f]
            acc.append(sum(recs[rng.randrange(len(recs))] for _ in recs) / len(recs))
        draws.append(sum(acc) / len(acc))
    draws.sort()
    lo = draws[int(alpha / 2 * n_boot)]
    hi = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"point": point, "lo": lo, "hi": hi, "n_clusters": len(fams),
            "n_records": len(rows)}


def exact_mcnemar(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sensitivity analysis treating RECORDS as independent (binary a/b)."""
    b = sum(1 for r in rows if float(r["a"]) > float(r["b"]))
    c = sum(1 for r in rows if float(r["b"]) > float(r["a"]))
    n = b + c
    if n == 0:
        return {"p": 1.0, "b": 0, "c": 0}
    from math import comb
    lo = min(b, c)
    p = 2.0 * sum(comb(n, k) for k in range(lo + 1)) / (2 ** n)
    return {"p": min(1.0, p), "b": b, "c": c}


def tost(rows: Sequence[Dict[str, Any]], delta: float,
         alpha: float = 0.05, seed: int = SEED) -> Dict[str, Any]:
    """Two one-sided tests for equivalence within +/- delta, using the
    family-cluster SE. Equivalent iff the (1-2*alpha) CI lies inside
    (-delta, delta)."""
    fm = list(family_means(rows).values())
    k = len(fm)
    if k < 2:
        return {"equivalent": None, "reason": "insufficient clusters", "k": k}
    mean = sum(fm) / k
    sd = math.sqrt(sum((x - mean) ** 2 for x in fm) / (k - 1))
    se = sd / math.sqrt(k)
    z = 1.645 if alpha == 0.05 else abs(_z_from_alpha(alpha))
    lo, hi = mean - z * se, mean + z * se
    return {"equivalent": bool(lo > -delta and hi < delta),
            "mean": mean, "se": se, "ci_lo": lo, "ci_hi": hi,
            "delta": delta, "k_clusters": k,
            "p_lower": 1 - _phi((mean + delta) / se) if se > 0 else 0.0,
            "p_upper": _phi((mean - delta) / se) if se > 0 else 0.0}


def _z_from_alpha(alpha: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _phi(mid) < 1 - alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def holm(pvalues: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Holm-Bonferroni within one prespecified family."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = {"p_raw": p, "p_holm": adj, "reject_05": adj < 0.05}
    return out


def power_tost(rows: Sequence[Dict[str, Any]], delta: float,
               alpha: float = 0.05) -> float:
    """Analytic TOST power at a true effect of zero, from the observed
    cluster-mean dispersion (used to decide verdict-vs-CI language)."""
    fm = list(family_means(rows).values())
    k = len(fm)
    if k < 2:
        return 0.0
    mean = sum(fm) / k
    sd = math.sqrt(sum((x - mean) ** 2 for x in fm) / (k - 1))
    se = sd / math.sqrt(k)
    if se <= 0:
        return 1.0
    z = 1.645 if alpha == 0.05 else abs(_z_from_alpha(alpha))
    return max(0.0, _phi((delta - z * se) / se) - _phi((-delta + z * se) / se))
