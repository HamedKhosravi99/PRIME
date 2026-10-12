#!/usr/bin/env python3
"""Risk-controlled conformal on the CONFIRMATORY split (results/tableCONF.json).

Why this exists. The `conformal` arm logs a model-elicited confidence per
claim but, by construction, never thresholds them: `_elicit_claim_confidences`
in src/prime_agents/arms.py says so, and calibration is an analysis-time step. On the
carried-forward family that step ran (experiments/analysis/build_conformal.py). On the
480-instance confirmatory split it never did, so the `conformal` row of the
external-comparator table scored the arm's RAW emissions -- direct answering
plus an unused elicitation turn. Reporting that row as the
conformal/calibrated-threshold family was not a fair comparison: the family's
whole content is the calibrated selection, and the row did not contain it.

Nothing new is run here. The confidences are already in the logs (475 of 480
episodes carry them; 5 errored), and this applies the same leave-one-out
cross-conformal rule build_conformal.py already uses, then scores the
resulting emission through build_w2a_tables.score -- the SAME scorer that
produced every other cell of that table, so the numbers are on one scale and
one estimand.

Two calibration granularities are computed because they answer different
questions and the honest reading needs both:

  instance   leave-one-instance-out within a domain. Matches the published
             method on the carried-forward family. Generous to the baseline:
             instances of one template family share route topology and probe
             set, so a held-out instance's near-twins sit in its own
             calibration set.
  family     leave-one-FAMILY-out within a domain. The clustering unit the
             rest of the paper uses for inference, and the one that does not
             leak family structure into the threshold.

The baseline is reported at its better granularity wherever they differ, so
no conclusion here rests on handicapping the comparator.

Usage: python3 experiments/analysis/build_conformal_confirmatory.py [--model qwen2.5-32b]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_ext_table as ext  # noqa: E402
import build_w2a_tables as bw  # noqa: E402

RESULTS = os.path.join(PILOT, "results")
ALPHAS = (0.05, 0.10, 0.20)


def risk_threshold(cal: List[Tuple[float, bool]], alpha: float) -> Optional[float]:
    """Smallest observed score tau whose selected set has unsupported
    fraction <= alpha; None if no threshold achieves it (=> abstain).

    Identical rule to experiments/analysis/build_conformal.py, restated rather than
    imported so this script does not depend on that one's I/O path.
    """
    if not cal:
        return None
    best = None
    for tau in sorted({s for s, _y in cal}):
        sel = [(s, y) for s, y in cal if s >= tau]
        if not sel:
            continue
        if sum(1 for _s, y in sel if not y) / len(sel) <= alpha:
            if best is None or tau < best:
                best = tau
    return best


def _usable(model: str) -> List[Dict[str, Any]]:
    """Confirmatory conformal episodes carrying elicited confidences."""
    out = []
    for e in ext.load(model, "conformal"):
        f = bw.FACTS.get(e.get("instance"))
        if not f or f["split"] != "heldout_balanced" or e.get("error"):
            continue
        if e.get("claim_confidences"):
            out.append(e)
    return out


def _labels(e: Dict[str, Any]) -> Dict[str, bool]:
    """Executable-oracle support under the authored protocol -- the same
    supp_a that ofar_a is defined against, so the calibration target and the
    reported endpoint are the same object."""
    return bw.FACTS[e["instance"]]["supp_a"]


def select(model: str, alpha: float, by: str) -> List[Dict[str, Any]]:
    """Re-emit every confirmatory conformal episode under calibrated
    selection, then score it with the shared scorer."""
    eps = _usable(model)
    by_dom: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for e in eps:
        by_dom[bw.FACTS[e["instance"]]["domain"]].append(e)

    rows = []
    for _dom, cell in sorted(by_dom.items()):
        for e in cell:
            f = bw.FACTS[e["instance"]]
            held = f["family"] if by == "family" else e["instance"]
            cal = [
                (float(s), bool(_labels(o).get(c, False)))
                for o in cell
                if (bw.FACTS[o["instance"]]["family"] if by == "family"
                    else o["instance"]) != held
                for c, s in o["claim_confidences"].items()
            ]
            tau = risk_threshold(cal, alpha)
            emitted = []
            if tau is not None:
                cands = [c for c, s in e["claim_confidences"].items()
                         if float(s) >= tau]
                if cands:
                    # the family's graded back-off: strongest claim clearing
                    # the calibrated threshold
                    emitted = [{"claim": max(
                        cands, key=lambda c: (f["levels"].get(c, 0), c))}]
            s = bw.score(dict(e, emitted=emitted))
            if s:
                s["n_cal"] = len(cal)
                s["tau"] = tau
                rows.append(s)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"status": "NOT RUN"}
    n = len(rows)
    m = lambda k: sum(r[k] for r in rows) / n
    avail = [r for r in rows if r["facts"]["ceiling"] > 0]
    emit = (sum(1 for r in avail if not r["abstain"]) / len(avail)) if avail else 0.0
    n_cal_min = min(r["n_cal"] for r in rows)
    return {
        "n_instances": n,
        "n_clusters": len({r["facts"]["family"] for r in rows}),
        "ofar_a": round(100 * m("ofar_a"), 1),
        "content_false": round(100 * m("content_false"), 1),
        "utility": round(m("utility"), 1),
        "emit_rate_avail": round(100 * emit, 1),
        "abstain_rate": round(100 * m("abstain"), 1),
        "cost": round(m("cost"), 2),
        "min_n_cal": n_cal_min,
        # the empirical-risk guarantee is meaningless below this floor
        "alpha_floor": round(1.0 / (n_cal_min + 1), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-32b")
    a = ap.parse_args(argv)

    out: Dict[str, Any] = {"model": a.model, "cells": {}}
    for by in ("instance", "family"):
        for alpha in ALPHAS:
            s = summarize(select(a.model, alpha, by))
            s["alpha"] = alpha
            s["calibration"] = by
            # does the baseline meet its OWN nominal level on this split?
            s["meets_nominal"] = (s.get("ofar_a", 100.0) / 100.0) <= alpha
            out["cells"]["%s|a%.2f" % (by, alpha)] = s

    raw = ext.cell(a.model, "conformal")
    out["raw_unthresholded"] = raw

    # WHY both rows look the way they do. The calibration rule is not what
    # fails here -- the score it calibrates is. This is the citable finding
    # and it belongs ahead of either row.
    eps = _usable(a.model)
    pairs = [(float(s), bool(_labels(e).get(c, False)))
             for e in eps for c, s in e["claim_confidences"].items()]
    lo_s, hi_s = min(s for s, _y in pairs), max(s for s, _y in pairs)
    at = lambda v: [y for s, y in pairs if s == v]
    n_allzero = sum(1 for e in eps
                    if all(float(v) == 0.0 for v in e["claim_confidences"].values()))
    dead_domains = sorted({
        bw.FACTS[e["instance"]]["domain"] for e in eps
        if all(float(v) == 0.0 for v in e["claim_confidences"].values())}
        - {bw.FACTS[e["instance"]]["domain"] for e in eps
           if any(float(v) != 0.0 for v in e["claim_confidences"].values())})
    out["score_diagnostic"] = {
        "n_claims": len(pairs),
        "n_episodes": len(eps),
        "pct_episodes_all_zero": round(100.0 * n_allzero / len(eps), 1),
        "domains_entirely_zero": dead_domains,
        "min_score": lo_s, "max_score": hi_s,
        "supported_at_min_pct": round(100.0 * sum(at(lo_s)) / len(at(lo_s)), 1),
        "supported_at_max_pct": round(100.0 * sum(at(hi_s)) / len(at(hi_s)), 1),
        "overall_supported_pct": round(
            100.0 * sum(1 for _s, y in pairs if y) / len(pairs), 1),
    }

    dst = os.path.join(RESULTS, "tableCONF.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=1, sort_keys=True)

    print("%-22s %6s %6s %6s %7s %6s %8s" %
          ("cell", "OFAR%", "Util", "Emit%", "nclust", "cost", "nominal?"))
    print("%-22s %6.1f %6.1f %6.1f %7d %6.2f %8s" %
          ("raw (no threshold)", raw["ofar_a"], raw["utility"],
           raw["emit_rate_avail"], raw["n_clusters"], raw["cost"], "n/a"))
    for k, s in sorted(out["cells"].items()):
        print("%-22s %6.1f %6.1f %6.1f %7d %6.2f %8s" %
              (k, s["ofar_a"], s["utility"], s["emit_rate_avail"],
               s["n_clusters"], s["cost"],
               "MET" if s["meets_nominal"] else "MISSED"))
    print("wrote %s" % dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
