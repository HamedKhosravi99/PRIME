"""Calibration of the v2 confirmatory endpoints (experiments/analysis/build_w2a_tables.py).

The three protocol endpoints must stand in the right logical relation to
each other on the generated benchmark, independently of any model run:

  * F*_B (conjunctive collapse) is STRICTLY STRONGER than F*_A, so every
    B-supported claim is A-supported (this is why "unsupported under both"
    was retired; docs/EXECUTION_PLAN.md, Part R4, Q1).
  * content-false (protocol-independent denotation evaluation) must never
    contradict A-support: a claim the authored protocol supports must not
    be false about the record.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "experiments", "analysis"), os.path.join(_P, "src", "prime_agents")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import build_w2a_tables as bw  # noqa: E402


def test_benchmark_facts_loaded():
    assert len(bw.FACTS) > 500, "confirmatory splits should be loaded"


def test_fb_is_strictly_stronger_than_fa():
    viol = [(f["id"], c) for f in bw.FACTS.values()
            for c in f["supp_a"] if f["supp_b"].get(c) and not f["supp_a"].get(c)]
    assert not viol, "B-supported but not A-supported: %s" % viol[:3]


def test_content_false_never_contradicts_support():
    bad = [(f["id"], c) for f in bw.FACTS.values()
           for c, v in f["content"].items() if v == "false" and f["supp_a"].get(c)]
    assert not bad, "supported yet content-false: %s" % bad[:3]


def test_content_true_implies_supported_on_generated_instances():
    """Generated denotations mirror the routes, so content-true and
    A-support coincide here; a divergence would mean the generator's
    denotation and its licenses disagree."""
    bad = [(f["id"], c) for f in bw.FACTS.values()
           for c, v in f["content"].items() if v == "true" and not f["supp_a"].get(c)]
    assert not bad, "content-true yet unsupported: %s" % bad[:3]


def test_endpoint_distribution_is_non_degenerate():
    c = Counter(v for f in bw.FACTS.values() for v in f["content"].values())
    assert c["true"] > 100 and c["false"] > 100, c
    assert c["unavailable"] > 0, "unavailable branch never exercised"


def test_ceiling_matches_supported_levels():
    for f in bw.FACTS.values():
        want = max([f["levels"][c] for c, ok in f["supp_a"].items() if ok] + [0])
        assert f["ceiling"] == want


# ---------------------------------------------------------------------------
# silence degeneracy (docs/EXECUTION_PLAN.md, Part 6b, FB-02)
# ---------------------------------------------------------------------------

def _rows_for(monkeypatch, tmp_path, emit_when_available):
    """Synthesise one arm that emits on a given fraction of ANSWERABLE
    instances and never emits an unsupported claim, so OFAR is 0.0 by
    construction and the only thing distinguishing the cells is silence."""
    import json
    import os
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    for dom in bw.DOMAINS:
        eps, seen = [], 0
        for f in bw.FACTS.values():
            if f["split"] != "heldout_balanced" or f["domain"] != dom:
                continue
            ok = [(f["levels"][c], c) for c, s in f["supp_a"].items() if s]
            emit = []
            if ok:
                seen += 1
                if seen <= emit_when_available * 1000:
                    emit = [{"claim": max(ok)[1]}]
            eps.append({"instance": f["id"], "arm": "x", "error": None,
                        "emitted": emit, "tool_calls": 1.0})
        d = os.path.join(str(tmp_path), "w2a", "m-%s" % dom, "a__B20")
        os.makedirs(d, exist_ok=True)
        json.dump({"run": {}, "episodes": eps}, open(os.path.join(d, "summary.json"), "w"))
    return bw.cell("m", "a__B20")


def test_silent_arm_scores_zero_ofar_but_is_flagged_degenerate(tmp_path, monkeypatch):
    c = _rows_for(monkeypatch, tmp_path, 0.0)
    assert c["ofar_a"] == 0.0, "a silent arm does score perfect OFAR"
    assert c["emit_rate_avail"] == 0.0
    assert c["degenerate"] is True, "silence must not read as safety"


def test_answering_arm_with_the_same_ofar_is_not_flagged(tmp_path, monkeypatch):
    c = _rows_for(monkeypatch, tmp_path, 1.0)
    assert c["ofar_a"] == 0.0            # identical column to the silent arm
    assert c["emit_rate_avail"] == 100.0
    assert c["degenerate"] is False


def test_emit_rate_is_measured_against_answerable_not_all_instances():
    """84 of 480 balanced instances admit no supported claim; abstaining on
    those is correct, so they must not count against an arm's emit rate."""
    n_avail = sum(1 for f in bw.FACTS.values()
                  if f["split"] == "heldout_balanced" and f["ceiling"] > 0)
    assert n_avail == 396 and n_avail < 480


def test_degenerate_threshold_is_registered_and_not_a_boundary_fit():
    assert bw.DEGENERATE_EMIT_RATE == 0.25


def test_equivalence_margin_comes_from_the_frozen_primary_panel(tmp_path, monkeypatch):
    """docs/ANALYSIS_PLAN.md, F3: ONE margin for the study, from the PRIMARY
    panel's dev split. The builder defaulted to the F2 floor of 5.0, so once
    the pilot landed the table would have tested equivalence against 5.0 while
    the prose quoted 6.16."""
    import json
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    assert bw.frozen_margin() == 5.0, "floor applies only before the pilot"
    json.dump({"MDE_delta": 6.16},
              open(str(tmp_path / "w0_pilots_qwen2.5-32b.json"), "w"))
    assert bw.frozen_margin() == 6.16


def test_margin_never_comes_from_the_secondary_panel(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    json.dump({"MDE_delta": 7.13},
              open(str(tmp_path / "w0_pilots_qwen2.5-7b.json"), "w"))
    assert bw.frozen_margin() == 5.0, \
        "the 7B panel must not be able to set the study's margin"


def test_corrupt_pilot_output_falls_back_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    open(str(tmp_path / "w0_pilots_qwen2.5-32b.json"), "w").write("not json")
    assert bw.frozen_margin() == 5.0


def test_errored_episodes_are_dropped_but_counted(tmp_path, monkeypatch):
    """A model-call timeout is not a decision by the agent, so it must not
    be scored as an abstention -- but n alone cannot distinguish 118 scored
    from 120 attempted, and silent attrition is how a confirmatory table
    starts lying about its denominator."""
    import json
    import os
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    for dom in bw.DOMAINS:
        eps = []
        for f in bw.FACTS.values():
            if f["split"] != "heldout_balanced" or f["domain"] != dom:
                continue
            eps.append({"instance": f["id"], "arm": "x", "emitted": [],
                        "tool_calls": 1.0,
                        "error": "APITimeoutError" if len(eps) < 2 else None})
        d = os.path.join(str(tmp_path), "w2a", "m-%s" % dom, "a__B20")
        os.makedirs(d, exist_ok=True)
        json.dump({"run": {}, "episodes": eps}, open(os.path.join(d, "summary.json"), "w"))
    c = bw.cell("m", "a__B20")
    assert c["n_errored"] == 2 * len(bw.DOMAINS)
    assert c["n_attempted"] == c["n_instances"] + c["n_errored"] == 480
    # the dropped episodes must not inflate abstention
    assert c["abstain"] == 100.0


def test_pooled_emission_cannot_hide_a_degenerate_domain(tmp_path, monkeypatch):
    """An arm answering 94% in one domain and 10% in another pools to ~52%,
    clearing the FB-02 threshold while half its data is silence. The pooled
    figure alone would report that arm as healthy."""
    import json
    import os
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    loud, quiet = bw.DOMAINS[0], bw.DOMAINS[1]
    for dom in bw.DOMAINS:
        eps, seen = [], 0
        for f in bw.FACTS.values():
            if f["split"] != "heldout_balanced" or f["domain"] != dom:
                continue
            ok = [(f["levels"][c], c) for c, s in f["supp_a"].items() if s]
            emit = []
            if ok and f["ceiling"] > 0:
                seen += 1
                answer = True if dom == loud else (seen % 10 == 0)
                if answer:
                    emit = [{"claim": max(ok)[1]}]
            eps.append({"instance": f["id"], "arm": "x", "error": None,
                        "emitted": emit, "tool_calls": 1.0})
        d = os.path.join(str(tmp_path), "w2a", "m-%s" % dom, "a__B20")
        os.makedirs(d, exist_ok=True)
        json.dump({"run": {}, "episodes": eps}, open(os.path.join(d, "summary.json"), "w"))
    c = bw.cell("m", "a__B20")
    assert quiet in c["degenerate_domains"], c["emit_rate_by_domain"]
    assert c["emit_rate_by_domain"][loud] == 100.0
    assert c["degenerate"] is False, "pooled rate clears the threshold"
    assert c["emit_rate_by_domain"][quiet] < 25.0
