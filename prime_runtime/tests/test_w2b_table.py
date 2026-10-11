"""W2b table path (analysis/build_w2b_table.py), validated before the runs.

Same rationale as the T1/T2 fixture: until W2b returns data every cell reads
NOT RUN, so a schema mismatch between what run_batch writes and what the
builder reads would surface only after the GPU hours were spent.
"""
from __future__ import annotations

import json
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(os.path.dirname(_TESTS))
for _x in (_P, os.path.join(_P, "analysis"), os.path.join(_P, "runner")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import build_w2a_tables as bw   # noqa: E402
import build_w2b_table as w2b   # noqa: E402

MODEL = "qwen2.5-fixture"


def _best(f):
    ok = [(f["levels"][c], c) for c, s in f["supp_a"].items() if s]
    return max(ok)[1] if ok else None


def _bad(f):
    b = sorted(c for c, s in f["supp_a"].items() if not s)
    return b[0] if b else None


def write(root, arm, behaviour, valid=True, eligible=True):
    for dom in bw.DOMAINS:
        eps = []
        for f in bw.FACTS.values():
            if f["split"] != "heldout_balanced" or f["domain"] != dom:
                continue
            good, bad = _best(f), _bad(f)
            if behaviour == "clean":
                em = [{"claim": good}] if good else []
            elif behaviour == "dirty" and bad:
                em = [{"claim": bad}]
            else:
                em = []
            eps.append({"instance": f["id"], "arm": arm, "error": None,
                        "emitted": em, "tool_calls": 2.0,
                        "induced_program_valid": valid,
                        "induction_eligibility": {c: eligible for c in f["levels"]}})
        d = os.path.join(root, "w2b", "%s-%s" % (MODEL, dom),
                         "%s__induced__B20" % arm)
        os.makedirs(d, exist_ok=True)
        json.dump({"run": {}, "episodes": eps},
                  open(os.path.join(d, "summary.json"), "w"))


def test_missing_runs_read_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(w2b, "RESULTS", str(tmp_path))
    assert w2b.cell(MODEL, "prime_guided") == {"status": "NOT RUN"}


def test_cell_reports_validity_and_eligibility(tmp_path, monkeypatch):
    monkeypatch.setattr(w2b, "RESULTS", str(tmp_path))
    write(str(tmp_path), "prime_guided_gated", "clean", valid=True, eligible=True)
    c = w2b.cell(MODEL, "prime_guided_gated")
    assert c["n_instances"] == 480 and c["n_clusters"] == 60
    assert c["program_valid"] == 100.0
    assert c["claim_eligibility"] == 100.0
    assert c["ofar_a"] == 0.0


def test_ineligible_claims_are_reported_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(w2b, "RESULTS", str(tmp_path))
    write(str(tmp_path), "prime_guided_gated", "clean", eligible=False)
    c = w2b.cell(MODEL, "prime_guided_gated")
    assert c["claim_eligibility"] == 0.0, \
        "a zero eligibility rate must be visible, not absent"


def test_invalid_programs_are_a_measured_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(w2b, "RESULTS", str(tmp_path))
    write(str(tmp_path), "prime_guided", "silent", valid=False)
    c = w2b.cell(MODEL, "prime_guided")
    assert c["program_valid"] == 0.0
    assert c["ofar_a"] == 0.0 and c["degenerate"] is True, \
        "all-invalid induction abstains, and that must read as degenerate"


def test_amplification_is_signed_so_the_gate_reads_positive(tmp_path, monkeypatch):
    """The paper's deployment rule: ungated guidance on an unverified induced
    program is dangerous. POSITIVE amplification = the gate prevented it."""
    monkeypatch.setattr(w2b, "RESULTS", str(tmp_path))
    write(str(tmp_path), "prime_guided", "dirty")          # ungated: unsupported
    write(str(tmp_path), "prime_guided_gated", "clean")    # gated: supported only
    rows = w2b.paired(MODEL, "prime_guided", "prime_guided_gated", "ofar_a")
    assert rows
    assert sum(r["a"] for r in rows) > sum(r["b"] for r in rows)
    rc = w2b.main(["--models", MODEL])
    assert rc == 0
    T4 = json.load(open(os.path.join(str(tmp_path), "tableT4.json")))
    amp = T4["amplification"][MODEL]
    assert amp["amplification_pp"] > 50.0, amp
    assert amp["n_clusters"] == 60
    assert "_conventions" in T4


def _amp(pp, p, lo, hi):
    return {"_route_coverage": {}, "cells": {
                "qwen2.5-32b|prime_guided_gated": {
                    "status": "ok", "n_instances": 320, "n_clusters": 45,
                    "program_valid": 100.0, "claim_eligibility": 36.2,
                    "ofar_a": 17.3, "content_false": 0.0, "utility": 29.1,
                    "emit_rate_avail": 89.7, "degenerate": False, "cost": 3.0}},
            "amplification": {"qwen2.5-32b": {
                "amplification_pp": pp, "ci": [lo, hi], "p_randomization": p,
                "discordant_clusters": 9, "n_clusters": 15,
                "n_instances": 119}}}


def test_a_gate_that_helps_is_reported_as_a_requirement():
    import fill_v2_tables as fv
    fv.PRIMARY = "qwen2.5-32b"
    out = fv.FILLERS["T4Text"]({"T1": None, "T2": None,
                                "T4": _amp(41.7, 0.0002, 33.0, 50.2)})
    assert "dangerous configuration" in out
    assert "deployment requirement" in out


def test_a_gate_that_backfires_is_reported_as_a_negative_result():
    """With coverage this low the gated arm often has NO eligible target, so
    guidance degrades to the unguided masked policy. Printing 'raises false
    authorization by -9.8 points' would be nonsense, and calling it a win
    would be worse."""
    import fill_v2_tables as fv
    fv.PRIMARY = "qwen2.5-32b"
    out = fv.FILLERS["T4Text"]({"T1": None, "T2": None,
                                "T4": _amp(-9.8, 0.0041, -17.8, -3.4)})
    assert "does not behave as designed" in out
    assert "9.8 points MORE false authorization" in out
    assert "negative result about the gate as specified" in out
    assert "-9.8 points" not in out
    assert "deployment requirement" not in out


def test_an_undetermined_gate_contrast_is_not_called_null():
    import fill_v2_tables as fv
    fv.PRIMARY = "qwen2.5-32b"
    out = fv.FILLERS["T4Text"]({"T1": None, "T2": None,
                                "T4": _amp(1.2, 0.61, -4.0, 6.4)})
    assert "underdetermined rather than null" in out


def test_gate_bound_conditioning_excludes_episodes_the_gate_cannot_bind_on(tmp_path, monkeypatch):
    """On an instance where NO claim is eligible the gated arm has no target
    and behaves as the unguided policy, so those episodes differ for a
    reason that is not gating. The eligibility map comes from the induced
    program, which both arms share from one per-family cache, so it is
    identical across arms and conditioning on it is legitimate."""
    import os
    monkeypatch.setattr(w2b, "RESULTS", str(tmp_path))
    for dom in bw.DOMAINS:
        for arm, mode in (("prime_guided", "clean"),
                          ("prime_guided_gated", "dirty")):
            eps = []
            for i, f in enumerate(bw.FACTS.values()):
                if f["split"] != "heldout_balanced" or f["domain"] != dom:
                    continue
                good, bad = _best(f), _bad(f)
                em = ([{"claim": bad}] if (mode == "dirty" and bad)
                      else ([{"claim": good}] if good else []))
                # alternate: none eligible / some eligible / all eligible
                cl = list(f["levels"])
                el = ({c: False for c in cl} if i % 3 == 0 else
                      ({c: True for c in cl} if i % 3 == 1 else
                       {c: (k > 0) for k, c in enumerate(cl)}))
                eps.append({"instance": f["id"], "arm": arm, "error": None,
                            "emitted": em, "tool_calls": 2.0,
                            "induced_program_valid": True,
                            "induction_eligibility": el})
            d = os.path.join(str(tmp_path), "w2b", "%s-%s" % (MODEL, dom),
                             "%s__induced__B20" % arm)
            os.makedirs(d, exist_ok=True)
            json.dump({"run": {}, "episodes": eps},
                      open(os.path.join(d, "summary.json"), "w"))

    everything = w2b.paired(MODEL, "prime_guided", "prime_guided_gated", "ofar_a")
    bound = w2b.paired(MODEL, "prime_guided", "prime_guided_gated", "ofar_a",
                       gate_bound_only=True)
    assert 0 < len(bound) < len(everything), (len(bound), len(everything))
    # only the mixed-eligibility third survives
    assert abs(len(bound) - len(everything) / 3) <= len(bw.DOMAINS)


def test_all_eligible_is_not_treated_as_the_gate_binding(tmp_path, monkeypatch):
    """If every claim is eligible the gate removes nothing, so those
    episodes are not evidence about gating either."""
    import os
    monkeypatch.setattr(w2b, "RESULTS", str(tmp_path))
    for dom in bw.DOMAINS:
        for arm in ("prime_guided", "prime_guided_gated"):
            eps = []
            for f in bw.FACTS.values():
                if f["split"] != "heldout_balanced" or f["domain"] != dom:
                    continue
                good = _best(f)
                eps.append({"instance": f["id"], "arm": arm, "error": None,
                            "emitted": [{"claim": good}] if good else [],
                            "tool_calls": 1.0, "induced_program_valid": True,
                            "induction_eligibility": {c: True for c in f["levels"]}})
            d = os.path.join(str(tmp_path), "w2b", "%s-%s" % (MODEL, dom),
                             "%s__induced__B20" % arm)
            os.makedirs(d, exist_ok=True)
            json.dump({"run": {}, "episodes": eps},
                      open(os.path.join(d, "summary.json"), "w"))
    assert w2b.paired(MODEL, "prime_guided", "prime_guided_gated", "ofar_a",
                      gate_bound_only=True) == []
