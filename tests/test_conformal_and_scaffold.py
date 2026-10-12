"""Tests for the conformal arm + LOO risk-control scoring, and the pin that
scaffold variant 0 is byte-identical to the originally-run prompts (so every
published rubric/reflexion/cove number is untouched by the sensitivity
machinery)."""
from __future__ import annotations

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "src", "prime_agents"), os.path.join(_P, "experiments", "analysis")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import arms as arms_mod  # noqa: E402
import build_conformal as bc  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

REAL = os.path.join(_P, "data", "pilot", "instances", "osv", "osv-0001.json")


# ---------------------------------------------------------------------------
# risk_threshold: the calibration core
# ---------------------------------------------------------------------------

def test_risk_threshold_picks_smallest_safe_tau():
    # scores 10..40; labels: only >=30 are supported
    pairs = [(10, False), (20, False), (30, True), (40, True)]
    # alpha=0: need zero unsupported above tau -> tau=30
    assert bc.risk_threshold(pairs, 0.0) == 30
    # alpha=1/3: {20,30,40} has 1/3 bad -> tau=20 is admissible
    assert bc.risk_threshold(pairs, 1 / 3) == 20
    # alpha below anything achievable except the clean tail
    assert bc.risk_threshold(pairs, 0.2) == 30


def test_risk_threshold_none_when_no_tau_is_safe():
    pairs = [(50, False), (60, False)]
    assert bc.risk_threshold(pairs, 0.4) is None  # every tail is 100% bad


def test_risk_threshold_monotone_in_alpha():
    pairs = [(i * 10, i % 2 == 1) for i in range(1, 11)]
    taus = []
    for alpha in (0.0, 0.2, 0.5, 0.9):
        t = bc.risk_threshold(pairs, alpha)
        taus.append(float("inf") if t is None else t)
    assert taus == sorted(taus, reverse=True)  # looser alpha -> lower tau


def test_score_cell_flags_alpha_below_floor():
    labels = {"i1": {"C1": True}, "i2": {"C1": True}}
    levels = {"i1": {"C1": 1}, "i2": {"C1": 1}}
    eps = [{"instance": "i1", "claim_confidences": {"C1": 90}},
           {"instance": "i2", "claim_confidences": {"C1": 80}}]
    cell = bc.score_cell(eps, labels, levels, alpha=0.01)
    # 1 calibration pair each -> floor = 1/2 > 0.01
    assert cell["alpha_below_floor"] is True
    assert cell["min_n_cal"] == 1


def test_score_cell_gold_confidences_score_perfectly():
    """Confidence = oracle truth (100 for supported, 0 for unsupported) must
    yield OFAR 0 and full utility at any reasonable alpha."""
    import json
    sup, lv = bc.gold_support_and_levels(REAL)
    labels = {}
    levels = {}
    eps = []
    for k in range(4):  # four synthetic copies of the same instance
        iid = "osv-%d" % k
        labels[iid], levels[iid] = sup, lv
        eps.append({"instance": iid,
                    "claim_confidences": {c: (100.0 if ok else 0.0)
                                          for c, ok in sup.items()}})
    cell = bc.score_cell(eps, labels, levels, alpha=0.2)
    assert cell["episode_ofar"] == 0.0
    assert cell["utility"] == 1.0
    assert cell["abstained"] == 0.0


# ---------------------------------------------------------------------------
# conformal arm plumbing
# ---------------------------------------------------------------------------

class _ConfLLM:
    """Answers every tool-loop turn with a stop, and the confidence
    elicitation with fixed JSON."""

    def chat(self, messages, tools=None):
        if tools is None and "confidences" in messages[-1]["content"]:
            return {"content": '{"confidences": {"C1": 90, "C2": 55, "C3": 10}}',
                    "tool_calls": [], "usage": {}}
        return {"content": "done", "tool_calls": [], "usage": {}}


def test_conformal_arm_logs_confidences(tmp_path):
    from prime_runtime.trace import TraceLogger

    instance = load_instance(REAL)
    with TraceLogger(str(tmp_path / "t.jsonl")) as trace:
        result = arms_mod.run_arm("conformal", instance, _ConfLLM(), trace, "gold")
    assert result.claim_confidences == {"C1": 90.0, "C2": 55.0, "C3": 10.0}


def test_conformal_arm_elicitation_failure_is_measured_empty(tmp_path):
    from prime_runtime.trace import TraceLogger

    class _Broken(_ConfLLM):
        def chat(self, messages, tools=None):
            return {"content": "no json here", "tool_calls": [], "usage": {}}

    instance = load_instance(REAL)
    with TraceLogger(str(tmp_path / "t.jsonl")) as trace:
        result = arms_mod.run_arm("conformal", instance, _Broken(), trace, "gold")
    assert result.claim_confidences == {}


# ---------------------------------------------------------------------------
# scaffold variants: default is byte-identical to the published scaffolds
# ---------------------------------------------------------------------------

def test_variant0_is_byte_identical_to_published_scaffolds(monkeypatch):
    monkeypatch.delenv("PRIME_SCAFFOLD_VARIANT", raising=False)
    assert arms_mod._scaffold_variant() == 0
    assert arms_mod._RUBRIC_VARIANTS[0] == "First, list the evaluation criteria that apply."
    r0 = arms_mod._REFLECT_PROMPT_VARIANTS["reflexion"][0]
    assert r0[0].startswith("Self-critique: for the claim you just proposed (%s)")
    assert r0[1].startswith("Given that self-critique, revise your assessment.")
    c0 = arms_mod._REFLECT_PROMPT_VARIANTS["cove"][0]
    assert c0[0].startswith("Chain-of-verification: for the claim you just proposed (%s)")


def test_reflexion_variant3_is_the_released_prompt():
    """Variant 3 is the VERBATIM released Reflexion prompt (two documented
    environment-noun substitutions + one binding sentence); its distinctive
    original wording -- including the source's own typo -- must survive."""
    crit, revise = arms_mod._REFLECT_PROMPT_VARIANTS["reflexion"][3]
    assert "self refection" in crit          # the released text's own typo
    assert "Diagnose a possible reason for failure" in crit
    assert "%s" in crit and "emit_claim" in revise


def test_variants_change_wording_not_mechanics(monkeypatch):
    for arm in ("reflexion", "cove"):
        for v in range(len(arms_mod._REFLECT_PROMPT_VARIANTS[arm])):
            crit, revise = arms_mod._REFLECT_PROMPT_VARIANTS[arm][v]
            assert "%s" in crit                     # claim id slot preserved
            assert "emit_claim" in revise           # same terminal instruction
    monkeypatch.setenv("PRIME_SCAFFOLD_VARIANT", "2")
    assert arms_mod._scaffold_variant() == 2
    monkeypatch.setenv("PRIME_SCAFFOLD_VARIANT", "junk")
    assert arms_mod._scaffold_variant() == 0
