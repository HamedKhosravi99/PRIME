"""Anchors for the standard-construction (S1) end-to-end pipeline
(experiments/analysis/openvocab_endtoend.py) -- the experiment that runs
open-vocabulary standard construction through the UNCHANGED masked runtime
and scores it against the authored standard by evidence signatures only.

Two anchors, no LLM anywhere (scripted episodes):
  - GOLD anchor: a proposal derived from the authored standard itself must
    compile, run, signature-map 1:1, and score utility 1.0 at OFAR 0 -- if it
    does not, the scorer is broken, not the model.
  - JUNK anchor: a proposal whose single atom probes a nonexistent field
    compiles (schema-valid) but its atom is F on the real record, so the
    unchanged mask must refuse every emission: 0 emissions, 0 false
    authorization. Soundness composes with self-proposed standards.
"""
from __future__ import annotations

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "src", "prime_agents"), os.path.join(_P, "experiments", "analysis")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import llm as llm_mod  # noqa: E402
import openvocab_endtoend as ov  # noqa: E402
from prime_runtime.predicates import parse_predicate_rule  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from run_episode import run_episode  # noqa: E402

REAL = os.path.join(_P, "data", "pilot", "instances", "osv", "osv-0001.json")


def _gold_proposal(inst):
    atoms = []
    for a in inst.atoms:
        ptype, expected = parse_predicate_rule(a.resolver.attest)
        atoms.append({"id": a.id,
                      "field_path": dict(a.resolver.params)["field_path"],
                      "predicate": ptype, "expected": expected})
    return {
        "claims": [{"id": c.id, "text": c.text, "level": c.level} for c in inst.claims],
        "order": [list(p) for p in inst.order],
        "atoms": atoms,
        "licenses": {cid: [list(r) for r in rts]
                     for cid, rts in inst.gold_licenses.items()},
    }


def _run(proposal, tmp_path):
    real = load_instance(REAL)
    syn = str(tmp_path / "syn.json")
    ok, err = ov.compile_synthetic(real, proposal, syn)
    assert ok, err
    syn_inst = load_instance(syn)
    fake = llm_mod.ScriptedLLM(llm_mod.build_dry_run_script(syn_inst), program_json="")
    ep = run_episode(syn, "prime", "gold", fake, str(tmp_path))
    return ov.score_episode(real, proposal, ep)


def test_gold_anchor_scores_perfect(tmp_path):
    real = load_instance(REAL)
    sc = _run(_gold_proposal(real), tmp_path)
    assert sc["episode_ofar_best"] == 0 and sc["episode_ofar_worst"] == 0
    assert sc["n_unmatched_emissions"] == 0
    assert sc["utility"] == 1.0
    assert sc["supported_level"] == sc["oracle_ceiling_level"] == 3


def test_junk_anchor_mask_refuses_everything(tmp_path):
    junk = {
        "claims": [{"id": "C1", "text": "x", "level": 1}],
        "order": [],
        "atoms": [{"id": "E1", "field_path": "no.such.field",
                   "predicate": "exists", "expected": None}],
        "licenses": {"C1": [["E1"]]},
    }
    sc = _run(junk, tmp_path)
    # the invented atom is F on the real record: the unchanged mask must
    # refuse every emission of a claim licensed only by it
    assert sc["n_emitted"] == 0
    assert sc["episode_ofar_best"] == 0 and sc["episode_ofar_worst"] == 0
    assert sc["utility"] == 0.0


def test_unknown_predicate_is_a_measured_gate(tmp_path):
    real = load_instance(REAL)
    bad = {"claims": [{"id": "C1", "text": "x", "level": 1}], "order": [],
           "atoms": [{"id": "E1", "field_path": "summary",
                      "predicate": "not_a_predicate", "expected": None}],
           "licenses": {"C1": [["E1"]]}}
    ok, err = ov.compile_synthetic(real, bad, str(tmp_path / "s.json"))
    assert not ok and "predicate" in err
