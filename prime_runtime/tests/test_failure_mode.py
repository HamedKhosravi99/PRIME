"""Failure-mode decomposition (analysis/failure_mode.py).

OFAR says an arm authorized something unsupported. It does not distinguish
over-reaching for a strong claim from asserting a weak one on an instance
where NOTHING is licensed and abstention is the only correct action. Those
are different defects; the ladder's arms exhibit different ones.
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
import failure_mode as fm       # noqa: E402

MODEL = "qwen2.5-fixture"


def write(root, arm, mode):
    """mode: 'dead' emits only where nothing is supported; 'overreach' emits
    the strongest claim regardless; 'clean' emits only supported claims."""
    for dom in bw.DOMAINS:
        eps = []
        for f in bw.FACTS.values():
            if f["split"] != "heldout_balanced" or f["domain"] != dom:
                continue
            ok = [(f["levels"][c], c) for c, s in f["supp_a"].items() if s]
            strongest = max((f["levels"][c], c) for c in f["levels"])[1]
            if mode == "dead":
                em = [{"claim": strongest}] if f["ceiling"] == 0 else []
            elif mode == "overreach":
                em = [{"claim": strongest}]
            else:
                em = [{"claim": max(ok)[1]}] if ok else []
            eps.append({"instance": f["id"], "arm": arm, "error": None,
                        "emitted": em, "tool_calls": 1.0})
        d = os.path.join(root, "w2a", "%s-%s" % (MODEL, dom), "%s__B20" % arm)
        os.makedirs(d, exist_ok=True)
        json.dump({"run": {}, "episodes": eps}, open(os.path.join(d, "summary.json"), "w"))


def test_missing_arm_reads_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    assert fm.profile(MODEL, "react")["status"] == "NOT RUN"


def test_emitting_only_where_nothing_is_supported_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    write(str(tmp_path), "react", "dead")
    p = fm.profile(MODEL, "react")
    assert p["n_dead"] == 84
    assert p["emitted_on_dead"] == 84 and p["dead_emit_rate"] == 1.0


def test_a_clean_arm_never_emits_on_dead_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    write(str(tmp_path), "masked_reactive_v", "clean")
    p = fm.profile(MODEL, "masked_reactive_v")
    assert p["dead_emit_rate"] == 0.0
    assert p["overreach_share"] == 0.0, "clean arm has no unsupported top-level"


def test_overreach_shows_as_unsupported_mass_at_the_top_level(tmp_path, monkeypatch):
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    write(str(tmp_path), "react_spec", "overreach")
    p = fm.profile(MODEL, "react_spec")
    assert p["emissions"] == 480
    assert p["overreach_share"] > 0.3, p
    assert p["by_level"], "level mix must be reported, not just a rate"


def test_clean_and_overreach_can_share_an_emission_count(tmp_path, monkeypatch):
    """The point of the decomposition: two arms can emit equally often and
    fail completely differently."""
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    write(str(tmp_path), "react_spec", "overreach")
    write(str(tmp_path), "prime_guided", "overreach")
    a = fm.profile(MODEL, "react_spec")
    write(str(tmp_path), "prime_guided", "clean")
    b = fm.profile(MODEL, "prime_guided")
    assert a["overreach_share"] > b["overreach_share"]
    assert a["dead_emit_rate"] > b["dead_emit_rate"]


def test_emissions_contradicted_by_the_agents_own_ledger_are_counted(tmp_path, monkeypatch):
    """The sharpest form of the gap: the agent established an atom FALSE
    itself, every route to the claim needs that atom, and it emitted the
    claim anyway. This is ignored information, not missing information, and
    it is the number that says being SHOWN the rule is not enforcement."""
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    import glob
    target = None
    # sorted: glob order differs between filesystems, and the target must not
    # depend on it. The claim must also be one the record does not support,
    # since only an unsupported emission can be contradicted by the ledger.
    for p in sorted(glob.glob(os.path.join(_P, "bench", "instances_v2", "security",
                                           "heldout_balanced", "*.json"))):
        inst = json.load(open(p))
        gold = {a["id"]: a["gold_value"] for a in inst["atoms"]}
        for cid, routes in inst["gold_licenses"].items():
            supported = any(all((gold.get(x[1:]) == "F") if x.startswith("!") else (gold.get(x) == "T")
                                for x in r) for r in routes)
            common = {x for x in set(routes[0]).intersection(*[set(r) for r in routes])
                      if not x.startswith("!")}
            if common and not supported:
                target = (p, inst, cid, sorted(common)[0])
                break
        if target:
            break
    assert target, "need an unsupported claim whose routes share an atom"
    path, inst, cid, blocking = target

    for dom in bw.DOMAINS:
        eps = []
        if dom == "security":
            eps.append({"instance": inst["id"], "arm": "x", "error": None,
                        "emitted": [{"claim": cid}], "tool_calls": 3.0,
                        "instance_path": path,
                        "final_ledger": {blocking: "F"}})
        d = os.path.join(str(tmp_path), "w2a", "%s-%s" % (MODEL, dom), "x__B20")
        os.makedirs(d, exist_ok=True)
        json.dump({"run": {}, "episodes": eps}, open(os.path.join(d, "summary.json"), "w"))

    p = fm.profile(MODEL, "x")
    assert p["emitted_against_own_ledger"] == 1, p
    ex = p["knew_examples"][0]
    assert ex["claim"] == cid and blocking in ex["false_atoms"]


def test_an_unknown_atom_state_does_not_count_as_knowing(tmp_path, monkeypatch):
    """Only an atom the agent established FALSE counts. Unresolved is not
    knowledge, and counting it would overstate the finding."""
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    import glob
    p0 = sorted(glob.glob(os.path.join(_P, "bench", "instances_v2", "security",
                                       "heldout_balanced", "*.json")))[0]
    inst = json.load(open(p0))
    cid = sorted(inst["gold_licenses"])[0]
    for dom in bw.DOMAINS:
        eps = []
        if dom == "security":
            eps.append({"instance": inst["id"], "arm": "x", "error": None,
                        "emitted": [{"claim": cid}], "tool_calls": 3.0,
                        "instance_path": p0,
                        "final_ledger": {a["id"]: "U" for a in inst["atoms"]}})
        d = os.path.join(str(tmp_path), "w2a", "%s-%s" % (MODEL, dom), "x__B20")
        os.makedirs(d, exist_ok=True)
        json.dump({"run": {}, "episodes": eps}, open(os.path.join(d, "summary.json"), "w"))
    assert fm.profile(MODEL, "x")["emitted_against_own_ledger"] == 0


def test_failed_lookups_are_measured_from_the_traces(tmp_path, monkeypatch):
    """Record schemas differ in guessability -- OSV exposes `published`,
    ClinicalTrials.gov nests protocolSection.identificationModule.briefTitle
    -- so an arm without resolver locators can be crippled in one domain and
    fine in another for reasons unrelated to support logic. Crediting the
    vocabulary rung with fixing JUDGEMENT when it fixed ADDRESSING would
    misattribute the ladder's first step."""
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    monkeypatch.setattr(fm, "PILOT", str(tmp_path))
    dom = bw.DOMAINS[0]
    d = os.path.join(str(tmp_path), "results", "w2a",
                     "%s-%s" % (MODEL, dom), "react__B20")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "x.jsonl"), "w") as fh:
        fh.write(json.dumps({"type": "action", "name": "inspect_record"}) + "\n")
        for _ in range(3):
            fh.write(json.dumps({"type": "observation",
                                 "data": {"error": "field not found"}}) + "\n")
        fh.write(json.dumps({"type": "observation",
                             "data": {"value": "ok"}}) + "\n")
    bad, tot = fm._lost_calls(MODEL, "react", dom)
    assert (bad, tot) == (3, 4)


def test_a_successful_lookup_is_not_counted_as_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "PILOT", str(tmp_path))
    dom = bw.DOMAINS[0]
    d = os.path.join(str(tmp_path), "results", "w2a",
                     "%s-%s" % (MODEL, dom), "react_spec__B20")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "y.jsonl"), "w") as fh:
        for _ in range(5):
            fh.write(json.dumps({"type": "observation",
                                 "data": {"value": 1}}) + "\n")
    assert fm._lost_calls(MODEL, "react_spec", dom) == (0, 5)
