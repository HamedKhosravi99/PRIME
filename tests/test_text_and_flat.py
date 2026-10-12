"""Text-evidence atoms (the model as resolver) and the flat evidence-gate arm."""
from __future__ import annotations

import glob
import json
import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(_TESTS)
for _p in (_PILOT, os.path.join(_PILOT, "src"), os.path.join(_PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

TEXT = sorted(glob.glob(os.path.join(_PILOT, "data", "obligationbench", "clinical", "heldout_text", "*.json")))
BAL = sorted(glob.glob(os.path.join(_PILOT, "data", "obligationbench", "security", "heldout_balanced", "*.json")))


def _run(arm, path, tmp_path, script):
    inst = load_instance(path)
    with TraceLogger(os.path.join(str(tmp_path), "%s.jsonl" % arm)) as trace:
        res = arms_mod.run_arm(arm, inst, llm_mod.ScriptedLLM(list(script)), trace, max_tool_calls=20)
    rows = [json.loads(l) for l in open(os.path.join(str(tmp_path), "%s.jsonl" % arm)) if l.strip()]
    return inst, res, rows


@pytest.mark.skipif(not TEXT, reason="text split not present")
def test_text_atom_attested_by_the_model_and_masked(tmp_path):
    path = TEXT[0]
    raw = json.load(open(path))
    gold = {a["id"]: a["gold_value"] for a in raw["atoms"]}
    inst = load_instance(path)
    e1 = next(a for a in inst.atoms if a.id == "E1")
    e2 = next(a for a in inst.atoms if a.id == "E2")
    src = dict(e2.hint["source"])
    text = arms_mod.ToolExecutor(inst).execute("read_text", src)["text"]
    quote = " ".join(text.split())[20:90]
    # the model reads, attests E2 with the gold value, then states C2
    script = [{"name": "inspect_record", "arguments": dict(e1.resolver.params)},
              {"name": "read_text", "arguments": src},
              {"name": "attest_atom", "arguments": {"atom_id": "E2", "value": gold["E2"], "quote": quote}},
              {"name": "emit_claim", "arguments": {"claim_id": "C2"}}]
    inst, res, rows = _run("prime_guided", path, tmp_path, script)
    atts = [(r["attestation"]["atom"], r["attestation"]["value"]) for r in rows if r.get("type") == "attestation"]
    assert ("E2", gold["E2"]) in atts
    if gold["E2"] == "T":
        assert [e["claim"] for e in res.emitted] == ["C2"]
    else:
        assert all(e["claim"] != "C2" for e in res.emitted)
    # a wrong attestation is recorded as given (resolver error), never silently corrected
    wrong = "F" if gold["E2"] == "T" else "T"
    script2 = script[:2] + [{"name": "attest_atom", "arguments": {"atom_id": "E2", "value": wrong, "quote": quote}}]
    inst, res2, rows2 = _run("prime_guided", path, tmp_path, script2)
    assert ("E2", wrong) in [(r["attestation"]["atom"], r["attestation"]["value"]) for r in rows2 if r.get("type") == "attestation"]
    # completion never attests a text atom
    inst, res3, rows3 = _run("prime_complete", path, tmp_path, [])
    assert not any(r.get("type") == "discharge" and r.get("tool") == "attest_atom" for r in rows3)


@pytest.mark.skipif(not BAL, reason="benchmark instances not present")
def test_flat_gate_states_only_the_top_conclusion(tmp_path):
    for path in BAL[:6]:
        inst, res, rows = _run("prime_flat", path, tmp_path, [])
        assert all(e["claim"] == "C3" for e in res.emitted)
        raw = json.load(open(path)); gv = {a["id"]: a["gold_value"] for a in raw["atoms"]}
        supp3 = any(all(gv.get(x) == "T" for x in r) for r in raw["gold_licenses"]["C3"])
        if res.emitted:
            assert supp3
