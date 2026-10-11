#!/usr/bin/env python3
"""Local tests of the wrapper (no GPU): the verifying executor's acceptance
rule, its blindness (what the verifier receives), and the end-to-end
monkeypatch path through runner.run_episode with a scripted agent.

Usage: python3 test_wrapper.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "runner"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime import goldledger  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from run_episode import run_episode  # noqa: E402

import blind_verifier  # noqa: E402
import verified_executor as VE  # noqa: E402


class FakeVerifier:
    """Answers with a fixed verdict and records exactly what it was shown."""
    def __init__(self, answer):
        self.answer = answer
        self.seen = []

    def chat(self, messages, tools=None):
        assert tools is None
        self.seen.append([dict(m) for m in messages])
        return {"content": "Because the text says so.\nANSWER: %s" % self.answer,
                "usage": {"prompt_tokens": 100, "completion_tokens": 12}}


def main() -> int:
    blind_verifier.check_frozen()
    path = sorted(glob.glob(os.path.join(PILOT, "bench", "instances_v2", "security", "heldout_text", "*.json")))[0]
    inst = load_instance(path)
    atom = next(a for a in inst.atoms if a.resolver.tool == "attest_atom")
    src = atom.hint["source"]
    text = inst.records[src["record_id"]][src["field_path"]]
    quote = " ".join(text.split())[:40]

    # 1. agreement -> accepted and recorded; disagreement -> rejected, nothing recorded
    for answer, expect in (("T", True), ("F", False), ("UNCLEAR", False)):
        VE.VERIFIER_LLM = FakeVerifier(answer)
        VE.REGISTRY.clear()
        ex = VE.VerifiedToolExecutor(inst)
        ex.execute("read_text", dict(src))
        obs = ex.execute("attest_atom", {"atom_id": atom.id, "value": "T", "quote": quote})
        assert obs["accepted"] is expect, (answer, obs)
        if not expect:
            assert obs["reason"] == VE.REJECT_REASON and "verdict" not in json.dumps(obs)
        ledger = Ledger(inst.atom_ids, inst.phi_patterns)
        applied = goldledger.delta(inst, ledger, "attest_atom", {"atom_id": atom.id, "value": "T", "quote": quote}, obs, 1)
        assert (len(applied) == 1) is expect and ledger.state(atom.id) == ("T" if expect else "U")
        # blindness: the verifier saw only system + one user message with condition and document
        seen = VE.VERIFIER_LLM.seen
        assert len(seen) == 1 and [m["role"] for m in seen[0]] == ["system", "user"]
        user = seen[0][1]["content"]
        assert atom.hint["what"] in user and text in user
        assert quote not in user.replace(text, "") and "value" not in user.lower().replace("value", "", 0)  # no attested value leaked
        assert "T\"" not in user and "attested" not in user.lower()
        ev = VE.REGISTRY[inst.id]
        assert len(ev) == 1 and ev[0]["invoked"] and ev[0]["agree"] is expect
    print("acceptance rule and blindness: OK")

    # 2. a second attestation of the same atom reuses the verdict (no new call)
    VE.VERIFIER_LLM = FakeVerifier("F"); VE.REGISTRY.clear()
    ex = VE.VerifiedToolExecutor(inst); ex.execute("read_text", dict(src))
    ex.execute("attest_atom", {"atom_id": atom.id, "value": "T", "quote": quote})
    ex.execute("attest_atom", {"atom_id": atom.id, "value": "T", "quote": " ".join(text.split())[10:60]})
    ev = VE.REGISTRY[inst.id]
    assert len(VE.VERIFIER_LLM.seen) == 1 and len(ev) == 2 and ev[1]["invoked"] is False
    # a flipped value that matches the verifier is accepted (agreement by the rule, symmetric)
    obs = ex.execute("attest_atom", {"atom_id": atom.id, "value": "F", "quote": quote})
    assert obs["accepted"] is True
    print("verdict reuse and symmetry: OK")

    # 3. end to end through run_episode with the executor installed: a
    #    scripted agent reads, attests E2=T, emits C2; verifier says F ->
    #    C2 never authorized -> fallback emits C1 (level 1)
    arms_mod.ToolExecutor = VE.VerifiedToolExecutor
    VE.VERIFIER_LLM = FakeVerifier("F"); VE.REGISTRY.clear()
    script = [{"name": "inspect_record", "arguments": {"record_id": src["record_id"], "field_path": "summary"}},
              {"name": "read_text", "arguments": dict(src)},
              {"name": "attest_atom", "arguments": {"atom_id": atom.id, "value": "T", "quote": quote}},
              {"name": "emit_claim", "arguments": {"claim_id": "C2"}}]
    with tempfile.TemporaryDirectory() as td:
        s = run_episode(path, "prime_guided", "gold", llm_mod.ScriptedLLM(script), td, max_tool_calls=20)
        assert not s.get("error"), s.get("error")
        assert [e["claim"] for e in s["emitted"]] == ["C1"], s["emitted"]
        assert s["refused_emissions"] >= 1
        rows = [json.loads(l) for l in open(os.path.join(td, inst.id + ".jsonl"))]
        atts = [r for r in rows if r["type"] == "attestation" and r["attestation"]["tool"] == "attest_atom"]
        assert not atts
        rej = [r for r in rows if r["type"] == "observation" and r["action"] == "attest_atom"]
        assert rej and rej[0]["data"]["reason"] == VE.REJECT_REASON
    # and with an agreeing verifier the same script authorizes C2
    VE.VERIFIER_LLM = FakeVerifier("T"); VE.REGISTRY.clear()
    with tempfile.TemporaryDirectory() as td:
        s = run_episode(path, "prime_guided", "gold", llm_mod.ScriptedLLM(script), td, max_tool_calls=20)
        assert [e["claim"] for e in s["emitted"]] == ["C2"], s["emitted"]
        # inspect, read, attest, emit are charged as in the original loop;
        # the verifier call is not (it is counted in the registry instead)
        assert s["tool_calls"] == 4, s["tool_calls"]
        assert len(VE.VERIFIER_LLM.seen) == 1
    print("end-to-end monkeypatch path: OK")
    print("all wrapper tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
