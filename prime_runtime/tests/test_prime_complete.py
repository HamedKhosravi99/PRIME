"""The prime_complete arm: PRIME's guidance plus scripted completion.

When the model stops short of the reachable ceiling, the runtime discharges
the open obligations itself (every one names a registered resolver call),
under the unchanged mask. Atoms whose locator is withheld (spec_hint) are
never discharged: finding their field needs the model."""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(os.path.dirname(_TESTS))
for _p in (_PILOT, os.path.join(_PILOT, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

BAL = sorted(glob.glob(os.path.join(_PILOT, "bench", "instances_v2", "security", "heldout_balanced", "*.json")))
LOC = sorted(glob.glob(os.path.join(_PILOT, "bench", "instances_v2", "security", "heldout_locator", "*.json")))


def _supported(path):
    raw = json.load(open(path, encoding="utf-8"))
    gv = {a["id"]: a["gold_value"] for a in raw["atoms"]}
    levels = {c["id"]: c["level"] for c in raw["claims"]}
    supp = {c: any(all(gv.get(x) == "T" for x in r) for r in routes)
            for c, routes in raw["gold_licenses"].items()}
    return supp, levels


def _run(arm, path, tmp_path, script=(), budget=20):
    instance = load_instance(path)
    with TraceLogger(os.path.join(str(tmp_path), "%s.jsonl" % arm)) as trace:
        result = arms_mod.run_arm(arm, instance, llm_mod.ScriptedLLM(list(script)), trace,
                                  max_tool_calls=budget)
    with open(os.path.join(str(tmp_path), "%s.jsonl" % arm), encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return result, rows


@pytest.mark.skipif(not BAL, reason="benchmark instances not present")
def test_completion_reaches_the_ceiling_when_the_model_does_nothing(tmp_path):
    # a task whose strongest conclusion is supported, so the ceiling is C3
    path = next(p for p in BAL if _supported(p)[0].get("C3"))
    supp, levels = _supported(path)
    guided, _ = _run("prime_guided", path, tmp_path)
    complete, rows = _run("prime_complete", path, tmp_path)
    assert guided.discharged_calls == 0
    assert complete.discharged_calls > 0
    assert any(r.get("type") == "discharge" for r in rows)
    assert [e["claim"] for e in complete.emitted] == ["C3"]
    # the model never called a tool: every call in the trace is the runtime's
    acts = [r for r in rows if r.get("type") == "action"]
    assert len(acts) == complete.discharged_calls
    assert complete.tool_calls == complete.discharged_calls <= 20


@pytest.mark.skipif(not BAL, reason="benchmark instances not present")
def test_completion_never_states_an_unsupported_conclusion(tmp_path):
    for path in BAL[:12]:
        supp, levels = _supported(path)
        res, _ = _run("prime_complete", path, tmp_path)
        for e in res.emitted:
            assert supp.get(e["claim"], False), (path, e)
        ceiling = max([levels[c] for c, ok in supp.items() if ok] + [0])
        got = max([levels[e["claim"]] for e in res.emitted] + [0])
        assert got == ceiling, (path, got, ceiling)


@pytest.mark.skipif(not BAL, reason="benchmark instances not present")
def test_completion_respects_the_budget(tmp_path):
    path = next(p for p in BAL if _supported(p)[0].get("C3"))
    res, _ = _run("prime_complete", path, tmp_path, budget=1)
    assert res.tool_calls <= 1 and res.discharged_calls <= 1


@pytest.mark.skipif(not LOC, reason="locator split not present")
def test_hinted_atoms_are_enumerated_by_completion(tmp_path):
    path = LOC[0]
    inst = load_instance(path)
    assert all(a.hint for a in inst.atoms if a.resolver is not None)
    res, rows = _run("prime_complete", path, tmp_path)
    # the model did nothing; the runtime enumerated candidates and reached the ceiling
    assert res.discharged_calls > 0 and res.tool_calls == res.discharged_calls
    supp, levels = _supported(path)
    ceiling = max([levels[c] for c, ok in supp.items() if ok] + [0])
    assert max([levels[e["claim"]] for e in res.emitted] + [0]) == ceiling
    d = [r for r in rows if r.get("type") == "discharge"]
    assert d and all(r["params"]["field_path"] in next(a for a in inst.atoms if a.id == r["atom"]).hint["candidates"] for r in d)
    # each candidate of an atom is tried at most once
    seen = collections.Counter((r["atom"], r["params"]["field_path"]) for r in d)
    assert max(seen.values()) == 1
    # the specification view shows the candidates and the description, not a single field
    spec = arms_mod._spec_vocab_block(inst)
    a = next(a for a in inst.atoms if a.resolver is not None)
    assert "field_path=one of" in spec and a.hint["what"] in spec
    for c in a.hint["candidates"]:
        assert c in spec
    assert a.resolver.params["field_path"] in a.hint["candidates"]


@pytest.mark.skipif(not LOC, reason="locator split not present")
def test_obligations_carry_candidate_bookkeeping():
    from prime_runtime import guidance as prime_guidance, status
    from prime_runtime.ledger import Ledger
    inst = load_instance(LOC[0])
    program = inst.gold_program()
    resolvers = status.resolver_table(inst)
    levels = {c.id: c.level for c in inst.claims}
    ledger = Ledger(inst.atom_ids, inst.phi_patterns)
    g = prime_guidance.reachable_ceiling(program, ledger, resolvers, levels, 20)
    assert g["blocking_atoms"]
    ex = arms_mod.ToolExecutor(inst)
    text0 = arms_mod._obligation_text(inst, g, ex)
    assert "must read" in text0 and "candidate fields not yet read" in text0
    assert "read without establishing it" not in text0
    # read a decoy candidate of the first open atom: it must show up as tried,
    # the atom must stay open, and the true field must remain untried
    from prime_runtime.schema import atom_of
    a = next(x for x in inst.atoms if x.id == atom_of(g["blocking_atoms"][0]))
    true = a.resolver.params["field_path"]
    decoy = next(c for c in a.hint["candidates"] if c != true)
    ex.execute(a.resolver.tool, dict(a.resolver.params, field_path=decoy))
    text1 = arms_mod._obligation_text(inst, g, ex)
    assert ("read without establishing it: " in text1) and (decoy in text1.split("read without establishing it: ")[1])
    assert true in text1.split("candidate fields not yet read: ")[1].split("]")[0]
    # the decoy read attests nothing
    from prime_runtime import goldledger
    obs = ex.execute(a.resolver.tool, dict(a.resolver.params, field_path=decoy))
    assert goldledger.delta(inst, ledger, a.resolver.tool, dict(a.resolver.params, field_path=decoy), obs, 1) == []


@pytest.mark.skipif(not BAL, reason="benchmark instances not present")
def test_forced_selection_stays_masked_when_nothing_is_authorized(tmp_path):
    # a task with no supported conclusion: completion cannot authorize anything,
    # and the end-of-episode forced selection must not state a claim unmasked
    path = next(p for p in BAL if not any(_supported(p)[0].values()))
    script = [{"name": "emit_claim", "arguments": {"claim_id": "C1"}}]
    res, rows = _run("prime_complete", path, tmp_path, script=script)
    assert res.emitted == []
    assert not any((r.get("claim_object") or {}).get("unmasked") for r in rows if r.get("type") == "emission")
