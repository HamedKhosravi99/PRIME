"""External-system arms (src/prime_agents/extsys.py): budget accounting, gating, and
the reporting rule of each instantiation, with a scripted model."""
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
import extsys  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

BAL = sorted(glob.glob(os.path.join(_PILOT, "data", "obligationbench", "*", "heldout_balanced", "*.json")))
pytestmark = pytest.mark.skipif(not BAL, reason="benchmark instances not present")

AGENTIC = ("ext_agentspec", "ext_agentspec_examine", "ext_researchloop", "ext_prooforstop")


def _gold(path):
    raw = json.load(open(path))
    gv = {a["id"]: a["gold_value"] for a in raw["atoms"]}
    lv = {c["id"]: c["level"] for c in raw["claims"]}
    supp = {c: any(all(gv.get(x) == "T" for x in r) for r in rs) for c, rs in raw["gold_licenses"].items()}
    return raw, gv, lv, supp


def _read(inst, aid):
    a = next(a for a in inst.atoms if a.id == aid)
    return {"name": a.resolver.tool, "arguments": dict(a.resolver.params)}


def _run(arm, path, tmp_path, script, budget=20):
    inst = load_instance(path)
    tp = os.path.join(str(tmp_path), "%s-%s.jsonl" % (arm, inst.id))
    with TraceLogger(tp) as trace:
        res = arms_mod.run_arm(arm, inst, llm_mod.ScriptedLLM(list(script)), trace, max_tool_calls=budget)
    rows = [json.loads(l) for l in open(tp) if l.strip()]
    return inst, res, rows


def _all_reads(inst):
    return [_read(inst, a.id) for a in inst.atoms if a.resolver is not None]


@pytest.mark.parametrize("arm", AGENTIC)
def test_unsupported_conclusion_never_stated(arm, tmp_path):
    """Claiming the strongest conclusion with no evidence is never reported,
    whatever the mechanism does with the attempt."""
    for path in BAL[::40]:
        raw, gv, lv, supp = _gold(path)
        top = max(lv, key=lambda c: lv[c])
        inst, res, rows = _run(arm, path, tmp_path, [{"name": "emit_claim", "arguments": {"claim_id": top}}])
        assert all(supp[e["claim"]] for e in res.emitted)
        assert res.tool_calls <= 20


@pytest.mark.parametrize("arm", AGENTIC)
def test_reads_then_strongest_supported(arm, tmp_path):
    """After reading every atom, a gated arm reports a supported conclusion
    and never more than the strongest one the record supports."""
    for path in BAL[::30]:
        raw, gv, lv, supp = _gold(path)
        inst = load_instance(path)
        ceil = max([lv[c] for c, ok in supp.items() if ok] + [0])
        best = [c for c in lv if lv[c] == ceil]
        script = _all_reads(inst) + ([{"name": "emit_claim", "arguments": {"claim_id": best[0]}}] if best else [])
        inst, res, rows = _run(arm, path, tmp_path, script)
        assert all(supp[e["claim"]] for e in res.emitted)
        assert res.tool_calls <= 20
        if ceil and arm in ("ext_researchloop", "ext_prooforstop"):
            # the projection / final gate reports the strongest admitted stage
            assert [lv[e["claim"]] for e in res.emitted] == [ceil]


def test_budget_counts_like_run_arm(tmp_path):
    path = BAL[0]
    inst = load_instance(path)
    reads = _all_reads(inst) * 3  # repeats hit the call-class cache
    for arm in AGENTIC:
        inst, res, rows = _run(arm, path, tmp_path, reads, budget=4)
        assert res.tool_calls <= 4
        acts = [r for r in rows if r.get("type") == "action"]
        assert len(acts) >= res.tool_calls


def test_agentspec_stop_ends_episode(tmp_path):
    for path in BAL[::25]:
        raw, gv, lv, supp = _gold(path)
        top = max(lv, key=lambda c: lv[c])
        inst, res, rows = _run("ext_agentspec", path, tmp_path,
                               [{"name": "emit_claim", "arguments": {"claim_id": top}}] + _all_reads(load_instance(path)))
        assert res.emitted == []
        assert res.ext.get("stopped_by_rule") == 1
        assert not any(r.get("type") == "observation" for r in rows)


def test_agentspec_examine_replans_and_continues(tmp_path):
    path = next(p for p in BAL if _gold(p)[3].get("C1"))
    raw, gv, lv, supp = _gold(path)
    inst = load_instance(path)
    c1_route = next(r for r in raw["gold_licenses"]["C1"] if all(gv.get(x) == "T" for x in r))
    script = ([{"name": "emit_claim", "arguments": {"claim_id": "C3"}}]
              + [_read(inst, a) for a in c1_route]
              + [{"name": "emit_claim", "arguments": {"claim_id": "C1"}}])
    inst, res, rows = _run("ext_agentspec_examine", path, tmp_path, script)
    assert res.ext.get("replans", 0) >= 1
    assert [e["claim"] for e in res.emitted] == ["C1"]


def test_prooforstop_blocks_then_reports_highest_admitted(tmp_path):
    path = next(p for p in BAL if _gold(p)[3].get("C1") and not _gold(p)[3].get("C3"))
    raw, gv, lv, supp = _gold(path)
    inst = load_instance(path)
    c1_route = next(r for r in raw["gold_licenses"]["C1"] if all(gv.get(x) == "T" for x in r))
    script = ([_read(inst, a) for a in c1_route]
              + [{"name": "emit_claim", "arguments": {"claim_id": "C3"}}] * 4)
    inst, res, rows = _run("ext_prooforstop", path, tmp_path, script)
    assert res.ext.get("blocks", 0) >= 1
    assert res.ext.get("stop_reason") in ("no_progress", "repair_cap")
    assert res.emitted and all(supp[e["claim"]] for e in res.emitted)


def test_researchloop_projection_ignores_overclaim(tmp_path):
    path = next(p for p in BAL if _gold(p)[3].get("C1") and not _gold(p)[3].get("C3"))
    raw, gv, lv, supp = _gold(path)
    inst = load_instance(path)
    c1_route = next(r for r in raw["gold_licenses"]["C1"] if all(gv.get(x) == "T" for x in r))
    script = [_read(inst, a) for a in c1_route] + [{"name": "emit_claim", "arguments": {"claim_id": "C3"}}]
    inst, res, rows = _run("ext_researchloop", path, tmp_path, script)
    assert res.ext.get("stated") == "C3" and res.ext.get("stated_not_admitted") == 1
    assert [e["claim"] for e in res.emitted] and all(supp[e["claim"]] for e in res.emitted)


def test_other_arms_unchanged_by_registration():
    for arm in ("prime_guided", "masked_reactive_v", "react_program_attested"):
        assert arm in arms_mod.ARMS
    assert set(extsys.EXT_ARMS) <= set(arms_mod.ARMS)
