"""A repeated call is served from the call-class cache and consumes no budget.

Each agent loop decides whether a call consumes budget before the call runs,
from that call's own class (``ToolExecutor.charges``). The logged runs decided
it from the cache status of the previously executed call, which charged a
repeat when the call before it was new and let the calls after a repeat go
free; docs/KNOWN_ISSUES.md gives the effect on the logged episodes.
"""
from __future__ import annotations

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(os.path.dirname(_TESTS))
for _x in (_P, os.path.join(_P, "runner")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

OSV = os.path.join(_P, "bench", "instances", "osv", "osv-0001.json")


def _read(rid, field):
    return {"name": "inspect_record", "arguments": {"record_id": rid, "field_path": field}}


def _run(arm, script, budget, tmp_path):
    inst = load_instance(OSV)
    with TraceLogger(str(tmp_path / ("%s.jsonl" % arm))) as trace:
        res = arms_mod.run_arm(arm, inst, llm_mod.ScriptedLLM(list(script)), trace, "gold",
                               max_tool_calls=budget)
    return res, str(tmp_path / ("%s.jsonl" % arm))


def _observed_fields(path):
    import json
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") == "observation":
                out.append((r.get("params") or {}).get("field_path"))
    return out


def test_executor_charges_new_calls_only():
    inst = load_instance(OSV)
    ex = arms_mod.ToolExecutor(inst)
    rid = sorted(inst.records)[0]
    a = {"record_id": rid, "field_path": "summary"}
    assert ex.charges("inspect_record", a)
    ex.execute("inspect_record", dict(a))
    assert not ex.charges("inspect_record", dict(a))           # a repeat is free
    assert ex.charges("inspect_record", {"record_id": rid, "field_path": "published"})
    assert ex.charges("emit_claim", {"claim_id": "C1"})         # never executed, always charged


def test_a_trailing_repeat_is_not_charged(tmp_path):
    rid = sorted(load_instance(OSV).records)[0]
    for arm in ("react", "react_program_attested", "masked_reactive_v"):
        res, _ = _run(arm, [_read(rid, "summary"), _read(rid, "summary")], 20, tmp_path)
        assert res.tool_calls == 1, (arm, res.tool_calls)


def test_a_repeat_leaves_budget_for_the_next_call(tmp_path):
    rid = sorted(load_instance(OSV).records)[0]
    script = [_read(rid, "summary"), _read(rid, "summary"), _read(rid, "published")]
    for arm in ("react", "masked_reactive_v"):
        res, trace = _run(arm, script, 2, tmp_path)
        assert res.tool_calls == 2, (arm, res.tool_calls)
        assert "published" in _observed_fields(trace), arm


def test_calls_after_a_repeat_are_charged(tmp_path):
    rid = sorted(load_instance(OSV).records)[0]
    script = [_read(rid, "summary"), _read(rid, "summary"), _read(rid, "published"), _read(rid, "details")]
    res, _ = _run("react", script, 20, tmp_path)
    assert res.tool_calls == 3
