"""Tests for the ``prime_guided`` arm (runner/arms.py): obligation-guided
completion's TERMINATION RULE enforcement, the guided system prompt, budget
exhaustion, and the unchanged safety mask. All scripted -- no API calls.

The unguided ``prime`` arm is exercised alongside ``prime_guided`` throughout
so the guided-vs-unguided ablation contrast and the "masking is unchanged"
claim are both pinned, not just asserted in prose.
"""

import glob
import json
import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(os.path.dirname(_TESTS))
_RUNNER = os.path.join(_PILOT, "runner")
for _p in (_PILOT, _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime import goldledger, oracle  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

BENCH = os.path.join(_PILOT, "bench", "instances")
OSV_0001 = os.path.join(BENCH, "osv", "osv-0001.json")
SWE_0001 = os.path.join(BENCH, "swe", "swe-0001.json")


def _run(arm, instance, script, out_dir, license_source="gold", max_tool_calls=arms_mod.MAX_TOOL_CALLS):
    trace_path = os.path.join(str(out_dir), "%s-%s.jsonl" % (arm, instance.id))
    client = llm_mod.ScriptedLLM(script)
    with TraceLogger(trace_path) as trace:
        result = arms_mod.run_arm(
            arm, instance, client, trace, license_source=license_source,
            max_tool_calls=max_tool_calls,
        )
    return result, trace_path


def _inspect(record_id, field_path):
    return {"name": "inspect_record", "arguments": {"record_id": record_id, "field_path": field_path}}


# ---------------------------------------------------------------------------
# Registration / ablation-pair sanity
# ---------------------------------------------------------------------------


def test_prime_guided_is_registered_and_prime_is_untouched():
    assert "prime_guided" in arms_mod.ARMS
    for arm in ("direct", "rubric", "react", "prime", "prime_nonminimal"):
        assert arm in arms_mod.ARMS


# ---------------------------------------------------------------------------
# TERMINATION RULE: premature emit_claim is held, not accepted, while a
# strictly stronger claim remains reachable within budget.
# ---------------------------------------------------------------------------


def test_premature_emit_is_held_then_succeeds_once_the_ceiling_is_reached(tmp_path):
    inst = load_instance(OSV_0001)
    rid = "GHSA-3pqx-4fqf-j49f"
    script = [
        {"name": "emit_claim", "arguments": {"claim_id": "C1"}},  # unauthorized: masked
        _inspect(rid, "affected.0.package.name"),  # E1
        _inspect(rid, "database_specific.severity"),  # E2 (+E3, shares field)
        {"name": "emit_claim", "arguments": {"claim_id": "C2"}},  # authorized but held: C3 still reachable
        _inspect(rid, "deployment.observed_version"),  # E4
        {"name": "emit_claim", "arguments": {"claim_id": "C3"}},  # now the ceiling: accepted
    ]
    result, trace_path = _run("prime_guided", inst, script, tmp_path)
    assert result.error == ""
    assert [e["claim"] for e in result.emitted] == ["C3"]
    assert result.refused == 1  # the premature C1 attempt: real mask refusal
    assert result.guided_holds == 1  # the premature (but authorized) C2 attempt: guided hold
    assert result.guided_stalls == 0

    replay = goldledger.reconstruct(trace_path, inst)
    emission = result.emitted[0]
    assert replay.complete_auth("C3", emission["step"]) == 1
    facts = replay.states_at(emission["step"])
    assert oracle.oracle_authorization(inst, "C3", facts) is True


def test_guided_hold_never_appears_when_the_script_already_waits_for_the_ceiling(tmp_path):
    """A script that gathers the FULL top-route before ever calling
    emit_claim triggers zero guided holds -- the mechanism only intervenes
    on genuine premature termination, matching the unguided prime arm's
    tool-call count exactly on this script."""
    inst = load_instance(SWE_0001)
    script = llm_mod.build_dry_run_script(inst)
    guided_result, _ = _run("prime_guided", inst, script, tmp_path)
    prime_result, _ = _run("prime", inst, script, tmp_path)
    assert guided_result.error == prime_result.error == ""
    assert [e["claim"] for e in guided_result.emitted] == [e["claim"] for e in prime_result.emitted]
    assert guided_result.guided_holds == 0
    assert guided_result.guided_stalls == 0
    assert guided_result.tool_calls == prime_result.tool_calls


# ---------------------------------------------------------------------------
# Budget exhaustion: the loop must still terminate and fall back to MaxAuth,
# never spin forever waiting for an unreachable ceiling.
# ---------------------------------------------------------------------------


def test_budget_exhaustion_falls_back_to_max_auth_not_a_held_claim(tmp_path):
    inst = load_instance(OSV_0001)
    rid = "GHSA-3pqx-4fqf-j49f"
    script = [_inspect(rid, "affected.0.package.name"), {"name": "emit_claim", "arguments": {"claim_id": "C1"}}]
    result, _ = _run("prime_guided", inst, script, tmp_path, max_tool_calls=1)
    assert result.error == ""
    assert [e["claim"] for e in result.emitted] == ["C1"]  # MaxAuth fallback, not held forever
    assert result.tool_calls == 1


def test_stall_bound_terminates_a_noncompliant_model(tmp_path):
    """A model that never calls a tool or emits anything gets nudged up to
    GUIDED_MAX_STALLS times, then the episode ends (bounded wall-clock/cost,
    never an infinite loop)."""
    inst = load_instance(OSV_0001)
    result, _ = _run("prime_guided", inst, [], tmp_path)  # empty script: never offers a tool call
    assert result.error == ""
    assert result.guided_stalls == arms_mod.GUIDED_MAX_STALLS
    assert result.tool_calls == 0
    assert result.emitted == []  # nothing authorized, forced-selection fallback also fails on prose


# ---------------------------------------------------------------------------
# Safety: the auth mask is exactly the same mechanism prime uses, so a
# genuinely unauthorized claim is still refused, never guided into being
# emitted.
# ---------------------------------------------------------------------------


def test_masking_still_refuses_an_unauthorized_claim_with_zero_evidence(tmp_path):
    inst = load_instance(OSV_0001)
    script = [{"name": "emit_claim", "arguments": {"claim_id": "C3"}}]
    result, _ = _run("prime_guided", inst, script, tmp_path)
    assert result.refused >= 1
    assert result.guided_holds == 0  # the mask fired first; guided-hold logic never reached
    assert result.emitted == []  # forced-selection fallback also can't authorize it


@pytest.mark.parametrize("path", sorted(
    glob.glob(os.path.join(BENCH, "osv", "*.json"))
    + glob.glob(os.path.join(BENCH, "fda", "*.json"))
    + glob.glob(os.path.join(BENCH, "fin", "*.json"))
))
def test_gold_dry_run_is_oracle_supported_under_prime_guided(path, tmp_path):
    """OFAR-preservation regression (the hard safety requirement): running
    prime_guided with the scripted GOLD-route dry run across every instance
    in the three losing domains must emit an oracle-supported top claim --
    the termination rule changes only WHEN the agent stops, never WHAT gets
    authorized (the mask is untouched)."""
    inst = load_instance(path)
    script = llm_mod.build_dry_run_script(inst)
    result, trace_path = _run("prime_guided", inst, script, tmp_path)
    assert result.error == ""
    assert len(result.emitted) >= 1
    replay = goldledger.reconstruct(trace_path, inst)
    for emission in result.emitted:
        assert replay.complete_auth(emission["claim"], emission["step"]) == 1
        facts = replay.states_at(emission["step"])
        assert oracle.oracle_authorization(inst, emission["claim"], facts) is True


# ---------------------------------------------------------------------------
# Guided system prompt content: ACTIVE OBLIGATION + TERMINATION RULE while
# stronger evidence remains reachable; a plain "may emit now" once the
# ceiling is reached.
# ---------------------------------------------------------------------------


def test_guided_prompt_states_the_obligation_while_stronger_claim_reachable():
    from prime_runtime import guidance
    from prime_runtime.interpreter import Interpreter
    from prime_runtime.ledger import Ledger

    inst = load_instance(OSV_0001)
    interp = Interpreter(inst, inst.gold_program())
    led = Ledger(inst.atom_ids, inst.phi_patterns)
    levels = guidance.claim_levels(inst)
    prompt = arms_mod._prime_guided_system_prompt(inst, interp, led, 20, levels)
    assert "ACTIVE OBLIGATION" in prompt
    assert "TERMINATION RULE" in prompt
    assert "C3" in prompt  # the ceiling at an empty ledger with ample budget


def test_guided_prompt_permits_emission_once_ceiling_is_reached():
    from prime_runtime import guidance
    from prime_runtime.interpreter import Interpreter
    from prime_runtime.ledger import Ledger

    inst = load_instance(OSV_0001)
    program = inst.gold_program()
    interp = Interpreter(inst, program)
    led = Ledger(inst.atom_ids, inst.phi_patterns)
    rid = "GHSA-3pqx-4fqf-j49f"
    executor = arms_mod.ToolExecutor(inst)
    step = 0
    for lit in min(inst.gold_licenses["C3"], key=len):
        atom = inst.atom(lit.lstrip("!"))
        step += 1
        obs = executor.execute(atom.resolver.tool, dict(atom.resolver.params))
        goldledger.delta(inst, led, atom.resolver.tool, dict(atom.resolver.params), obs, step)
    levels = guidance.claim_levels(inst)
    prompt = arms_mod._prime_guided_system_prompt(inst, interp, led, 20, levels)
    assert "ACTIVE OBLIGATION" not in prompt
    assert "may emit_claim now" in prompt
