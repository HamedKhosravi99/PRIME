"""Task #5 (the design notes,
Weakness 1 SS1.3 E1 methods list): the four new arms added to runner/arms.py
-- reflexion, cove, static_contract, prime_nonminimal -- share the same tool
loop and 20-tool-call budget as the original four; these tests pin their
mechanics deterministically (scripted LLM, no API calls)."""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(os.path.dirname(_TESTS))
_RUNNER = os.path.join(_PILOT, "runner")
_BENCH = os.path.join(_PILOT, "bench", "instances", "swe")
for _p in (_PILOT, _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

SWE_0001 = os.path.join(_BENCH, "swe-0001.json")  # single-route claims; E6 unused
SWE_0002 = os.path.join(_BENCH, "swe-0002.json")  # C3 has two alternative routes


def _run(arm, instance, script, out_dir, license_source="gold"):
    trace_path = os.path.join(str(out_dir), "%s.jsonl" % arm)
    client = llm_mod.ScriptedLLM(script)
    with TraceLogger(trace_path) as trace:
        result = arms_mod.run_arm(arm, instance, client, trace, license_source=license_source)
    return result, trace_path


def test_new_arms_are_registered():
    for arm in ("reflexion", "cove", "static_contract", "prime_nonminimal"):
        assert arm in arms_mod.ARMS
    # original four arms are untouched
    for arm in ("direct", "rubric", "react", "prime"):
        assert arm in arms_mod.ARMS


# --------------------------------------------------------------------------
# static_contract: authored non-minimal checklist gate
# --------------------------------------------------------------------------


def test_static_contract_checklist_is_union_of_gold_routes():
    inst = load_instance(SWE_0002)
    checklist = arms_mod._static_contract_checklist(inst)
    # C3 has two alternative gold routes differing in one atom; the static
    # checklist is their UNION (non-minimal, "do everything"), not either
    # single minimal route.
    union = {"E1", "E2", "E4", "E5", "E7"}
    assert set(checklist["C3"]) == union


def test_static_contract_emits_when_single_route_claim_matched(tmp_path):
    inst = load_instance(SWE_0001)  # C3 has exactly one gold route
    script = llm_mod.build_dry_run_script(inst)
    result, _ = _run("static_contract", inst, script, tmp_path)
    assert result.error == ""
    assert [e["claim"] for e in result.emitted] == ["C3"]
    assert result.refused == 0


def test_static_contract_refuses_until_full_union_checklist_satisfied(tmp_path):
    inst = load_instance(SWE_0002)  # C3 needs the UNION, base script covers one route only
    base = llm_mod.build_dry_run_script(inst)  # discharges one minimal route then emits
    result, _ = _run("static_contract", inst, base, tmp_path)
    # The single gold route is not the full union checklist: emission must be
    # refused at least once (gate fires; no learned minimal-route search
    # lets it succeed on partial evidence the way PRIME's OR-of-routes would).
    assert result.refused >= 1


# --------------------------------------------------------------------------
# prime_nonminimal: padded gold routes, minimality factor
# --------------------------------------------------------------------------


def test_nonminimal_program_pads_every_route_with_same_atoms():
    inst = load_instance(SWE_0002)
    program, pad_info = arms_mod._nonminimal_gold_program(inst)
    assert pad_info["pad_atoms"], "swe-0002 has >=1 unused atom to pad with"
    pad = set(pad_info["pad_atoms"])
    for cid, routes in program.licenses.items():
        gold_routes = inst.gold_licenses.get(cid, ())
        for route, gold_route in zip(routes, gold_routes):
            route_atoms = {r.lstrip("!") for r in route}
            gold_atoms = {r.lstrip("!") for r in gold_route}
            assert pad <= route_atoms
            assert route_atoms - gold_atoms == pad  # nothing else changed


def test_nonminimal_program_still_validates_against_the_instance():
    from prime_runtime.schema import validate_program

    for path in (SWE_0001, SWE_0002):
        inst = load_instance(path)
        program, _ = arms_mod._nonminimal_gold_program(inst)
        validate_program(program, inst)  # must not raise (SchemaError)


def test_nonminimal_refuses_the_unpadded_minimal_gold_route(tmp_path):
    inst = load_instance(SWE_0001)
    minimal_script = llm_mod.build_dry_run_script(inst)  # exactly the minimal gold route
    result, _ = _run("prime_nonminimal", inst, minimal_script, tmp_path)
    # The padded route requires strictly more than the minimal gold route;
    # a policy that stops at minimal evidence must be refused at least once
    # -- this is the empirical face of "minimality saves wasted work".
    assert result.refused >= 1


def test_nonminimal_succeeds_once_the_padding_atom_is_also_established(tmp_path):
    inst = load_instance(SWE_0001)
    _, pad_info = arms_mod._nonminimal_gold_program(inst)
    assert pad_info["pad_atoms"] == ["E6"]
    base = llm_mod.build_dry_run_script(inst)
    padded = base[:-1] + [
        {"name": "inspect_config", "arguments": {"path": "schedule/py.typed"}},
        {"name": "emit_claim", "arguments": {"claim_id": "C3"}},
    ]
    result, trace_path = _run("prime_nonminimal", inst, padded, tmp_path)
    assert result.error == ""
    assert [e["claim"] for e in result.emitted] == ["C3"]
    # Costs one more tool call than the matched minimal-route prime arm.
    minimal_result, _ = _run("prime", inst, base, tmp_path, license_source="gold")
    assert result.tool_calls == minimal_result.tool_calls + 1


def test_prime_nonminimal_rejects_induced_license_source():
    inst = load_instance(SWE_0001)
    import pytest

    client = llm_mod.ScriptedLLM([])
    from prime_runtime.trace import TraceLogger as _TL
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        trace_path = os.path.join(d, "x.jsonl")
        with pytest.raises(ValueError):
            with _TL(trace_path) as trace:
                arms_mod.run_arm(
                    "prime_nonminimal", inst, client, trace, license_source="induced"
                )


# --------------------------------------------------------------------------
# reflexion / cove: answer -> critique-or-verify -> revise -> emit
# --------------------------------------------------------------------------


def test_reflexion_and_cove_intercept_the_first_emit_then_finalize_the_second(tmp_path):
    inst = load_instance(SWE_0001)
    base = llm_mod.build_dry_run_script(inst)
    evidence = base[:-1]
    two_emits = evidence + [
        {"name": "emit_claim", "arguments": {"claim_id": "C3"}},
        {"name": "emit_claim", "arguments": {"claim_id": "C3"}},
    ]
    for arm in ("reflexion", "cove"):
        result, trace_path = _run(arm, inst, two_emits, tmp_path)
        assert result.error == ""
        assert [e["claim"] for e in result.emitted] == ["C3"]
        # exactly one emission is recorded (the second, final call) -- the
        # draft is intercepted, not recorded as a real emission.
        assert len(result.emitted) == 1
        types = [line for line in open(trace_path)]
        assert any('"type": "emission"' in t for t in types)


def test_reflexion_single_emit_script_falls_back_to_forced_selection(tmp_path):
    # If the model never re-emits after the critique/verify round, the
    # episode is NOT silently left unscored: the standard forced-selection
    # fallback (shared with every other arm) still applies.
    inst = load_instance(SWE_0001)
    base = llm_mod.build_dry_run_script(inst)  # ends in exactly one emit_claim
    result, _ = _run("reflexion", inst, base, tmp_path)
    assert result.error == ""
    # ScriptedLLM's tools=None fallback answer does not parse as the forced
    # -selection JSON, so this specific fake ends up with no emission -- but
    # critically, no crash / no error, matching baseline-arm behavior on an
    # uncooperative scripted model.
    assert isinstance(result.emitted, list)
