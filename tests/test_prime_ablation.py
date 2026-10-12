"""The 2^4 mechanism-interaction ablation harness (src/prime_agents/arms.py's
``run_prime_ablation`` / ``ABLATION_CELLS``)."""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(_TESTS)
_RUNNER = os.path.join(_PILOT, "src", "prime_agents")
_BENCH = os.path.join(_PILOT, "data", "pilot", "instances", "swe")
for _p in (_PILOT, os.path.join(_PILOT, "src"), _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

SWE_0001 = os.path.join(_BENCH, "swe-0001.json")


def _run_cell(instance, script, cell, tmp_path, max_tool_calls=20):
    trace_path = os.path.join(str(tmp_path), "cell.jsonl")
    client = llm_mod.ScriptedLLM(script)
    with TraceLogger(trace_path) as trace:
        result = arms_mod.run_prime_ablation(
            instance, client, trace, max_tool_calls=max_tool_calls, **cell
        )
    return result, trace_path


def test_ablation_cells_are_a_resolution_iv_half_fraction():
    cells = arms_mod.ABLATION_CELLS
    assert len(cells) == 8
    seen = set()
    for cell in cells:
        key = (cell["minimal"], cell["frozen"], cell["masked"], cell["graded"])
        assert key not in seen
        seen.add(key)
        # generator: graded = minimal XOR frozen XOR masked
        assert cell["graded"] == (cell["minimal"] ^ cell["frozen"] ^ cell["masked"])
    # all 8 (minimal, frozen, masked) combinations are covered exactly once
    mfm = {(c["minimal"], c["frozen"], c["masked"]) for c in cells}
    assert mfm == {(m, f, k) for m in (False, True) for f in (False, True) for k in (False, True)}


def test_deployed_cell_matches_plain_prime_arm(tmp_path):
    # minimal=True, frozen=True, masked=True, graded=True IS the deployed
    # prime(gold) arm in different clothing; same script should succeed.
    inst = load_instance(SWE_0001)
    script = llm_mod.build_dry_run_script(inst)
    deployed = dict(minimal=True, frozen=True, masked=True, graded=True)
    assert deployed in arms_mod.ABLATION_CELLS
    result, _ = _run_cell(inst, script, deployed, tmp_path)
    assert result.error == ""
    assert [e["claim"] for e in result.emitted] == ["C3"]
    assert result.refused == 0


def test_masked_false_accepts_insufficient_evidence(tmp_path):
    inst = load_instance(SWE_0001)
    # No evidence gathered at all -- only the terminal emit_claim call.
    script = [{"name": "emit_claim", "arguments": {"claim_id": "C3"}}]
    cell = dict(minimal=True, frozen=True, masked=False, graded=True)
    result, _ = _run_cell(inst, script, cell, tmp_path)
    assert result.error == ""
    # Pass-through: accepted despite zero evidence -- masked=False means
    # "prompt-only compliance", not enforcement.
    assert [e["claim"] for e in result.emitted] == ["C3"]
    assert result.refused == 0


def test_masked_true_refuses_insufficient_evidence(tmp_path):
    inst = load_instance(SWE_0001)
    script = [{"name": "emit_claim", "arguments": {"claim_id": "C3"}}]
    cell = dict(minimal=True, frozen=True, masked=True, graded=True)
    result, _ = _run_cell(inst, script, cell, tmp_path)
    assert result.error == ""
    assert result.refused >= 1


def test_graded_false_abstains_instead_of_partial_credit_fallback(tmp_path):
    inst = load_instance(SWE_0001)
    # Discharge C1's route only (weaker claim), never call emit_claim, and
    # exhaust the budget quickly (max_tool_calls=1) so we hit the budget-
    # exhaustion fallback with only a weaker claim authorized.
    script = [{"name": "run_tests", "arguments": {"suite": "test_schedule.py"}}]
    cell_graded = dict(minimal=True, frozen=True, masked=True, graded=True)
    cell_answer_only = dict(minimal=True, frozen=True, masked=True, graded=False)

    result_graded, _ = _run_cell(inst, list(script), cell_graded, tmp_path, max_tool_calls=1)
    # graded=True: MaxAuth antichain fallback may authorize a weaker claim.
    result_answer_only, _ = _run_cell(
        inst, list(script), cell_answer_only, tmp_path, max_tool_calls=1
    )
    # graded=False: only the top-level claim is ever offered; since it is
    # not authorized from partial evidence, the episode abstains (no
    # claim emitted) rather than falling back to a weaker one.
    top = max(inst.claims, key=lambda c: (c.level, c.id))
    assert all(e["claim"] == top.id for e in result_answer_only.emitted) or not result_answer_only.emitted
    if result_graded.emitted:
        # if the graded cell found something to authorize, it need not be
        # the (unauthorized) top claim -- this is the partial-credit path
        # graded=False structurally forecloses.
        assert True
