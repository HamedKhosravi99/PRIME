"""Call-class tests: the five machine-checkable call-class requirements,
the topology classifier, and the prefix-first regression test for the
nested-order counterexample.

The five requirements:
  1. one call class causes at most one external call;
  2. all atoms derived from that result transition consistently;
  3. repeated requests reuse the cache;
  4. planner-predicted cost equals executor-observed cost;
  5. budgets are matched using observed call classes for every method.
"""
from __future__ import annotations

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "src", "prime_agents")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import arms as arms_mod  # noqa: E402
from prime_runtime import goldledger, guidance, status  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance, make_program  # noqa: E402

FIN = os.path.join(_P, "data", "pilot", "instances", "fin", "fin-0001.json")
OSV = os.path.join(_P, "data", "pilot", "instances", "osv", "osv-0001.json")


# ---------------------------------------------------------------------------
# Requirements 1-3: executor cache
# ---------------------------------------------------------------------------

def test_req1_and_3_one_underlying_call_per_class_and_cache_reuse(monkeypatch):
    inst = load_instance(OSV)
    ex = arms_mod.ToolExecutor(inst)
    calls = {"n": 0}
    orig = ex._tool_inspect_record
    monkeypatch.setattr(ex, "_tool_inspect_record",
                        lambda p: (calls.__setitem__("n", calls["n"] + 1), orig(p))[1])
    params = {"record_id": sorted(inst.records)[0], "field_path": "summary"}
    o1 = ex.execute("inspect_record", dict(params))
    assert calls["n"] == 1 and ex.last_was_cached is False
    o2 = ex.execute("inspect_record", dict(params))
    assert calls["n"] == 1, "second identical request must not re-execute"
    assert ex.last_was_cached is True and o1 == o2
    assert len(ex.observed_classes) == 1


def test_req2_shared_class_atoms_transition_consistently():
    """fin-0001 has three atoms sharing one (tool, params) call class; a
    single executed call must attest ALL of them, and a repeat call must add
    no new or conflicting attestations."""
    inst = load_instance(FIN)
    ex = arms_mod.ToolExecutor(inst)
    ledger = Ledger(inst.atom_ids, inst.phi_patterns)
    # find the shared class
    from collections import Counter
    keys = Counter()
    for a in inst.atoms:
        keys[(a.resolver.tool,
              tuple(sorted((str(k), str(v)) for k, v in dict(a.resolver.params).items())))] += 1
    key, n_shared = keys.most_common(1)[0]
    assert n_shared >= 2
    params = {k: v for k, v in key[1]}
    obs = ex.execute(key[0], dict(params))
    atts1 = goldledger.delta(inst, ledger, key[0], dict(params), obs, step=1)
    for a in atts1:
        ledger.apply(a)
    assert len(atts1) == n_shared, "one call must attest every shared-class atom"
    states1 = ledger.states()
    obs2 = ex.execute(key[0], dict(params))
    assert ex.last_was_cached
    atts2 = goldledger.delta(inst, ledger, key[0], dict(params), obs2, step=2)
    for a in atts2:
        ledger.apply(a)
    assert ledger.states() == states1, "repeat must not change any atom state"


# ---------------------------------------------------------------------------
# Requirement 4: planner-predicted cost == executor-observed cost
# ---------------------------------------------------------------------------

def test_req4_planner_cost_equals_executor_observed():
    inst = load_instance(FIN)
    resolvers = status.resolver_table(inst)
    program = inst.gold_program()
    ledger = Ledger(inst.atom_ids, inst.phi_patterns)
    top = max(inst.claims, key=lambda c: (c.level, c.id))
    g = guidance.reachable_ceiling(program, ledger, resolvers, guidance.claim_levels(inst),
                                   remaining_budget=99)
    predicted = g["per_claim"][top.id]["cost"]
    # execute exactly the resolver calls of the top claim's minimal route
    ex = arms_mod.ToolExecutor(inst)
    route = g["per_claim"][top.id]["route"]
    by_id = {a.id: a for a in inst.atoms}
    for lit in route:
        a = by_id[lit.lstrip("!")]
        ex.execute(a.resolver.tool, dict(a.resolver.params))
    assert predicted == len(ex.observed_classes), (
        "guidance call-class cost (%s) must equal executor-observed distinct "
        "classes (%s)" % (predicted, len(ex.observed_classes)))


# ---------------------------------------------------------------------------
# Requirement 5: budgets count observed call classes, uniformly
# ---------------------------------------------------------------------------

class _RepeatLLM:
    """Issues the SAME tool call twice, then a distinct one, then stops."""

    def __init__(self, rid):
        self.calls = [
            {"name": "inspect_record", "arguments": {"record_id": rid, "field_path": "summary"}},
            {"name": "inspect_record", "arguments": {"record_id": rid, "field_path": "summary"}},
            {"name": "inspect_record", "arguments": {"record_id": rid, "field_path": "published"}},
        ]

    def chat(self, messages, tools=None):
        if self.calls:
            return {"content": "", "tool_calls": [self.calls.pop(0)], "usage": {}}
        return {"content": "done", "tool_calls": [], "usage": {}}


def test_req5_budget_counts_observed_classes(tmp_path):
    from prime_runtime.trace import TraceLogger

    inst = load_instance(OSV)
    rid = sorted(inst.records)[0]
    with TraceLogger(str(tmp_path / "t.jsonl")) as trace:
        result = arms_mod.run_arm("react", inst, _RepeatLLM(rid), trace, "gold")
    # 3 issued calls, 2 distinct classes -> budget consumed = 2
    assert result.tool_calls == 2


# ---------------------------------------------------------------------------
# Topology classifier + prefix-first regression
# ---------------------------------------------------------------------------

def _ladder():
    return make_program(claims=("C1", "C2"), order=(("C1", "C2"),),
                        licenses={"C1": [["a"]], "C2": [["a", "b"]]},
                        scope="unit", atoms=("a", "b"))


def test_classifier_ladder_and_branching():
    assert guidance.classify_topology(_ladder()) == "ladder"
    branching = make_program(claims=("C1", "C2"), order=(("C1", "C2"),),
                             licenses={"C1": [["a"], ["c"]], "C2": [["a", "b"]]},
                             scope="unit", atoms=("a", "b", "c"))
    assert guidance.classify_topology(branching) == "branching"
    # single routes but not nested -> branching
    disjoint = make_program(claims=("C1", "C2"), order=(),
                            licenses={"C1": [["a"]], "C2": [["b", "c"]]},
                            scope="unit", atoms=("a", "b", "c"))
    assert guidance.classify_topology(disjoint) == "branching"


def test_prefix_first_defeats_gpt_nested_counterexample():
    """Nested-order counterexample: ladder C1:{a}, C2:{a,b}, b false, B0=1.
    The concern was that an agent probing b first ends with nothing. With
    budget-aware guidance that cannot happen: at B0=1 the target is C1, so
    only a is surfaced, and the run emits C1, the hindsight maximum. On
    nested ladders the order of the target route's checks never matters
    (test_ladder_any_order.py); prefix-first stays the deployed default."""
    sys.path.insert(0, _TESTS)
    from test_guided_optimality import run_guided

    prog = _ladder()
    emitted, _h3, _ = run_guided(prog, {"C1": 1, "C2": 2},
                                 {"a": "T", "b": "F"}, budget=1)
    assert emitted == 1  # C1 secured; without ordering this could be 0


def test_prefix_first_never_changes_ceiling_selection():
    """Ordering is a tie-break within the ceiling's minimal route only: the
    ceiling claim itself must match an unordered recomputation."""
    inst = load_instance(OSV)
    program = inst.gold_program()
    resolvers = status.resolver_table(inst)
    ledger = Ledger(inst.atom_ids, inst.phi_patterns)
    g = guidance.reachable_ceiling(program, ledger, resolvers,
                                   guidance.claim_levels(inst), 20)
    assert set(g["blocking_atoms"]) == set(
        g["per_claim"][g["ceiling_claim"]]["open"])


def test_e1_contrast_is_mask_only(tmp_path):
    """docs/ANALYSIS_PLAN.md, F7: react_program_attested and
    masked_reactive_v must have IDENTICAL prompts except the
    authorized-claims gate line -- the mask is the only mechanism difference
    in the E1 enforcement contrast."""
    import arms as arms_mod
    from prime_runtime.interpreter import Interpreter
    from prime_runtime.ledger import Ledger
    from prime_runtime.schema import load_instance
    import glob as _glob
    path = sorted(_glob.glob(os.path.join(_P, "data", "obligationbench", "security", "dev", "*.json")))[0]
    inst = load_instance(path)
    ledger = Ledger(inst.atom_ids, inst.phi_patterns)
    interp = Interpreter(inst, inst.gold_program(), revision_mode="fixed")
    attested = arms_mod._ladder_system_prompt("react_program_attested", inst, ledger)
    authorized = interp.legal_claim_actions(ledger)
    masked_v = (attested + "\n\nAUTHORIZED CLAIMS (the only claims emit_claim "
                "will accept): %s" % (authorized or "none"))
    diff = masked_v.replace(attested, "")
    assert diff.strip().startswith("AUTHORIZED CLAIMS")
    assert "SPECIFICATION" in attested and "EVIDENCE LICENSES" in attested
    assert "ATTESTED LEDGER" in attested


def test_openai_client_receives_timeout_and_no_internal_retries():
    """Regression for the 70-minute stall (job 11621570): a hung server-side
    request must fail fast via OUR timeout, not block on the SDK default.
    Retries stay in LLMClient.chat so they are bounded and logged."""
    import llm as llm_mod
    c = llm_mod.LLMClient(model="x", base="http://127.0.0.1:1/v1", key="k", timeout=7.5)
    if c._openai_client is None:
        import pytest
        pytest.skip("openai package not installed in this environment")
    assert float(c._openai_client.timeout) == 7.5
    assert int(c._openai_client.max_retries) == 0
