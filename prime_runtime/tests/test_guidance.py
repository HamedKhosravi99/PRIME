"""Tests for prime_runtime.guidance: the reachable-ceiling / minimal-route
computation behind the ``prime_guided`` arm's obligation-guided completion
mechanism.

Two layers:
  - direct unit tests on ``reachable_ceiling`` against a small synthetic
    nested-route program (full control of ledger states, including a
    route-goes-dead/repair-blocked recompute case that is awkward to force
    on a real instance);
  - the same computation against real committed programs on the smoke
    instance (bench/instances/_smoke_record/rec-0001.json) and one real
    instance per losing domain (osv/fda/fin), discharging the actual gold
    route through the real ToolExecutor/goldledger path (no fabricated
    facts)."""

import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(os.path.dirname(_TESTS))
_RUNNER = os.path.join(_PILOT, "runner")
for _p in (_PILOT, _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime import goldledger, guidance, status  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance, make_program  # noqa: E402

import arms as arms_mod  # noqa: E402

BENCH = os.path.join(_PILOT, "bench", "instances")
REC_0001 = os.path.join(BENCH, "_smoke_record", "rec-0001.json")
OSV_0001 = os.path.join(BENCH, "osv", "osv-0001.json")
FDA_0001 = os.path.join(BENCH, "fda", "fda-0001.json")
FIN_0001 = os.path.join(BENCH, "fin", "fin-0001.json")

LEVELS = {"C1": 1, "C2": 2, "C3": 3}


def _att(atom, value, step=1):
    return {"atom": atom, "value": value, "tool": "t", "params": {}, "output_digest": "", "step": step}


def _resolvers(atoms, **overrides):
    table = {a: {"tool": "t", "enabled": True, "impossible": False} for a in atoms}
    for atom, spec in overrides.items():
        table[atom].update(spec)
    return table


def _toy_program():
    """Same shape as every record-domain instance (osv/fda/fin/_smoke_record):
    C1 <=* C2 <=* C3, nested minimal routes E1 / E1,E2 / E1,E2,E3."""
    return make_program(
        claims=("C1", "C2", "C3"),
        order=(("C1", "C2"), ("C2", "C3")),
        licenses={"C1": [["E1"]], "C2": [["E1", "E2"]], "C3": [["E1", "E2", "E3"]]},
        scope="unit",
        atoms=("E1", "E2", "E3"),
    )


# ---------------------------------------------------------------------------
# Direct unit tests (synthetic nested-route program)
# ---------------------------------------------------------------------------


def test_ceiling_on_empty_ledger_with_ample_budget_is_the_top_claim():
    program = _toy_program()
    led = Ledger(program.atoms, [])
    res = _resolvers(program.atoms)
    g = guidance.reachable_ceiling(program, led, res, LEVELS, remaining_budget=10)
    assert g["ceiling_claim"] == "C3"
    assert g["minimal_route"] == ["E1", "E2", "E3"]
    assert g["blocking_atoms"] == ["E1", "E2", "E3"]
    assert g["authorized_max_claim"] is None
    assert g["must_continue"] is True


def test_ceiling_is_capped_by_remaining_budget():
    program = _toy_program()
    led = Ledger(program.atoms, [])
    res = _resolvers(program.atoms)
    g = guidance.reachable_ceiling(program, led, res, LEVELS, remaining_budget=1)
    assert g["ceiling_claim"] == "C1"  # only the 1-atom route fits budget=1
    assert g["blocking_atoms"] == ["E1"]
    assert g["must_continue"] is True

    g0 = guidance.reachable_ceiling(program, led, res, LEVELS, remaining_budget=0)
    assert g0["ceiling_claim"] is None  # nothing authorized, nothing affordable
    assert g0["must_continue"] is False


def test_ceiling_recomputes_upward_as_atoms_resolve_then_terminates_at_the_top():
    program = _toy_program()
    led = Ledger(program.atoms, [])
    res = _resolvers(program.atoms)
    led.apply(_att("E1", "T", 1))
    g = guidance.reachable_ceiling(program, led, res, LEVELS, remaining_budget=10)
    assert g["authorized_max_claim"] == "C1"
    assert g["ceiling_claim"] == "C3"
    assert g["blocking_atoms"] == ["E2", "E3"]  # E1 already true: dropped from the obligation
    assert g["must_continue"] is True

    led.apply(_att("E2", "T", 2))
    led.apply(_att("E3", "T", 3))
    g2 = guidance.reachable_ceiling(program, led, res, LEVELS, remaining_budget=10)
    assert g2["ceiling_claim"] == "C3"
    assert g2["authorized_max_claim"] == "C3"
    assert g2["blocking_atoms"] == []
    assert g2["must_continue"] is False  # nothing stronger left: termination permitted


def test_ceiling_drops_when_the_only_route_becomes_repair_blocked():
    """An atom resolving AGAINST the polarity every route needs removes
    every claim built on it -- the 'ceiling recomputes downward' branch of
    the termination rule. No repair tool is exposed to the agent in this
    benchmark family, so a repair-blocked route is treated as unreachable
    (guidance._route_open_atoms's documented limitation), not silently
    left out of the obligation."""
    program = _toy_program()
    led = Ledger(program.atoms, [])
    res = _resolvers(program.atoms)
    led.apply(_att("E1", "F", 1))  # every route needs E1=T; now opposed
    g = guidance.reachable_ceiling(program, led, res, LEVELS, remaining_budget=10)
    assert g["ceiling_claim"] is None
    assert g["authorized_max_claim"] is None
    assert g["must_continue"] is False
    assert g["per_claim"]["C1"]["cost"] is None
    assert g["per_claim"]["C3"]["cost"] is None


def test_ceiling_ignores_a_dead_alternate_route_and_uses_the_live_one():
    program = make_program(
        claims=("C1",), order=(),
        licenses={"C1": [["E1"], ["!E1", "E2"]]},
        scope="unit", atoms=("E1", "E2"),
    )
    led = Ledger(program.atoms, [])
    res = _resolvers(program.atoms)
    led.apply(_att("E1", "T", 1))  # kills route 2 (needs !E1); completes route 1
    g = guidance.reachable_ceiling(program, led, res, {"C1": 1}, remaining_budget=10)
    assert g["ceiling_claim"] == "C1"
    assert g["blocking_atoms"] == []
    assert g["must_continue"] is False


def test_claim_levels_reads_instance_claim_levels():
    inst = load_instance(REC_0001)
    assert guidance.claim_levels(inst) == {"C1": 1, "C2": 2, "C3": 3}


# ---------------------------------------------------------------------------
# Real instances: smoke record + one per losing domain (osv/fda/fin), real
# gold program, real resolver table, real ToolExecutor/goldledger evidence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [REC_0001, OSV_0001, FDA_0001, FIN_0001])
def test_ceiling_on_real_instances_start_and_after_gold_route(path):
    inst = load_instance(path)
    program = inst.gold_program()
    resolvers = status.resolver_table(inst)
    levels = guidance.claim_levels(inst)
    led = Ledger(inst.atom_ids, inst.phi_patterns)
    top = max(inst.claims, key=lambda c: c.level)
    top_route = min(inst.gold_licenses[top.id], key=len)

    g_start = guidance.reachable_ceiling(
        program, led, resolvers, levels, remaining_budget=len(inst.atom_ids)
    )
    assert g_start["ceiling_claim"] == top.id
    assert set(g_start["blocking_atoms"]) == {a.lstrip("!") for a in top_route}
    assert g_start["must_continue"] is True
    for lit in g_start["blocking_atoms"]:
        assert g_start["tools"][lit] == inst.atom(lit).resolver.tool

    # Discharge exactly the gold route via the real tool executor + gold
    # ledger delta path (never fabricated facts).
    executor = arms_mod.ToolExecutor(inst)
    step = 0
    for lit in top_route:
        atom = inst.atom(lit.lstrip("!"))
        step += 1
        obs = executor.execute(atom.resolver.tool, dict(atom.resolver.params))
        goldledger.delta(inst, led, atom.resolver.tool, dict(atom.resolver.params), obs, step)

    g_end = guidance.reachable_ceiling(program, led, resolvers, levels, remaining_budget=10)
    assert g_end["ceiling_claim"] == top.id
    assert g_end["authorized_max_claim"] == top.id
    assert g_end["blocking_atoms"] == []
    assert g_end["must_continue"] is False


@pytest.mark.parametrize("path", [OSV_0001, FDA_0001, FIN_0001])
def test_ceiling_after_only_c1_evidence_never_regresses_below_authorized(path):
    """Regression for the measured failure this mechanism targets: PRIME
    under-claims by stopping at C1 even though a stronger claim's atoms are
    still gatherable. After only C1's own gold route is resolved, the
    ceiling must never be WEAKER than what just got authorized, and if
    anything stronger remains reachable, must_continue must fire so the
    (unguided) satisficing-at-C1 failure mode cannot recur.

    (financial instances may authorize a stronger claim than C1 directly
    from C1's own route -- inspect_record deltas fire every atom whose
    resolver params match the executed call, so a single field read can
    settle several threshold predicates over the same field at once,
    see goldledger.delta; that is a domain property, not a bug here, so
    this test asserts the general invariant rather than assuming C1 is
    exactly the authorized ceiling.)"""
    inst = load_instance(path)
    program = inst.gold_program()
    resolvers = status.resolver_table(inst)
    levels = guidance.claim_levels(inst)
    led = Ledger(inst.atom_ids, inst.phi_patterns)
    c1_route = min(inst.gold_licenses["C1"], key=len)
    top = max(inst.claims, key=lambda c: c.level)

    executor = arms_mod.ToolExecutor(inst)
    step = 0
    for lit in c1_route:
        atom = inst.atom(lit.lstrip("!"))
        step += 1
        obs = executor.execute(atom.resolver.tool, dict(atom.resolver.params))
        goldledger.delta(inst, led, atom.resolver.tool, dict(atom.resolver.params), obs, step)

    g = guidance.reachable_ceiling(
        program, led, resolvers, levels, remaining_budget=len(inst.atom_ids)
    )
    assert levels[g["authorized_max_claim"]] >= 1  # C1's own route is authorized at least
    assert levels[g["ceiling_claim"]] >= levels[g["authorized_max_claim"]]  # never regresses
    if g["authorized_max_claim"] == top.id:
        assert g["must_continue"] is False  # nothing stronger left: fine to stop
    else:
        assert g["must_continue"] is True  # a stronger claim remains reachable
        assert g["ceiling_claim"] == top.id  # budget is generous (len(atoms))
        assert g["blocking_atoms"]
