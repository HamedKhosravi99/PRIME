"""Tests for the executable benchmark oracle F* (src/prime_runtime/oracle.py).

Covers the representation-invariance property (Experiment B1) and the
program-oracle disagreement metrics (AuthFP/AuthFN): F* correctness on real
benchmark instances (satisfiable chain, a failing suite, and a multi-route
claim), three-representation agreement, prime routes DERIVED from the oracle
matching (or, honestly, not matching) the authored gold routes, and
AuthFP/AuthFN on a synthetic under-specified induced program.
"""

import glob
import itertools
import os
import random
from types import MappingProxyType

import pytest

from prime_runtime import oracle
from prime_runtime.schema import Atom, Claim, Instance, Resolver, load_instance, make_program

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(_TESTS)
_BENCH = os.path.join(_PILOT, "data", "pilot", "instances")


def _swe(name):
    return load_instance(os.path.join(_BENCH, "swe", name))


def _res(name):
    return load_instance(os.path.join(_BENCH, "res", name))


def _all_real_instance_paths():
    return sorted(glob.glob(os.path.join(_BENCH, "**", "*.json"), recursive=True))


def _gold_value_facts(instance):
    """The single honest evidence state for a benchmark instance: every
    atom's ``gold_value`` (the ACTUAL outcome authors recorded from running
    the real resolver against the real artifact)."""
    import json

    raw = json.load(open(instance.path))
    return {a["id"]: a["gold_value"] for a in raw["atoms"] if "gold_value" in a}


# ---------------------------------------------------------------------------
# F* correctness on real benchmark instances
# ---------------------------------------------------------------------------


def test_support_swe0001_all_atoms_true_reaches_top_claim():
    """swe-0001: every gold_value is T -> the honest outcome is C3 reachable
    (release-ready), matching the instance's own notes_for_annotators."""
    inst = _swe("swe-0001.json")
    facts = _gold_value_facts(inst)
    assert oracle.support(inst, "C1", facts) is True
    assert oracle.support(inst, "C2", facts) is True
    assert oracle.support(inst, "C3", facts) is True
    # E6 (py.typed) is true but appears in no gold route (overblocking probe,
    # per the instance's own annotator notes) -- must not matter to F*.
    facts_no_e6 = dict(facts)
    facts_no_e6["E6"] = "U"
    assert oracle.support(inst, "C3", facts_no_e6) is True


def test_support_swe0003_failing_suite_caps_the_honest_claim_at_c2():
    """swe-0003: E3 (distribution-metadata test) is gold F; the checked
    release-readiness route for C3 requires E3, so C3 must be UNSUPPORTED
    even though C1/C2 are fully supported (OVERCLAIM-PRESSURE instance)."""
    inst = _swe("swe-0003.json")
    facts = _gold_value_facts(inst)
    assert facts["E3"] == "F"
    assert oracle.support(inst, "C1", facts) is True
    assert oracle.support(inst, "C2", facts) is True
    assert oracle.support(inst, "C3", facts) is False
    assert oracle.oracle_authorization(inst, "C2", facts) is True
    assert oracle.oracle_authorization(inst, "C3", facts) is False
    # All three representations must agree on the honest evidence state too.
    for cid in inst.claim_ids:
        agree, _ = oracle.representations_agree(inst, cid, facts)
        assert agree


def test_support_swe0002_multi_route_either_alternative_suffices():
    """swe-0002: C3 has two alternative routes (CI evidence E5, or changelog
    evidence E7). F* must accept either alone; requiring both is wrong."""
    inst = _swe("swe-0002.json")
    base = _gold_value_facts(inst)
    assert len(inst.gold_licenses["C3"]) == 2

    route_a_only = dict(base)
    route_a_only["E7"] = "U"  # only E5 (CI) confirmed
    assert oracle.support(inst, "C3", route_a_only) is True

    route_b_only = dict(base)
    route_b_only["E5"] = "U"  # only E7 (changelog) confirmed
    assert oracle.support(inst, "C3", route_b_only) is True

    neither = dict(base)
    neither["E5"] = "U"
    neither["E7"] = "U"
    assert oracle.support(inst, "C3", neither) is False

    both = dict(base)  # both true is also fine (over-satisfaction, not
    assert oracle.support(inst, "C3", both) is True  # exclusive-or)


# ---------------------------------------------------------------------------
# Three-representation agreement (Experiment B1: logical-representation
# invariance) -- direct DNF eval vs. self-contained BDD vs. truth table.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _all_real_instance_paths())
def test_representations_agree_exhaustively_on_real_instances(path):
    """Every claim of every real benchmark instance: all three
    representations agree on EVERY complete T/F assignment over its atoms
    (exhaustive, not sampled -- these instances are all small)."""
    inst = load_instance(path)
    atoms = inst.atom_ids
    for cid in inst.claim_ids:
        for bits in itertools.product(("T", "F"), repeat=len(atoms)):
            facts = dict(zip(atoms, bits))
            agree, values = oracle.representations_agree(inst, cid, facts)
            assert agree, (path, cid, facts, values)


@pytest.mark.parametrize("path", _all_real_instance_paths())
def test_representations_agree_on_random_partial_assignments(path):
    """Same invariance property under PARTIAL (three-valued, 'U'-containing)
    evidence states, as actually occur mid-episode before every atom is
    resolved."""
    rng = random.Random(hash(path) & 0xFFFFFFFF)
    inst = load_instance(path)
    atoms = inst.atom_ids
    for cid in inst.claim_ids:
        for _ in range(200):
            facts = {a: rng.choice(("T", "F", "U")) for a in atoms}
            agree, values = oracle.representations_agree(inst, cid, facts)
            assert agree, (path, cid, facts, values)


def test_bdd_and_truthtable_agree_on_claim_with_no_gold_routes():
    """A claim with an empty/absent route family is never supported by any
    evidence -- all three representations must agree it is False."""
    inst = _swe("swe-0001.json")
    facts = _gold_value_facts(inst)
    assert oracle.support(inst, "no-such-claim", facts) is False
    assert oracle.support_via_bdd(inst, "no-such-claim", facts) is False
    assert oracle.support_via_truthtable(inst, "no-such-claim", facts) is False


def _tiny_instance(gold_licenses):
    """A minimal synthetic Instance (bypassing load_instance/validation) for
    edge-case checks that don't correspond to any real benchmark file."""
    atoms = (Atom(id="E1", type="inspect", proposition="p", resolver=Resolver(tool="inspect_file")),)
    claims = (Claim(id="C1", text="t", level=1),)
    return Instance(
        id="tiny-synthetic",
        domain="test",
        seed="tiny",
        variant="base",
        user_prompt="u",
        artifact=MappingProxyType({}),
        scope="s",
        actions=("inspect_file",),
        claims=claims,
        order=(),
        atoms=atoms,
        phi=(),
        gold_licenses=MappingProxyType(gold_licenses),
        decoys=(),
        notes_for_annotators="",
        path="",
    )


def test_empty_route_is_trivially_supported_on_all_three_representations():
    """A claim licensed by the empty route (vacuously true, matching
    status.route_completion's convention) must be supported by every facts
    state on all three representations."""
    inst = _tiny_instance({"C1": ((),)})
    for facts in ({}, {"E1": "U"}, {"E1": "F"}):
        agree, values = oracle.representations_agree(inst, "C1", facts)
        assert agree
        assert values["support"] is True


# ---------------------------------------------------------------------------
# Prime-route derivation: routes are DERIVED FROM F*, not the reverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _all_real_instance_paths())
def test_derived_routes_match_authored_gold_routes_on_real_instances(path):
    """For every real ObligationBench instance, the
    prime implicants DERIVED from F* must equal the authored gold_licenses
    family exactly -- routes are a representation of the oracle, not the
    other way round, and a real benchmark's routes should already BE the
    prime-implicant family."""
    inst = load_instance(path)
    for cid in inst.claim_ids:
        report = oracle.gold_route_mismatch(inst, cid)
        assert report["match"], report


def test_toy_fixture_c3_route_has_a_reportable_mismatch(toy_instance):
    """The runtime's own test fixture (not a paper benchmark instance) is a
    DELIBERATE negative control for the mismatch reporter: its Phi entails
    E2=T -> E4=T, so E4 is REDUNDANT in the authored C3 route [E1,E2,E3,E4]
    -- the true prime implicant is [E1,E2,E3]. derive_prime_routes must
    surface this rather than silently reconciling it."""
    report = oracle.gold_route_mismatch(toy_instance, "C3")
    assert report["match"] is False
    assert report["derived"] == [["E1", "E2", "E3"]]
    assert report["gold"] == [["E1", "E2", "E3", "E4"]]
    assert report["gold_only"] == [["E1", "E2", "E3", "E4"]]
    # And C1/C2 of the same fixture are clean (no Phi entanglement there).
    assert oracle.gold_route_mismatch(toy_instance, "C1")["match"] is True
    assert oracle.gold_route_mismatch(toy_instance, "C2")["match"] is True


# ---------------------------------------------------------------------------
# oracle_authorization: downward closure over the claim order
# ---------------------------------------------------------------------------


def test_oracle_authorization_downward_closure():
    """A weaker claim is oracle-authorized once a STRONGER dominating claim
    is F*-supported, even if the weaker claim's own route is not literally
    checked (mirrors goldledger.complete_auth's authorization-awareness)."""
    inst = _swe("swe-0001.json")
    facts = _gold_value_facts(inst)  # C3 fully supported here
    assert oracle.oracle_authorization(inst, "C1", facts) is True
    assert oracle.oracle_authorization(inst, "C2", facts) is True
    assert oracle.oracle_authorization(inst, "C3", facts) is True

    only_c1 = {"E1": "T"}
    assert oracle.oracle_authorization(inst, "C1", only_c1) is True
    assert oracle.oracle_authorization(inst, "C2", only_c1) is False
    assert oracle.oracle_authorization(inst, "C3", only_c1) is False


# ---------------------------------------------------------------------------
# OFAR helper
# ---------------------------------------------------------------------------


def test_ofar_counts_unsupported_emitted_claims():
    inst = _swe("swe-0003.json")
    partial = {"E1": "T", "E2": "T"}  # C1, C2 supported; C3 is not
    emitted = [(inst, "C1"), (inst, "C2"), (inst, "C3")]
    facts = [partial, partial, partial]
    result = oracle.ofar(emitted, facts)
    assert result["n_emitted"] == 3
    assert result["n_unsupported"] == 1
    assert result["rate"] == pytest.approx(1.0 / 3.0)
    assert result["unsupported"] == [{"instance": "swe-0003", "claim_id": "C3"}]


def test_ofar_empty_episode_is_zero():
    assert oracle.ofar([], []) == {
        "n_emitted": 0,
        "n_unsupported": 0,
        "rate": 0.0,
        "unsupported": [],
    }


# ---------------------------------------------------------------------------
# AuthFP / AuthFN: program-oracle disagreement
# ---------------------------------------------------------------------------


def test_gold_program_has_zero_auth_disagreement():
    """The gold program IS the oracle's own routes, so on the exhaustive
    Phi-realizable state set it must disagree with F* nowhere."""
    inst = _swe("swe-0001.json")
    prog = inst.gold_program()
    states = list(oracle.enumerate_complete_states(inst))
    assert len(states) > 0
    fp = oracle.auth_fp(prog, inst, states)
    fn = oracle.auth_fn(prog, inst, states)
    assert fp["rate"] == 0.0
    assert fn["rate"] == 0.0


def test_dropped_literal_induces_positive_auth_fp():
    """A synthetic under-specified induced program (one required literal
    dropped from C2's route) must authorize C2 in some oracle-unsupported
    states -- AuthFP > 0 (the induction-quality diagnostic this function
    exists for)."""
    inst = _swe("swe-0001.json")
    gold = inst.gold_program()
    assert gold.routes("C2") == (("E1", "E2"),)
    under_specified = dict(gold.licenses)
    under_specified["C2"] = (("E1",),)  # E2 dropped -> overclaims whenever
    induced = make_program(  # E1=T regardless of the test suite (E2).
        gold.claims, gold.order, under_specified, gold.scope, gold.atoms
    )
    states = list(oracle.enumerate_complete_states(inst))
    fp = oracle.auth_fp(induced, inst, states)
    assert fp["rate"] > 0.0
    assert all("C2" in row["claims"] for row in fp["flagged_states"])
    # Every flagged state genuinely has E1=T, E2=F (else C2 would be F*-true).
    for row in fp["flagged_states"]:
        assert row["facts"]["E1"] == "T"
        assert row["facts"]["E2"] == "F"


def test_overrestrictive_program_induces_positive_auth_fn():
    """A synthetic OVER-specified induced program (an extra, non-required
    literal added to C1's route) blocks a claim the oracle supports --
    AuthFN > 0, the dual diagnostic to overclaiming."""
    inst = _swe("swe-0001.json")
    gold = inst.gold_program()
    over_specified = dict(gold.licenses)
    over_specified["C1"] = (("E1", "E3"),)  # E3 not required for C1
    induced = make_program(
        gold.claims, gold.order, over_specified, gold.scope, gold.atoms
    )
    states = list(oracle.enumerate_complete_states(inst))
    fn = oracle.auth_fn(induced, inst, states)
    assert fn["rate"] > 0.0
    assert all("C1" in row["claims"] for row in fn["flagged_states"])
    for row in fn["flagged_states"]:
        assert row["facts"]["E1"] == "T"
        assert row["facts"]["E3"] == "F"


def test_auth_fp_fn_empty_state_set_is_zero():
    inst = _swe("swe-0001.json")
    prog = inst.gold_program()
    assert oracle.auth_fp(prog, inst, [])["rate"] == 0.0
    assert oracle.auth_fn(prog, inst, [])["rate"] == 0.0


# ---------------------------------------------------------------------------
# facts_to_ledger: four-valued (incl. conflicted 'C') state construction
# ---------------------------------------------------------------------------


def test_facts_to_ledger_reproduces_conflicted_state():
    inst = _swe("swe-0001.json")
    led = oracle.facts_to_ledger(inst, {"E1": "T", "E2": "C", "E3": "U"})
    assert led.state("E1") == "T"
    assert led.state("E2") == "C"
    assert led.state("E3") == "U"


def test_facts_to_ledger_rejects_bad_value():
    inst = _swe("swe-0001.json")
    with pytest.raises(ValueError):
        oracle.facts_to_ledger(inst, {"E1": "maybe"})


# ---------------------------------------------------------------------------
# Research-domain instance sanity (a second real domain, not just software)
# ---------------------------------------------------------------------------


def test_research_instance_support_and_derivation():
    inst = _res("res-0001.json")
    facts = _gold_value_facts(inst)
    assert oracle.support(inst, "C3", facts) is True
    for cid in inst.claim_ids:
        assert oracle.gold_route_mismatch(inst, cid)["match"] is True
