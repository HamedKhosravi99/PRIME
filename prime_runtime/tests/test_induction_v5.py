"""Tests for the W3 conservative induction stack (analysis/induction_v5.py).

Everything here is LLM-free: the state-labeling and counterfactual probes
are driven by scripted oracles so the MECHANICAL parts (enumeration, DNF
compilation, lints, merge, eligibility) are pinned exactly.
"""
from __future__ import annotations

import glob
import json
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(os.path.dirname(_TESTS))
for _x in (_P, os.path.join(_P, "runner"), os.path.join(_P, "analysis")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import induction_v5 as iv  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

V2 = sorted(glob.glob(os.path.join(_P, "bench", "instances_v2", "security", "dev", "*.json")))


class _GoldLabeler:
    """Scripted 'model' that labels a state supported iff some GOLD route of
    the claim is contained in it. Used ONLY to verify that the compiler
    reconstructs the gold routes from perfect labels."""

    def __init__(self, instance):
        self.inst = instance

    def chat(self, messages, tools=None):
        text = messages[-1]["content"]
        cid = text.split("Candidate claim ")[1].split(" ")[0] if "Candidate claim " in text \
            else text.split("Claim ")[1].split(":")[0]
        est = {ln.strip().split(" ")[0] for ln in text.splitlines()
               if ln.strip().startswith("E") and ln.rstrip().endswith("ESTABLISHED")
               and "NOT ESTABLISHED" not in ln}
        ok = any(set(r) <= est for r in self.inst.gold_licenses.get(cid, []))
        return {"content": json.dumps({"supported": ok}), "tool_calls": [], "usage": {}}


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------

def test_realizable_states_is_exhaustive_and_excludes_unavailable():
    inst = load_instance(V2[0])
    states = iv.realizable_states(inst)
    n_free = sum(1 for a in inst.atoms if a.resolver is not None)
    assert len(states) == 2 ** n_free
    impossible = {a.id for a in inst.atoms if a.resolver is None}
    for st in states:
        assert not (st & impossible)


def test_describe_state_leaks_no_values_or_gold():
    inst = load_instance(V2[0])
    st = iv.realizable_states(inst)[5]
    txt = iv.describe_state(inst, st)
    raw = json.load(open(V2[0]))
    assert "gold_value" not in txt and "gold_licenses" not in txt
    # no record VALUE strings appear
    rec = list(raw["records"].values())[0]
    for v in json.dumps(rec).split('"'):
        if len(v) > 25 and v in txt:
            raise AssertionError("record value leaked into state description: %r" % v[:40])


# ---------------------------------------------------------------------------
# aggregation + DNF compilation
# ---------------------------------------------------------------------------

def test_aggregate_policies():
    assert iv.aggregate([True, True, True], "unanimous") is True
    assert iv.aggregate([True, True, False], "unanimous") is False
    assert iv.aggregate([True, True, False], "majority") is True
    assert iv.aggregate([True, False, False], "majority") is False
    assert iv.aggregate([True, True, True, False, False], "k_of_n", k=3) is True
    assert iv.aggregate([], "unanimous") is False


def test_compile_routes_recovers_gold_from_perfect_labels():
    """With gold-consistent labels the compiler must return exactly the gold
    routes (upward-closed, minimal) -- the mechanical half is correct."""
    inst = load_instance(V2[0])
    states = iv.realizable_states(inst)
    for c in inst.claims:
        gold = [sorted(r) for r in inst.gold_licenses[c.id]]
        accepted = {st for st in states if any(set(r) <= st for r in gold)}
        got = iv.compile_routes_exact(accepted, states, list(inst.atom_ids))
        assert sorted(got) == sorted(gold), (c.id, got, gold)


def test_compile_routes_is_sound_under_noisy_labels():
    """A single spuriously-accepted state must not create a route that a
    refuting state contradicts (soundness = upward closure)."""
    inst = load_instance(V2[0])
    states = iv.realizable_states(inst)
    c = inst.claims[-1]
    gold = [set(r) for r in inst.gold_licenses[c.id]]
    accepted = {st for st in states if any(r <= st for r in gold)}
    # spuriously accept ONE small state
    small = min((st for st in states if st not in accepted), key=len)
    accepted = accepted | {small}
    got = iv.compile_routes_exact(accepted, states, list(inst.atom_ids))
    for r in got:
        for st in states:
            if set(r) <= st:
                assert st in accepted, "unsound route %s admitted" % r


# ---------------------------------------------------------------------------
# linter
# ---------------------------------------------------------------------------

def test_lint_clean_on_gold_program():
    inst = load_instance(V2[0])
    lic = {cid: [list(r) for r in rs] for cid, rs in inst.gold_licenses.items()}
    findings = iv.lint_program(inst, lic)
    kinds = {f["lint"] for f in findings}
    # generated instances carry a deliberate distractor atom -> that lint is
    # expected; nothing else may fire
    assert kinds <= {"unused_vocabulary_atom"}, findings


def test_lint_catches_each_defect_class():
    inst = load_instance(V2[0])
    gold = {cid: [list(r) for r in rs] for cid, rs in inst.gold_licenses.items()}
    atoms = list(inst.atom_ids)

    routeless = dict(gold); routeless["C2"] = []
    assert any(f["lint"] == "routeless_claim" for f in iv.lint_program(inst, routeless))

    unknown = dict(gold); unknown["C1"] = [["E999"]]
    assert any(f["lint"] == "unknown_atom" for f in iv.lint_program(inst, unknown))

    dominated = dict(gold)
    base = gold["C1"][0]
    dominated["C1"] = [list(base), sorted(set(base) | {atoms[-1]})]
    assert any(f["lint"] == "dominated_route" for f in iv.lint_program(inst, dominated))

    incoherent = dict(gold)
    incoherent["C3"] = [[a for a in atoms if a not in set(gold["C1"][0])][:2]]
    assert any(f["lint"] == "order_incoherent" for f in iv.lint_program(inst, incoherent))


# ---------------------------------------------------------------------------
# safe merge + eligibility
# ---------------------------------------------------------------------------

def test_safe_merge_unions_and_drops_dominated():
    a = {"C1": [["E1"]], "C2": [["E1", "E2"]]}
    b = {"C1": [["E1", "E3"]], "C2": [["E1", "E4"]]}
    m = iv.safe_merge([a, b])
    assert set(m["C1"][0]) == {"E1", "E3"}
    assert set(m["C2"][0]) == {"E1", "E2", "E4"}


def test_eligibility_excludes_flagged_claims():
    lic = {"C1": [["E1"]], "C2": [["E1", "E2"]], "C3": [["E1", "E2", "E3"]]}
    lint = [{"lint": "dominated_route", "claim": "C2"}]
    cf = {"necessity_failures": [{"claim": "C3", "route": ["E1"], "literal": "E1"}],
          "completeness_failures": []}
    e = iv.eligibility(None, lic, lint, cf)
    assert e == {"C1": True, "C2": False, "C3": False}


# ---------------------------------------------------------------------------
# end-to-end with the scripted labeler
# ---------------------------------------------------------------------------

def test_state_first_induction_end_to_end_recovers_gold():
    inst = load_instance(V2[0])
    lic = iv.induce_state_first(inst, _GoldLabeler(inst), policy="unanimous", draws=1)
    for cid, routes in inst.gold_licenses.items():
        assert sorted(sorted(r) for r in routes) == sorted(lic[cid]), (cid, lic[cid])
    findings = iv.lint_program(inst, lic)
    assert {f["lint"] for f in findings} <= {"unused_vocabulary_atom"}


# ---------------------------------------------------------------------------
# W3 freeze gate findings (analysis/w3_freeze.py)
# ---------------------------------------------------------------------------

def test_route_width_is_not_capped_below_the_corpus_maximum():
    """compile_routes_exact defaulted to max_width=6, but 189 gold routes in
    ObligationBench are 7 atoms wide. Under that cap those claims compiled
    to NOTHING, so an induced program could not license them -- which would
    have surfaced in W2b as an induction-quality result rather than a
    configuration cap."""
    import itertools
    atoms = list("abcdefg")                     # 7 atoms, as in the corpus
    full = frozenset(atoms)
    # every subset is realizable and only the FULL state is accepted, so no
    # proper subset of the vocabulary is a sound route: width 7 is forced
    states = [frozenset(c) for n in range(len(atoms) + 1)
              for c in itertools.combinations(atoms, n)]
    accepted = {full}
    routes = iv.compile_routes_exact(accepted, states, atoms)
    assert routes == [sorted(atoms)], routes
    # the old default silently produced nothing at all
    assert iv.compile_routes_exact(accepted, states, atoms, max_width=6) == []


def test_width_cap_still_applies_to_large_vocabularies():
    """The cap is a real tradeoff above _UNCAPPED_VOCAB; it must not vanish."""
    atoms = ["e%d" % i for i in range(_big())]
    assert len(atoms) > iv._UNCAPPED_VOCAB
    states = [frozenset(atoms)]
    routes = iv.compile_routes_exact({frozenset(atoms)}, states, atoms)
    assert all(len(r) <= 6 for r in routes)


def _big():
    return iv._UNCAPPED_VOCAB + 1


def test_unused_vocabulary_lint_does_not_gate_guidance():
    """ObligationBench deliberately seeds distractor atoms, so
    `unused_vocabulary_atom` fires on gold for 472 of 480 balanced instances.
    It names an atom rather than a claim, which is the only reason guidance
    is not switched off corpus-wide -- adding a `claim` key to that finding
    would silently disable guidance everywhere."""
    lint = [{"lint": "unused_vocabulary_atom", "atom": "E8"}]
    elig = iv.eligibility(_inst(), {"C1": [["E1"]]}, lint, {})
    assert elig["C1"] is True


def test_unresolvable_atom_lint_does_gate_guidance():
    """The evidence-unavailable stratum: an atom with no resolver can never
    be established, so refusing to guide toward claims that need it is the
    fail-closed behaviour, and must keep working."""
    lint = [{"lint": "unresolvable_atom_in_route", "claim": "C1", "atom": "E4"}]
    elig = iv.eligibility(_inst(), {"C1": [["E1", "E4"]]}, lint, {})
    assert elig["C1"] is False


def _inst():
    import glob
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    p = sorted(glob.glob(os.path.join(
        here, "bench", "instances_v2", "*", "heldout_balanced", "*.json")))[0]
    return iv.load_instance(p)
