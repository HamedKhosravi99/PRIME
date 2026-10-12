"""W2b plumbing: the W3 conservative induction stack must be REACHABLE from
the runner.

experiments/analysis/induction_v5.py was built, tested and frozen, but nothing in
src/prime_agents/ ever imported it: `--license-source induced` always called the
one-shot induce_program in arms.py. A W2b run launched in that state would
have consumed GPU measuring the legacy induction while the paper described
the conservative one, and the failure would have been invisible -- both
paths produce a valid Program and a normal-looking episode.
"""
from __future__ import annotations

import glob
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "src", "prime_agents"), os.path.join(_P, "experiments", "analysis")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import arms  # noqa: E402
import run_batch  # noqa: E402
import run_episode as re_mod  # noqa: E402


def _instance():
    return sorted(glob.glob(os.path.join(
        _P, "data", "obligationbench", "security", "dev", "*.json")))[0]


def test_induction_method_is_threaded_through_every_layer():
    import inspect
    for fn in (re_mod.run_episode, arms.run_arm):
        assert "induction_method" in inspect.signature(fn).parameters, fn


def test_default_is_the_published_oneshot_condition():
    """Switching the default would silently change every published induced
    number, so the new path must be opt-in."""
    import inspect
    for fn in (re_mod.run_episode, arms.run_arm):
        p = inspect.signature(fn).parameters["induction_method"]
        assert p.default == "oneshot", fn


def test_state_first_actually_calls_the_w3_stack(monkeypatch, tmp_path):
    """The point of the wiring: prove induce_state_first is reached, not
    that an episode merely completes."""
    import induction_v5 as iv
    called = {}
    real = iv.induce_state_first

    def spy(instance, llm, policy="unanimous", draws=3):
        called["hit"] = (policy, draws)
        return real(instance, llm, policy=policy, draws=draws)

    monkeypatch.setattr(iv, "induce_state_first", spy)
    rc = run_batch.main([
        "--dry-run", "--arm", "prime", "--license-source", "induced",
        "--induction-method", "state_first",
        "--instances", _instance(), "--out", str(tmp_path)])
    assert rc == 0
    assert called.get("hit"), "state_first did not reach induce_state_first"


def test_oneshot_does_not_call_the_w3_stack(monkeypatch, tmp_path):
    import induction_v5 as iv
    called = {}
    monkeypatch.setattr(iv, "induce_state_first",
                        lambda *a, **k: called.setdefault("hit", True) or {})
    run_batch.main([
        "--dry-run", "--arm", "prime", "--license-source", "induced",
        "--instances", _instance(), "--out", str(tmp_path)])
    assert not called, "the default path must not run the W3 stack"


def test_episode_records_the_method_and_eligibility(tmp_path):
    import json
    run_batch.main([
        "--dry-run", "--arm", "prime", "--license-source", "induced",
        "--induction-method", "state_first",
        "--instances", _instance(), "--out", str(tmp_path)])
    e = json.load(open(os.path.join(str(tmp_path), "summary.json")))["episodes"][0]
    assert e["induction_method"] == "state_first"
    assert "induction_eligibility" in e, \
        "the per-claim guidance gate must be recorded, not recomputed later"


# ---------------------------------------------------------------------------
# prime_guided_gated: the induced-condition primary (docs/EXECUTION_PLAN.md,
# Part R, P8)
# ---------------------------------------------------------------------------

def _run(tmp_path, arm, method="state_first"):
    import json
    run_batch.main([
        "--dry-run", "--arm", arm, "--license-source", "induced",
        "--induction-method", method,
        "--instances", _instance(), "--out", str(tmp_path)])
    return json.load(open(os.path.join(str(tmp_path), "summary.json")))["episodes"][0]


def test_gated_arm_is_registered_and_runnable():
    assert "prime_guided_gated" in arms.ARMS
    # run_batch used to keep its OWN copy of this list, which drifted
    assert run_batch.arms_mod.ARMS is arms.ARMS


def test_gated_arm_targets_only_eligible_claims(monkeypatch, tmp_path):
    """The mechanism, tested directly rather than through a fixture whose
    induced program happens to license nothing: with a program that DOES
    authorize the top claim, guidance must target only the claims the W3
    gates passed. The fifth wave's failure mode was guiding an induced
    program toward a claim whose routes never passed counterfactual
    validation, which turned 0% OFAR into 100%.
    """
    from prime_runtime.schema import load_instance
    inst = load_instance(_instance())
    gold = {c.id: [list(r) for r in inst.gold_licenses.get(c.id, [])]
            for c in inst.claims}
    program = arms._build_validated_program(inst, gold)
    top = max(inst.claims, key=lambda c: c.level).id
    elig = {c.id: (c.id != top) for c in inst.claims}   # top claim BLOCKED

    seen = {}

    def fake(instance, llm, policy="unanimous", draws=3):
        return program, elig

    monkeypatch.setattr(arms, "induce_program_v5", fake)
    real_ceiling = arms.prime_guidance.reachable_ceiling

    def spy_ceiling(prog, ledger, resolvers, claim_levels, budget):
        seen.setdefault("levels", dict(claim_levels))
        return real_ceiling(prog, ledger, resolvers, claim_levels, budget)

    monkeypatch.setattr(arms.prime_guidance, "reachable_ceiling", spy_ceiling)
    _run(tmp_path, "prime_guided_gated")
    assert seen.get("levels") is not None, "guidance never computed a ceiling"
    assert top not in seen["levels"], \
        "gated arm targeted the claim the W3 gates rejected"
    assert set(seen["levels"]) == {c for c in elig if elig[c]}


def test_ungated_guided_arm_targets_every_claim(monkeypatch, tmp_path):
    """The contrast that makes the gate measurable: the ungated arm is kept
    as the amplification diagnostic, not deleted."""
    from prime_runtime.schema import load_instance
    inst = load_instance(_instance())
    gold = {c.id: [list(r) for r in inst.gold_licenses.get(c.id, [])]
            for c in inst.claims}
    program = arms._build_validated_program(inst, gold)
    top = max(inst.claims, key=lambda c: c.level).id
    elig = {c.id: (c.id != top) for c in inst.claims}
    seen = {}
    monkeypatch.setattr(arms, "induce_program_v5",
                        lambda *a, **k: (program, elig))
    real_ceiling = arms.prime_guidance.reachable_ceiling
    monkeypatch.setattr(arms.prime_guidance, "reachable_ceiling",
                        lambda p, l, r, cl, b: (seen.setdefault("levels", dict(cl)),
                                                real_ceiling(p, l, r, cl, b))[1])
    _run(tmp_path, "prime_guided")
    assert top in seen.get("levels", {}), \
        "ungated arm must still target the top claim"


def test_gating_only_applies_under_state_first_induction(tmp_path):
    """One-shot induction has no eligibility map; the gated arm must not
    silently behave as if everything were eligible."""
    e = _run(tmp_path, "prime_guided_gated", method="oneshot")
    assert e["induction_eligibility"] is None


# ---------------------------------------------------------------------------
# Induce once per template family (docs/EXECUTION_PLAN.md, Part 6b, FB-05)
# ---------------------------------------------------------------------------

def test_induction_is_cached_per_family_not_per_instance(tmp_path, monkeypatch):
    """Per-instance induction repeats the same elicitation 4-15 times: within
    a family the prompt and claim texts are identical and only the record
    LOCATOR differs. At 1,152 model calls per induction that is ~154
    GPU-hours instead of ~6 for W2b."""
    import glob
    import json
    import os
    from prime_runtime.schema import load_instance
    import induction_v5 as iv

    by_fam = {}
    for p in sorted(glob.glob(os.path.join(
            _P, "data", "obligationbench", "security", "heldout_balanced", "*.json"))):
        f = json.loads(open(p).read())
        fam = json.loads(f["notes_for_annotators"])["family"]
        by_fam.setdefault(fam, []).append(p)
    fam, paths = next((f, ps) for f, ps in by_fam.items() if len(ps) >= 3)

    calls = {"n": 0}

    def fake(instance, llm, policy="unanimous", draws=3):
        calls["n"] += 1
        return {c.id: [[instance.atom_ids[0]]] for c in instance.claims}

    monkeypatch.setattr(iv, "induce_state_first", fake)
    monkeypatch.setattr(arms, "_INDUCTION_CACHE", {})
    monkeypatch.setattr(arms, "_INDUCTION_CACHE_DIR", str(tmp_path))
    llm = object()
    for p in paths[:3]:
        arms.induce_program_v5(load_instance(p), llm)
    assert calls["n"] == 1, \
        "induced %d times for one family; expected 1" % calls["n"]


def test_different_families_induce_separately(tmp_path, monkeypatch):
    import glob
    import json
    import os
    from prime_runtime.schema import load_instance
    import induction_v5 as iv

    seen, paths = {}, []
    for p in sorted(glob.glob(os.path.join(
            _P, "data", "obligationbench", "security", "heldout_balanced", "*.json"))):
        fam = json.loads(json.loads(open(p).read())["notes_for_annotators"]
                         if False else open(p).read())
        fam = json.loads(json.loads(open(p).read())["notes_for_annotators"])["family"]
        if fam not in seen:
            seen[fam] = p
            paths.append(p)
        if len(paths) == 3:
            break

    calls = {"n": 0}
    monkeypatch.setattr(iv, "induce_state_first",
                        lambda inst, llm, policy="unanimous", draws=3:
                        (calls.__setitem__("n", calls["n"] + 1),
                         {c.id: [[inst.atom_ids[0]]] for c in inst.claims})[1])
    monkeypatch.setattr(arms, "_INDUCTION_CACHE", {})
    monkeypatch.setattr(arms, "_INDUCTION_CACHE_DIR", str(tmp_path))
    llm = object()
    for p in paths:
        arms.induce_program_v5(load_instance(p), llm)
    assert calls["n"] == 3, "distinct families must induce separately"


def test_induction_cache_survives_across_processes(tmp_path, monkeypatch):
    """Each ARM is a separate run_batch process, so an in-memory cache alone
    still re-induces every family once per arm -- 3x the model calls for
    programs that are identical by construction (19 GPU-hours instead of 6)."""
    import glob
    import os
    from prime_runtime.schema import load_instance
    import induction_v5 as iv

    monkeypatch.setattr(arms, "_INDUCTION_CACHE_DIR", str(tmp_path))
    inst = load_instance(sorted(glob.glob(os.path.join(
        _P, "data", "obligationbench", "security", "heldout_balanced", "*.json")))[0])

    calls = {"n": 0}

    def fake(instance, llm, policy="unanimous", draws=3):
        calls["n"] += 1
        return {c.id: [[instance.atom_ids[0]]] for c in instance.claims}

    monkeypatch.setattr(iv, "induce_state_first", fake)

    class LLM:
        model = "qwen2.5-32b"

    monkeypatch.setattr(arms, "_INDUCTION_CACHE", {})
    arms.induce_program_v5(inst, LLM())
    assert calls["n"] == 1
    assert os.listdir(str(tmp_path)), "nothing written to the cache"

    # a FRESH process: in-memory cache empty, disk cache warm
    monkeypatch.setattr(arms, "_INDUCTION_CACHE", {})
    arms.induce_program_v5(inst, LLM())
    assert calls["n"] == 1, "re-induced despite a warm disk cache"


def test_cache_is_keyed_by_model_and_induction_config(tmp_path, monkeypatch):
    """A different model or policy must never silently reuse another's
    induced programs."""
    import glob
    import os
    from prime_runtime.schema import load_instance
    import induction_v5 as iv

    monkeypatch.setattr(arms, "_INDUCTION_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(arms, "_INDUCTION_CACHE", {})
    inst = load_instance(sorted(glob.glob(os.path.join(
        _P, "data", "obligationbench", "security", "heldout_balanced", "*.json")))[0])
    calls = {"n": 0}
    monkeypatch.setattr(iv, "induce_state_first",
                        lambda i, l, policy="unanimous", draws=3:
                        (calls.__setitem__("n", calls["n"] + 1),
                         {c.id: [[i.atom_ids[0]]] for c in i.claims})[1])

    class A:
        model = "qwen2.5-32b"

    class B:
        model = "qwen2.5-7b"

    arms.induce_program_v5(inst, A())
    arms.induce_program_v5(inst, B())
    assert calls["n"] == 2, "different models shared a cache entry"
    monkeypatch.setattr(arms, "_INDUCTION_CACHE", {})
    arms.induce_program_v5(inst, A(), policy="majority")
    assert calls["n"] == 3, "different policies shared a cache entry"


def test_an_unnamed_model_is_never_cached(tmp_path, monkeypatch):
    """Structural guard against the contamination that already happened:
    test and stub clients have no .model, and caching their output hands
    fabricated programs to a later confirmatory run."""
    import glob
    import os
    from prime_runtime.schema import load_instance
    import induction_v5 as iv

    monkeypatch.setattr(arms, "_INDUCTION_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(arms, "_INDUCTION_CACHE", {})
    monkeypatch.setattr(iv, "induce_state_first",
                        lambda i, l, policy="unanimous", draws=3:
                        {c.id: [] for c in i.claims})
    inst = load_instance(sorted(glob.glob(os.path.join(
        _P, "data", "obligationbench", "security", "heldout_balanced", "*.json")))[0])
    arms.induce_program_v5(inst, object())          # no .model attribute
    assert not os.listdir(str(tmp_path)), \
        "an unnamed model's induction reached the on-disk cache"
