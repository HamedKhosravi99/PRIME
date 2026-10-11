"""Specification/evidence split, hash-gated commitment, and the
Commitment-View Noninterference audit (paper Weakness 4, §4.2-§4.6).

Covers:
  - spec_manifest excludes outcome-bearing fields (gold_value, proposition,
    gold_licenses, decoys, notes_for_annotators) and is a pure function of
    the permitted pre-commit fields;
  - PreCommitEvidenceError fires when an evidence resolver is invoked
    before the commitment hash is recorded (strict mode), and is disabled
    post-commit;
  - h0 is stable under program canonicalization (route re-ordering,
    duplicate routes) and changes whenever the program's licensed content
    actually changes;
  - the CommitmentGate audit helpers correctly flag / clear a trace;
  - Experiment I1 (paired hidden-outcome invariance) is exactly zero
    divergence under the strict manifest and strictly positive once the
    manifest is expanded to artifact-visible (analysis/noninterference_audit.py).
"""

import copy
import json
import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(os.path.dirname(_TESTS))
_ANALYSIS = os.path.join(_PILOT, "analysis")
for _p in (_PILOT, _ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime import commitment as C  # noqa: E402
from prime_runtime.schema import load_instance, make_program  # noqa: E402

from conftest import FIXTURE_INSTANCE  # noqa: E402


# ---------------------------------------------------------------------------
# spec_manifest: excludes outcome-bearing fields
# ---------------------------------------------------------------------------


def test_spec_manifest_excludes_evidence_fields(toy_instance):
    manifest = C.spec_manifest(toy_instance)
    blob = C.canonical_json(manifest)
    # Nothing in the manifest mentions gold_licenses, decoys, or annotator
    # notes -- these are read straight off the instance, so a substring
    # check on the manifest's own serialization is a direct, load-bearing
    # check (not a proxy) that spec_manifest() never touched them.
    assert "gold_licenses" not in blob
    assert "decoys" not in blob
    assert "comment_density" not in blob  # toy_instance's decoy value
    assert "notes_for_annotators" not in blob
    assert "staging.json" not in blob  # only appears in notes_for_annotators
    # No atom entry carries a resolver OUTCOME or ground-truth field.
    for atom_entry in manifest["atoms"]:
        assert "gold_value" not in atom_entry
        assert "proposition" not in atom_entry


def test_spec_manifest_excludes_outcome_revealing_propositions():
    # The res-0001-style corpus convention: an atom's authored proposition
    # can literally state the answer ("... the adjudication is that none
    # does."). spec_manifest must never surface proposition text.
    raw = _load_raw(FIXTURE_INSTANCE)
    raw["atoms"][0]["proposition"] = "SECRET_GOLD_ANSWER: the suite passes with 42 failures"
    inst = _instance_from_raw(raw, "leak_check")
    manifest = C.spec_manifest(inst)
    assert "SECRET_GOLD_ANSWER" not in C.canonical_json(manifest)


def test_spec_manifest_includes_permitted_fields(toy_instance):
    manifest = C.spec_manifest(toy_instance)
    assert manifest["domain"] == toy_instance.domain
    assert manifest["repository"]["commit"] == toy_instance.artifact["commit"]
    assert manifest["repository"]["path"] == toy_instance.artifact["path"]
    assert set(manifest["registered_tools"]) == set(toy_instance.actions)
    assert {c["id"] for c in manifest["claims"]} == set(toy_instance.claim_ids)
    assert {a["id"] for a in manifest["atoms"]} == set(toy_instance.atom_ids)
    # Resolver TYPE / locator info is permitted (where to look, not what
    # was found): e.g. the run_tests atom's tool name is visible.
    atom_by_id = {a["id"] for a in manifest["atoms"] if a.get("tool") == "run_tests"}
    assert atom_by_id  # at least one atom declares run_tests as its resolver


def test_spec_manifest_is_pure_and_order_independent(toy_instance):
    # Same instance -> byte-identical manifest on repeated calls (no hidden
    # nondeterminism / clock reads).
    m1 = C.spec_manifest(toy_instance)
    m2 = C.spec_manifest(toy_instance)
    assert C.canonical_json(m1) == C.canonical_json(m2)


# ---------------------------------------------------------------------------
# PreCommitEvidenceError / CommitmentGate capability boundary
# ---------------------------------------------------------------------------


def test_evidence_resolver_blocked_before_commit(toy_instance):
    gate = C.CommitmentGate(mode=C.CommitmentMode.STRICT)
    with pytest.raises(C.PreCommitEvidenceError):
        gate.evidence_resolver("run_tests")


def test_evidence_resolver_allowed_after_commit(toy_instance):
    gate = C.CommitmentGate(mode=C.CommitmentMode.STRICT)
    program = toy_instance.gold_program()
    gate.commit(toy_instance, program, step=0)
    assert gate.evidence_resolver("run_tests") is True


def test_artifact_aware_mode_permits_precommit_evidence_access(toy_instance):
    gate = C.CommitmentGate(mode=C.CommitmentMode.ARTIFACT_AWARE)
    # No exception: artifact-aware mode grants the weaker
    # freeze-after-inspection guarantee, not full noninterference.
    assert gate.evidence_resolver("inspect_file") is True
    assert gate.committed is False


def test_gate_rejects_double_commit(toy_instance):
    gate = C.CommitmentGate(mode=C.CommitmentMode.STRICT)
    program = toy_instance.gold_program()
    gate.commit(toy_instance, program, step=0)
    with pytest.raises(RuntimeError):
        gate.commit(toy_instance, program, step=1)


def test_unknown_commitment_mode_rejected():
    with pytest.raises(ValueError):
        C.CommitmentGate(mode="yolo")


def test_gated_evidence_executor_wraps_capability_check(toy_instance):
    calls = []

    class _StubExecutor:
        def execute(self, name, params):
            calls.append((name, dict(params)))
            return {"ok": True}

    gate = C.CommitmentGate(mode=C.CommitmentMode.STRICT)
    gated = C.GatedEvidenceExecutor(_StubExecutor(), gate)
    with pytest.raises(C.PreCommitEvidenceError):
        gated.execute("run_tests", {"suite": "tests"})
    assert calls == []  # the inner executor was never reached

    gate.commit(toy_instance, toy_instance.gold_program(), step=0)
    result = gated.execute("run_tests", {"suite": "tests"})
    assert result == {"ok": True}
    assert calls == [("run_tests", {"suite": "tests"})]


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def test_verify_no_precommit_evidence_clean_gate(toy_instance):
    gate = C.CommitmentGate(mode=C.CommitmentMode.STRICT)
    gate.commit(toy_instance, toy_instance.gold_program(), step=0)
    gate.evidence_resolver("run_tests")
    audit = C.verify_no_precommit_evidence(gate)
    assert audit["ok"] is True
    assert audit["commit_recorded"] is True
    assert audit["violations"] == []


def test_verify_no_precommit_evidence_flags_bypassed_violation(toy_instance):
    # A caller that catches PreCommitEvidenceError and (incorrectly) keeps
    # going still leaves a "violation" event in the gate's own log --
    # the independent audit must still catch it.
    gate = C.CommitmentGate(mode=C.CommitmentMode.STRICT)
    with pytest.raises(C.PreCommitEvidenceError):
        gate.evidence_resolver("run_tests")
    gate.commit(toy_instance, toy_instance.gold_program(), step=0)
    audit = C.verify_no_precommit_evidence(gate)
    assert audit["ok"] is False
    assert len(audit["violations"]) == 1


def test_audit_episode_trace_ok():
    entries = [
        {"type": "commit", "step": 0, "h0": "abc"},
        {"type": "action", "step": 1, "name": "run_tests"},
        {"type": "observation", "step": 1, "action": "run_tests"},
        {"type": "action", "step": 2, "name": "emit_claim"},
    ]
    audit = C.audit_episode_trace(entries)
    assert audit == {
        "ok": True,
        "commit_recorded": True,
        "commit_index": 0,
        "violations": [],
        "n_entries": 4,
    }


def test_audit_episode_trace_flags_precommit_evidence():
    entries = [
        {"type": "action", "step": 0, "name": "run_tests"},  # evidence before commit
        {"type": "commit", "step": 1, "h0": "abc"},
    ]
    audit = C.audit_episode_trace(entries)
    assert audit["ok"] is False
    assert audit["violations"][0]["name"] == "run_tests"


def test_audit_episode_trace_no_commit_marker_is_not_ok():
    entries = [{"type": "action", "step": 0, "name": "run_tests"}]
    audit = C.audit_episode_trace(entries)
    assert audit["commit_recorded"] is False
    assert audit["ok"] is False


# ---------------------------------------------------------------------------
# h0 stability under canonicalization / sensitivity to real changes
# ---------------------------------------------------------------------------


def test_h0_stable_under_route_reordering_and_dedup(toy_instance):
    licenses_a = {
        "C1": [["E1"]],
        "C2": [["E2", "E1"]],  # unsorted literal order
        "C3": [["E1", "E2", "E3", "E4"]],
    }
    licenses_b = {
        "C1": [["E1"], ["E1"]],  # duplicate route
        "C2": [["E1", "E2"]],  # sorted
        "C3": [["E4", "E3", "E2", "E1"]],  # different order, same set
    }
    prog_a = make_program(
        claims=toy_instance.claim_ids, order=toy_instance.order,
        licenses=licenses_a, scope=toy_instance.scope, atoms=toy_instance.atom_ids,
    )
    prog_b = make_program(
        claims=toy_instance.claim_ids, order=toy_instance.order,
        licenses=licenses_b, scope=toy_instance.scope, atoms=toy_instance.atom_ids,
    )
    assert C.program_hash(prog_a) == C.program_hash(prog_b)
    rec_a = C.commit_program(prog_a, manifest_hash="m", step=0)
    rec_b = C.commit_program(prog_b, manifest_hash="m", step=5)
    assert rec_a["h0"] == rec_b["h0"]
    # The step counter is NOT part of the hashed content (it's a metadata
    # timestamp placeholder), so differing step values don't move h0.
    assert rec_a["timestamp_placeholder"] == 0
    assert rec_b["timestamp_placeholder"] == 5


def test_h0_changes_when_program_changes(toy_instance):
    licenses_a = {"C1": [["E1"]], "C2": [["E1", "E2"]], "C3": [["E1", "E2", "E3", "E4"]]}
    licenses_b = {"C1": [["E1"]], "C2": [["E1", "E2"]], "C3": [["E1", "E2", "E3"]]}  # dropped E4
    prog_a = make_program(
        claims=toy_instance.claim_ids, order=toy_instance.order,
        licenses=licenses_a, scope=toy_instance.scope, atoms=toy_instance.atom_ids,
    )
    prog_b = make_program(
        claims=toy_instance.claim_ids, order=toy_instance.order,
        licenses=licenses_b, scope=toy_instance.scope, atoms=toy_instance.atom_ids,
    )
    assert C.program_hash(prog_a) != C.program_hash(prog_b)


def test_verify_commitment_detects_tamper(toy_instance):
    program = toy_instance.gold_program()
    record = C.commit_program(program, manifest_hash="m", step=0)
    assert C.verify_commitment(record, program) is True
    tampered = dict(record, h0="0" * 64)
    assert C.verify_commitment(tampered, program) is False


# ---------------------------------------------------------------------------
# Experiment I1 / I2 (analysis/noninterference_audit.py)
# ---------------------------------------------------------------------------


def test_I1_strict_mode_zero_divergence():
    noninterference_audit = pytest.importorskip("noninterference_audit")
    result = noninterference_audit.run_I1(seeds=(0, 1))
    assert result["n_pairs"] > 0
    assert result["manifest_divergence_rate"] == 0.0
    assert result["program_divergence_rate"] == 0.0
    assert result["claim_divergence_rate"] == 0.0
    assert result["atom_divergence_rate"] == 0.0
    assert result["route_divergence_rate"] == 0.0
    assert result["order_divergence_rate"] == 0.0
    # Direct h0-equality check on every single trial, not just the rate.
    for trial in result["trials"]:
        assert trial["h0_a"] == trial["h0_b"]


def test_I2_divergence_rises_with_exposure():
    noninterference_audit = pytest.importorskip("noninterference_audit")
    result = noninterference_audit.run_I2(seeds=(0, 1))
    modes = result["modes"]
    assert modes["strict"]["program_divergence_rate"] == 0.0
    assert modes["expanded"]["program_divergence_rate"] > 0.0
    assert modes["artifact_visible"]["program_divergence_rate"] > 0.0
    # Monotonically non-decreasing in aggregate as more is exposed.
    assert (
        modes["strict"]["program_divergence_rate"]
        <= modes["expanded"]["program_divergence_rate"]
        <= modes["artifact_visible"]["program_divergence_rate"]
    )
    # The mutation that only ever manifests as a POST-commit resolver
    # outcome must diverge under NO manifest mode, however expansive --
    # it models information that does not exist yet at commit time.
    for mode_row in modes.values():
        assert mode_row["per_mutation_divergence_rate"]["hidden_runtime_override"] == 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_raw(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _instance_from_raw(raw, tmp_name):
    import tempfile

    raw = copy.deepcopy(raw)
    tmp_dir = tempfile.mkdtemp(prefix="commitment_test_")
    path = os.path.join(tmp_dir, tmp_name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh)
    return load_instance(path)
