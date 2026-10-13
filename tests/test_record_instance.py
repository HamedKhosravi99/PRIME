"""Structured-record capability tests (Tier-M infra): the generic
``inspect_record`` resolver, the ``records`` schema field, and the
end-to-end pipeline on the synthetic smoke instance
``data/pilot/instances/_smoke_record/rec-0001.json`` (design rationale in
src/prime_runtime/predicates.py).

Covers: nested field-path resolution (dict + list mixed paths, not-found
cases); goldledger firing record_predicate atoms through the real
ToolExecutor (mirrors src/suffice/construction/scaffold_repo.py's verifier); a full scripted
prime-arm episode end to end (build_dry_run_script -> run_arm -> gold
replay); the exact-param-attestation-trap regression guard (record atoms
must declare ONLY stable identifying params, never the predicate/expected
value); schema.py's defensive record_id check; and src/suffice/validate.py +
src/suffice/construction/scaffold_repo.py as subprocesses on the smoke instance.
"""

import json
import os
import subprocess
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PILOT = os.path.dirname(_TESTS)
_RUNNER = os.path.join(_PILOT, "src", "prime_agents")
_BENCH = os.path.join(_PILOT, "src", "suffice", "construction")
for _p in (_PILOT, os.path.join(_PILOT, "src"), _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from prime_runtime import goldledger  # noqa: E402
from prime_runtime.schema import SchemaError, load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

REC_0001 = os.path.join(_PILOT, "data", "pilot", "instances", "_smoke_record", "rec-0001.json")
VALIDATE_PY = os.path.join(_PILOT, "src", "suffice", "validate.py")
SCAFFOLD_PY = os.path.join(_BENCH, "scaffold_repo.py")


@pytest.fixture
def rec_instance():
    return load_instance(REC_0001)


# ---------------------------------------------------------------------------
# schema.py: the ``records`` field and its defensive validation
# ---------------------------------------------------------------------------


def test_instance_records_populated(rec_instance):
    assert set(rec_instance.records.keys()) == {"SMOKE-VULN-0001"}
    rec = rec_instance.records["SMOKE-VULN-0001"]
    assert rec["package"]["name"] == "acme-widget"
    assert rec["severity"]["cvss_score"] == 9.8


def test_non_record_instance_has_empty_records(toy_instance):
    assert dict(toy_instance.records) == {}


def test_schema_rejects_inspect_record_atom_with_undeclared_record_id(tmp_path):
    with open(REC_0001, encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["atoms"][0]["resolver"]["params"]["record_id"] = "NO-SUCH-RECORD"
    p = tmp_path / "bad_record.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="not declared in instance records"):
        load_instance(str(p))


# ---------------------------------------------------------------------------
# src/prime_agents/arms.py: _resolve_field_path (nested dict/list dotted paths)
# ---------------------------------------------------------------------------


def test_resolve_field_path_nested_dict_and_list():
    record = {
        "a": {"b": [{"c": 1}, {"c": 2}]},
        "top": "value",
    }
    assert arms_mod._resolve_field_path(record, "top") == ("value", True)
    assert arms_mod._resolve_field_path(record, "a.b.0.c") == (1, True)
    assert arms_mod._resolve_field_path(record, "a.b.1.c") == (2, True)
    assert arms_mod._resolve_field_path(record, "a.b.-1.c") == (2, True)
    assert arms_mod._resolve_field_path(record, "") == (record, True)


def test_resolve_field_path_not_found():
    record = {"a": {"b": 1}}
    assert arms_mod._resolve_field_path(record, "a.missing") == (None, False)
    assert arms_mod._resolve_field_path(record, "a.b.c") == (None, False)  # b is not a container
    assert arms_mod._resolve_field_path(record, "a.b.99") == (None, False)


def test_tool_inspect_record_success_and_errors(rec_instance):
    ex = arms_mod.ToolExecutor(rec_instance)
    obs = ex.execute("inspect_record", {"record_id": "SMOKE-VULN-0001", "field_path": "severity.level"})
    assert obs == {
        "record_id": "SMOKE-VULN-0001", "field_path": "severity.level", "value": "CRITICAL",
    }
    missing_record = ex.execute("inspect_record", {"record_id": "NOPE", "field_path": "x"})
    assert missing_record["error"] == "record not found"
    missing_field = ex.execute(
        "inspect_record", {"record_id": "SMOKE-VULN-0001", "field_path": "no.such.field"}
    )
    assert missing_field["error"] == "field not found"


# ---------------------------------------------------------------------------
# goldledger: record_predicate atoms fire through the REAL ToolExecutor,
# reproducing src/suffice/construction/scaffold_repo.py's verifier logic directly.
# ---------------------------------------------------------------------------


def test_every_atom_resolves_to_its_declared_gold_value(rec_instance):
    from prime_runtime.ledger import Ledger

    raw_gold = {a["id"]: a["gold_value"] for a in json.load(open(REC_0001))["atoms"]}
    ex = arms_mod.ToolExecutor(rec_instance)
    ledger = Ledger(rec_instance.atom_ids, rec_instance.phi_patterns)
    for atom in rec_instance.atoms:
        params = dict(atom.resolver.params)
        obs = ex.execute(atom.resolver.tool, params)
        applied = goldledger.delta(rec_instance, ledger, atom.resolver.tool, params, obs, 1)
        got = next((a["value"] for a in applied if a["atom"] == atom.id), None)
        assert got == raw_gold[atom.id], (atom.id, obs, applied)


def test_a_real_models_natural_call_still_fires_the_atom(rec_instance):
    """Exact-param-attestation-trap regression check (the bug this design was
    built to avoid): a tool call carrying ONLY the two stable identifying
    params a real model would naturally supply -- with no knowledge of the
    predicate type or expected value baked into the atom -- still fires the
    atom. If this ever required extra params (predicate/expected/threshold)
    the atom would silently never fire against real model tool calls."""
    ex = arms_mod.ToolExecutor(rec_instance)
    from prime_runtime.ledger import Ledger

    ledger = Ledger(rec_instance.atom_ids, rec_instance.phi_patterns)
    # A model naturally supplies record_id + field_path only -- it has no way
    # to know (and is never told) the predicate type or expected threshold.
    natural_call_params = {"record_id": "SMOKE-VULN-0001", "field_path": "severity.cvss_score"}
    obs = ex.execute("inspect_record", natural_call_params)
    applied = goldledger.delta(rec_instance, ledger, "inspect_record", natural_call_params, obs, 1)
    assert any(a["atom"] == "E3" and a["value"] == "T" for a in applied)


def test_record_atoms_declare_only_stable_identifying_params(rec_instance):
    """Grammar-level guard against the trap re-appearing: every inspect_record
    atom's resolver.params must be exactly {record_id, field_path} -- never
    a predicate type, expected value, or threshold (those live only in the
    atom's attest rule, which the calling model never has to reproduce)."""
    for atom in rec_instance.atoms:
        if atom.resolver is not None and atom.resolver.tool == "inspect_record":
            assert set(atom.resolver.params.keys()) == {"record_id", "field_path"}, atom.id


# ---------------------------------------------------------------------------
# End-to-end scripted episode (mirrors test_dry_run.py / test_new_arms.py)
# ---------------------------------------------------------------------------


def test_prime_gold_episode_end_to_end(rec_instance, tmp_path):
    script = llm_mod.build_dry_run_script(rec_instance)
    client = llm_mod.ScriptedLLM(script)
    trace_path = str(tmp_path / "rec-0001.jsonl")
    with TraceLogger(trace_path) as trace:
        result = arms_mod.run_arm("prime", rec_instance, client, trace, license_source="gold")
    assert result.error == ""
    assert [e["claim"] for e in result.emitted] == ["C3"]

    replay = goldledger.reconstruct(trace_path, rec_instance)
    emission = result.emitted[0]
    assert replay.complete_auth("C3", emission["step"]) == 1
    assert replay.completion_at_final() == 1.0


def test_direct_arm_is_unsupported_on_the_smoke_instance(rec_instance, tmp_path):
    """The direct arm answers with no evidence at all -- gold scoring must
    flag it, exactly as for the existing domains (test_dry_run.py)."""
    client = llm_mod.ScriptedLLM(llm_mod.build_dry_run_script(rec_instance))
    trace_path = str(tmp_path / "rec-0001-direct.jsonl")
    with TraceLogger(trace_path) as trace:
        result = arms_mod.run_arm("direct", rec_instance, client, trace, license_source="gold")
    assert [e["claim"] for e in result.emitted] == ["C3"]
    replay = goldledger.reconstruct(trace_path, rec_instance)
    emission = result.emitted[0]
    assert replay.complete_auth("C3", emission["step"]) == 0


# ---------------------------------------------------------------------------
# src/suffice/validate.py and src/suffice/construction/scaffold_repo.py as subprocesses (the real
# offline authoring tools) on the smoke instance -- regression protection
# for the two extended, standalone src/suffice/construction/ scripts.
# ---------------------------------------------------------------------------


def test_bench_validate_passes_on_smoke_instance():
    proc = subprocess.run(
        [sys.executable, VALIDATE_PY, REC_0001],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_bench_scaffold_verify_passes_on_smoke_instance():
    proc = subprocess.run(
        [sys.executable, SCAFFOLD_PY, "verify", REC_0001],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFY: PASS" in proc.stdout
    assert "MISMATCH" not in proc.stdout
