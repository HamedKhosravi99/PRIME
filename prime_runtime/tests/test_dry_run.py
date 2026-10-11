"""End-to-end dry run: the full runner pipeline on the fixture instance with
the scripted deterministic fake LLM — no API calls (SPEC.md SSRunner)."""

import json
import os
import subprocess
import sys

from conftest import FIXTURE_INSTANCE

_PILOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_BATCH = os.path.join(_PILOT, "runner", "run_batch.py")


def test_dry_run_prime_gold(tmp_path):
    out_dir = str(tmp_path / "results")
    proc = subprocess.run(
        [
            sys.executable,
            RUN_BATCH,
            "--dry-run",
            "--arm",
            "prime",
            "--license-source",
            "gold",
            "--out",
            out_dir,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    summary_path = os.path.join(out_dir, "summary.json")
    assert os.path.exists(summary_path)
    with open(summary_path) as fh:
        summary = json.load(fh)
    assert summary["run"]["arm"] == "prime"
    assert summary["run"]["model"] == "scripted"
    (episode,) = summary["episodes"]
    assert episode["error"] == ""
    assert episode["tool_calls"] <= 20

    # The scripted policy discharges C3's gold route then emits C3, and the
    # method-independent gold replay must agree (CompleteAuth_gold = 1).
    emitted = episode["emitted"]
    assert [e["claim"] for e in emitted] == ["C3"]
    assert emitted[0]["complete_auth_gold"] == 1

    # Trace exists and contains actions, observations, attestations, and the
    # masked emission.
    trace_path = os.path.join(out_dir, "toy-0001.jsonl")
    assert os.path.exists(trace_path)
    types = set()
    with open(trace_path) as fh:
        for line in fh:
            types.add(json.loads(line)["type"])
    assert {"action", "observation", "attestation", "emission"} <= types


def test_dry_run_direct_arm_baseline(tmp_path):
    """The direct arm emits without evidence; gold scoring flags it as an
    unsupported claim (CompleteAuth_gold = 0)."""
    out_dir = str(tmp_path / "results")
    proc = subprocess.run(
        [
            sys.executable,
            RUN_BATCH,
            "--dry-run",
            "--arm",
            "direct",
            "--license-source",
            "gold",
            "--out",
            out_dir,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(os.path.join(out_dir, "summary.json")) as fh:
        summary = json.load(fh)
    (episode,) = summary["episodes"]
    emitted = episode["emitted"]
    assert [e["claim"] for e in emitted] == ["C3"]
    assert emitted[0]["complete_auth_gold"] == 0
