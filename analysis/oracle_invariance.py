#!/usr/bin/env python3
"""Oracle representation-invariance audit (Experiments B1 / B5).

the design notes Weakness 7 §7.7:

    Experiment B1 (logical-representation invariance): transform the same
    support function among DNF / CNF / BDD / executable code; verify
    identical scores for all logged traces.

    Experiment B5 (oracle perturbation audit): apply semantics-preserving
    transformations to the oracle implementation; all result tables must
    remain byte-identical; publish a hash of the generated scores.

For every logged trace under ``pilot/results/**/*.jsonl`` belonging to a
DEPLOYED arm (i.e. excluding the ``drift__*``/``advdrift__*`` standard-drift
ablation arms under ``adversarial/`` and the model dirs, which are
deliberately adversarial and not deployment candidates — see the
integrated-plan "deployed fixed arm" convention already used by
``analysis/drift_table.py``):

  1. reconstruct the method-independent gold ledger from the trace via
     ``goldledger.reconstruct`` (reused, not reimplemented — same function
     the runner itself uses to score CompleteAuth_gold);
  2. for every emitted claim, take the fail-closed CLOSURE state
     (``closure.cl``) of the gold ledger at that claim's emission step —
     the same "evidence state omega_t" ``auth.lic``/``auth.auth_ids`` use to
     authorize a claim under a program — as the ``facts`` argument to the
     oracle;
  3. score that (instance, claim, facts) triple three ways: ``support``,
     ``support_via_bdd``, ``support_via_truthtable``; assert all three
     agree;
  4. separately (once per distinct instance encountered, not per trace),
     verify ``derive_prime_routes`` reproduces the authored gold routes.

Never reads an emitting arm's own attestation/emission claims of support —
only the reconstructed gold ledger, exactly like ``goldledger``/``aggregate``
already do (SPEC.md: "no fabricated results anywhere").
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PILOT)

from prime_runtime import goldledger, oracle, schema  # noqa: E402
from prime_runtime.closure import cl, is_inconsistent  # noqa: E402
from prime_runtime.trace import read_trace  # noqa: E402

DEFAULT_RESULTS = os.path.join(PILOT, "results")
DEFAULT_BENCH = os.path.join(PILOT, "bench", "instances")
DEFAULT_FIXTURE = os.path.join(PILOT, "prime_runtime", "tests", "fixtures", "instance_toy.json")

# Standard-drift / adversarial-reviser ablation arms are deliberately
# non-deployed exploit-pressure conditions (memo Weakness 2 §2.3, C5) --
# excluded here, matching how analysis/drift_table.py treats them as a
# separate mechanism experiment, not a deployment candidate.
_NON_DEPLOYED_ARM_PREFIXES = ("drift__", "advdrift__")


def is_deployed_trace(path: str, results_root: str) -> bool:
    rel = os.path.relpath(path, results_root)
    parts = rel.split(os.sep)
    if "adversarial" in parts[:-1]:
        return False
    arm_dir = parts[-2] if len(parts) >= 2 else ""
    return not arm_dir.startswith(_NON_DEPLOYED_ARM_PREFIXES)


def build_instance_index(bench_root: str, fixture_path: str) -> dict:
    """instance id -> loaded Instance, scanned from every bench instance
    file plus the runtime's own test fixture (exercised by the *__scripted
    smoke-test traces, which are legitimate deployed-arm dry runs)."""
    index = {}
    paths = sorted(glob.glob(os.path.join(bench_root, "**", "*.json"), recursive=True))
    if os.path.isfile(fixture_path):
        paths.append(fixture_path)
    for path in paths:
        try:
            inst = schema.load_instance(path)
        except schema.SchemaError as exc:
            print("WARNING: could not load instance %s: %r" % (path, exc), file=sys.stderr)
            continue
        index[inst.id] = inst
    return index


def _meta_instance_id(entries):
    for e in entries:
        if e.get("type") == "meta" and e.get("instance"):
            return e["instance"]
    return None


def facts_at_step(replay: "goldledger.GoldReplay", step: int) -> dict:
    """The fail-closed closure state of the gold ledger at ``step`` — the
    same evidence-state object ``auth.lic``/``auth.auth_ids`` consult to
    authorize a claim under a program (auth.py, closure.py; imported, not
    modified). Inconsistent (Phi-UNSAT) ledgers authorize nothing under the
    runtime's own semantics, so they are treated as fully unresolved here."""
    led = replay.ledger_at(step)
    closed = cl(led)
    if is_inconsistent(closed):
        return {a: "U" for a in replay.instance.atom_ids}
    return dict(closed)


def audit_traces(results_root: str, instance_index: dict) -> dict:
    all_jsonl = sorted(glob.glob(os.path.join(results_root, "**", "*.jsonl"), recursive=True))
    deployed = [p for p in all_jsonl if is_deployed_trace(p, results_root)]
    skipped_nondeployed = len(all_jsonl) - len(deployed)

    n_traces = 0
    n_skipped_no_instance = 0
    n_claims_scored = 0
    n_agree = 0
    n_oracle_supported = 0
    disagreements = []
    score_records = []  # for the B5 score hash

    for path in deployed:
        entries = read_trace(path)
        inst_id = _meta_instance_id(entries)
        instance = instance_index.get(inst_id) if inst_id else None
        if instance is None:
            n_skipped_no_instance += 1
            continue
        n_traces += 1
        replay = goldledger.reconstruct(entries, instance)
        for entry in entries:
            if entry.get("type") != "emission":
                continue
            step = int(entry.get("step", 0))
            facts = facts_at_step(replay, step)
            claim_ids = entry.get("claims") or []
            for claim_id in claim_ids:
                if claim_id not in instance.claim_ids:
                    continue  # not a scorable typed claim (defensive)
                agree, values = oracle.representations_agree(instance, claim_id, facts)
                n_claims_scored += 1
                if agree:
                    n_agree += 1
                else:
                    disagreements.append(
                        {
                            "trace": os.path.relpath(path, PILOT),
                            "instance": instance.id,
                            "claim_id": claim_id,
                            "step": step,
                            "facts": facts,
                            "values": values,
                        }
                    )
                oracle_auth = oracle.oracle_authorization(instance, claim_id, facts)
                if oracle_auth:
                    n_oracle_supported += 1
                score_records.append(
                    [
                        os.path.relpath(path, PILOT),
                        step,
                        claim_id,
                        bool(values["support"]),
                        bool(values["bdd"]),
                        bool(values["truthtable"]),
                    ]
                )

    score_records.sort()
    scores_blob = json.dumps(score_records, sort_keys=True)
    scores_hash = hashlib.sha256(scores_blob.encode("utf-8")).hexdigest()

    return {
        "n_trace_files_total": len(all_jsonl),
        "n_trace_files_nondeployed_excluded": skipped_nondeployed,
        "n_trace_files_deployed": len(deployed),
        "n_traces_scored": n_traces,
        "n_traces_skipped_unknown_instance": n_skipped_no_instance,
        "n_emitted_claims_scored": n_claims_scored,
        "n_emitted_claims_oracle_supported": n_oracle_supported,
        "n_representation_agreements": n_agree,
        "agreement_rate": (n_agree / n_claims_scored) if n_claims_scored else 1.0,
        "disagreements": disagreements,
        "scores_sha256": scores_hash,
    }


def audit_route_derivation(instance_index: dict) -> dict:
    reports = []
    for inst_id in sorted(instance_index):
        instance = instance_index[inst_id]
        for claim_id in instance.claim_ids:
            reports.append(oracle.gold_route_mismatch(instance, claim_id))
    mismatches = [r for r in reports if not r["match"]]
    return {
        "n_instances": len(instance_index),
        "n_claims_checked": len(reports),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
    }


def render_report(trace_audit: dict, route_audit: dict) -> str:
    lines = []
    lines.append("# Oracle invariance audit (Experiments B1 / B5)")
    lines.append("")
    lines.append("## B1 — logical-representation invariance (support vs. BDD vs. truth table)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append("| trace files under results/**/*.jsonl | %d |" % trace_audit["n_trace_files_total"])
    lines.append(
        "| excluded (drift__/advdrift__ non-deployed arms) | %d |"
        % trace_audit["n_trace_files_nondeployed_excluded"]
    )
    lines.append("| deployed-arm trace files | %d |" % trace_audit["n_trace_files_deployed"])
    lines.append("| traces scored (instance resolved) | %d |" % trace_audit["n_traces_scored"])
    lines.append(
        "| traces skipped (unknown instance id) | %d |" % trace_audit["n_traces_skipped_unknown_instance"]
    )
    lines.append("| emitted claims scored | %d |" % trace_audit["n_emitted_claims_scored"])
    lines.append(
        "| emitted claims oracle-supported (informational) | %d |"
        % trace_audit["n_emitted_claims_oracle_supported"]
    )
    lines.append("| representation agreements | %d |" % trace_audit["n_representation_agreements"])
    lines.append("| **agreement rate** | **%.4f%%** |" % (100.0 * trace_audit["agreement_rate"]))
    lines.append("| disagreements | %d |" % len(trace_audit["disagreements"]))
    lines.append("| scores SHA-256 (B5 hash) | `%s` |" % trace_audit["scores_sha256"])
    lines.append("")
    if trace_audit["disagreements"]:
        lines.append("### Disagreements (should be empty)")
        lines.append("")
        for d in trace_audit["disagreements"][:50]:
            lines.append("- %r" % d)
        lines.append("")
    lines.append("## Route-derivation check (routes DERIVED from F*, per instance)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append("| instances checked | %d |" % route_audit["n_instances"])
    lines.append("| (instance, claim) pairs checked | %d |" % route_audit["n_claims_checked"])
    lines.append("| mismatches (derived prime routes != authored gold routes) | %d |" % route_audit["n_mismatches"])
    lines.append("")
    if route_audit["mismatches"]:
        lines.append("### Mismatches (benchmark-authoring findings — reported, not silently fixed)")
        lines.append("")
        for m in route_audit["mismatches"]:
            lines.append(
                "- **%s / %s**: derived=%s gold=%s (derived_only=%s, gold_only=%s)"
                % (m["instance"], m["claim_id"], m["derived"], m["gold"], m["derived_only"], m["gold_only"])
            )
        lines.append("")
    else:
        lines.append("No mismatches.")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--bench", default=DEFAULT_BENCH)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    instance_index = build_instance_index(args.bench, args.fixture)
    if not instance_index:
        print("REFUSING: no benchmark instances found under %s" % args.bench, file=sys.stderr)
        sys.exit(2)

    trace_audit = audit_traces(args.results, instance_index)
    if trace_audit["n_traces_scored"] == 0:
        print("REFUSING: no deployed-arm traces with a resolvable instance found under %s" % args.results, file=sys.stderr)
        sys.exit(2)
    route_audit = audit_route_derivation(instance_index)

    report = render_report(trace_audit, route_audit)
    print(report)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump({"trace_audit": trace_audit, "route_audit": route_audit}, fh, indent=2, sort_keys=True)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")

    if trace_audit["disagreements"] or route_audit["n_mismatches"]:
        # Non-fatal: mismatches are reported findings, not a failed run --
        # but signal via exit code for CI/automation that something needs
        # human review (memo §7.2: "do not silently fix").
        sys.exit(1)


if __name__ == "__main__":
    main()
