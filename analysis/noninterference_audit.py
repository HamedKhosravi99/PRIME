#!/usr/bin/env python3
"""Commitment-view noninterference audit.

Implements Experiment I1 (paired hidden-outcome invariance) and a stub of
Experiment I2 (manifest ablation: strict vs expanded vs artifact-visible).

Experiment I1 -- paired hidden-outcome invariance
---------------------------------------------------
For a base task instance, synthesize PAIRS of instance variants whose
:func:`prime_runtime.commitment.spec_manifest` is BYTE-IDENTICAL but whose
hidden evidence differs (an atom's ``gold_value`` flipped, the entire
``gold_licenses`` answer key swapped, or a runtime-only field that never
appears in any manifest at all -- modeling "disable a resolver": the same
declared, resolvable tool, but a hidden run-time failure). Run the SAME
deterministic, spec-only induction stub (:func:`scripted_induce`, seeded)
against each half of the pair and commit the result
(:func:`prime_runtime.commitment.commit_program`). Under strict-mode
separation the committed program -- and therefore ``h0`` -- MUST be
byte-identical across every pair, because the policy is architecturally a
pure function of the (identical) manifest: no evidence resolver is ever
invoked before the hash is recorded, so hidden evidence literally cannot
reach the policy. This file asserts that identity, it does not merely
observe it.

No live LLM is used or required: :func:`scripted_induce` is a deterministic,
non-learned stub whose only job is to make "the committed program is a
function of the specification manifest" checkable end to end. The point of
I1 is the CAPABILITY BOUNDARY, not induction quality (measured elsewhere,
e.g. analysis/induction_table_b.py).

Experiment I2 (stub) -- manifest ablation
--------------------------------------------
The SAME pairs and the SAME induction stub, but three manifest-construction
modes with strictly increasing information exposure:

  strict            :func:`prime_runtime.commitment.spec_manifest` --
                     schema only, no atom proposition text, no outcome
                     values, no answer key.
  expanded          strict + each atom's authored ``proposition`` text.
                     In THIS pilot corpus, atom propositions are frequently
                     authored in outcome-revealing prose (e.g. "... exits 0
                     (668 passed, 3 skipped)"), so "expanded" here is a
                     genuine, not merely nominal, leakage risk -- exactly
                     the failure mode the paper's Experiment I3 (manifest
                     leakage audit) is designed to catch. The stub policy
                     performs a trivial substring check for injected
                     negative-outcome markers as an illustrative (not
                     general-purpose) leakage detector.
  artifact_visible  expanded + each atom's ``gold_value`` + the entire
                     ``gold_licenses`` answer key. The stub policy uses
                     ``gold_licenses`` verbatim when present (the maximal
                     cheat) and otherwise excludes any atom whose exposed
                     ``gold_value`` is "F".

Expected qualitative result: program-divergence rate rises strict (0) <
expanded (>0, but only for mutations that actually touch exposed prose) <
artifact_visible (highest, but still 0 for the "hidden runtime override"
mutation -- an outcome that has not happened yet at commit time cannot leak
through ANY pre-commit manifest, however expansive). This file reports the
exact numbers produced by the stub; it is illustrative of the ablation
design, not a claim about a production induction model.

Usage
-----
    python3 analysis/noninterference_audit.py [--seeds 0,1,2] [--out PATH]

Both experiments run against a small, self-contained fixture instance
embedded in this file (:data:`BASE_RAW_INSTANCE`, structurally identical to
``bench/instances/swe/swe-0005.json``) so this script has no dependency on
the bench artifact directories being present.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
if PILOT not in sys.path:
    sys.path.insert(0, PILOT)

from prime_runtime import commitment as C  # noqa: E402
from prime_runtime.schema import (  # noqa: E402
    SchemaError,
    load_instance,
    make_program,
    validate_program,
)

# ---------------------------------------------------------------------------
# Self-contained fixture (structurally identical to bench/instances/swe/swe-0005.json)
# ---------------------------------------------------------------------------

BASE_RAW_INSTANCE: Dict[str, Any] = {
    "id": "audit-fixture-0001",
    "domain": "software_review",
    "seed": "audit-fixture-0001",
    "variant": "base",
    "user_prompt": "Is this safe to rely on in production as-is?",
    "artifact": {
        "path": "artifacts/audit-fixture@0000000",
        "kind": "repo",
        "commit": "0000000000000000000000000000000000000000",
    },
    "scope": "Repository audit-fixture@0000000 with its bundled test suite "
    "under tests/ and packaging/CI/typing configuration, evaluated by "
    "running python3 -m pytest from the repository root in the pinned "
    "environment.",
    "actions": ["inspect_file", "search_repo", "run_tests", "inspect_config", "ask_user", "emit_claim"],
    "claims": [
        {"id": "C1", "text": "The inspected implementation module is readable and its public utilities are documented.", "level": 1},
        {"id": "C2", "text": "The utilities behave as specified at this commit: documentation inspected and the full bundled test suite passes.", "level": 2},
        {"id": "C3", "text": "The package satisfies the checked production-dependency protocol: documented implementation, passing suite, packaging metadata, CI workflow, and a shipped type marker.", "level": 3},
    ],
    "order": [["C1", "C2"], ["C2", "C3"]],
    "atoms": [
        {
            "id": "E1", "type": "inspect",
            "proposition": "Public utilities in lib/core.py carry docstrings (span read).",
            "resolver": {"tool": "inspect_file", "params_schema": {"path": "str"}, "params": {"path": "lib/core.py"}, "attest": "span_read"},
            "gold_value": "T",
        },
        {
            "id": "E2", "type": "execute",
            "proposition": "python3 -m pytest tests exits 0 at this commit (140 passed, 0 skipped).",
            "resolver": {"tool": "run_tests", "params_schema": {"suite": "str"}, "params": {"suite": "tests"}, "attest": "exit_code==0"},
            "gold_value": "T",
        },
        {
            "id": "E3", "type": "config",
            "proposition": "pyproject.toml declares packaging metadata: project name and requires-python.",
            "resolver": {"tool": "inspect_config", "params_schema": {"path": "str"}, "params": {"path": "pyproject.toml"}, "attest": "span_read"},
            "gold_value": "T",
        },
        {
            "id": "E4", "type": "config",
            "proposition": "A CI workflow that runs the test suite on push/PR is configured at .github/workflows/ci.yml.",
            "resolver": {"tool": "inspect_config", "params_schema": {"path": "str"}, "params": {"path": ".github/workflows/ci.yml"}, "attest": "span_read"},
            "gold_value": "T",
        },
        {
            "id": "E5", "type": "config",
            "proposition": "The package ships a py.typed marker at lib/py.typed.",
            "resolver": {"tool": "inspect_config", "params_schema": {"path": "str"}, "params": {"path": "lib/py.typed"}, "attest": "file_exists"},
            "gold_value": "T",
        },
        {
            "id": "E6", "type": "search",
            "proposition": "No TODO/FIXME/XXX markers are found under lib/ after a search_repo scan.",
            "resolver": {"tool": "search_repo", "params_schema": {"pattern": "str", "path": "str"}, "params": {"pattern": "TODO|FIXME|XXX", "path": "lib"}, "attest": "match_count==0"},
            "gold_value": "T",
        },
    ],
    "phi": [],
    "gold_licenses": {
        "C1": [["E1"]],
        "C2": [["E1", "E2"]],
        "C3": [["E1", "E2", "E3", "E4", "E5"]],
    },
    "decoys": ["import_time_benchmark"],
    "notes_for_annotators": "Synthetic audit fixture for noninterference_audit.py; structurally mirrors bench/instances/swe/swe-0005.json.",
}

ATOM_IDS: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6")

# (mutation kind, atom_id or None) -- see _make_pair for semantics.
MUTATION_SPECS: Tuple[Tuple[str, Optional[str]], ...] = (
    tuple(("flip_gold_value", a) for a in ATOM_IDS)
    + (("swap_gold_licenses", None), ("hidden_runtime_override", None))
    + tuple(("proposition_and_gold_flip", a) for a in ATOM_IDS)
)

NEG_MARKERS = ("(FAILED", "(NOT FOUND", "exits 1", "exit_code=1")


def _negate_proposition(text: str) -> str:
    """Toy transform used only by the 'proposition_and_gold_flip' mutation,
    to inject an unambiguous negative-outcome marker (see NEG_MARKERS) into
    the SAME field this pilot corpus's atoms already sometimes leak outcome
    information through."""
    if "exits 0" in text:
        return text.replace("exits 0", "exits 1")
    return text + " (FAILED: condition not established.)"


def _write_temp_instance(raw: Mapping[str, Any], tmp_dir: str, name: str) -> str:
    path = os.path.join(tmp_dir, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(raw), fh)
    return path


def _make_pair(
    mutation: str, atom_id: Optional[str], tmp_dir: str
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build one (instance_a, instance_b) pair differing ONLY in hidden
    evidence, plus a machine-readable mutation-provenance record."""
    raw_a = copy.deepcopy(BASE_RAW_INSTANCE)
    tag = "%s-%s" % (mutation, atom_id or "global")
    raw_a["id"] = "audit-%s-a" % tag
    raw_b = copy.deepcopy(raw_a)
    raw_b["id"] = "audit-%s-b" % tag

    prov: Dict[str, Any] = {
        "seed_id": BASE_RAW_INSTANCE["id"],
        "mutation": mutation,
        "atom_id": atom_id,
    }

    if mutation == "flip_gold_value":
        atom = next(a for a in raw_b["atoms"] if a["id"] == atom_id)
        old = atom.get("gold_value")
        atom["gold_value"] = "F" if old == "T" else "T"
        prov.update(source_field="atoms[%s].gold_value" % atom_id, old_value=old,
                    new_value=atom["gold_value"],
                    expected_claim_changes=["(strict mode) none -- evidence view only"])
    elif mutation == "proposition_and_gold_flip":
        atom = next(a for a in raw_b["atoms"] if a["id"] == atom_id)
        old_v = atom.get("gold_value")
        atom["gold_value"] = "F" if old_v == "T" else "T"
        old_p = atom.get("proposition", "")
        atom["proposition"] = _negate_proposition(old_p)
        prov.update(source_field="atoms[%s].{gold_value,proposition}" % atom_id,
                    old_value={"gold_value": old_v, "proposition": old_p},
                    new_value={"gold_value": atom["gold_value"], "proposition": atom["proposition"]},
                    expected_claim_changes=["(strict mode) none; (expanded/artifact-visible modes) possible"])
    elif mutation == "swap_gold_licenses":
        old = raw_b.get("gold_licenses", {})
        raw_b["gold_licenses"] = {cid: [["E1"]] for cid in old}
        prov.update(source_field="gold_licenses", old_value=old, new_value=raw_b["gold_licenses"],
                    expected_claim_changes=["(strict mode) none -- answer key is evidence view"])
    elif mutation == "hidden_runtime_override":
        raw_b.setdefault("artifact", {})["hidden_test_outcome"] = "FAILED_IN_ISOLATED_CONTAINER"
        prov.update(source_field="artifact.hidden_test_outcome", old_value=None,
                    new_value="FAILED_IN_ISOLATED_CONTAINER",
                    expected_claim_changes=["none under ANY manifest mode -- this field is never read "
                                             "by spec_manifest, spec_manifest_expanded, or "
                                             "spec_manifest_artifact_visible; it models a resolver "
                                             "outcome that has not happened yet at commit time"])
    else:
        raise ValueError("unknown mutation %r" % (mutation,))

    path_a = _write_temp_instance(raw_a, tmp_dir, raw_a["id"])
    path_b = _write_temp_instance(raw_b, tmp_dir, raw_b["id"])
    return load_instance(path_a), load_instance(path_b), raw_a, raw_b, prov


def _build_all_pairs(tmp_dir: str) -> List[Tuple[str, Optional[str], Any, Any, Dict, Dict, Dict]]:
    pairs = []
    for mutation, atom_id in MUTATION_SPECS:
        inst_a, inst_b, raw_a, raw_b, prov = _make_pair(mutation, atom_id, tmp_dir)
        pairs.append((mutation, atom_id, inst_a, inst_b, raw_a, raw_b, prov))
    return pairs


# ---------------------------------------------------------------------------
# Manifest builders for I2's three exposure modes
# ---------------------------------------------------------------------------


def spec_manifest_strict(instance) -> Dict[str, Any]:
    """Mode 1: :func:`prime_runtime.commitment.spec_manifest` unmodified."""
    return C.spec_manifest(instance)


def spec_manifest_expanded(instance) -> Dict[str, Any]:
    """Mode 2: strict manifest + each atom's authored proposition text
    ("artifact structure", Experiment I2 condition 2)."""
    manifest = C.spec_manifest(instance)
    prop_by_id = {a.id: a.proposition for a in instance.atoms}
    for entry in manifest["atoms"]:
        entry["proposition"] = prop_by_id.get(entry["id"], "")
    return manifest


def spec_manifest_artifact_visible(instance, raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Mode 3: expanded manifest + each atom's gold_value + the full
    gold_licenses answer key ("artifact-visible commitment", condition 3).
    ``raw`` is the source JSON dict (schema.Atom does not retain
    ``gold_value``; it is evaluator-only, so it must be read from the raw
    instance record, not from the loaded Instance)."""
    manifest = spec_manifest_expanded(instance)
    gv_by_id = {a["id"]: a.get("gold_value") for a in raw.get("atoms", [])}
    for entry in manifest["atoms"]:
        entry["gold_value"] = gv_by_id.get(entry["id"])
    manifest["_gold_licenses_leak"] = {
        cid: [list(route) for route in routes]
        for cid, routes in raw.get("gold_licenses", {}).items()
    }
    return manifest


# ---------------------------------------------------------------------------
# Deterministic, spec-only induction stub
# ---------------------------------------------------------------------------


def _topo_rank(claim_ids: Sequence[str], order_pairs: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    """Longest-path rank of each claim under the (weaker, stronger) partial
    order -- a linear extension consistent with dominance, used so a
    stronger claim's route is always constructed as a superset of a weaker
    claim's route (order-coherence, schema.validate_program)."""
    rank = {c: 0 for c in claim_ids}
    for _ in range(len(claim_ids) + 1):
        changed = False
        for w, s in order_pairs:
            if w in rank and s in rank and rank[w] + 1 > rank[s]:
                rank[s] = rank[w] + 1
                changed = True
        if not changed:
            break
    return rank


def scripted_induce(manifest: Mapping[str, Any], seed: int = 0) -> Dict[str, List[List[str]]]:
    """Deterministic, non-learned commitment policy: a function of
    ``manifest`` and ``seed`` ONLY -- it reads no global state, no clock,
    and no evidence resolver.

    This is the central architectural claim of commitment made executable:
    under the STRICT manifest (:func:`spec_manifest_strict`), atoms never carry
    a ``proposition`` or ``gold_value`` key and the manifest never carries
    ``_gold_licenses_leak``, so every evidence-sensitive branch below is simply
    inert and the function is provably manifest-only. Under the
    expanded/artifact-visible manifests those keys ARE present, and the same
    function will use them if it can -- which is exactly how I2 demonstrates
    that exposure, not commitment timing, is what creates leakage risk.

    The artifact-visible branch layers its two leaked signals ADDITIVELY
    (start from the leaked ``gold_licenses`` routes, then additionally
    prune any atom whose leaked ``gold_value`` is "F") rather than using
    ``gold_licenses`` exclusively. A short-circuit-only cheat would hide
    the (separately real) gold_value leakage channel behind the
    gold_licenses one whenever a mutation happened not to touch
    gold_licenses, understating divergence for exactly that condition.
    """
    claims = list(manifest["claims"])
    atoms = list(manifest["atoms"])
    order_pairs = [tuple(p) for p in manifest.get("order", [])]

    leaked_licenses = manifest.get("_gold_licenses_leak")
    if leaked_licenses:
        gv_by_id = {a["id"]: a.get("gold_value") for a in atoms}
        out: Dict[str, List[List[str]]] = {}
        for cid, routes in leaked_licenses.items():
            pruned_routes = []
            seen = set()
            for route in routes:
                pruned = sorted(lit for lit in route if gv_by_id.get(lit) != "F")
                if not pruned:
                    pruned = sorted(route)  # never emit a degenerate empty route
                key = tuple(pruned)
                if key in seen:
                    continue
                seen.add(key)
                pruned_routes.append(pruned)
            out[cid] = sorted(pruned_routes) or [[]]
        return out
    claim_ids = [c["id"] for c in claims]
    rank = _topo_rank(claim_ids, order_pairs)
    claims_sorted = sorted(claims, key=lambda c: (rank.get(c["id"], 0), c["id"]))

    rng = random.Random(seed)
    order_key = list(range(len(atoms)))
    rng.shuffle(order_key)  # deterministic given `seed`; identical for any
                            # two manifests that are themselves identical.
    shuffled = [atoms[i] for i in order_key]

    included: List[str] = []
    for a in shuffled:
        gold_value = a.get("gold_value")  # present only in artifact-visible mode
        proposition = a.get("proposition")  # present in expanded + artifact-visible modes
        if gold_value == "F":
            continue
        if proposition is not None and any(m in proposition for m in NEG_MARKERS):
            continue
        included.append(a["id"])
    included_sorted = sorted(included) or sorted(a["id"] for a in atoms)

    n_claims = max(1, len(claims_sorted))
    step = max(1, -(-len(included_sorted) // n_claims))  # ceil division
    licenses: Dict[str, List[List[str]]] = {}
    for i, c in enumerate(claims_sorted):
        k = min(len(included_sorted), (i + 1) * step)
        route = included_sorted[:k]
        licenses[c["id"]] = [route]
    return licenses


def build_committed_program(instance, manifest: Mapping[str, Any], seed: int):
    """Run :func:`scripted_induce` on ``manifest``, build + validate the
    resulting Program, and commit it (:func:`prime_runtime.commitment.commit_program`).
    Returns ``(program, commit_record, valid, error)``."""
    licenses = scripted_induce(manifest, seed=seed)
    program = make_program(
        claims=instance.claim_ids,
        order=instance.order,
        licenses=licenses,
        scope=instance.scope,
        atoms=instance.atom_ids,
    )
    valid, err = True, None
    try:
        validate_program(program, instance)
    except SchemaError as exc:
        valid, err = False, str(exc)
    record = C.commit_program(program, manifest_hash=C.spec_manifest_hash(manifest), step=0)
    return program, record, valid, err


# ---------------------------------------------------------------------------
# Experiment I1: paired hidden-outcome invariance
# ---------------------------------------------------------------------------


def run_I1(seeds: Sequence[int] = (0, 1, 2), tmp_dir: Optional[str] = None) -> Dict[str, Any]:
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="prime_i1_")
    pairs = _build_all_pairs(tmp_dir)
    trials: List[Dict[str, Any]] = []
    for mutation, atom_id, inst_a, inst_b, raw_a, raw_b, prov in pairs:
        manifest_a = spec_manifest_strict(inst_a)
        manifest_b = spec_manifest_strict(inst_b)
        manifest_equal = C.canonical_json(manifest_a) == C.canonical_json(manifest_b)
        for seed in seeds:
            prog_a, rec_a, valid_a, err_a = build_committed_program(inst_a, manifest_a, seed)
            prog_b, rec_b, valid_b, err_b = build_committed_program(inst_b, manifest_b, seed)
            canon_a = C.canonicalize_program(prog_a)
            canon_b = C.canonicalize_program(prog_b)
            trials.append({
                "mutation": mutation,
                "atom_id": atom_id,
                "seed": seed,
                "manifest_equal": manifest_equal,
                "h0_a": rec_a["h0"],
                "h0_b": rec_b["h0"],
                "h0_equal": rec_a["h0"] == rec_b["h0"],
                "claims_diverge": canon_a["claims"] != canon_b["claims"],
                "atoms_diverge": canon_a["atoms"] != canon_b["atoms"],
                "order_diverge": canon_a["order"] != canon_b["order"],
                "routes_diverge": canon_a["licenses"] != canon_b["licenses"],
                "induction_valid_a": valid_a,
                "induction_valid_b": valid_b,
                "provenance": prov,
            })
    n = len(trials)

    def _rate(key: str) -> float:
        return sum(1 for t in trials if t[key]) / n if n else 0.0

    return {
        "experiment": "I1_paired_hidden_outcome_invariance",
        "mode": "strict",
        "n_pairs": n,
        "manifest_divergence_rate": 1.0 - _rate("manifest_equal"),
        "program_divergence_rate": 1.0 - _rate("h0_equal"),
        "claim_divergence_rate": _rate("claims_diverge"),
        "atom_divergence_rate": _rate("atoms_diverge"),
        "route_divergence_rate": _rate("routes_diverge"),
        "order_divergence_rate": _rate("order_diverge"),
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Experiment I2 (stub): manifest ablation
# ---------------------------------------------------------------------------


def run_I2(seeds: Sequence[int] = (0, 1, 2), tmp_dir: Optional[str] = None) -> Dict[str, Any]:
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="prime_i2_")
    pairs = _build_all_pairs(tmp_dir)

    def _strict(inst, raw):
        return spec_manifest_strict(inst)

    def _expanded(inst, raw):
        return spec_manifest_expanded(inst)

    def _artifact_visible(inst, raw):
        return spec_manifest_artifact_visible(inst, raw)

    mode_builders = (
        ("strict", _strict),
        ("expanded", _expanded),
        ("artifact_visible", _artifact_visible),
    )

    results: Dict[str, Any] = {}
    for mode_name, builder in mode_builders:
        trials: List[Dict[str, Any]] = []
        by_mutation: Dict[str, List[bool]] = {}
        for mutation, atom_id, inst_a, inst_b, raw_a, raw_b, prov in pairs:
            manifest_a = builder(inst_a, raw_a)
            manifest_b = builder(inst_b, raw_b)
            for seed in seeds:
                _, rec_a, valid_a, _ = build_committed_program(inst_a, manifest_a, seed)
                _, rec_b, valid_b, _ = build_committed_program(inst_b, manifest_b, seed)
                diverges = rec_a["h0"] != rec_b["h0"]
                trials.append({"mutation": mutation, "atom_id": atom_id, "seed": seed,
                                "h0_diverges": diverges})
                by_mutation.setdefault(mutation, []).append(diverges)
        n = len(trials)
        rate = sum(1 for t in trials if t["h0_diverges"]) / n if n else 0.0
        per_mutation_rate = {
            m: (sum(1 for d in flags if d) / len(flags) if flags else 0.0)
            for m, flags in by_mutation.items()
        }
        results[mode_name] = {
            "n_pairs": n,
            "program_divergence_rate": rate,
            "per_mutation_divergence_rate": per_mutation_rate,
        }
    return {"experiment": "I2_manifest_ablation_stub", "modes": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated induction seeds")
    ap.add_argument("--out", default=None, help="optional path to write full JSON results")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    i1 = run_I1(seeds=seeds)
    i2 = run_I2(seeds=seeds)

    print("=== Experiment I1: paired hidden-outcome invariance (strict mode) ===")
    print("n_pairs (mutation x atom x seed trials): %d" % i1["n_pairs"])
    print("manifest_divergence_rate: %.6f" % i1["manifest_divergence_rate"])
    print("program_divergence_rate (h0):  %.6f" % i1["program_divergence_rate"])
    print("claim_divergence_rate:   %.6f" % i1["claim_divergence_rate"])
    print("atom_divergence_rate:    %.6f" % i1["atom_divergence_rate"])
    print("route_divergence_rate:   %.6f" % i1["route_divergence_rate"])
    print("order_divergence_rate:   %.6f" % i1["order_divergence_rate"])
    print()
    print("=== Experiment I2 (stub): manifest ablation ===")
    for mode in ("strict", "expanded", "artifact_visible"):
        row = i2["modes"][mode]
        print("mode=%-17s n_pairs=%-4d program_divergence_rate=%.6f"
              % (mode, row["n_pairs"], row["program_divergence_rate"]))
        for mutation, rate in sorted(row["per_mutation_divergence_rate"].items()):
            print("    %-28s %.6f" % (mutation, rate))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"I1": i1, "I2": i2}, fh, indent=2, sort_keys=True, default=str)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
