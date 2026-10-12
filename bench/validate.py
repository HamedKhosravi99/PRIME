#!/usr/bin/env python3
"""ObligationBench-pilot instance validator (Agent B, bench/).

Checks, per instance JSON under instances/swe/:
  1.  parses as JSON and conforms to bench/schema.json (jsonschema if
      available, else a minimal structural fallback);
  2.  referential integrity: unique claim/atom ids; order pairs, phi keys,
      gold_license keys and route literals all reference declared ids;
      every claim has a gold license family; resolver tools are listed
      actions; decoys are not listed actions;
  3.  artifact integrity: artifact dir exists, contains PROVENANCE, and the
      dir's short sha prefixes the declared commit;
  4.  path existence: every 'path'/'suite' resolver param names files that
      exist in the artifact checkout (suite split on whitespace, ::nodeid
      stripped);
  5.  realizability: every gold route admits a full assignment (route atoms
      True) violating no phi forbid-pattern;
  6.  order-coherence (sufficient condition): every route of the
      stronger claim literal-contains some route of each weaker claim in the
      transitive closure of `order`; levels must agree with `order`;
  7.  minimality modulo phi against f_bench: removing
      any literal from any route must break sufficiency (brute force over
      realizable assignments; atom count <= 8); plus semantic-dominance:
      no route of a claim may be dominated by a strictly weaker sufficient
      route of the same claim;
  8.  noev variants: exactly one atom has resolver null; that atom occurs in
      every route of the top-level claim and in no route of any other claim;
  9.  gold_value present exactly on resolver-bearing atoms.

Exit 0 iff all instances pass. Usage: python3 validate.py [instance.json ...]
"""
import itertools
import json
import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(BENCH, "schema.json")
INST_DIR = os.path.join(BENCH, "instances", "swe")

# Optional: reuse prime_runtime's closed predicate-grammar parser for a
# deeper check of 'record_predicate:*' attest rules (structured-record
# instances). This validator otherwise has no runtime import, so degrade
# gracefully (skip that one check) if prime_runtime is not importable.
_PILOT = os.path.dirname(BENCH)
if _PILOT not in sys.path:
    sys.path.insert(0, _PILOT)
try:
    from prime_runtime.predicates import parse_predicate_rule as _parse_predicate_rule
except ImportError:  # pragma: no cover - validator degrades gracefully
    _parse_predicate_rule = None


def _resolve_field_path(record, field_path):
    """Walk a dotted field path ('a.b.0.c') through nested dict/list JSON,
    mirroring runner.arms._resolve_field_path (duplicated here, not
    imported, so this validator keeps no hard runtime dependency).
    Returns (value, found)."""
    if not field_path:
        return record, True
    cur = record
    for part in field_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return None, False
        else:
            return None, False
    return cur, True


def load_schema_checker():
    schema = json.load(open(SCHEMA))
    try:
        import jsonschema

        def check(doc):
            v = jsonschema.Draft202012Validator(schema)
            return [f"schema: {e.json_path}: {e.message}" for e in v.iter_errors(doc)]
        return check, "jsonschema"
    except ImportError:  # minimal structural fallback
        def check(doc):
            errs = []
            for k in schema["required"]:
                if k not in doc:
                    errs.append(f"schema(fallback): missing required field {k}")
            return errs
        return check, "fallback"


def realizable_assignments(atom_ids, phi):
    """All total {atom: bool} assignments violating no forbid pattern."""
    out = []
    for bits in itertools.product([False, True], repeat=len(atom_ids)):
        w = dict(zip(atom_ids, bits))
        ok = True
        for rule in phi:
            if all(w[a] == (v == "T") for a, v in rule["forbid"].items()):
                ok = False
                break
        if ok:
            out.append(w)
    return out


def satisfies(w, route):
    return all(w[a] for a in route)


def sufficient(term, routes, omega):
    """term (set of atoms=T) sufficient for f_bench = OR(routes), modulo phi."""
    return all(any(satisfies(w, r) for r in routes)
               for w in omega if satisfies(w, term))


def validate(path, schema_check):
    errs = []
    try:
        inst = json.load(open(path))
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"]
    errs += schema_check(inst)
    if errs:
        return errs  # structural errors first; later checks assume shape

    claims = {c["id"]: c for c in inst["claims"]}
    atoms = {a["id"]: a for a in inst["atoms"]}
    if len(claims) != len(inst["claims"]):
        errs.append("duplicate claim ids")
    if len(atoms) != len(inst["atoms"]):
        errs.append("duplicate atom ids")
    if inst["id"] != (inst["seed"] if inst["variant"] == "base"
                      else f"{inst['seed']}__{inst['variant']}"):
        errs.append("id/seed/variant naming mismatch")

    # --- referential integrity
    for w, s in inst["order"]:
        for c in (w, s):
            if c not in claims:
                errs.append(f"order references unknown claim {c}")
        if w in claims and s in claims and claims[w]["level"] >= claims[s]["level"]:
            errs.append(f"order pair [{w},{s}] disagrees with levels")
    for rule in inst["phi"]:
        for a in rule["forbid"]:
            if a not in atoms:
                errs.append(f"phi references unknown atom {a}")
    if set(inst["gold_licenses"]) != set(claims):
        errs.append("gold_licenses keys != claim ids")
    for c, routes in inst["gold_licenses"].items():
        for r in routes:
            for lit in r:
                if lit not in atoms:
                    errs.append(f"route literal {lit} of {c} is not a declared atom")
    for a in inst["atoms"]:
        res = a["resolver"]
        if res is not None:
            if res["tool"] not in inst["actions"]:
                errs.append(f"atom {a['id']} resolver tool {res['tool']} not in actions")
            if "gold_value" not in a and a["type"] != "user":
                errs.append(f"atom {a['id']} has a resolver but no gold_value")
        elif "gold_value" in a:
            errs.append(f"atom {a['id']} has resolver null but a gold_value")
    for d in inst["decoys"]:
        if d in inst["actions"]:
            errs.append(f"decoy {d} duplicates a listed action")

    # --- structured-record atoms (Tier-M predicate grammar: inspect_record /
    # record_predicate:*; prime_runtime/predicates.py)
    records = inst.get("records", {})
    for a in inst["atoms"]:
        res = a["resolver"]
        if not res or res.get("tool") != "inspect_record":
            continue
        params = res.get("params", {})
        rid = params.get("record_id")
        fpath = params.get("field_path")
        if not rid:
            errs.append(f"atom {a['id']}: inspect_record resolver missing params.record_id")
        elif rid not in records:
            errs.append(f"atom {a['id']}: record_id {rid!r} not declared in top-level records")
        if not fpath:
            errs.append(f"atom {a['id']}: inspect_record resolver missing params.field_path")
        elif rid in records:
            _, found = _resolve_field_path(records[rid], fpath)
            if not found:
                errs.append(f"atom {a['id']}: field_path {fpath!r} does not resolve in record {rid!r}")
        attest = res.get("attest", "")
        if attest.startswith("record_predicate:") and _parse_predicate_rule is not None:
            if _parse_predicate_rule(attest) is None:
                errs.append(f"atom {a['id']}: malformed record_predicate attest rule {attest!r}")

    # --- artifact + path existence
    art = os.path.join(BENCH, inst["artifact"]["path"])
    short = inst["artifact"]["path"].rsplit("@", 1)[-1]
    if not os.path.isdir(art):
        errs.append(f"artifact dir missing: {inst['artifact']['path']}")
    else:
        if not os.path.isfile(os.path.join(art, "PROVENANCE")):
            errs.append("artifact missing PROVENANCE file")
        # git-repo artifacts embed the short sha as dir@<sha> and a 40-hex
        # commit; non-git artifacts (research paper_with_corpus, arxiv: commit)
        # skip this check.
        if "@" in inst["artifact"]["path"] and not inst["artifact"]["commit"].startswith(short):
            errs.append("artifact dir short sha does not prefix commit")
        for a in inst["atoms"]:
            res = a["resolver"]
            if not res:
                continue
            params = res.get("params", {})
            targets = []
            if "path" in params:
                targets.append(params["path"])
            if "suite" in params:
                targets += [t.split("::")[0] for t in params["suite"].split()]
            for t in targets:
                if not os.path.exists(os.path.join(art, t)):
                    errs.append(f"atom {a['id']}: referenced path does not exist in artifact: {t}")

    # --- logic checks over realizable assignments
    atom_ids = sorted(atoms)
    omega = realizable_assignments(atom_ids, inst["phi"])
    if not omega:
        errs.append("phi unsatisfiable: no realizable assignments")
        return errs

    for c, routes in inst["gold_licenses"].items():
        seen = set()
        for r in routes:
            key = frozenset(r)
            if key in seen:
                errs.append(f"{c}: duplicate route {sorted(r)}")
            seen.add(key)
        # realizability of each route under phi
        for r in routes:
            if not any(satisfies(w, r) for w in omega):
                errs.append(f"{c}: route {sorted(r)} unrealizable under phi")
        # minimality: literal-deletion first filter against f_bench modulo phi
        for r in routes:
            for lit in r:
                weakened = [x for x in r if x != lit]
                if sufficient(weakened, routes, omega):
                    errs.append(f"{c}: literal {lit} redundant in route {sorted(r)} (modulo phi)")
        # semantic dominance between routes of the same claim
        for p, q in itertools.permutations(routes, 2):
            p_ent_q = all(satisfies(w, q) for w in omega if satisfies(w, p))
            q_ent_p = all(satisfies(w, p) for w in omega if satisfies(w, q))
            if p_ent_q and not q_ent_p:
                errs.append(f"{c}: route {sorted(p)} dominated by weaker route {sorted(q)}")

    # --- order-coherence (sufficient condition): transitive closure of order
    closure = {(w, s) for w, s in inst["order"]}
    changed = True
    while changed:
        changed = False
        for (a, b), (c, d) in itertools.product(list(closure), repeat=2):
            if b == c and (a, d) not in closure:
                closure.add((a, d))
                changed = True
    for w, s in closure:
        for rs in inst["gold_licenses"].get(s, []):
            if not any(set(rw) <= set(rs) for rw in inst["gold_licenses"].get(w, [])):
                errs.append(f"order-coherence: route {sorted(rs)} of {s} "
                            f"contains no route of weaker claim {w}")

    # --- noev variant checks
    if inst["variant"] == "noev":
        disabled = [a["id"] for a in inst["atoms"] if a["resolver"] is None]
        if len(disabled) != 1:
            errs.append(f"noev variant must disable exactly one resolver, got {disabled}")
        else:
            d = disabled[0]
            top = max(claims.values(), key=lambda c: c["level"])["id"]
            for r in inst["gold_licenses"][top]:
                if d not in r:
                    errs.append(f"noev: disabled atom {d} missing from top-claim route {sorted(r)}"
                                " (gold C3 would remain reachable)")
            for c, routes in inst["gold_licenses"].items():
                if c != top and any(d in r for r in routes):
                    errs.append(f"noev: disabled atom {d} appears in route of lower claim {c}"
                                " (downgrade target would also be unreachable)")
    return errs


def main(argv):
    import glob as _glob
    files = argv or sorted(
        _glob.glob(os.path.join(BENCH, "instances", "*", "*.json")))
    schema_check, mode = load_schema_checker()
    print(f"validator: schema engine = {mode}; {len(files)} instance file(s)")
    failed = 0
    for f in files:
        errs = validate(f, schema_check)
        rel = os.path.relpath(f, BENCH)
        if errs:
            failed += 1
            print(f"FAIL {rel}")
            for e in errs:
                print(f"   - {e}")
        else:
            print(f"OK   {rel}")
    print(f"{len(files) - failed}/{len(files)} instances valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
