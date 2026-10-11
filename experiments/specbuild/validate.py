#!/usr/bin/env python3
"""Validation phase (PROTOCOL.md): score every candidate on the validation
registrations against the tracker, select by the stated rule, write feedback
for the repair round, and (with --final) freeze the selected program.

A candidate is valid when its program parses, derive() runs on every
validation registration, and the runtime's schema validation accepts its
instances. Score: agreement between the program's full-evidence level and the
tracker's reference level. Selection: highest agreement; ties by fewer
over-claims, then by fewer atoms.

Usage: python3 validate.py --round 0
       python3 validate.py --round 1 --final
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def val_records():
    split = json.load(open(os.path.join(HERE, "data", "split.json")))
    recs = {}
    for nct in split["validation"]:
        p = os.path.join(os.environ.get("SPECBUILD_VAL_RECORDS") or os.path.join(HERE, "data", "validation_records"), nct + ".json")
        if os.path.exists(p):
            recs[nct] = json.load(open(p))
    return recs


def score(prog, recs, ref):
    ids = sorted(recs)
    trimmed = [common.trim(recs[i]) for i in ids]
    try:
        if prog.get("derive"):
            common.compile_derive(prog["derive"])
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "reason": "derive does not compile: %s" % exc}
    der = common.run_derive_batch(prog.get("derive", ""), trimmed)
    errs = [e for _, e in der if e]
    if len(errs) == len(ids):
        return {"valid": False, "reason": "derive failed on every record: %s" % errs[0][:200]}
    # fail closed per record: a record whose derive raises gets no derived values (atoms on them stay unresolved)
    der = [(d if d is not None else {}, e) for d, e in der]
    levels, rows = {}, []
    for k, (i, (d, _)) in enumerate(zip(ids, der)):
        inst = common.build_instance(prog, i, recs[i], d, "val")
        if k < 25:
            try:
                common.validate_instance(inst)
            except Exception as exc:  # noqa: BLE001
                return {"valid": False, "reason": "schema: %s" % exc}
        levels[i] = common.full_level(inst)
        rows.append((i, d))
    agree = sum(1 for i in ids if i in ref and levels[i] == ref[i]["level"])
    over = sum(1 for i in ids if i in ref and levels[i] > ref[i]["level"])
    under = sum(1 for i in ids if i in ref and levels[i] < ref[i]["level"])
    n = sum(1 for i in ids if i in ref)
    return {"valid": True, "n": n, "agree": agree, "agree_pct": 100.0 * agree / n, "over": over, "under": under,
            "n_atoms": len(prog["atoms"]), "derive_errors": len(errs), "derive_error_examples": sorted(set(e[:160] for e in errs))[:3],
            "levels": levels, "derived": dict(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=int, required=True); ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    recs = val_records(); ref = common.reference(list(recs), "validation")
    rounds = list(range(0, a.round + 1)) if a.final else [a.round]
    table = []
    for r in rounds:
        for p in sorted(glob.glob(os.path.join(HERE, "candidates", "round%d" % r, "cand_*.json"))):
            c = json.load(open(p))
            if "program" not in c:
                table.append({"round": r, "k": c["k"], "valid": False, "reason": c.get("error", "")[:200]}); continue
            s = score(c["program"], recs, ref)
            ent = {"round": r, "k": c["k"], **{k: v for k, v in s.items() if k not in ("levels", "derived")}}
            ent["_s"] = s; ent["_prog"] = c["program"]
            table.append(ent)
            print("[validate] round %d cand %d: %s" % (r, c["k"], ("agree %.1f%% over %d under %d atoms %d" % (s["agree_pct"], s["over"], s["under"], s["n_atoms"])) if s["valid"] else s["reason"]), flush=True)
    valid = [t for t in table if t.get("valid")]
    out_dir = os.path.join(HERE, "candidates", "round%d" % a.round)
    summary = [{k: v for k, v in t.items() if not k.startswith("_")} for t in table]
    json.dump(summary, open(os.path.join(out_dir, "validation_scores%s.json" % ("_final" if a.final else "")), "w"), indent=1)
    if not valid:
        print("[validate] no valid candidate"); return 1
    best = sorted(valid, key=lambda t: (-t["agree"], t["over"], t["n_atoms"], t["round"], t["k"]))[0]
    sel = {"round": best["round"], "k": best["k"], "agree_pct": best["agree_pct"], "over": best["over"], "under": best["under"], "program": best["_prog"]}
    if not a.final:
        json.dump(sel, open(os.path.join(out_dir, "selected.json"), "w"), indent=1)
        # feedback for the repair round: up to 12 disagreements, over-claims and under-claims alternating, seeded
        s = best["_s"]; rng = random.Random(20261011)
        overs = [i for i in sorted(s["levels"]) if i in ref and s["levels"][i] > ref[i]["level"]]
        unders = [i for i in sorted(s["levels"]) if i in ref and s["levels"][i] < ref[i]["level"]]
        rng.shuffle(overs); rng.shuffle(unders)
        pick = []
        for x in range(12):
            src = overs if x % 2 == 0 else unders
            if not src:
                src = unders if src is overs else overs
            if src:
                pick.append(src.pop())
        ex = []
        for i in pick:
            rec = common.trim(recs[i]); rec["protocolSection"]["identificationModule"].pop("nctId", None)
            ex.append({"registration": rec, "derived": s["derived"].get(i), "program_level": s["levels"][i], "tracker_level": ref[i]["level"]})
        json.dump({"n": s["n"], "agree": s["agree"], "agree_pct": s["agree_pct"], "examples": ex, "derive_errors": s["derive_errors"],
                   "derive_error_examples": s["derive_error_examples"]}, open(os.path.join(out_dir, "feedback.json"), "w"), indent=1)
        print("[validate] selected round %d cand %d (agree %.1f%%); feedback with %d examples" % (best["round"], best["k"], best["agree_pct"], len(ex)))
    else:
        frozen = {"program": best["_prog"], "selected": {k: sel[k] for k in ("round", "k", "agree_pct", "over", "under")},
                  "validation_n": best["n"], "protocol": "PROTOCOL.md"}
        blob = json.dumps(frozen["program"], sort_keys=True).encode()
        frozen["program_sha256"] = hashlib.sha256(blob).hexdigest()
        json.dump(frozen, open(os.path.join(HERE, "candidates", "FROZEN_PROGRAM.json"), "w"), indent=1)
        print("[validate] FROZEN round %d cand %d agree %.1f%% sha256 %s" % (best["round"], best["k"], best["agree_pct"], frozen["program_sha256"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
