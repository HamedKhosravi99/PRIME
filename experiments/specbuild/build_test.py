#!/usr/bin/env python3
"""Test phase, part 1 (PROTOCOL.md): PRIME instances for the 1,999 held-out
registrations under (A) the paper's program, through its unchanged builder,
whose exclusions are abstentions, and (B) the frozen program of the
validation phase. Also draws the agent subset: 600 registrations, uniformly by
id with seed 20261011, before any label is read. Reads no tracker status.

Usage: python3 build_test.py [--frozen candidates/FROZEN_PROGRAM.json]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.environ.get("SPECBUILD_PILOT") or os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PILOT, "src", "obligationbench", "construction", "fdaaa"))
import common  # noqa: E402
import build_fdaaa_instances as BA  # noqa: E402  (read-only import of the paper's builder)

REC = os.environ.get("SPECBUILD_TEST_RECORDS") or os.path.join(PILOT, "experiments", "fdaaa_heldout", "results", "records")
OUT = os.path.join(HERE, "data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", default=os.path.join(HERE, "candidates", "FROZEN_PROGRAM.json"))
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(REC, "NCT*.json")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    sponsor = {r["registry_id"]: r.get("sponsor_slug") or "unknown" for r in csv.DictReader(open(os.path.join(common.EXT, "trialstracker_2026-08-03.csv")))}
    subset = sorted(random.Random(20261011).sample(ids, 600))
    json.dump({"seed": 20261011, "n_test": len(ids), "agent_subset": subset}, open(os.path.join(OUT, "test_split.json"), "w"), indent=1)
    studies = {i: json.load(open(p)) for i, p in zip(ids, paths)}
    man = {"n_test": len(ids), "A": collections.Counter(), "B": collections.Counter()}

    # (A) the paper's program
    da = os.path.join(OUT, "test_A"); shutil.rmtree(da, ignore_errors=True); os.makedirs(da)
    for i in ids:
        s = studies[i]; c = BA.compile_contract(s)
        why = None
        if not c["adjudicable"]:
            why = "withheld"
        elif not c["act"]:
            why = "not-act"
        elif not c["due"]:
            why = "no-deadline"
        elif c["verdict"] == "not_yet_due":
            why = "not-due"
        if why is None:
            inst = BA.build_instance(i, s, c, sponsor.get(i, "unknown"))
            gv = {x["id"]: x["gold_value"] for x in inst["atoms"]}
            if any(gv.get(e) != "T" for e in ("E1", "E2", "E3")):
                why = "applicability-undeterminable"
            else:
                json.dump(inst, open(os.path.join(da, inst["id"] + ".json"), "w"), indent=1, sort_keys=True)
                man["A"]["instance"] += 1
        if why:
            man["A"][why] += 1

    # (B) the frozen program
    db = os.path.join(OUT, "test_B"); shutil.rmtree(db, ignore_errors=True); os.makedirs(db)
    frozen = json.load(open(a.frozen)); prog = frozen["program"]
    der = common.run_derive_batch(prog.get("derive", ""), [common.trim(studies[i]) for i in ids], timeout=300.0)
    for k, (i, (d, err)) in enumerate(zip(ids, der)):
        if err:
            man["B"]["derive-error (fail-closed: no derived values)"] += 1; d = {}
        inst = common.build_instance(prog, i, studies[i], d, "specB")
        if k < 50:
            common.validate_instance(inst)
        json.dump(inst, open(os.path.join(db, inst["id"] + ".json"), "w"), indent=1, sort_keys=True)
        man["B"]["instance"] += 1

    # agent-run copies restricted to the subset
    for tag, d in (("A", da), ("B", db)):
        dd = d + "_agent"; shutil.rmtree(dd, ignore_errors=True); os.makedirs(dd)
        for p in glob.glob(os.path.join(d, "*.json")):
            if json.load(open(p))["seed"] in subset:
                shutil.copy(p, dd)
        man[tag]["agent_instances"] = len(glob.glob(os.path.join(dd, "*.json")))
    man["frozen_sha256"] = frozen.get("program_sha256")
    man = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in man.items()}
    json.dump(man, open(os.path.join(OUT, "test_manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
