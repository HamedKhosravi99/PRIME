#!/usr/bin/env python3
"""PRIME runtime outcomes on SUFFICE tasks, for cross-implementation checks.

SUFFICE's scorer (``src/suffice/__init__.py`` here; ``src/suffice/`` in the
standalone benchmark) re-implements the runtime's evidence handling, so that scoring
needs no part of PRIME. This module computes, with the runtime itself (the tool
executor, the resolvers of the gold ledger, the four-valued ledger, and the
authorization check), the outcomes any scorer must agree with:

``reads``     for every task of the record splits, and for each resolver field,
              locator candidate, a missing field, and an unknown record: the
              observation (a digest) and the checks the read settles on a fresh
              ledger;
``auth``      for the reference state and 12 seeded check-value states per task:
              the conclusions the state authorizes, weaker ones included;
``attest``    for each check of the text-evidence split: acceptance and the check's
              ledger state after each step of a fixed sequence (attest before
              reading, a short quote, a quote not in the text, a bad value, a check
              that is not a text check, the verbatim quote, a repeat, and a
              contradiction);
``episodes``  when the logs are present: for each completed episode of the 15
              reference agents on the balanced split, the checks its reads settled,
              the conclusions they authorize, and the distinct call classes the
              executor charged.

    python3 src/suffice/construction/conformance.py --write PATH.json.gz

writes the fixture that the benchmark's tests/test_conformance.py reads;
tests/test_suffice_conformance.py compares the same outcomes
with ``src/suffice/__init__.py`` directly.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "prime_agents"), os.path.join(ROOT, "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
from prime_runtime import auth as prime_auth  # noqa: E402
from prime_runtime import goldledger  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

DOMAINS = ("security", "clinical", "drugsafety", "financial")
RECORD_SPLITS = ("dev", "val", "heldout_balanced", "heldout_natural", "heldout_locator", "heldout_onebranch")
TEXT_SPLIT = "heldout_text"
N_STATES = 12


def task_paths(splits: Iterable[str]) -> List[str]:
    out: List[str] = []
    for dom in DOMAINS:
        for split in splits:
            out += sorted(glob.glob(os.path.join(ROOT, "data", "suffice", dom, split, "*.json")))
    return out


def digest(obs: Dict[str, Any]) -> List[str]:
    if obs.get("error") is not None:
        return ["error", str(obs["error"])]
    return ["value", hashlib.sha1(json.dumps(obs.get("value"), sort_keys=True, default=str).encode("utf-8")).hexdigest()]


def probes(raw: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The reads to compare: every resolver field and locator candidate of the
    task, a missing field of its record, and its first field on an unknown record."""
    out: List[Tuple[str, str]] = []
    for a in raw["atoms"]:
        r = a.get("resolver") or {}
        if r.get("tool") != "inspect_record":
            continue
        rid = str((r.get("params") or {}).get("record_id", raw.get("seed")))
        out.append((rid, str(r["params"]["field_path"])))
        for c in ((a.get("spec_hint") or {}).get("candidates") or []):
            out.append((rid, str(c)))
    if out:
        out.append((out[0][0], "no.such.field"))
        out.append(("NO-SUCH-RECORD", out[0][1]))
    return list(dict.fromkeys(out))


def settle(inst, ex, ledger: Ledger, name: str, params: Dict[str, Any], step: int) -> Dict[str, Any]:
    obs = ex.execute(name, dict(params))
    for att in goldledger.delta(inst, ledger, name, dict(params), obs, step):
        ledger.apply(att)
    return obs


def read_outcomes(path: str) -> List[List[Any]]:
    inst = load_instance(path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    rows = []
    for rid, fp in probes(raw):
        ex = arms_mod.ToolExecutor(inst)
        ledger = Ledger(inst.atom_ids, inst.phi_patterns)
        obs = settle(inst, ex, ledger, "inspect_record", {"record_id": rid, "field_path": fp}, 1)
        states = {a: v for a, v in ledger.states().items() if v != "U"}
        rows.append([rid, fp, digest(obs), states])
    return rows


def auth_outcomes(path: str) -> List[List[Any]]:
    inst = load_instance(path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    program = inst.gold_program()
    ids = [a["id"] for a in raw["atoms"]]
    rng = random.Random("%s|auth" % raw["id"])
    states = [{a["id"]: a["gold_value"] for a in raw["atoms"] if a.get("gold_value") in ("T", "F")}]
    for _ in range(N_STATES):
        states.append({a: v for a in ids for v in [rng.choice(("T", "F", None))] if v})
    out = []
    for st in states:
        ledger = Ledger(inst.atom_ids, inst.phi_patterns)
        for k, (a, v) in enumerate(sorted(st.items())):
            ledger.apply({"atom": a, "value": v, "tool": "conformance", "params": {}, "step": k + 1})
        out.append([st, sorted(prime_auth.auth_ids(program, ledger))])
    return out


def attest_outcomes(path: str) -> List[List[Any]]:
    """Per text-evidence check: the sequence of (action, args, accepted, state)."""
    inst = load_instance(path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    other = next((a["id"] for a in raw["atoms"] if (a.get("resolver") or {}).get("tool") != "attest_atom"), None)
    out = []
    for a in raw["atoms"]:
        if (a.get("resolver") or {}).get("tool") != "attest_atom":
            continue
        src = (a.get("spec_hint") or {}).get("source") or {}
        rid, fp = str(src.get("record_id", "")), str(src.get("field_path", ""))
        text = " ".join(str(goldledger_text(raw, rid, fp) or "").split())
        quote = text[:48].strip()
        seq = [("attest_atom", [a["id"], "T", quote]),
               ("read_text", [rid, fp]),
               ("attest_atom", [a["id"], "T", "short"]),
               ("attest_atom", [a["id"], "T", "zq " + quote[::-1] + " qz"]),
               ("attest_atom", [a["id"], "maybe", quote]),
               ("attest_atom", [other or "E0", "T", quote]),
               ("attest_atom", [a["id"], "T", quote]),
               ("attest_atom", [a["id"], "T", quote]),
               ("attest_atom", [a["id"], "F", quote])]
        ex = arms_mod.ToolExecutor(inst)
        ledger = Ledger(inst.atom_ids, inst.phi_patterns)
        steps = []
        for k, (name, args) in enumerate(seq):
            params = ({"atom_id": args[0], "value": args[1], "quote": args[2]} if name == "attest_atom"
                      else {"record_id": args[0], "field_path": args[1]})
            obs = settle(inst, ex, ledger, name, params, k + 1)
            steps.append([name, args, obs.get("accepted") if name == "attest_atom" else None,
                          ledger.states()[a["id"]]])
        out.append([a["id"], steps])
    return out


def goldledger_text(raw: Dict[str, Any], rid: str, fp: str) -> Optional[str]:
    v, ok = arms_mod._resolve_field_path((raw.get("records") or {}).get(rid), fp)
    return v if ok and isinstance(v, str) else None


def episode_outcomes(results: str) -> Dict[str, Dict[str, List[Any]]]:
    """For each completed episode of the reference agents (reproduce_tables.AGENTS):
    the replayed reads' settled checks, authorized conclusions, and charged classes."""
    import reproduce_tables as rt
    paths = {}
    for dom in DOMAINS:
        for p in glob.glob(os.path.join(ROOT, "data", "suffice", dom, "heldout_balanced", "*.json")):
            paths[os.path.basename(p)[:-5]] = p
    out: Dict[str, Dict[str, List[Any]]] = {}
    for agent, (_name, folder, _exp) in rt.AGENTS.items():
        rows: Dict[str, List[Any]] = {}
        for dom in DOMAINS:
            run = os.path.join(results, folder, "%s-%s" % (rt.MODEL, dom), "%s__%s" % (agent, rt.BUDGET))
            with open(os.path.join(run, "summary.json"), encoding="utf-8") as fh:
                episodes = json.load(fh)["episodes"]
            for e in episodes:
                if e.get("error") or e.get("instance") not in paths:
                    continue
                inst = load_instance(paths[e["instance"]])
                ex = arms_mod.ToolExecutor(inst)
                ledger = Ledger(inst.atom_ids, inst.phi_patterns)
                step = 0
                with open(os.path.join(run, e["instance"] + ".jsonl"), encoding="utf-8") as fh:
                    for line in fh:
                        r = json.loads(line)
                        if r.get("type") == "action" and r.get("name") == "inspect_record":
                            p = r.get("params") or {}
                            step += 1
                            settle(inst, ex, ledger, "inspect_record",
                                   {"record_id": str(p.get("record_id", "")), "field_path": str(p.get("field_path", ""))}, step)
                states = {a: v for a, v in ledger.states().items() if v != "U"}
                logged = {a: v for a, v in (e.get("final_ledger") or {}).items() if v != "U"}
                rows[e["instance"]] = [states, sorted(prime_auth.auth_ids(inst.gold_program(), ledger)),
                                       len(ex.observed_classes), logged]
        out[agent] = rows
    return out


def build(results: Optional[str]) -> Dict[str, Any]:
    fx: Dict[str, Any] = {"reads": {}, "auth": {}, "attest": {}, "episodes": {}}
    for p in task_paths(RECORD_SPLITS):
        tid = os.path.basename(p)[:-5]
        fx["reads"][tid] = read_outcomes(p)
        fx["auth"][tid] = auth_outcomes(p)
    for p in task_paths((TEXT_SPLIT,)):
        tid = os.path.basename(p)[:-5]
        fx["attest"][tid] = attest_outcomes(p)
        fx["auth"][tid] = auth_outcomes(p)
    if results and os.path.isdir(os.path.join(results, "w2a")):
        fx["episodes"] = episode_outcomes(results)
    return fx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", required=True, help="output path (.json.gz)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    a = ap.parse_args(argv)
    fx = build(a.results)
    with gzip.open(a.write, "wt", encoding="utf-8") as fh:
        json.dump(fx, fh, sort_keys=True, separators=(",", ":"))
    print("tasks with reads %d, auth %d, text %d; episodes for %d agents -> %s"
          % (len(fx["reads"]), len(fx["auth"]), len(fx["attest"]), len(fx["episodes"]), a.write))
    return 0


if __name__ == "__main__":
    sys.exit(main())
