#!/usr/bin/env python3
"""Secondary analysis of the end-to-end test (run after score_e2e.py).

Reads the test instances, the agent summaries, the fetched test records, and
the tracker labels, and writes reported/analysis.json with:
- route check: emissions without a fully established route, per program and arm;
- episode cost: tool calls, runtime-completion calls, and tokens per episode;
- coverage of the fixed policy on registrations the tracker counts as due;
- full-evidence level agreement on due and not-yet-due registrations;
- the unsupported conclusions of the fixed policy by tracker status, stated
  conclusion, and whether a certification of delay was filed;
- adverse-against-adverse classification agreement: program A from its
  builder's verdicts (the paper's measure); program B, which has no
  compliant/overdue verdict, from its levels with a due-but-not-late
  registration counted compliant when the registry records a results
  submission and overdue otherwise;
- the validation sample's status mix.

Usage: python3 analyze_e2e.py
"""
from __future__ import annotations

import ast
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

RECDIR = os.environ.get("SPECBUILD_TEST_RECORDS") or os.path.join(common.PILOT, "experiments", "fdaaa_heldout", "results", "records")
ARMS = ("react_program_attested", "prime_guided", "prime_complete")


def _lit(v, default):
    if v in (None, ""):
        return default
    return ast.literal_eval(v) if isinstance(v, str) else v


def has_cert(nct):
    r = json.load(open(os.path.join(RECDIR, nct + ".json")))
    return bool(r.get("protocolSection", {}).get("statusModule", {}).get("dispFirstSubmitDate"))


def main() -> int:
    ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RECDIR, "NCT*.json")))
    ref = common.reference(ids, "test")
    out = {"n_test": len(ids), "programs": {}}
    for tag in ("A", "B"):
        inst = {}
        for p in glob.glob(os.path.join(HERE, "data", "test_%s" % tag, "*.json")):
            x = json.load(open(p)); inst[x["seed"]] = x
        level = {i: common.full_level(inst[i]) if i in inst else 0 for i in ids}
        due = [i for i in ids if ref[i]["level"] >= 2]
        notdue = [i for i in ids if ref[i]["level"] == 1]
        unsup = [i for i in ids if level[i] > ref[i]["level"]]
        prog = {
            "coverage_on_due": {"n": len(due), "covered": sum(1 for i in due if level[i] >= 1)},
            "level_agreement_due": 100 * sum(level[i] == ref[i]["level"] for i in due) / len(due),
            "level_agreement_not_due": 100 * sum(level[i] == ref[i]["level"] for i in notdue) / len(notdue),
            "unsupported_fixed": {
                "n": len(unsup),
                "with_certification": sum(1 for i in unsup if has_cert(i)),
                "by_status_and_stated": dict(collections.Counter("%s->C%d" % (ref[i]["status"], level[i]) for i in unsup)),
            },
        }
        # classification agreement, adverse against adverse
        verdict = {}
        for i, x in inst.items():
            if tag == "A":
                verdict[i] = json.loads(x["notes_for_annotators"])["verdict"]
            elif level[i] >= 2:
                rs = x["records"][i].get("protocolSection", {}).get("statusModule", {}).get("resultsFirstSubmitDate")
                verdict[i] = "late" if level[i] == 3 else ("compliant" if rs else "overdue")
        agree = sum(1 for i in verdict if (verdict[i] in ("late", "overdue")) == ref[i]["adverse"])
        prog["classification_agreement"] = {"classified": len(verdict), "agree": agree, "pct": 100 * agree / max(1, len(verdict)),
                                            "verdicts": dict(collections.Counter(verdict.values())),
                                            "definition": "builder verdicts" if tag == "A" else "levels; due-not-late split by a recorded results submission"}
        # agent arms: route check and cost
        agent_inst = {}
        for p in glob.glob(os.path.join(HERE, "data", "test_%s_agent" % tag, "*.json")):
            x = json.load(open(p)); agent_inst[x["id"]] = x
        arms = {}
        for arm in ARMS:
            p = os.path.join(HERE, "results", tag, arm + "__B20", "summary.json")
            if not os.path.exists(p):
                continue
            eps = json.load(open(p))["episodes"]
            n_emit = no_route = errors = calls = completion = tokens = 0
            for e in eps:
                errors += bool(e.get("error"))
                calls += int(e.get("tool_calls") or 0)
                completion += int(e.get("discharged_calls") or 0)
                t = _lit(e.get("tokens"), {})
                tokens += t.get("prompt_tokens", 0) + t.get("completion_tokens", 0)
                led = _lit(e.get("final_ledger"), {})
                routes_of = agent_inst[e["instance"]]["gold_licenses"]
                for c in _lit(e.get("emitted"), []):
                    n_emit += 1
                    ok = any(all(led.get(a.lstrip("!")) == ("F" if a.startswith("!") else "T") for a in r) for r in routes_of.get(c["claim"], []))
                    no_route += not ok
            n = len(eps)
            arms[arm] = {"episodes": n, "episode_errors": errors, "emissions": n_emit, "emissions_without_route": no_route,
                         "tool_calls_per_episode": calls / n, "completion_calls_per_episode": completion / n, "tokens_per_episode": tokens / n}
        prog["arms"] = arms
        out["programs"][tag] = prog
    split = json.load(open(os.path.join(HERE, "data", "split.json")))
    vids = split["validation"]
    vref = common.reference(vids, "validation")
    vrec = os.path.join(HERE, "data", "validation_records")
    out["validation_sample"] = {
        "n": len(vids),
        "status": dict(collections.Counter(r["status"] for r in vref.values())),
        "due": sum(1 for r in vref.values() if r["level"] >= 2),
        "with_certification": sum(1 for i in vids if os.path.exists(os.path.join(vrec, i + ".json")) and json.load(open(os.path.join(vrec, i + ".json"))).get("protocolSection", {}).get("statusModule", {}).get("dispFirstSubmitDate")),
    }
    out["test_sample"] = {"status": dict(collections.Counter(r["status"] for r in ref.values())),
                          "due": sum(1 for r in ref.values() if r["level"] >= 2),
                          "with_certification": sum(1 for i in ids if has_cert(i))}
    os.makedirs(os.path.join(HERE, "reported"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "reported", "analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
