#!/usr/bin/env python3
"""FDAAA: the frozen program on untouched registrations.

PROTOCOL, frozen before any label is read (2026-10-10):
  population   trials listed in the FDAAA TrialsTracker export of 3 August
               2026 (bench/fdaaa/external/trialstracker_2026-08-03.csv) that
               are NOT in the 6,000-registration development pool and whose
               sponsor_slug does not occur in that pool
  sample       2,000 of them, drawn uniformly at random with seed 20261010
               from the sorted list of eligible registry ids
  records      fetched from the public ClinicalTrials.gov API v2, one request
               per trial, stored verbatim (results/records/<NCT>.json); a
               trial whose fetch fails is reported as such, never backfilled
  program      the FDAAA program exactly as committed: compile_contract and
               build_instance imported, unmodified, from
               bench/fdaaa/build_fdaaa_instances.py (reading D: calendar
               year, certification timeliness, no administrative grace; the
               runtime's predicate grammar decides every atom)
  selection    the builder's own: drop withheld, not applicable (not an ACT),
               no deadline, not yet due; then the runtime applicability check
               (E1, E2, E3 all attest T); the sponsor-size and cap rules of
               the instance builder are NOT applied (they shaped the 319 for
               clustering, not for classification)
  reference    the tracker's status, read only after classification:
               adverse = {reported-late, overdue, overdue-cancelled}; the
               program's adverse = {late, overdue}
  reported     agreement adverse-vs-adverse with a Wilson interval, the
               three-way cross-tabulation, coverage (share of fetched trials
               that reach a classification), attrition by reason, and the
               same figures restricted to the tracker statuses the original
               319 covered (reported, reported-late, overdue)
No reading is reselected, no threshold tuned, nothing rerun after reading
the labels. The snapshot date of the program (3 August 2026) is kept; a
trial whose deadline falls after that date is "not yet due" exactly as in
the builder.

Usage: python3 heldout_eval.py fetch   (network; resumable)
       python3 heldout_eval.py score   (CPU; writes results/heldout_eval.json)
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PILOT, "bench", "fdaaa"))
sys.path.insert(0, PILOT)
import build_fdaaa_instances as B  # noqa: E402  (read-only import of the frozen builder)

EXT = os.path.join(PILOT, "bench", "fdaaa", "external")
RES = os.path.join(HERE, "results"); REC = os.path.join(RES, "records")
SEED = 20261010; N_SAMPLE = 2000
API = "https://clinicaltrials.gov/api/v2/studies/%s"
ADVERSE_TRACKER = {"reported-late", "overdue", "overdue-cancelled"}
ADVERSE_OURS = {"late", "overdue"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def sample_ids():
    pool = json.load(open(os.path.join(EXT, "ctgov_v2_6000.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(EXT, "trialstracker_2026-08-03.csv"), encoding="utf-8")))
    pool_sponsors = {r["sponsor_slug"] for r in rows if r["registry_id"] in pool}
    eligible = sorted(r["registry_id"] for r in rows
                      if r["registry_id"] not in pool and (r.get("sponsor_slug") or "unknown") not in pool_sponsors)
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(eligible, N_SAMPLE))
    return chosen, {"tracker_rows": len(rows), "pool": len(pool), "pool_sponsors": len(pool_sponsors),
                    "eligible_outside_pool_and_sponsors": len(eligible), "sampled": len(chosen), "seed": SEED}


def fetch():
    os.makedirs(REC, exist_ok=True)
    ids, meta = sample_ids()
    json.dump({"meta": meta, "ids": ids}, open(os.path.join(RES, "sample.json"), "w"), indent=1)
    done = failed = 0; t0 = time.time()
    for i, nct in enumerate(ids, 1):
        path = os.path.join(REC, nct + ".json")
        if os.path.exists(path):
            done += 1; continue
        r = subprocess.run(["curl", "-s", "--max-time", "40", API % nct], capture_output=True, text=True)
        try:
            doc = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
        except json.JSONDecodeError:
            doc = None
        if doc is None or "protocolSection" not in doc:
            failed += 1
            with open(os.path.join(RES, "fetch_failures.txt"), "a") as fh:
                fh.write(nct + "\n")
        else:
            json.dump(doc, open(path, "w"), indent=0)
            done += 1
        if i % 100 == 0:
            print("[fetch] %d/%d fetched=%d failed=%d %.0fs" % (i, len(ids), done, failed, time.time() - t0), flush=True)
        time.sleep(0.2)
    print("[fetch] complete fetched=%d failed=%d" % (done, failed), flush=True)


def score():
    ids = json.load(open(os.path.join(RES, "sample.json")))["ids"]
    tracker = {r["registry_id"]: r for r in csv.DictReader(open(os.path.join(EXT, "trialstracker_2026-08-03.csv"), encoding="utf-8"))}
    attrition = collections.Counter(); classified = {}
    for nct in ids:
        path = os.path.join(REC, nct + ".json")
        if not os.path.exists(path):
            attrition["fetch-failed"] += 1; continue
        study = json.load(open(path))
        c = B.compile_contract(study)
        if not c["adjudicable"]:
            attrition["withheld"] += 1; continue
        if not c["act"]:
            attrition["not-act"] += 1; continue
        if not c["due"]:
            attrition["no-deadline"] += 1; continue
        if c["verdict"] == "not_yet_due":
            attrition["not-due"] += 1; continue
        inst = B.build_instance(nct, study, c, tracker.get(nct, {}).get("sponsor_slug") or "unknown")
        gv = {a["id"]: a["gold_value"] for a in inst["atoms"]}
        if any(gv.get(e) != "T" for e in ("E1", "E2", "E3")):
            attrition["applicability-undeterminable"] += 1; continue
        classified[nct] = {"verdict": c["verdict"], "atoms": gv, "cert_filed": c["cert_filed"], "cert_valid": c["cert_valid"]}
    # ---- labels are read only from here on ----
    conf = collections.Counter(); agree = checked = 0
    for nct, r in classified.items():
        ext = tracker[nct]["status"]
        conf[(r["verdict"], ext)] += 1
        checked += 1; agree += int((r["verdict"] in ADVERSE_OURS) == (ext in ADVERSE_TRACKER))
    # the original 319 covered tracker statuses reported / reported-late / overdue only
    sub = {n: r for n, r in classified.items() if tracker[n]["status"] in ("reported", "reported-late", "overdue")}
    agree_sub = sum(int((r["verdict"] in ADVERSE_OURS) == (tracker[n]["status"] in ADVERSE_TRACKER)) for n, r in sub.items())
    out = {"protocol": __doc__.split("Usage:")[0], "n_sampled": len(ids), "attrition": dict(attrition),
           "n_classified": len(classified), "coverage_of_fetched": len(classified) / max(1, len(ids) - attrition["fetch-failed"]),
           "agreement_adverse": agree / max(1, checked), "agreement_ci": wilson(agree, checked), "n_checked": checked,
           "agreement_on_original_statuses": agree_sub / max(1, len(sub)), "agreement_on_original_statuses_ci": wilson(agree_sub, len(sub)),
           "n_original_statuses": len(sub),
           "confusion": {"%s|%s" % k: v for k, v in sorted(conf.items())},
           "verdicts": dict(collections.Counter(r["verdict"] for r in classified.values())),
           "tracker_statuses": dict(collections.Counter(tracker[n]["status"] for n in classified)),
           "with_certification": sum(1 for r in classified.values() if r["cert_filed"]),
           "disagreements_with_certification": sum(1 for n, r in classified.items() if r["cert_filed"] and ((r["verdict"] in ADVERSE_OURS) != (tracker[n]["status"] in ADVERSE_TRACKER)))}
    json.dump(out, open(os.path.join(RES, "heldout_eval.json"), "w"), indent=1, sort_keys=True)
    print("sampled %d; attrition %s; classified %d (coverage of fetched %.1f%%)" % (len(ids), dict(attrition), len(classified), 100 * out["coverage_of_fetched"]))
    print("agreement adverse-vs-adverse: %d/%d = %.2f%% [%.2f, %.2f]" % (agree, checked, 100 * out["agreement_adverse"], 100 * out["agreement_ci"][0], 100 * out["agreement_ci"][1]))
    print("on the tracker statuses the original 319 covered: %d/%d = %.2f%% [%.2f, %.2f]" % (
        agree_sub, len(sub), 100 * out["agreement_on_original_statuses"], 100 * out["agreement_on_original_statuses_ci"][0], 100 * out["agreement_on_original_statuses_ci"][1]))
    print("confusion (ours|tracker):", out["confusion"])
    print("verdicts:", out["verdicts"], "| tracker:", out["tracker_statuses"], "| certifications:", out["with_certification"], "disagreements among them:", out["disagreements_with_certification"])


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    {"fetch": fetch, "score": score}[sys.argv[1]]()
