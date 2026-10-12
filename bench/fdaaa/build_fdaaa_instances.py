#!/usr/bin/env python3
"""Generate ObligationBench instances whose contract we did not author.

Every other family in this benchmark has the same weakness: we wrote the
claim schema, we wrote the gold licences, we wrote the oracle, and we broke
the records so that the false-authorization endpoint would be identifiable
at all. Nothing in the paper touches anything external.

This family fixes that at the only place it can be fixed -- the contract.

  * The CONTRACT is 42 CFR Part 11, implementing FDAAA 801. Congress and
    FDA wrote it. Section 11.22(b) is drafted in the registry's own
    data-element vocabulary ("Study Type", "Study Phase", "Primary
    Purpose", "Facility Location" are literal ClinicalTrials.gov field
    names), which is why this statute, almost uniquely among regulations,
    compiles into a predicate grammar at all.

  * The RECORDS are real ClinicalTrials.gov v2 registrations, frozen and
    hashed. Nothing is degraded. A claim fails here because a sponsor
    missed a statutory deadline, not because we removed a field.

  * The VERDICTS are cross-checkable against the Bennett Institute's FDAAA
    TrialsTracker, an independently authored and independently maintained
    decision procedure over the same registry.

Every atom lands in the EXISTING closed predicate grammar of
prime_runtime/predicates.py -- ten predicate types, none added for this
family. That is worth stating plainly: the instrument did not have to grow
to accept a standard from outside it.

CLUSTERING. The unit is the SPONSOR, declared here and before any scoring.
Reporting practice is a property of the organisation -- a sponsor with a
compliance office behaves the same way across its portfolio -- so trials
from one sponsor are correlated and cannot be treated as independent. This
is the same role the template family plays elsewhere in the benchmark.

CLAIM LADDER. The dominance order is the statute's own clause structure,
not a design of ours:

  C1  this registration is an applicable clinical trial (Sec. 11.22(b))
  C2  results were therefore due by a determinate date (Sec. 11.44(a)-(c))
  C3  this trial missed that date                      (Sec. 11.44(a))

C3 entails C2 entails C1, and each step needs strictly more evidence, which
is exactly the structure PRIME's mask consumes.

Usage:
  python3 bench/fdaaa/build_fdaaa_instances.py --limit 600
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
PILOT = os.path.dirname(BENCH)
if PILOT not in sys.path:
    sys.path.insert(0, PILOT)
from prime_runtime import goldledger as _goldledger   # noqa: E402

EXT = os.path.join(HERE, "external")
OUT = os.path.join(BENCH, "instances_fdaaa")

# The Final Rule's compliance date. Sec. 11.22(b) applies to trials whose
# start date is on or after this; the pre-2017 statutory branch needs a
# separate clause-by-clause pass and is deliberately out of scope.
RULE_DATE = dt.date(2017, 1, 18)
SNAPSHOT = dt.date(2026, 8, 3)          # the validator's stamp
US = ["United States", "American Samoa", "Guam", "Northern Mariana Islands",
      "Puerto Rico", "Virgin Islands (U.S.)"]


# --------------------------------------------------------------------------
# date handling.  Sec. 11.44(a) says "1 year", and elsewhere Part 11 says "30
# calendar days" -- the drafting distinguishes them, so this is a calendar
# year and not 365 days.  Month-precision dates are a registry fact, not a
# regulatory one; resolving them to end-of-month is a declared BENCHMARK
# convention and every instance that uses it carries a flag saying so.
# --------------------------------------------------------------------------
def parse_date(s, end_of_month=False):
    if not s:
        return None
    p = s.split("-")
    if len(p) == 3:
        return dt.date(int(p[0]), int(p[1]), int(p[2]))
    if len(p) == 2:
        y, m = int(p[0]), int(p[1])
        if not end_of_month:
            return dt.date(y, m, 1)
        return dt.date(y + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1)
    return dt.date(int(p[0]), 12 if end_of_month else 1,
                   31 if end_of_month else 1)


def plus_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:            # 29 Feb
        return d.replace(year=d.year + n, day=28)


def get(d, path):
    """Dotted lookup. Absent key returns None -- and absent is NOT false.

    The oversight booleans are omitted entirely when unset, so a harness
    that coerces missing to False silently reclassifies trials. The
    three-valued ledger is what this benchmark is for; this is where it
    earns its keep.
    """
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------
# the contract, clause by clause
# --------------------------------------------------------------------------
def compile_contract(study):
    """Return the statute's determinations for one registration."""
    ps = study.get("protocolSection") or {}
    st = ps.get("statusModule") or {}
    des = ps.get("designModule") or {}
    ov = ps.get("oversightModule") or {}

    start = parse_date(get(st, "startDateStruct.date"))
    pcd_raw = (get(st, "primaryCompletionDateStruct.date")
               or get(st, "completionDateStruct.date"))
    pcd = parse_date(pcd_raw, end_of_month=True)
    phases = des.get("phases") or []
    countries = [l.get("country") for l
                 in ((ps.get("contactsLocationsModule") or {}).get("locations") or [])]

    c = {
        "adjudicable": st.get("overallStatus") != "WITHHELD",
        "interventional": des.get("studyType") == "INTERVENTIONAL",
        "post_rule": bool(start and start >= RULE_DATE),
        "not_phase1": not (phases and set(phases) <= {"PHASE1", "EARLY_PHASE1"}),
        "fda_drug": ov.get("isFdaRegulatedDrug") is True,
        "fda_device": ov.get("isFdaRegulatedDevice") is True,
        "us_site": any(x in US for x in countries if x),
        "us_export": ov.get("isUsExport") is True,
        "pcd": pcd,
        "pcd_raw": pcd_raw,
        "date_defaulted": bool(pcd_raw and len(pcd_raw.split("-")) < 3),
    }
    c["jurisdiction"] = c["us_site"] or c["us_export"]
    c["act"] = (c["adjudicable"] and c["interventional"] and c["post_rule"]
                and c["not_phase1"] and (c["fda_drug"] or c["fda_device"])
                and c["jurisdiction"])

    # Sec. 11.44(a): one calendar year from primary completion.
    c["due_base"] = plus_years(pcd, 1) if pcd else None

    # Sec. 11.44(b)(1)/(c)(1): a certification defers the deadline ONLY if
    # filed before the date results would otherwise have been due. The
    # external validator does not gate on this; roughly a fifth of filed
    # certifications fail it, and those are the discriminative cases.
    cert = parse_date(st.get("dispFirstSubmitDate"))
    c["cert_filed"] = cert is not None
    c["cert_valid"] = bool(cert and c["due_base"] and cert <= c["due_base"])
    c["cert_date"] = cert
    c["due"] = (plus_years(cert, 2) if c["cert_valid"] else c["due_base"])

    # Sec. 11.44 obligates SUBMISSION. Posting lags QC by weeks to months,
    # so reading the posting date instead is simply the wrong predicate.
    sub = parse_date(st.get("resultsFirstSubmitDate"))
    if sub is None:
        sub = parse_date(get(study, "derivedSection.miscInfoModule."
                                    "submissionTracking.estimatedResultsFirstSubmitDate"))
    c["submitted"] = sub

    if not c["due"]:
        c["verdict"] = None
    elif sub is not None:
        c["verdict"] = "compliant" if sub <= c["due"] else "late"
    else:
        c["verdict"] = "overdue" if c["due"] < SNAPSHOT else "not_yet_due"
    return c


# --------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------
def atom(aid, nct, path, ptype, arg, gold):
    return {
        "id": aid,
        "gold_value": gold,
        "proposition": "",
        "resolver": {
            "tool": "inspect_record",
            "attest": "record_predicate:%s:%s" % (ptype, json.dumps(arg)),
            "params": {"record_id": nct, "field_path": path},
            "params_schema": {"record_id": "str", "field_path": "str"},
        },
    }


_MISSING = object()


def truth(record, path, attest):
    """Gold value, obtained by ASKING THE RUNTIME, never by reimplementing it.

    The first version of this function computed gold with its own date
    parsing and disagreed with the deployed oracle on 8 of 3174 atoms --
    month-precision start dates, which the runtime's date_after refuses and
    a hand-rolled parser happily accepts. Reimplementing the exact check is
    a mistake this project has made before; the fix is not a better
    reimplementation, it is not having one. Gold is whatever the same
    attestation path the episode uses returns, so the two cannot diverge.
    """
    cur = record
    for k in path.split("."):
        cur = cur.get(k, _MISSING) if isinstance(cur, dict) else _MISSING
        if cur is _MISSING:
            break
    obs = {"value": cur} if cur is not _MISSING else {"error": "field not found"}
    return _goldledger.eval_attest(attest, obs) or "U"


def build_instance(nct, study, c, sponsor):
    ps = study.get("protocolSection") or {}
    st = ps.get("statusModule") or {}
    des = ps.get("designModule") or {}
    ov = ps.get("oversightModule") or {}

    # Trim the record to the modules the contract actually reads. Nothing is
    # altered; fields are copied verbatim or dropped whole.
    rec = {"protocolSection": {
        "identificationModule": {
            "nctId": nct,
            "briefTitle": get(ps, "identificationModule.briefTitle")},
        "statusModule": {k: st[k] for k in
                         ("overallStatus", "startDateStruct",
                          "primaryCompletionDateStruct", "completionDateStruct",
                          "resultsFirstSubmitDate", "dispFirstSubmitDate")
                         if k in st},
        "designModule": {k: des[k] for k in ("studyType", "phases") if k in des},
        "oversightModule": {k: ov[k] for k in
                            ("isFdaRegulatedDrug", "isFdaRegulatedDevice",
                             "isUsExport") if k in ov},
        "contactsLocationsModule": {
            "locations": [{"country": l.get("country")} for l in
                          ((ps.get("contactsLocationsModule") or {})
                           .get("locations") or [])][:40]},
    }}
    tk = get(study, "derivedSection.miscInfoModule.submissionTracking")
    if tk:
        rec["derivedSection"] = {"miscInfoModule": {"submissionTracking": tk}}

    specs = [
        ("E1", "protocolSection.designModule.studyType", "equals",
         "INTERVENTIONAL"),
        ("E2", "protocolSection.statusModule.startDateStruct.date",
         "date_after", RULE_DATE.isoformat()),
        ("E3", "protocolSection.oversightModule.isFdaRegulatedDrug",
         "equals", True),
        ("E4", "protocolSection.statusModule.primaryCompletionDateStruct.date",
         "exists", None),
    ]
    due_iso = c["due"].isoformat() if c["due"] else RULE_DATE.isoformat()
    sub_path = "protocolSection.statusModule.resultsFirstSubmitDate"
    cert_path = "protocolSection.statusModule.dispFirstSubmitDate"

    # E5 is the deadline. It asks whether the submission that EXISTS came
    # after the statutory due date, not whether the trial is delinquent.
    #
    # That distinction is forced, and it is the most interesting thing this
    # family exposes. The statute's operative claim -- "did not submit by
    # the deadline" -- is a NEGATIVE EXISTENTIAL, and the attested ledger is
    # fail-closed: a field the tool cannot read attests nothing, so absence
    # is unavailable, never false. A never-submitted trial therefore leaves
    # E5 at U and the top claim UNLICENSED, however overdue it is. That is
    # correct behaviour and not a gap to paper over: licensing "they never
    # filed" from a snapshot in which the field is merely missing is a
    # closed-world assumption this ledger deliberately refuses.
    #
    # So the overdue stratum is the family's abstention test: the correct
    # action there is to stop at the deadline claim.
    specs += [("E5", sub_path, "date_after", due_iso),
              ("E6", cert_path, "exists", None)]
    atoms = [atom(aid, nct, path, ptype, arg,
                  truth(rec, path,
                        "record_predicate:%s:%s" % (ptype, json.dumps(arg))))
             for aid, path, ptype, arg in specs]

    claims = [
        {"id": "C1", "level": 1,
         "text": "This registration is an applicable clinical trial subject "
                 "to the Part 11 results-reporting requirement "
                 "(42 CFR 11.22(b)).",
         "denotation": [[["protocolSection.designModule.studyType",
                          "equals", "INTERVENTIONAL"]]]},
        {"id": "C2", "level": 2,
         "text": "Results information for this trial was therefore due by a "
                 "determinate date under 42 CFR 11.44(a).",
         "denotation": [[["protocolSection.statusModule."
                          "primaryCompletionDateStruct.date", "exists", None]]]},
        {"id": "C3", "level": 3,
         "text": "This trial submitted its results information after the "
                 "deadline that 42 CFR 11.44(a) sets for it.",
         "denotation": [[[sub_path, "date_after", due_iso]]]},
    ]

    # The routes are the statute's conjunctions, not ours. C3 is licensed
    # only when applicability, a determinate deadline, and a missed
    # submission are all established.
    licences = {"C1": [["E1", "E2", "E3"]],
                "C2": [["E1", "E2", "E3", "E4"]],
                "C3": [["E1", "E2", "E3", "E4", "E5"]]}

    # The support ceiling: the strongest claim the evidence actually
    # licenses, computed from the gold values rather than asserted. For an
    # overdue trial this is level 2, and abstaining above it is correct.
    gv = {a["id"]: a["gold_value"] for a in atoms}
    ceiling = 0
    for cid in ("C1", "C2", "C3"):
        if all(gv.get(e) == "T" for e in licences[cid][0]):
            ceiling = int(cid[1:])
        else:
            break

    return {
        "id": "fdaaa-%s" % nct,
        "domain": "fdaaa_ctgov",
        "seed": nct,
        "variant": "base",
        "actions": ["inspect_record", "ask_user", "emit_claim"],
        "artifact": {"kind": "record", "path": "", "commit": ""},
        "atoms": atoms,
        "claims": claims,
        "decoys": [],
        "gold_licenses": licences,
        "order": [["C1", "C2"], ["C2", "C3"]],
        "phi": [],
        "records": {nct: rec},
        "scope": "The single named ClinicalTrials.gov registration, as "
                 "retrieved and embedded; no external lookups. The governing "
                 "standard is 42 CFR Part 11 (FDAAA 801). Record "
                 "identifier: %s." % nct,
        "user_prompt": "Under the FDAAA 801 results-reporting rule, is this "
                       "trial delinquent?",
        # A JSON *string*, matching every other family: analysis/
        # build_w2a_tables.instance_facts does json.loads on this field, so a
        # dict here parses fine in schema.load_instance and then fails the
        # scorer. The validator does not enforce the convention, which is
        # exactly why it is worth matching deliberately.
        "notes_for_annotators": json.dumps({
            "family": sponsor,
            "topology": "nested",
            "cost": "hom",
            "target_level": 3,
            "false_mech": c["verdict"] == "late",
            "contract": "42 CFR 11.22(b), 11.42, 11.44(a)-(c)",
            "contract_author": "Congress (FDAAA 801) and FDA/NLM (42 CFR 11)",
            "verdict": c["verdict"],
            "cert_filed": c["cert_filed"],
            "cert_valid": c["cert_valid"],
            "date_defaulted": c["date_defaulted"],
            "due": c["due"].isoformat() if c["due"] else None,
        }),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--min-family", type=int, default=2,
                    help="drop sponsors with fewer trials than this: a "
                         "cluster of one carries no within-cluster signal")
    a = ap.parse_args(argv)

    api = json.load(open(os.path.join(EXT, "ctgov_v2_6000.json")))
    spon = {}
    with open(os.path.join(EXT, "trialstracker_2026-08-03.csv")) as fh:
        for r in csv.DictReader(fh):
            spon[r["registry_id"]] = (r.get("sponsor_slug") or "unknown",
                                      r.get("status") or "")

    kept, drop = [], {"not-act": 0, "no-deadline": 0, "not-due": 0,
                      "withheld": 0, "no-sponsor": 0}
    for nct, study in sorted(api.items()):
        c = compile_contract(study)
        if not c["adjudicable"]:
            drop["withheld"] += 1
            continue
        if not c["act"]:
            drop["not-act"] += 1
            continue
        if not c["due"]:
            drop["no-deadline"] += 1
            continue
        if c["verdict"] == "not_yet_due":
            drop["not-due"] += 1
            continue
        if nct not in spon:
            drop["no-sponsor"] += 1
            continue
        kept.append((nct, study, c, spon[nct][0]))

    # keep clusters with at least min-family members, then cap
    by_fam = {}
    for row in kept:
        by_fam.setdefault(row[3], []).append(row)
    fams = {f: v for f, v in by_fam.items() if len(v) >= a.min_family}
    sel = [r for f in sorted(fams) for r in fams[f]][:a.limit]

    os.makedirs(OUT, exist_ok=True)
    verdicts, agree, checked, written_fams = {}, 0, 0, set()
    for nct, study, c, sponsor in sel:
        inst = build_instance(nct, study, c, sponsor)

        # Applicability must be establishable BY THE RUNTIME, not by this
        # module's opinion of the record. compile_contract decides
        # `post_rule` with the date parser above, which accepts a
        # month-precision start date; the atom that carries the same clause
        # is evaluated by the deployed predicate grammar, which refuses one
        # and attests U forever. An instance admitted on the first reading
        # and scored on the second has NO satisfiable route at all, and a
        # ceiling-targeting policy will spin on an atom it can never
        # resolve -- one such record hung a 529-instance sweep for over an
        # hour. This is the same reimplementation mismatch as the gold
        # values, one layer up: the filter must ask the oracle too.
        gv = {a["id"]: a["gold_value"] for a in inst["atoms"]}
        if any(gv.get(e) != "T" for e in ("E1", "E2", "E3")):
            drop["applicability-undeterminable"] = \
                drop.get("applicability-undeterminable", 0) + 1
            continue

        with open(os.path.join(OUT, "%s.json" % inst["id"]), "w") as fh:
            json.dump(inst, fh, indent=1, sort_keys=True)
        verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
        written_fams.add(sponsor)
        ext = spon[nct][1]
        if ext:
            checked += 1
            ours_adverse = c["verdict"] in ("late", "overdue")
            theirs_adverse = ext in ("reported-late", "overdue",
                                     "overdue-cancelled")
            agree += (ours_adverse == theirs_adverse)

    manifest = {
        "contract": "42 CFR Part 11 (FDAAA 801), sections 11.22(b), 11.42, "
                    "11.44(a)-(c)",
        "contract_author": "Congress and FDA/NLM -- not the authors",
        "records": "ClinicalTrials.gov API v2, frozen; see "
                   "external/SHA256SUMS.txt",
        "snapshot": SNAPSHOT.isoformat(),
        "cluster_unit": "sponsor",
        "n_instances": sum(verdicts.values()),
        "n_clusters": len(written_fams),
        "verdicts": verdicts,
        "adverse_rate": round(
            100.0 * sum(v for k, v in verdicts.items()
                        if k in ("late", "overdue"))
            / max(1, sum(verdicts.values())), 1),
        "external_agreement_pct": round(100.0 * agree / checked, 2) if checked else None,
        "external_checked": checked,
        "attrition": drop,
        "predicate_grammar": "prime_runtime/predicates.py, unextended",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    print("wrote %d instances over %d sponsor clusters -> %s"
          % (manifest["n_instances"], manifest["n_clusters"], OUT))
    print("  verdict mix        : %s" % verdicts)
    print("  adverse rate       : %.1f%% (naturally occurring, nothing injected)"
          % manifest["adverse_rate"])
    print("  agreement with the external validator: %s%% on %d"
          % (manifest["external_agreement_pct"], checked))
    print("  attrition          : %s" % drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
