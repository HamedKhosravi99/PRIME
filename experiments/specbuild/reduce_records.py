#!/usr/bin/env python3
"""Reduce registrations to the modules both programs read (for shipping to the cluster),
and verify the reduction changes nothing: trim() and the paper's builder give identical
output on full and reduced records."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PILOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PILOT, "bench", "fdaaa"))
import common, build_fdaaa_instances as BA

def reduce(s):
    ps = s.get("protocolSection") or {}; out = {"protocolSection": {}}
    for m in ("identificationModule", "statusModule", "designModule", "oversightModule"):
        if m in ps: out["protocolSection"][m] = ps[m]
    out["protocolSection"]["contactsLocationsModule"] = {"locations": [{"country": l.get("country")} for l in ((ps.get("contactsLocationsModule") or {}).get("locations") or [])]}
    out["protocolSection"]["armsInterventionsModule"] = {"interventions": [{"type": i.get("type")} for i in ((ps.get("armsInterventionsModule") or {}).get("interventions") or [])]}
    out["protocolSection"]["sponsorCollaboratorsModule"] = {"leadSponsor": {"class": ((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("class")}}
    tk = ((s.get("derivedSection") or {}).get("miscInfoModule") or {}).get("submissionTracking")
    if tk: out["derivedSection"] = {"miscInfoModule": {"submissionTracking": tk}}
    if "hasResults" in s: out["hasResults"] = s["hasResults"]
    return out

for name, src in (("test", os.path.join(PILOT, "experiments", "fdaaa_heldout", "results", "records")), ("validation", os.path.join(HERE, "data", "validation_records"))):
    dst = os.path.join(HERE, "data", "reduced", name); os.makedirs(dst, exist_ok=True); bad = 0; n = 0
    for p in sorted(glob.glob(os.path.join(src, "NCT*.json"))):
        s = json.load(open(p)); r = reduce(s); nct = os.path.basename(p)[:-5]; n += 1
        if json.dumps(common.trim(s), sort_keys=True) != json.dumps(common.trim(r), sort_keys=True): bad += 1
        c1, c2 = BA.compile_contract(s), BA.compile_contract(r)
        if repr(c1) != repr(c2): bad += 1
        elif c1["due"]:
            if json.dumps(BA.build_instance(nct, s, c1, "x"), sort_keys=True) != json.dumps(BA.build_instance(nct, r, c2, "x"), sort_keys=True): bad += 1
        json.dump(r, open(os.path.join(dst, nct + ".json"), "w"))
    print(name, n, "records reduced; mismatches:", bad)
