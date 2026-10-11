"""Shared pieces of the specification-construction study (see PROTOCOL.md).

trim()             the embedded registration: the API v2 fields a reviewer of
                   42 CFR Part 11 would consult, copied verbatim
catalog()          the field catalog given to the creator (paths, types,
                   enumerated values seen in development records, descriptions)
run_derive()       the creator's derive(record) in a restricted sandbox
build_instance()   a program (atoms, routes, optional derive) applied to one
                   registration -> a PRIME instance validated by the runtime
full_level()       strongest conclusion the program licenses when every atom
                   is resolved by its deterministic resolver
reference_level()  the tracker's level (3 reported-late, 2 results due,
                   1 not yet due); test labels only through split="test"
"""
from __future__ import annotations

import ast
import collections
import csv
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.environ.get("SPECBUILD_PILOT") or os.path.dirname(os.path.dirname(HERE))
if PILOT not in sys.path:
    sys.path.insert(0, PILOT)
from prime_runtime import goldledger  # noqa: E402
from prime_runtime.predicates import PREDICATE_TYPES  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

EXT = os.path.join(PILOT, "bench", "fdaaa", "external")
SNAPSHOT = "2026-08-03"
LEVEL = {"C1": 1, "C2": 2, "C3": 3}
CLAIMS = [
    {"id": "C1", "level": 1, "text": "This registration is an applicable clinical trial subject to the results-reporting requirement of 42 CFR Part 11."},
    {"id": "C2", "level": 2, "text": "Results information for this trial was due by a determinate date that had passed by 3 August 2026."},
    {"id": "C3", "level": 3, "text": "Results information for this trial was submitted after the deadline that 42 CFR Part 11 sets for it."},
]

STATUS_KEYS = ("overallStatus", "startDateStruct", "primaryCompletionDateStruct", "completionDateStruct", "resultsFirstSubmitDate",
               "resultsFirstPostDateStruct", "dispFirstSubmitDate", "studyFirstSubmitDate")
DESCRIPTIONS = {
    "protocolSection.identificationModule.briefTitle": "Short title of the study.",
    "protocolSection.statusModule.overallStatus": "Recruitment status of the study as a whole.",
    "protocolSection.statusModule.startDateStruct.date": "Study start date: the date the first participant was enrolled (actual or estimated); YYYY-MM-DD or YYYY-MM.",
    "protocolSection.statusModule.startDateStruct.type": "Whether the start date is ACTUAL or ESTIMATED.",
    "protocolSection.statusModule.primaryCompletionDateStruct.date": "Primary completion date: the date the final participant was examined or received an intervention for the primary outcome (actual or estimated); YYYY-MM-DD or YYYY-MM.",
    "protocolSection.statusModule.primaryCompletionDateStruct.type": "Whether the primary completion date is ACTUAL or ESTIMATED.",
    "protocolSection.statusModule.completionDateStruct.date": "Study completion date (actual or estimated); YYYY-MM-DD or YYYY-MM.",
    "protocolSection.statusModule.completionDateStruct.type": "Whether the study completion date is ACTUAL or ESTIMATED.",
    "protocolSection.statusModule.resultsFirstSubmitDate": "Date the responsible party first submitted results information to ClinicalTrials.gov (absent if never submitted).",
    "protocolSection.statusModule.resultsFirstPostDateStruct.date": "Date results information was first posted on ClinicalTrials.gov after quality control.",
    "protocolSection.statusModule.dispFirstSubmitDate": "Date the responsible party first submitted a certification or request to delay the submission of results information (absent if none).",
    "protocolSection.statusModule.studyFirstSubmitDate": "Date the study was first submitted for registration.",
    "protocolSection.designModule.studyType": "Study type: INTERVENTIONAL, OBSERVATIONAL, or EXPANDED_ACCESS.",
    "protocolSection.designModule.phases": "List of study phases (for example PHASE1, PHASE2, PHASE3, PHASE4, EARLY_PHASE1, NA).",
    "protocolSection.designModule.designInfo.primaryPurpose": "Primary purpose of an interventional study (for example TREATMENT, PREVENTION, DEVICE_FEASIBILITY).",
    "protocolSection.oversightModule.isFdaRegulatedDrug": "Whether the study studies a U.S. FDA-regulated drug product (absent if not reported).",
    "protocolSection.oversightModule.isFdaRegulatedDevice": "Whether the study studies a U.S. FDA-regulated device product (absent if not reported).",
    "protocolSection.oversightModule.isUnapprovedDevice": "Whether a device product studied is not approved or cleared by FDA.",
    "protocolSection.oversightModule.isPpsd": "Whether the study is a pediatric postmarket surveillance of a device product.",
    "protocolSection.oversightModule.isUsExport": "Whether a product under investigation is manufactured in and exported from the U.S. for study in another country.",
    "protocolSection.contactsLocationsModule.locations": "List of facility locations; each has a country.",
    "protocolSection.armsInterventionsModule.interventions": "List of interventions; each has a type (DRUG, DEVICE, BIOLOGICAL, PROCEDURE, ...).",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.class": "Class of the lead sponsor (INDUSTRY, NIH, FED, OTHER, ...).",
    "derivedSection.miscInfoModule.submissionTracking.estimatedResultsFirstSubmitDate": "Date results information was first submitted, when results are submitted but not yet posted (set by ClinicalTrials.gov).",
    "hasResults": "Whether posted results exist on ClinicalTrials.gov.",
}


def _get(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def trim(study):
    """The embedded registration: modules a Part 11 reviewer consults, values copied verbatim."""
    ps = study.get("protocolSection") or {}
    st = ps.get("statusModule") or {}; de = ps.get("designModule") or {}; ov = ps.get("oversightModule") or {}
    rec = {"protocolSection": {
        "identificationModule": {"nctId": _get(ps, "identificationModule.nctId"), "briefTitle": _get(ps, "identificationModule.briefTitle")},
        "statusModule": {k: st[k] for k in STATUS_KEYS if k in st},
        "designModule": {k: de[k] for k in ("studyType", "phases") if k in de},
        "oversightModule": {k: ov[k] for k in ("isFdaRegulatedDrug", "isFdaRegulatedDevice", "isUnapprovedDevice", "isPpsd", "isUsExport") if k in ov},
        "contactsLocationsModule": {"locations": [{"country": l.get("country")} for l in ((ps.get("contactsLocationsModule") or {}).get("locations") or [])][:60]},
        "armsInterventionsModule": {"interventions": [{"type": i.get("type")} for i in ((ps.get("armsInterventionsModule") or {}).get("interventions") or [])][:20]},
        "sponsorCollaboratorsModule": {"leadSponsor": {"class": _get(ps, "sponsorCollaboratorsModule.leadSponsor.class")}},
    }}
    if "designInfo" in de and "primaryPurpose" in (de.get("designInfo") or {}):
        rec["protocolSection"]["designModule"]["designInfo"] = {"primaryPurpose": de["designInfo"]["primaryPurpose"]}
    tk = _get(study, "derivedSection.miscInfoModule.submissionTracking.estimatedResultsFirstSubmitDate")
    if tk:
        rec["derivedSection"] = {"miscInfoModule": {"submissionTracking": {"estimatedResultsFirstSubmitDate": tk}}}
    rec["hasResults"] = bool(study.get("hasResults"))
    return rec


def catalog(records):
    """Field catalog for the creator: every leaf path of the embedded record, with type and observed enum values."""
    types, vals = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)

    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, p + "." + k if p else k)
        elif isinstance(o, list):
            types[p]["list"] += 1
            for x in o[:5]:
                walk(x, p + ".N")
        else:
            types[p][type(o).__name__] += 1
            if isinstance(o, (str, bool)) and len(str(o)) < 30:
                vals[p][o] += 1
    for r in records:
        walk(r, "")
    out = []
    for p in sorted(types):
        if p.endswith("nctId"):
            continue
        ent = {"path": p, "type": "/".join(sorted(types[p])), "description": DESCRIPTIONS.get(p.split(".N")[0]) if ".N" in p else DESCRIPTIONS.get(p)}
        leaf = p.split(".")[-1].lower()
        if "date" in leaf or p.endswith("briefTitle"):
            ent["values"] = "dates" if "date" in leaf else "free text"
        elif len(vals[p]) and len(vals[p]) <= 15:
            ent["values"] = [v for v, _ in vals[p].most_common()]
        out.append(ent)
    return out


# ---------------------------------------------------------------- sandbox
_SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map", "max", "min",
    "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "None", "True", "False", "ValueError", "KeyError",
    "TypeError", "Exception", "IndexError", "AttributeError")}
import calendar as _calendar  # noqa: E402
import _strptime as _strptime_mod  # noqa: E402  (datetime.strptime imports it lazily through __import__)
import time as _time  # noqa: E402  (datetime.strftime loads it through __import__; harmless on its own)
_ALLOWED_MODULES = {"datetime": _dt, "math": math, "re": re, "json": json, "calendar": _calendar, "time": _time}
# internal modules the allowed ones load on demand; returned already imported, never executed from the sandbox
_INTERNAL_MODULES = {"_strptime": _strptime_mod}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[name]
    if name in _INTERNAL_MODULES:
        return _INTERNAL_MODULES[name]
    raise ImportError("import of %r is not allowed" % name)


def check_code(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id in ("open", "exec", "eval", "compile", "globals", "locals", "getattr", "setattr", "__import__"):
            raise ValueError("name %r is not allowed" % node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"NCT\d{8}", node.value):
            raise ValueError("registration identifiers may not appear in the program")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "derive" for n in tree.body):
        raise ValueError("derive(record) is not defined")
    return tree


def compile_derive(code):
    check_code(code)
    g = {"__builtins__": dict(_SAFE, __import__=_restricted_import), "datetime": _dt, "date": _dt.date, "timedelta": _dt.timedelta,
         "math": math, "re": re, "json": json, "calendar": _calendar}
    exec(compile(code, "<derive>", "exec"), g)  # noqa: S102 (restricted globals, AST-checked, no file or network access)
    return g["derive"]


def _jsonable(v):
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()[:10]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _derive_worker(code, rec, q):
    try:
        f = compile_derive(code)
        out = f(json.loads(json.dumps(rec)))
        if not isinstance(out, dict):
            raise TypeError("derive must return a dict")
        q.put(("ok", {str(k): _jsonable(v) for k, v in out.items()}))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", "%s: %s" % (type(exc).__name__, exc)))


def run_derive(code, rec, timeout=5.0):
    """derive(record) in a child process with restricted builtins and a time limit."""
    if not code:
        return {}, None
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    q = ctx.Queue(); p = ctx.Process(target=_derive_worker, args=(code, rec, q)); p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(); return None, "timeout"
    try:
        status, val = q.get_nowait()
    except Exception:  # noqa: BLE001
        return None, "no result"
    return (val, None) if status == "ok" else (None, val)


def run_derive_batch(code, recs, timeout=60.0):
    """derive over many records in one sandboxed child (fast path); falls back per record on failure."""
    if not code:
        return [({}, None) for _ in recs]
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()

    def worker(code, recs, q):
        try:
            f = compile_derive(code)
        except Exception as exc:  # noqa: BLE001
            q.put([(None, "%s: %s" % (type(exc).__name__, exc))] * len(recs)); return
        res = []
        for r in recs:
            try:
                out = f(json.loads(json.dumps(r)))
                if not isinstance(out, dict):
                    raise TypeError("derive must return a dict")
                res.append(({str(k): _jsonable(v) for k, v in out.items()}, None))
            except Exception as exc:  # noqa: BLE001
                res.append((None, "%s: %s" % (type(exc).__name__, exc)))
        q.put(res)
    q = ctx.Queue(); p = ctx.Process(target=worker, args=(code, recs, q)); p.start()
    try:
        res = q.get(timeout=timeout)
    except Exception:  # noqa: BLE001
        res = None
    p.join(1)
    if p.is_alive():
        p.terminate()
    if res is None:
        return [run_derive(code, r) for r in recs]
    return res


# ---------------------------------------------------------------- program -> instance
def validate_program_shape(prog):
    atoms = prog.get("atoms"); routes = prog.get("routes")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("atoms must be a non-empty list")
    ids = [a.get("id") for a in atoms]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("atom ids must be unique strings")
    for a in atoms:
        if a.get("predicate") not in PREDICATE_TYPES:
            raise ValueError("unknown predicate %r" % a.get("predicate"))
        if not isinstance(a.get("field"), str) or not a["field"]:
            raise ValueError("atom %s has no field" % a.get("id"))
    if not isinstance(routes, dict) or set(routes) != {"C1", "C2", "C3"}:
        raise ValueError("routes must give C1, C2 and C3")
    for c, rs in routes.items():
        if not isinstance(rs, list) or not rs or not all(isinstance(r, list) and r for r in rs):
            raise ValueError("routes for %s must be a non-empty list of non-empty lists" % c)
        for r in rs:
            for l in r:
                if l.lstrip("!") not in ids:
                    raise ValueError("route literal %r names no atom" % l)
    if len(atoms) > 12:
        raise ValueError("at most 12 atoms")
    return True


def attest_rule(a):
    arg = None if a["predicate"] == "exists" else a.get("argument")
    return "record_predicate:%s:%s" % (a["predicate"], json.dumps(arg))


def truth(rec, path, rule):
    """Gold value through the runtime's own path resolver and predicate grammar (never a reimplementation)."""
    sys.path.insert(0, os.path.join(PILOT, "runner")) if os.path.join(PILOT, "runner") not in sys.path else None
    from arms import _resolve_field_path  # noqa: E402
    value, found = _resolve_field_path(rec, path)
    obs = {"value": value} if found else {"error": "field not found"}
    return goldledger.eval_attest(rule, obs) or "U"


def build_instance(prog, nct, study, derived, tag):
    rec = trim(study)
    if derived is not None:
        rec["derived"] = derived
    atoms = []
    for a in prog["atoms"]:
        rule = attest_rule(a)
        atoms.append({"id": a["id"], "gold_value": truth(rec, a["field"], rule), "proposition": a.get("meaning", ""),
                      "resolver": {"tool": "inspect_record", "attest": rule, "params": {"record_id": nct, "field_path": a["field"]},
                                   "params_schema": {"record_id": "str", "field_path": "str"}}})
    return {
        "id": "%s-%s" % (tag, nct), "domain": "fdaaa_specbuild", "seed": nct, "variant": "base",
        "actions": ["inspect_record", "emit_claim"], "artifact": {"kind": "record", "path": "", "commit": ""},
        "atoms": atoms, "claims": [dict(c, denotation=[]) for c in CLAIMS], "decoys": [],
        "gold_licenses": {c: [list(r) for r in prog["routes"][c]] for c in ("C1", "C2", "C3")},
        "order": [["C1", "C2"], ["C2", "C3"]], "phi": [], "records": {nct: rec},
        "scope": "The single named ClinicalTrials.gov registration, as retrieved and embedded (field \"derived\" holds values the committed program computes from the record); no external lookups. The governing standard is 42 CFR Part 11 (FDAAA 801). Record identifier: %s." % nct,
        "user_prompt": "Under the FDAAA 801 results-reporting rule (42 CFR Part 11), what does this registration establish about its results-reporting obligation?",
        "notes_for_annotators": json.dumps({"family": "specbuild", "topology": "nested", "cost": "hom", "program": tag}),
    }


def validate_instance(inst):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(inst, fh); p = fh.name
    try:
        load_instance(p)
    finally:
        os.unlink(p)


def full_level(inst):
    gv = {a["id"]: a["gold_value"] for a in inst["atoms"]}
    lv = 0
    for c, routes in inst["gold_licenses"].items():
        if any(all(gv.get(l.lstrip("!")) == ("F" if l.startswith("!") else "T") for l in r) for r in routes):
            lv = max(lv, LEVEL[c])
    return lv


# ---------------------------------------------------------------- reference (tracker)
_TRACKER = None


def _tracker():
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = {r["registry_id"]: r for r in csv.DictReader(open(os.path.join(EXT, "trialstracker_2026-08-03.csv")))}
    return _TRACKER


def reference(ids, split):
    """Tracker reference for the given ids. Test labels are read only with split='test' (final scoring)."""
    assert split in ("validation", "test")
    t = _tracker(); out = {}
    for i in ids:
        r = t.get(i)
        if r is None:
            continue
        lv = 3 if r["status"] == "reported-late" else 2 if r["results_due"] == "True" else 1
        out[i] = {"level": lv, "status": r["status"], "adverse": r["status"] in ("reported-late", "overdue", "overdue-cancelled")}
    return out
