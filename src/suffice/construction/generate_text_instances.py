#!/usr/bin/env python3
"""The text-evidence split (heldout_text): atoms whose evidence is free text
and whose resolver is the model.

For clinical registrations the source is the eligibility-criteria text; for
security advisories the advisory's details text. Each text atom states a
condition the text either establishes or not (e.g. "enrollment is restricted
to adults"); the agent reads the text with read_text and attests the atom
with attest_atom, quoting a span. Ground truth never needs a judge: every
condition has a structured twin in the registry record (minimumAge, sex,
healthyVolunteers; the registered package name and the fixed and first
affected versions in the affected ranges), which is withheld from the embedded
record so the agent cannot read it, and the atom's gold value is computed
from the twin. Violations occur naturally:
nothing is degraded; the standard's required values are fixed per family
and the records vary.

Routes (chain): C1 = [E1] (a field atom, the title / summary is present);
C2 = [E1, E2]; C3 = [E1, E2, E3, E4] with E2..E4 text atoms.

Usage: python3 src/suffice/construction/generate_text_instances.py --per-domain 120
Writes data/suffice/<domain>/heldout_text/<id>.json, ids from 7001.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_record_instances import get_path, slice_into, SEED, OUT, format_predicate_rule  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

ACTIONS = ["inspect_record", "read_text", "attest_atom", "ask_user", "emit_claim"]


def _age_years(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(year|month|week|day)", s.lower())
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return v if unit == "year" else v / {"month": 12.0, "week": 52.0, "day": 365.0}[unit]


# Per domain: the source text field, a field atom for C1, and the text atoms
# with their condition in words and the executable truth over the hidden twin.
DOMAINS: Dict[str, Dict[str, Any]] = {
    "clinical": {
        "user_prompt": "Is this trial registration current and complete enough for eligibility screening?",
        "scope": "The single named ClinicalTrials.gov study record, as retrieved and embedded; no external lookups.",
        "source": "protocolSection.eligibilityModule.eligibilityCriteria",
        "field_atom": ("protocolSection.identificationModule.briefTitle", "exists"),
        "twins": ["protocolSection.eligibilityModule.minimumAge", "protocolSection.eligibilityModule.sex",
                  "protocolSection.eligibilityModule.healthyVolunteers"],
        "text_atoms": [
            ("the eligibility criteria restrict enrollment to adults, that is, a minimum age of 18 years or more",
             lambda rec: (lambda a: None if a is None else a >= 18)(_age_years(get_path(rec, "protocolSection.eligibilityModule.minimumAge")[0]))),
            ("the eligibility criteria restrict enrollment to one sex only (only male or only female participants)",
             lambda rec: (lambda s: None if s is None else s in ("MALE", "FEMALE"))(get_path(rec, "protocolSection.eligibilityModule.sex")[0])),
            ("the eligibility criteria accept healthy volunteers",
             lambda rec: (lambda h: None if h is None else bool(h))(get_path(rec, "protocolSection.eligibilityModule.healthyVolunteers")[0])),
        ],
        "required": {"C2": ["T"], "C3": ["T", "F", "F"]},
        "claim_text": {
            1: "The registration identifies the study with a readable title (as inspected).",
            2: "The registration is screening-ready for an adult cohort: identified, and the eligibility text restricts enrollment to adults.",
            3: "The registration satisfies the checked adult-cohort protocol: identified, adults only, open to both sexes, and not a healthy-volunteer study, as established from the eligibility text.",
        },
    },
    "security": {
        "user_prompt": "Is this security advisory complete and current enough to act on for the affected package?",
        "scope": "The single named OSV advisory record, as retrieved and embedded; no external lookups.",
        "source": "details",
        "field_atom": ("summary", "exists"),
        "twins": ["affected.0.package.name", "affected.*.ranges.*.events.*.fixed", "affected.*.ranges.*.events.*.introduced"],
        "text_atoms": [
            ("the details text names the affected package (the package itself, by its registered name or a distinctive part of it, not only the vendor or the ecosystem)",
             lambda rec: _package_named(rec)),
            ("the details text states a version in which the issue is fixed",
             lambda rec: _fixed_named(rec)),
            ("the details text states the first affected version, the version in which the issue was introduced",
             lambda rec: _introduced_named(rec)),
        ],
        "required": {"C2": ["T"], "C3": ["T", "T", "T"]},
        "claim_text": {
            1: "The advisory has a readable summary (as inspected).",
            2: "The advisory is triage-ready: it has a summary and its text names the affected package.",
            3: "The advisory satisfies the checked completeness protocol: summary present, and the text names the package, a fixed version, and the first affected version.",
        },
    },
}


def _fixed_versions(rec) -> List[str]:
    out = []
    for a in rec.get("affected") or []:
        for rg in a.get("ranges") or []:
            for ev in rg.get("events") or []:
                if ev.get("fixed"):
                    out.append(str(ev["fixed"]))
    return out


def _fixed_named(rec) -> Optional[bool]:
    fx = _fixed_versions(rec)
    if not fx:
        return None  # no structured twin: the condition cannot be scored
    text = (rec.get("details") or "")
    return any(v in text for v in fx)


# Tokens of a registered package name that do not identify the package on
# their own (ecosystem prefixes, hosting, generic suffixes).
_GENERIC = {"org", "com", "net", "io", "github", "gitlab", "www", "core", "lib", "libs", "plugin", "plugins",
            "python", "py", "js", "node", "java", "go", "rs", "api", "sdk", "client", "server", "cli", "common",
            "commons", "utils", "util", "tools", "main", "base", "dev", "src", "pkg", "package", "module", "modules",
            "apache", "google", "microsoft", "aws", "azure", "jenkins", "ci", "types", "contrib", "ext", "extra"}


def _package_tokens(name: str) -> List[str]:
    toks = [t for t in re.split(r"[\s/:@._\-]+", name.lower()) if t]
    return [t for t in toks if len(t) >= 4 and t not in _GENERIC]


def _package_named(rec) -> Optional[bool]:
    """The structured twin of 'the text names the affected package': the
    registered name, the name with -/_ interchanged, or one of its
    distinctive tokens (vendor, hosting, and generic suffix tokens removed)
    occurs in the details text. A name with no distinctive token cannot be
    scored and the task is dropped."""
    name = (get_path(rec, "affected.0.package.name")[0] or "")
    if not name:
        return None
    text = (rec.get("details") or "").lower()
    norm = lambda x: x.lower().replace("_", "-")
    if norm(name) in norm(text) or name.lower().replace("-", "_") in text:
        return True
    toks = _package_tokens(name)
    if not toks:
        return None
    return any(re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", text) for t in toks)


def _introduced_named(rec) -> Optional[bool]:
    """The first affected version(s) in the advisory's ranges occur in the
    text. An advisory whose ranges start at "0" (every version affected)
    has no such version and the task is dropped."""
    vs = []
    for a in rec.get("affected") or []:
        for rg in a.get("ranges") or []:
            for ev in rg.get("events") or []:
                v = ev.get("introduced")
                if v and str(v) != "0":
                    vs.append(str(v))
    if not vs:
        return None
    text = rec.get("details") or ""
    return any(v in text for v in vs)


def build(domain: str, row: Dict[str, Any], seq: int) -> Tuple[Optional[Dict[str, Any]], str]:
    spec = DOMAINS[domain]
    rec = row["record"]
    rid = row["record_id"]
    text, ok = get_path(rec, spec["source"])
    if not ok or not isinstance(text, str) or len(text.strip()) < 200:
        return None, "no-text"
    fpath, fop = spec["field_atom"]
    fval, fok = get_path(rec, fpath)
    if not fok or fval in (None, "", []):
        return None, "field-missing"
    truths = []
    for what, fn in spec["text_atoms"]:
        v = fn(rec)
        if v is None:
            return None, "twin-missing"
        truths.append(bool(v))
    rec_slice: Dict[str, Any] = {}
    slice_into(rec_slice, rec, fpath)
    slice_into(rec_slice, rec, spec["source"])
    # the structured twins are NOT embedded: the agent cannot read them
    atoms = [{"id": "E1", "type": "record", "proposition": "",
              "resolver": {"tool": "inspect_record", "params_schema": {"field_path": "str", "record_id": "str"},
                           "params": {"field_path": fpath, "record_id": rid}, "attest": format_predicate_rule(fop, None)},
              "gold_value": "T"}]
    req = spec["required"]
    for i, ((what, _fn), truth) in enumerate(zip(spec["text_atoms"], truths)):
        aid = "E%d" % (i + 2)
        atoms.append({"id": aid, "type": "text", "proposition": "",
                      "resolver": {"tool": "attest_atom", "params_schema": {"atom_id": "str", "value": "str", "quote": "str"},
                                   "params": {"atom_id": aid}, "attest": "attested"},
                      "gold_value": "T" if truth else "F",
                      "spec_hint": {"what": what, "source": {"record_id": rid, "field_path": spec["source"]}}})
    # routes with the family's required polarity: a literal is E_k when the
    # required value is T and !E_k when it is F
    def lit(k, v):
        return ("E%d" % k) if v == "T" else ("!E%d" % k)
    routes = {"C1": [["E1"]],
              "C2": [["E1", lit(2, req["C2"][0])]],
              "C3": [["E1"] + [lit(k + 2, v) for k, v in enumerate(req["C3"])]]}
    claims = [{"id": c, "text": spec["claim_text"][i + 1], "level": i + 1, "denotation": []}
              for i, c in enumerate(("C1", "C2", "C3"))]
    iid = "%s-v2-%04d" % (domain[:3], seq)
    inst = {
        "id": iid, "domain": "%s_v2" % domain, "seed": rid, "variant": "text",
        "user_prompt": spec["user_prompt"], "artifact": {"path": "", "kind": "record", "commit": ""},
        "scope": spec["scope"] + " Record identifier: %s." % rid,
        "actions": ACTIONS, "claims": claims, "order": [["C1", "C2"], ["C2", "C3"]],
        "atoms": atoms, "phi": [], "gold_licenses": routes, "decoys": [], "records": {rid: rec_slice},
        "notes_for_annotators": json.dumps({"family": "%s-text" % domain, "topology": "ladder", "cost": "text",
                                            "target_level": None, "false_mech": False, "text": True,
                                            "truths": truths, "twins": spec["twins"]}),
    }
    return inst, ""


def achieved_level(inst) -> int:
    gv = {a["id"]: a["gold_value"] for a in inst["atoms"]}
    def holds(lit):
        neg = lit.startswith("!")
        v = gv.get(lit.lstrip("!"))
        return (v == "F") if neg else (v == "T")
    levels = {c["id"]: c["level"] for c in inst["claims"]}
    return max([levels[c] for c, rs in inst["gold_licenses"].items() if any(all(holds(x) for x in r) for r in rs)] + [0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+", default=["clinical", "security"])
    ap.add_argument("--per-domain", type=int, default=120)
    ap.add_argument("--seq-base", type=int, default=7000)
    args = ap.parse_args()
    tmp = os.path.join(OUT, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    for domain in args.domains:
        rows = [json.loads(l) for l in open(os.path.join(PILOT, "data", "harvest", domain, "records.jsonl"), encoding="utf-8")]
        rng = random.Random(SEED + 7)
        rng.shuffle(rows)
        sub = os.path.join(OUT, domain, "heldout_text")
        os.makedirs(sub, exist_ok=True)
        n = 0; att: Dict[str, int] = {}; levels: Dict[int, int] = {}
        seq = args.seq_base
        for row in rows:
            if n >= args.per_domain:
                break
            seq += 1
            inst, why = build(domain, row, seq)
            if inst is None:
                att[why] = att.get(why, 0) + 1
                continue
            p = os.path.join(tmp, inst["id"] + ".json")
            json.dump(inst, open(p, "w", encoding="utf-8"), default=str)
            try:
                load_instance(p, validate=True)
            except Exception as exc:  # noqa: BLE001
                att["schema"] = att.get("schema", 0) + 1
                os.remove(p)
                continue
            os.remove(p)
            with open(os.path.join(sub, inst["id"] + ".json"), "w", encoding="utf-8") as fh:
                json.dump(inst, fh, indent=1, sort_keys=True, default=str)
            n += 1
            lv = achieved_level(inst)
            levels[lv] = levels.get(lv, 0) + 1
        print("[%s] heldout_text=%d attrition=%s ceiling levels=%s" % (domain, n, att, dict(sorted(levels.items()))), file=sys.stderr)
    os.rmdir(tmp) if not os.listdir(tmp) else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
