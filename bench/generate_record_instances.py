#!/usr/bin/env python3
"""Template-family instance generator (docs/EXECUTION_PLAN.md, Part 4).

Generates ObligationBench record-domain instances from the harvested public
records (bench/harvest_records.py), under the frozen v5 design:

  * >=16 branching + >=8 nested template FAMILIES per domain, enumerated
    deterministically from a per-domain probe pool; a family = (topology,
    cost profile, probe composition) fingerprint;
  * full 2x2 {nested,branching} x {homogeneous,heterogeneous call-class
    cost} cross with matched difficulty covariates (Part R4, Q2);
  * every held-out record belongs to EXACTLY ONE family (Part R4, Q3);
  * balanced-diagnostic strata 30/30/25/15 (C3/C2max/C1max/none) with
    exactly ~40% of instances containing a FALSE load-bearing atom (the
    remaining degradation realized by authored unavailability, resolver:
    null -- the schema's existing evidence-unavailable mechanism);
  * a natural-prevalence set with NO support rebalancing (truthy-derived
    predicates; the record's own field availability determines the level);
  * machine-readable claim DENOTATIONS (DNF over (field, predicate,
    expected)) for the content-false endpoint;
  * family splits 40/10/50 dev/val/heldout, families held out whole,
    replacement forbidden;
  * verify-or-drop with two-type attrition accounting (Part R, P10).

Every instance passes schema validation AND an atom-value verify (resolver
executed through the real ToolExecutor; observed truth must equal the
stored gold_value). No LLM, no judge, no hand-authored per-instance content.

Outputs:
  bench/instances_v2/<domain>/<split>/<id>.json
  bench/instances_v2/manifest.json      (counts, strata, attrition, splits)

Usage: python3 bench/generate_record_instances.py [--domains ...]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import glob
import json
import os
import random
import sys
import zlib
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, os.path.join(PILOT, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime.predicates import evaluate_predicate, format_predicate_rule  # noqa: E402
from prime_runtime.schema import SchemaError, load_instance  # noqa: E402

SEED = 20260801
# Offsets of the family-enumeration seed, per domain. Earlier versions drew them as
# hash(domain) % 1000, which Python randomizes in every process (PYTHONHASHSEED), so
# each run enumerated different families from the same seed. These are the offsets
# the published files were generated with, recovered by matching the enumerated
# families against them: the primary corpus (run of 2026-08-01) and the one-branch
# variant (run of 2026-10-08, which drew its own offsets, so its family ids name
# other probe sets than the same ids in the primary corpus). A domain not listed
# gets a stable offset from its name.
FAMILY_SEED_OFFSET = {"security": 472, "clinical": 363, "drugsafety": 94, "financial": 382}
ONE_BRANCH_FAMILY_SEED_OFFSET = {"security": 965, "clinical": 547, "drugsafety": 464, "financial": 866}


def family_rng(domain: str, offsets: Dict[str, int] = FAMILY_SEED_OFFSET) -> random.Random:
    off = offsets.get(domain, zlib.crc32(domain.encode("utf-8")) % 1000)
    return random.Random(SEED + off)
OUT = os.path.join(HERE, "instances_v2")

# ---------------------------------------------------------------------------
# Per-domain probe pools. A probe = a named, reliably-present field with a
# value kind that determines (a) which predicate makes it TRUE from the
# record's own value, (b) which plausible predicate makes it FALSE, and
# (c) whether an "unavailable" variant exists (field absent in some records
# or authored resolver:null).
# kind: str_exists | str_equals | date | numeric | bool
# ---------------------------------------------------------------------------
POOLS: Dict[str, Dict[str, Any]] = {
    "security": {
        "user_prompt": "Is this security advisory complete and current enough to act on for the affected package?",
        "scope": "The single named OSV advisory record, as retrieved and embedded; no external lookups.",
        "claim_text": {
            1: "The advisory identifies an affected package with a readable summary (as inspected).",
            2: "The advisory is triage-ready: affected package identified, summary present, and its review/severity metadata are populated as inspected.",
            3: "The advisory satisfies the checked currency-and-completeness protocol: triage-ready per the inspected fields, with consistent publication metadata and populated reference and classification fields.",
        },
        "probes": [
            ("pkg_name", "affected.0.package.name", "str_exists"),
            ("summary", "summary", "str_exists"),
            ("details", "details", "str_exists"),
            ("ecosystem", "affected.0.package.ecosystem", "str_equals"),
            ("published", "published", "date"),
            ("modified", "modified", "date"),
            ("reviewed", "database_specific.github_reviewed", "bool"),
            ("severity", "database_specific.severity", "str_exists"),
            ("ref_type", "references.0.type", "str_exists"),
            ("ref_url", "references.0.url", "str_exists"),
            ("cwe", "database_specific.cwe_ids.0", "str_exists"),
            ("cvss", "severity.0.score", "str_exists"),
            ("alias", "aliases.0", "str_exists"),
            ("schema_v", "schema_version", "str_exists"),
        ],
        "optionalish": ["cwe", "cvss", "alias"],
    },
    "clinical": {
        "user_prompt": "Is this trial registration current and complete enough for eligibility screening?",
        "scope": "The single named ClinicalTrials.gov study record, as retrieved and embedded; no external lookups.",
        "claim_text": {
            1: "The registration identifies the study with a readable title and declared status (as inspected).",
            2: "The registration is screening-ready: identified, status declared, study type and eligibility fields populated as inspected.",
            3: "The registration satisfies the checked currency-and-completeness protocol: screening-ready per the inspected fields, with consistent submission dates and populated eligibility metadata.",
        },
        "probes": [
            ("nct", "protocolSection.identificationModule.nctId", "str_exists"),
            ("title", "protocolSection.identificationModule.briefTitle", "str_exists"),
            ("status", "protocolSection.statusModule.overallStatus", "str_exists"),
            ("verified", "protocolSection.statusModule.statusVerifiedDate", "str_exists"),
            ("first_submit", "protocolSection.statusModule.studyFirstSubmitDate", "date"),
            ("last_update", "protocolSection.statusModule.lastUpdateSubmitDate", "date"),
            ("study_type", "protocolSection.designModule.studyType", "str_equals"),
            ("summary", "protocolSection.descriptionModule.briefSummary", "str_exists"),
            ("eligibility", "protocolSection.eligibilityModule.eligibilityCriteria", "str_exists"),
            ("sex", "protocolSection.eligibilityModule.sex", "str_exists"),
            ("official", "protocolSection.identificationModule.officialTitle", "str_exists"),
            ("min_age", "protocolSection.eligibilityModule.minimumAge", "str_exists"),
            ("healthy", "protocolSection.eligibilityModule.healthyVolunteers", "bool"),
        ],
        "optionalish": ["min_age", "official", "healthy"],
    },
    "drugsafety": {
        "user_prompt": "Is this recall record complete and correctly classified for downstream pharmacy action?",
        "scope": "The single named openFDA enforcement record, as retrieved and embedded; no external lookups.",
        "claim_text": {
            1: "The recall is identified with a stated classification (as inspected).",
            2: "The recall is action-ready: identified, classified, with product description, reason, and status populated as inspected.",
            3: "The recall satisfies the checked completeness protocol: action-ready per the inspected fields, with consistent dates and populated distribution and firm fields.",
        },
        "probes": [
            ("recall_no", "recall_number", "str_exists"),
            ("classification", "classification", "str_equals"),
            ("status", "status", "str_equals"),
            ("product", "product_description", "str_exists"),
            ("reason", "reason_for_recall", "str_exists"),
            ("firm", "recalling_firm", "str_exists"),
            # openFDA dates are compact YYYYMMDD; the runtime date parser is
            # ISO-only, so we compare them NUMERICALLY (identical ordering)
            # rather than changing frozen predicate semantics.
            ("init_date", "recall_initiation_date", "date_compact"),
            ("report_date", "report_date", "date_compact"),
            ("dist", "distribution_pattern", "str_exists"),
            ("voluntary", "voluntary_mandated", "str_exists"),
            ("country", "country", "str_equals"),
            ("event", "event_id", "str_exists"),
            ("quantity", "product_quantity", "str_exists"),
            ("code_info", "code_info", "str_exists"),
        ],
        "optionalish": ["quantity", "code_info", "dist"],
    },
    "financial": {
        "user_prompt": "Are this filer's reported balance-sheet figures present and within the declared plausibility protocol for the latest filings?",
        "scope": "The single named SEC companyfacts record, as retrieved and embedded; no external lookups.",
        "claim_text": {
            1: "The filer is identified and reports a total-assets figure (as inspected).",
            2: "The filer's core balance-sheet figures are present and positive as inspected under the declared protocol.",
            3: "The filer satisfies the checked reporting protocol: core figures present, positive, and within the declared magnitude bounds, with cash and equity fields populated as inspected.",
        },
        "probes": [
            # pool rebuilt from measured coverage over the 400-record
            # harvest (>=71% each; entityName/cik 100%)
            ("entity", "entityName", "str_exists"),
            ("cik", "cik", "numeric"),
            ("assets", "facts.us-gaap.Assets.units.USD.0.val", "numeric"),
            ("l_and_e", "facts.us-gaap.LiabilitiesAndStockholdersEquity.units.USD.0.val", "numeric"),
            ("net_income", "facts.us-gaap.NetIncomeLoss.units.USD.0.val", "numeric"),
            ("aoci", "facts.us-gaap.AccumulatedOtherComprehensiveIncomeLossNetOfTax.units.USD.0.val", "numeric"),
            ("equity", "facts.us-gaap.StockholdersEquity.units.USD.0.val", "numeric"),
            ("eps", "facts.us-gaap.EarningsPerShareBasic.units.USD/shares.0.val", "numeric"),
            ("retained", "facts.us-gaap.RetainedEarningsAccumulatedDeficit.units.USD.0.val", "numeric"),
            ("cash", "facts.us-gaap.CashAndCashEquivalentsAtCarryingValue.units.USD.0.val", "numeric"),
            ("ppe", "facts.us-gaap.PropertyPlantAndEquipmentNet.units.USD.0.val", "numeric"),
            ("shares_issued", "facts.us-gaap.CommonStockSharesIssued.units.shares.0.val", "numeric"),
            ("ppe_dep", "facts.us-gaap.AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment.units.USD.0.val", "numeric"),
        ],
        "optionalish": ["shares_issued", "ppe_dep", "ppe"],
    },
}

ACTIONS = ["inspect_record", "ask_user", "emit_claim"]

# What each probe must read, in words, for the locator-uncertain split: the
# specification view there names the quantity and lists candidate fields
# (the true one among them) but not which field holds it.
WHAT: Dict[str, Dict[str, str]] = {
    "security": {
        "pkg_name": "the name of the affected package", "summary": "the advisory's one-line summary",
        "details": "the advisory's detailed description", "ecosystem": "the ecosystem of the affected package",
        "published": "the date the advisory was published", "modified": "the date the advisory was last modified",
        "reviewed": "whether the advisory has been reviewed by the database maintainers",
        "severity": "the maintainers' severity label", "ref_type": "the type of the first reference",
        "ref_url": "the URL of the first reference", "cwe": "the first CWE identifier",
        "cvss": "the first CVSS score string", "alias": "the first alias identifier (such as a CVE)",
        "schema_v": "the schema version of the record",
    },
    "clinical": {
        "nct": "the NCT identifier", "title": "the brief title", "status": "the overall recruitment status",
        "verified": "the date the status was last verified", "first_submit": "the date the registration was first submitted",
        "last_update": "the date of the most recent update submission", "study_type": "the study type",
        "summary": "the brief summary", "eligibility": "the eligibility criteria text",
        "sex": "the sex eligibility field", "official": "the official title", "min_age": "the minimum age for eligibility",
        "healthy": "whether healthy volunteers are accepted",
    },
    "drugsafety": {
        "recall_no": "the recall number", "classification": "the recall classification",
        "status": "the recall status", "product": "the description of the recalled product",
        "reason": "the stated reason for the recall", "firm": "the recalling firm",
        "init_date": "the date the recall was initiated", "report_date": "the date the recall was reported",
        "dist": "the distribution pattern", "voluntary": "whether the recall was voluntary or mandated",
        "country": "the country", "event": "the event identifier", "quantity": "the product quantity",
        "code_info": "the lot or code information",
    },
    "financial": {
        "entity": "the filer's name", "cik": "the filer's CIK number", "assets": "the latest reported total assets",
        "l_and_e": "the latest reported total liabilities and stockholders' equity",
        "net_income": "the latest reported net income or loss",
        "aoci": "the latest reported accumulated other comprehensive income or loss",
        "equity": "the latest reported stockholders' equity", "eps": "the latest reported basic earnings per share",
        "retained": "the latest reported retained earnings or accumulated deficit",
        "cash": "the latest reported cash and cash equivalents", "ppe": "the latest reported net property, plant and equipment",
        "shares_issued": "the latest reported number of common shares issued",
        "ppe_dep": "the latest reported accumulated depreciation on property, plant and equipment",
    },
}


def derive_locator_split(domains, per_domain: int, n_candidates: int = 3) -> int:
    """heldout_locator: the balanced tasks with the atoms' exact fields
    withheld from the specification view. Each resolvable atom gets a
    ``spec_hint`` {what, candidates}: the quantity it must read, in words,
    and ``n_candidates`` field paths in a seeded random order, the true one
    among them; the decoys are other fields of the same domain that are
    present on the record and are not read by any atom of the task, chosen
    from the same value kind when possible, and their values are added to
    the embedded record so that absence gives nothing away. Resolvers keep
    the true locator, so attestation and ground truth are unchanged; ids are
    the balanced ids plus 6000."""
    mpath = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"seed": SEED, "domains": {}}
    manifest.setdefault("locator", {})
    tmpdir = os.path.join(OUT, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    for domain in domains:
        pool = {p[0]: p for p in POOLS[domain]["probes"]}
        path_of = {p[1]: p[0] for p in POOLS[domain]["probes"]}
        rows = {}
        with open(os.path.join(HERE, "harvest", domain, "records.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows[r["record_id"]] = r["record"]
        src = sorted(glob.glob(os.path.join(OUT, domain, "heldout_balanced", "*.json")))[:per_domain]
        sub = os.path.join(OUT, domain, "heldout_locator")
        os.makedirs(sub, exist_ok=True)
        n_ok = n_fail = 0
        for p in src:
            inst = json.load(open(p, encoding="utf-8"))
            record = rows.get(inst["seed"])
            if record is None:
                n_fail += 1
                continue
            rid = inst["seed"]
            used_paths = {a["resolver"]["params"]["field_path"] for a in inst["atoms"] if a.get("resolver")}
            # the fields of atoms declared unavailable (resolver None) must stay
            # absent from the embedded record; they are recoverable from the
            # claim denotations, whose entries follow the route literals in order
            unavail_paths = set()
            routes_by_claim = inst["gold_licenses"]
            for c in inst["claims"]:
                for route, conj in zip(routes_by_claim.get(c["id"], []), c.get("denotation") or []):
                    for lit, (fp, _op, _exp) in zip(route, conj):
                        if fp not in used_paths:
                            unavail_paths.add(fp)
            rec_slice = inst["records"][rid]
            for a in inst["atoms"]:
                if not a.get("resolver"):
                    continue
                true_path = a["resolver"]["params"]["field_path"]
                probe = path_of.get(true_path)
                if probe is None:
                    continue
                kind = pool[probe][2]
                rng = random.Random("%s|%s|%s" % (SEED, inst["id"], a["id"]))
                same = [q for q in POOLS[domain]["probes"] if q[2] == kind and q[1] not in used_paths]
                other = [q for q in POOLS[domain]["probes"] if q[2] != kind and q[1] not in used_paths]
                present = lambda q: get_path(record, q[1])[1] and get_path(record, q[1])[0] not in (None, "", [])
                decoys = [q for q in same if present(q)]
                rng.shuffle(decoys)
                extra = [q for q in other if present(q)]
                rng.shuffle(extra)
                decoys = [q for q in decoys + extra if q[1] not in unavail_paths][:n_candidates - 1]
                for q in decoys:
                    slice_into(rec_slice, record, q[1])
                cands = [true_path] + [q[1] for q in decoys]
                rng.shuffle(cands)
                a["spec_hint"] = {"what": WHAT[domain][probe], "candidates": cands}
            seq = int(inst["id"].rsplit("-", 1)[1]) + 6000
            new_id = "%s-v2-%04d" % (domain[:3], seq)
            meta = json.loads(inst["notes_for_annotators"])
            meta["locator"] = True
            meta["source_instance"] = inst["id"]
            inst["id"] = new_id
            inst["notes_for_annotators"] = json.dumps(meta)
            ok, why = verify(inst, tmpdir)
            if not ok:
                n_fail += 1
                print("[%s] %s: %s" % (domain, new_id, why), file=sys.stderr)
                continue
            with open(os.path.join(sub, new_id + ".json"), "w", encoding="utf-8") as fh:
                json.dump(inst, fh, indent=1, sort_keys=True, default=str)
            n_ok += 1
        manifest["locator"][domain] = {"count": n_ok, "failed": n_fail, "n_candidates": n_candidates,
                                       "source": "heldout_balanced"}
        print("[%s] heldout_locator=%d failed=%d" % (domain, n_ok, n_fail), file=sys.stderr)
    os.rmdir(tmpdir) if not os.listdir(tmpdir) else None
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print("wrote %s/manifest.json" % OUT, file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# field access (mirrors runner/arms._resolve_field_path semantics)
# ---------------------------------------------------------------------------
def get_path(record: Any, path: str) -> Tuple[Any, bool]:
    cur = record
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None, False
    return cur, True


def slice_into(dst: Dict[str, Any], record: Any, path: str) -> None:
    """Copy record[path] into dst preserving the nested/indexed structure."""
    parts = path.split(".")
    cur_src, cur_dst = record, dst
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(cur_src, dict):
            if part not in cur_src:
                return
            nxt = cur_src[part]
            if last:
                cur_dst[part] = nxt
            else:
                cur_dst = cur_dst.setdefault(part, [] if isinstance(nxt, list) else {})
            cur_src = nxt
        elif isinstance(cur_src, list) and part.isdigit():
            idx = int(part)
            if idx >= len(cur_src):
                return
            nxt = cur_src[idx]
            while len(cur_dst) <= idx:
                cur_dst.append({} if isinstance(nxt, dict) else ([] if isinstance(nxt, list) else None))
            if last:
                cur_dst[idx] = nxt
            else:
                if not isinstance(cur_dst[idx], (dict, list)):
                    cur_dst[idx] = {} if isinstance(nxt, dict) else []
                cur_dst = cur_dst[idx]
            cur_src = nxt
        else:
            return


# ---------------------------------------------------------------------------
# predicate derivation: TRUE from the record's own value; FALSE plausibly
# ---------------------------------------------------------------------------
def true_pred(kind: str, value: Any) -> Tuple[str, Any]:
    if kind == "str_exists":
        return ("exists", None)
    if kind == "str_equals":
        return ("equals", value)
    if kind == "bool":
        return ("equals", value)
    if kind == "date":
        d = str(value)[:10]
        year = int(d[:4]) + 2
        return ("date_before", "%04d-01-01" % year)
    if kind == "date_compact":
        return ("numeric_le", int(str(value)[:8]) + 20000)  # ~2 years later
    if kind == "numeric":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return ("exists", None)
        bound = 0 if v >= 0 else v * 2 - 1
        return ("numeric_ge", bound)
    return ("exists", None)


def false_pred(kind: str, value: Any, donor: Any) -> Tuple[str, Any]:
    """A PLAUSIBLE predicate the record falsifies (never gibberish)."""
    if kind in ("str_equals", "bool", "str_exists"):
        d = donor if (donor is not None and donor != value) else "%s-alt" % value
        return ("equals", d)
    if kind == "date":
        d = str(value)[:10]
        year = max(1990, int(d[:4]) - 1)
        return ("date_before", "%04d-01-01" % year)
    if kind == "date_compact":
        return ("numeric_le", int(str(value)[:8]) - 10000)  # ~1 year earlier
    if kind == "numeric":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return ("equals", donor if donor is not None else 0)
        return ("numeric_ge", abs(v) * 10 + 1e9)
    return ("equals", "__none__")


# ---------------------------------------------------------------------------
# family enumeration
# ---------------------------------------------------------------------------
def enumerate_families(domain: str, rng: random.Random) -> List[Dict[str, Any]]:
    """Deterministic family list: >=16 branching + >=8 nested, split evenly
    across the cost 2x2. A family fixes: topology, cost profile, and the
    probe composition of each level (C1: 1 probe; C2: +2; C3: +2..3; for
    branching, an alternative C2 branch of 2 probes)."""
    pool = [p for p in POOLS[domain]["probes"]]
    names = [p[0] for p in pool]
    fams: List[Dict[str, Any]] = []
    combos = list(itertools.combinations(range(len(pool)), 6))
    rng.shuffle(combos)
    seen = set()
    for topo, need in (("branching", 20), ("nested", 10)):
        made = 0
        for combo in combos:
            if made >= need:
                break
            sel = [names[i] for i in combo]
            key = (topo, tuple(sorted(sel)))
            if key in seen:
                continue
            seen.add(key)
            cost = "het" if (made % 2 == 0) else "hom"
            c1 = sel[:1]
            if topo == "nested":
                fam = {"routes": {"C1": [c1], "C2": [sel[:3]], "C3": [sel[:6]]}}
            else:
                # branching: two C2 alternatives sharing the C1 prefix;
                # C3 extends branch A (order-coherent: contains a C2 route)
                fam = {"routes": {"C1": [c1],
                                  "C2": [sel[:3], [sel[0], sel[3], sel[4]]],
                                  "C3": [sel[:3] + [sel[5]]]}}
            if cost == "het":
                # heterogeneous call-class cost: duplicate-field probe pair
                # (two predicates on ONE field -> one call class): realized
                # at instantiation by adding a second predicate on sel[1]'s
                # field into C3's route.
                fam["het_pair_on"] = sel[1]
            fam.update({"family_id": "%s-%s-%s-%02d" % (domain, topo, cost, made),
                        "domain": domain, "topology": topo, "cost": cost,
                        "probes": sel})
            fams.append(fam)
            made += 1
    return fams


# ---------------------------------------------------------------------------
# instance construction
# ---------------------------------------------------------------------------
def build_instance(domain: str, fam: Dict[str, Any], rec_row: Dict[str, Any],
                   target_level: Optional[int], false_mech: bool,
                   donor_rec: Any, seq: int,
                   one_branch: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
    """Build one instance JSON dict, or (None, reason) on construction
    failure. target_level None => natural (truthy predicates, no
    engineering). false_mech: degrade via FALSE atom (else via authored
    unavailability).

    one_branch (branching families only): degrade ONE probe on exactly one
    of the two alternative routes to C2, chosen so that the probe also lies
    on C3's route. The other alternative stays satisfiable, so C2 is
    supported through it alone and C3 is not: the agent has to abandon the
    failed branch and find the surviving one. The balanced split never
    contains this case (there the degraded atom is on every C2 route or on
    none); this is the heldout_onebranch split."""
    pool = {p[0]: p for p in POOLS[domain]["probes"]}
    record = rec_row["record"]
    rid = rec_row["record_id"]

    # every probe used by the family must resolve on this record
    used = list(dict.fromkeys([n for routes in fam["routes"].values()
                               for r in routes for n in r]))
    values = {}
    for n in used:
        v, ok = get_path(record, pool[n][1])
        if not ok or v in (None, "", []):
            return None, "field-missing:%s" % n
    for n in used:
        values[n] = get_path(record, pool[n][1])[0]

    # choose the degraded atom (the false/unavailable load-bearing one)
    atoms_spec: Dict[str, Dict[str, Any]] = {}
    degrade_probe = None
    if one_branch:
        c1p = fam["routes"]["C1"][0][0]
        c2_routes = fam["routes"]["C2"]
        if len(c2_routes) < 2:
            return None, "no-branch"
        c3set = {n for r in fam["routes"]["C3"] for n in r}
        kill = None
        for r in c2_routes:
            shared = [n for n in r if n != c1p and n in c3set
                      and not all(n in r2 for r2 in c2_routes)]
            if shared:
                kill = shared[0]
                break
        if kill is None:
            return None, "no-degradable-c2"
        degrade_probe = [kill]
    elif target_level is not None and target_level < 3:
        # level 2: kill a C3-only probe; level 1: kill a C2 probe not in C1;
        # level 0: kill the C1 probe. For branching, level<=1 must kill the
        # SHARED prefix (else the other branch survives).
        c1p = fam["routes"]["C1"][0][0]
        c2_only = [n for r in fam["routes"]["C2"] for n in r if n != c1p]
        c3_only = [n for r in fam["routes"]["C3"] for n in r
                   if all(n not in r2 for r2 in fam["routes"]["C2"]) and n != c1p]
        if target_level == 2:
            cand = c3_only or c2_only
        elif target_level == 1:
            # kill one probe in EVERY C2 route (branching: one per branch)
            cand = None
        else:
            cand = [c1p]
        if target_level == 1:
            kill = []
            for r in fam["routes"]["C2"]:
                ks = [n for n in r if n != c1p]
                if not ks:
                    return None, "no-degradable-c2"
                kill.append(ks[0])
            degrade_probe = kill
        else:
            if not cand:
                return None, "no-degradable-c3"
            degrade_probe = [cand[0]]

    donor_vals = {}
    for n in used:
        dv, ok = get_path(donor_rec, pool[n][1]) if donor_rec is not None else (None, False)
        donor_vals[n] = dv if ok else None

    # atom list: E1.. in used order (+ heterogeneous twin + distractor)
    eid = 0
    id_of = {}
    for n in used:
        eid += 1
        aid = "E%d" % eid
        id_of[n] = aid
        kind = pool[n][2]
        if degrade_probe and n in degrade_probe:
            if false_mech:
                op, exp = false_pred(kind, values[n], donor_vals.get(n))
                atoms_spec[aid] = {"probe": n, "op": op, "exp": exp, "avail": True}
            else:
                atoms_spec[aid] = {"probe": n, "op": "exists", "exp": None, "avail": False}
        else:
            op, exp = true_pred(kind, values[n])
            atoms_spec[aid] = {"probe": n, "op": op, "exp": exp, "avail": True}

    routes = {c: [[id_of[n] for n in r] for r in rs]
              for c, rs in fam["routes"].items()}

    # heterogeneous cost: add a second TRUE predicate on het_pair_on's field
    # to C3's first route (same field -> same call class as its twin)
    if fam.get("het_pair_on"):
        n = fam["het_pair_on"]
        if n in id_of and atoms_spec[id_of[n]]["avail"]:
            eid += 1
            twin = "E%d" % eid
            kind = pool[n][2]
            v = values[n]
            if kind == "date":
                op, exp = "date_after", "1980-01-01"
            elif kind == "date_compact":
                op, exp = "numeric_ge", 19800101
            elif kind == "numeric":
                try:
                    op, exp = "numeric_le", abs(float(v)) * 10 + 1e9
                except (TypeError, ValueError):
                    op, exp = "exists", None
            else:
                op, exp = "exists", None
            atoms_spec[twin] = {"probe": n, "op": op, "exp": exp, "avail": True}
            routes["C3"] = [r + [twin] for r in routes["C3"]]

    # distractor: a pool probe NOT in the family, truthy, non-route
    spare = [p for p in POOLS[domain]["probes"] if p[0] not in used]
    if spare:
        n, path, kind = spare[seq % len(spare)]
        v, ok = get_path(record, path)
        if ok and v not in (None, "", []):
            eid += 1
            aid = "E%d" % eid
            op, exp = true_pred(kind, v)
            atoms_spec[aid] = {"probe": n, "op": op, "exp": exp, "avail": True,
                               "distractor": True}

    # gold values + record slice
    rec_slice: Dict[str, Any] = {}
    atoms_json = []
    for aid in sorted(atoms_spec, key=lambda x: int(x[1:])):
        sp = atoms_spec[aid]
        path = pool[sp["probe"]][1]
        if not sp["avail"]:
            atoms_json.append({"id": aid, "type": "record", "proposition": "",
                               "resolver": None, "gold_value": "F"})
            continue
        slice_into(rec_slice, record, path)
        v, ok = get_path(record, path)
        verdict = bool(ok) and bool(evaluate_predicate(sp["op"], sp["exp"], v))
        atoms_json.append({
            "id": aid, "type": "record", "proposition": "",
            "resolver": {"tool": "inspect_record",
                         "params_schema": {"field_path": "str", "record_id": "str"},
                         "params": {"field_path": path, "record_id": rid},
                         "attest": format_predicate_rule(sp["op"], sp["exp"])},
            "gold_value": "T" if verdict else "F"})

    texts = POOLS[domain]["claim_text"]
    claims = [{"id": c, "text": texts[i + 1], "level": i + 1,
               "denotation": [[[pool[atoms_spec[a]["probe"]][1],
                                atoms_spec[a]["op"], atoms_spec[a]["exp"]]
                               for a in r] for r in routes[c]]}
              for i, c in enumerate(("C1", "C2", "C3"))]

    iid = "%s-v2-%04d" % (domain[:3], seq)
    inst = {
        "id": iid, "domain": "%s_v2" % domain, "seed": rid, "variant": "base",
        "user_prompt": POOLS[domain]["user_prompt"],
        "artifact": {"path": "", "kind": "record", "commit": ""},
        # Record identifier IS spec-view content (artifact identity /
        # locator params are permitted; field VALUES are not). Omitting it
        # starved every masked arm in the first dev pilot: models
        # hallucinated record_ids, no tool call ever matched an atom's
        # resolver params, and the ledger never attested anything.
        "scope": POOLS[domain]["scope"] + " Record identifier: %s." % rid,
        "actions": ACTIONS,
        "claims": claims,
        "order": [["C1", "C2"], ["C2", "C3"]],
        "atoms": atoms_json,
        "phi": [],
        "gold_licenses": routes,
        "decoys": [],
        "records": {rid: rec_slice},
        "notes_for_annotators": json.dumps({
            "family": fam["family_id"], "topology": fam["topology"],
            "cost": fam["cost"], "target_level": target_level,
            "false_mech": bool(false_mech and (one_branch or (
                target_level is not None and target_level < 3))),
            # the key appears on one-branch tasks only, as in the published files
            **({"one_branch": True} if one_branch else {})}),
    }
    return inst, ""


def achieved_level(inst: Dict[str, Any]) -> int:
    gv = {a["id"]: a["gold_value"] for a in inst["atoms"]}
    lvl = 0
    levels = {c["id"]: c["level"] for c in inst["claims"]}
    for cid, routes in inst["gold_licenses"].items():
        if any(all(gv.get(x) == "T" for x in r) for r in routes):
            lvl = max(lvl, levels[cid])
    return lvl


def satisfiable_routes(inst: Dict[str, Any], cid: str) -> int:
    gv = {a["id"]: a["gold_value"] for a in inst["atoms"]}
    return sum(1 for r in inst["gold_licenses"][cid]
               if all(gv.get(x) == "T" for x in r))


def generate_one_branch(domains, per_domain: int, seq_base: int) -> int:
    """The heldout_onebranch split: branching held-out families, records not
    used by any existing instance of the domain, one C2 alternative degraded.
    Nothing already on disk is touched; ids start at seq_base so they cannot
    collide with the balanced and natural ids."""
    rng = random.Random(SEED)
    tmpdir = os.path.join(OUT, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    mpath = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"seed": SEED, "domains": {}}
    manifest.setdefault("one_branch", {})
    for domain in domains:
        fams = enumerate_families(domain, family_rng(domain, ONE_BRANCH_FAMILY_SEED_OFFSET))
        heldout = {fid for fid, info in manifest["domains"][domain]["families"].items()
                   if info["split"] == "heldout" and info["topology"] == "branching"}
        fams = [f for f in fams if f["family_id"] in heldout]
        used_records = set()
        for split in ("dev", "val", "heldout_balanced", "heldout_natural", "heldout_onebranch"):
            for p in glob.glob(os.path.join(OUT, domain, split, "*.json")):
                try:
                    used_records.add(json.load(open(p, encoding="utf-8"))["seed"])
                except (OSError, ValueError, KeyError):
                    continue
        rows = [json.loads(l) for l in
                open(os.path.join(HERE, "harvest", domain, "records.jsonl"))]
        rng.shuffle(rows)
        rows = [r for r in rows if r["record_id"] not in used_records]
        sub = os.path.join(OUT, domain, "heldout_onebranch")
        have = len(glob.glob(os.path.join(sub, "*.json")))
        att = {"field-missing": 0, "no-degradable": 0, "schema": 0,
               "verify-mismatch": 0, "stratum-miss": 0}
        n_false = 0
        seq = have
        ri = 0
        fi = 0
        while have < per_domain and ri < len(rows) and fams:
            f = fams[fi % len(fams)]
            fi += 1
            row = rows[ri]
            ri += 1
            fm = FALSE_MECH_CYCLE[seq % len(FALSE_MECH_CYCLE)]
            donor = rows[ri % len(rows)]["record"] if len(rows) > 1 else None
            inst, why = build_instance(domain, f, row, 2, fm, donor,
                                       seq_base + seq + 1, one_branch=True)
            if inst is None:
                att["field-missing" if why.startswith("field-missing") else "no-degradable"] += 1
                continue
            if achieved_level(inst) != 2 or satisfiable_routes(inst, "C2") != 1:
                att["stratum-miss"] += 1
                continue
            ok, why = verify(inst, tmpdir)
            if not ok:
                att["schema" if why.startswith("schema") else "verify-mismatch"] += 1
                continue
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, inst["id"] + ".json"), "w", encoding="utf-8") as fh:
                json.dump(inst, fh, indent=1, sort_keys=True, default=str)
            have += 1
            seq += 1
            n_false += int(json.loads(inst["notes_for_annotators"])["false_mech"])
        manifest["one_branch"][domain] = {
            "count": have, "families": sorted(f["family_id"] for f in fams),
            "false_mech_instances": n_false, "construction_attrition": att,
            "seq_base": seq_base}
        print("[%s] heldout_onebranch=%d false-mech=%d attrition=%s" %
              (domain, have, n_false, att), file=sys.stderr)
    os.rmdir(tmpdir) if not os.listdir(tmpdir) else None
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print("wrote %s/manifest.json" % OUT, file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# verify-or-drop (schema + executor replay)
# ---------------------------------------------------------------------------
def verify(inst: Dict[str, Any], tmpdir: str) -> Tuple[bool, str]:
    path = os.path.join(tmpdir, inst["id"] + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(inst, fh, default=str)
    try:
        obj = load_instance(path, validate=True)
    except (SchemaError, Exception) as exc:  # noqa: BLE001
        os.remove(path)
        return False, "schema:%s" % exc
    import arms as arms_mod
    from prime_runtime import goldledger
    from prime_runtime.ledger import Ledger
    ex = arms_mod.ToolExecutor(obj)
    ledger = Ledger(obj.atom_ids, obj.phi_patterns)
    for a in obj.atoms:
        if a.resolver is None:
            continue
        obs = ex.execute(a.resolver.tool, dict(a.resolver.params))
        for att in goldledger.delta(obj, ledger, a.resolver.tool,
                                    dict(a.resolver.params), obs, step=1):
            ledger.apply(att)
    gv = {x["id"]: x["gold_value"] for x in inst["atoms"]}
    for a in obj.atoms:
        if a.resolver is None:
            continue
        st = ledger.state(a.id)
        want = gv[a.id]
        if st != want:
            os.remove(path)
            return False, "verify-mismatch:%s observed=%s gold=%s" % (a.id, st, want)
    os.remove(path)
    return True, ""


# ---------------------------------------------------------------------------
# main generation loop
# ---------------------------------------------------------------------------
STRATA = [3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 3, 2, 1, 0, 3, 2]  # ~30/30/25/15 cycle
FALSE_MECH_CYCLE = [True, True, False, True, False, True, False]  # ~57% of degraded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+", default=list(POOLS))
    ap.add_argument("--balanced-per-domain", type=int, default=120)
    ap.add_argument("--natural-per-domain", type=int, default=60)
    ap.add_argument("--dev-per-domain", type=int, default=48)
    ap.add_argument("--natural-topup", action="store_true",
                    help="fill ONLY heldout_natural, using records beyond "
                    "--records-offset (previously generated dev/val/balanced "
                    "instances stay byte-identical)")
    ap.add_argument("--records-offset", type=int, default=0)
    ap.add_argument("--seq-base", type=int, default=0)
    ap.add_argument("--locator-per-domain", type=int, default=0,
                    help="derive ONLY the heldout_locator split (N balanced tasks "
                    "per domain, 120 = all) with the atoms' exact fields withheld "
                    "from the specification view (spec_hint: what + candidates)")
    ap.add_argument("--one-branch-per-domain", type=int, default=0,
                    help="generate ONLY the heldout_onebranch split (N tasks "
                    "per domain) from branching held-out families and unused "
                    "records; existing splits stay byte-identical; ids start "
                    "at --seq-base (9000 if unset)")
    args = ap.parse_args()
    if args.locator_per_domain:
        return derive_locator_split(args.domains, args.locator_per_domain)
    if args.one_branch_per_domain:
        return generate_one_branch(args.domains, args.one_branch_per_domain,
                                   args.seq_base or 9000)

    rng = random.Random(SEED)
    mpath = os.path.join(OUT, "manifest.json")
    manifest: Dict[str, Any] = (json.load(open(mpath)) if os.path.exists(mpath)
                                else {"seed": SEED, "domains": {}})
    manifest["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmpdir = os.path.join(OUT, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    for domain in args.domains:
        fams = enumerate_families(domain, family_rng(domain))
        # family splits 40/10/50 per topology
        splits: Dict[str, str] = {}
        for topo in ("branching", "nested"):
            fs = [f for f in fams if f["topology"] == topo]
            n_dev = max(1, int(0.4 * len(fs)))
            n_val = max(1, int(0.1 * len(fs)))
            for i, f in enumerate(fs):
                splits[f["family_id"]] = ("dev" if i < n_dev else
                                          "val" if i < n_dev + n_val else "heldout")
        rows = [json.loads(l) for l in
                open(os.path.join(HERE, "harvest", domain, "records.jsonl"))]
        if args.natural_topup:
            rows = rows[args.records_offset:]
        rng.shuffle(rows)
        # record -> exactly one family (round-robin over families)
        assign: Dict[str, List[Dict[str, Any]]] = {f["family_id"]: [] for f in fams}
        for i, row in enumerate(rows):
            assign[fams[i % len(fams)]["family_id"]].append(row)

        att = {"field-missing": 0, "no-degradable": 0, "schema": 0,
               "verify-mismatch": 0, "stratum-miss": 0}
        counts = {"dev": 0, "val": 0, "heldout_balanced": 0, "heldout_natural": 0}
        topo_counts: Dict[Tuple[str, str], int] = {}
        if args.natural_topup:
            # count existing natural instances so the quota tops up
            existing = glob_natural = os.path.join(OUT, domain, "heldout_natural")
            n_have = len([f for f in os.listdir(existing)]) if os.path.isdir(existing) else 0
            counts["heldout_natural"] = n_have
            counts["dev"] = counts["val"] = counts["heldout_balanced"] = 10**9  # closed
        strata_out: Dict[str, int] = {"3": 0, "2": 0, "1": 0, "0": 0}
        n_false_lb = 0
        seq = 0
        quota = {"dev": args.dev_per_domain, "heldout_balanced": args.balanced_per_domain,
                 "heldout_natural": args.natural_per_domain, "val": 12}
        topo_quota = dict(quota)

        # K1 stratification: alternate topologies so every bucket fills
        # ~50/50 branching/nested (the old family-sequential loop exhausted
        # quotas on branching families alone -- caught by the disk recount).
        b_f = [f for f in fams if f["topology"] == "branching"]
        n_f = [f for f in fams if f["topology"] == "nested"]
        order: List[Dict[str, Any]] = []
        for i in range(max(len(b_f), len(n_f))):
            if i < len(b_f):
                order.append(b_f[i])
            if i < len(n_f):
                order.append(n_f[i])
        cursors = {f["family_id"]: 0 for f in fams}
        progressed = True
        passes = 0
        while progressed and passes < 400:
            progressed = False
            passes += 1
            for f in order:
                split = splits[f["family_id"]]
                pool_rows = assign[f["family_id"]]
                di = cursors[f["family_id"]]
                if di >= len(pool_rows):
                    continue
                row = pool_rows[di]
                cursors[f["family_id"]] = di + 1
                progressed = True
                bucket = None
                if split == "dev" and counts["dev"] < quota["dev"]:
                    bucket = "dev"
                elif split == "val" and counts["val"] < quota["val"]:
                    bucket = "val"
                elif split == "heldout":
                    if counts["heldout_balanced"] < quota["heldout_balanced"]:
                        bucket = "heldout_balanced"
                    elif counts["heldout_natural"] < quota["heldout_natural"]:
                        bucket = "heldout_natural"
                if bucket is None:
                    continue
                # K1 frozen stratification: each bucket fills ~50/50 by
                # topology; a family whose (bucket, topology) sub-quota is
                # full is closed for that bucket.
                tkey = (bucket, f["topology"])
                if topo_counts.get(tkey, 0) >= (quota[bucket] + 1) // 2:
                    continue
                natural = bucket == "heldout_natural"
                tgt = None if natural else STRATA[(counts[bucket]) % len(STRATA)]
                fm = FALSE_MECH_CYCLE[di % len(FALSE_MECH_CYCLE)]
                donor = pool_rows[(di + 1) % len(pool_rows)]["record"] if len(pool_rows) > 1 else None
                seq += 1
                eff_seq = args.seq_base + seq if args.natural_topup else seq
                inst, why = build_instance(domain, f, row, tgt, fm, donor, eff_seq)
                if inst is None:
                    att["field-missing" if why.startswith("field-missing") else "no-degradable"] += 1
                    continue
                if tgt is not None and achieved_level(inst) != tgt:
                    att["stratum-miss"] += 1
                    continue
                ok, why = verify(inst, tmpdir)
                if not ok:
                    att["schema" if why.startswith("schema") else "verify-mismatch"] += 1
                    continue
                sub = os.path.join(OUT, domain, bucket)
                os.makedirs(sub, exist_ok=True)
                with open(os.path.join(sub, inst["id"] + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(inst, fh, indent=1, sort_keys=True, default=str)
                counts[bucket] += 1
                topo_counts[(bucket, f["topology"])] = topo_counts.get((bucket, f["topology"]), 0) + 1
                lvl = achieved_level(inst)
                strata_out[str(lvl)] += 1
                meta = json.loads(inst["notes_for_annotators"])
                if meta["false_mech"]:
                    n_false_lb += 1

        manifest["domains"][domain] = {
            "n_families": len(fams),
            "families": {f["family_id"]: {"topology": f["topology"], "cost": f["cost"],
                                          "split": splits[f["family_id"]],
                                          "probes": f["probes"]} for f in fams},
            "counts": counts, "achieved_strata": strata_out,
            "false_load_bearing_instances": n_false_lb,
            "construction_attrition": att,
        }
        print("[%s] %s strata=%s false-lb=%d attrition=%s" %
              (domain, counts, strata_out, n_false_lb, att), file=sys.stderr)

    os.rmdir(tmpdir) if not os.listdir(tmpdir) else None
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print("wrote %s/manifest.json" % OUT, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
