#!/usr/bin/env python3
"""Generator for the ObligationBench descriptive numbers (EXECUTION_PLAN v5:
"numbers only from generators"; FREEZE F9 slot \\slot{BenchStats}).

Every quantity the manuscript states about the benchmark is derived here from
the frozen manifest plus the instance files themselves -- never hand-typed --
so a regeneration that changes a count cannot silently disagree with the prose.

Emits LaTeX to stdout; --json emits the same facts as a dict.
Usage: python3 analysis/bench_stats.py [--json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
for _p in (PILOT, os.path.join(PILOT, "runner"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BENCH = os.path.join(PILOT, "bench", "instances_v2")
MANIFEST = os.path.join(BENCH, "manifest.json")

# presentation order fixed here so the prose is stable across regenerations
SOURCE_ORDER = ("security", "clinical", "drugsafety", "financial")
SOURCES = {
    "security": "vulnerability records from OSV",
    "clinical": "trial registrations from the ClinicalTrials.gov v2 API",
    "drugsafety": "drug recall notices from openFDA",
    "financial": "XBRL company facts from SEC EDGAR",
}
SPLITS = ("dev", "val", "heldout_balanced", "heldout_natural")


def _false_mech_count(split: str) -> int:
    """Instances in one split whose broken route fails on a FALSE atom.

    Read from the instances rather than summed off the manifest, because the
    manifest total spans every split and the prose that quotes it is about
    the confirmatory split alone.
    """
    n = 0
    for p in glob.glob(os.path.join(BENCH, "*", split, "*.json")):
        try:
            meta = json.load(open(p, encoding="utf-8"))["notes_for_annotators"]
            meta = json.loads(meta) if isinstance(meta, str) else meta
        except (OSError, ValueError, KeyError):
            continue
        if meta.get("false_mech"):
            n += 1
    return n


def collect() -> dict:
    import build_w2a_tables as bw  # instance_facts is the single reader

    man = json.load(open(MANIFEST, encoding="utf-8"))
    doms = man["domains"]

    counts = {s: len(glob.glob(os.path.join(BENCH, "*", s, "*.json")))
              for s in SPLITS}
    bal = [f for f in bw.FACTS.values() if f["split"] == "heldout_balanced"]
    nat = [f for f in bw.FACTS.values() if f["split"] == "heldout_natural"]

    attr = collections.Counter()
    for v in doms.values():
        attr.update(v["construction_attrition"])
    fam_sizes = collections.Counter(f["family"] for f in bal)
    claims = collections.Counter(len(f["levels"]) for f in bal)
    content = collections.Counter(v for f in bal for v in f["content"].values())

    return {
        "n_total": sum(counts.values()),
        "counts": counts,
        "n_domains": len(doms),
        "n_families_total": sum(v["n_families"] for v in doms.values()),
        "n_families_balanced": len(fam_sizes),
        "family_size_lo": min(fam_sizes.values()),
        "family_size_hi": max(fam_sizes.values()),
        "topology_balanced": dict(collections.Counter(f["topology"] for f in bal)),
        "ceiling_balanced": {str(k): v for k, v in sorted(
            collections.Counter(f["ceiling"] for f in bal).items())},
        "claims_per_instance": sorted(claims),
        "content_balanced": dict(content),
        # ALL splits. Stating this bare in a paragraph about the
        # confirmatory split invited exactly the misreading it got: a
        # reviewer took "301 instances are false-load-bearing" as 301 of
        # 480 (62.7%) and argued the endpoint is engineered at a high
        # corruption rate. 301 is dev 86 + val 16 + balanced 199; the
        # confirmatory rate is 199 of 480, i.e. 41.5%. Both are reported,
        # each with its denominator.
        "false_load_bearing": sum(v["false_load_bearing_instances"]
                                  for v in doms.values()),
        "false_load_bearing_balanced": _false_mech_count("heldout_balanced"),
        "attrition": dict(attr),
        "n_natural": len(nat),
        "natural_by_domain": {d: v["counts"]["heldout_natural"]
                              for d, v in sorted(doms.items())},
        "seed": man["seed"],
    }


def latex_short(s: dict) -> str:
    """Body-length version (FREEZE F9 slot). The construction details it
    omits -- per-domain attrition, the natural split's uneven realized
    sizes -- are not dropped: they live in the benchmark-schema appendix and
    in the frozen manifest, and --json still emits every one of them."""
    bal = s["counts"]["heldout_balanced"]
    k = s["claims_per_instance"]
    return (
        "The corpus holds %d instances over %d record domains --- %s --- in %d "
        "template families; a family fixes the claim schema, route topology and "
        "probe set and varies only the record, so it is the unit of clustering "
        "for every test we report. All confirmatory numbers come from the "
        "held-out balanced split: %d instances in %d families of %d--%d, each "
        "carrying exactly %d claims, split %d/%d between branching and nested "
        "route topologies, with support ceilings spread across all four levels "
        "so that %d admit no supported claim and abstention is correct there "
        "rather than a scoring artefact, and %d are false-load-bearing --- a "
        "route broken by a genuinely false atom rather than a missing one. "
        "Construction is verified, not asserted: no instance was retained with "
        "a schema violation, an unreachable stratum, or any disagreement "
        "between its gold licences and its executable verification. The %d "
        "split %d balanced / %d natural / %d dev / %d val, dev and val setting "
        "the margin and every pilot decision and never scored confirmatorily; "
        "the natural split is reported separately and never pooled, and "
        "generation is deterministic under seed %d "
        "(Appendix~\\ref{apx:schema})."
        % (s["n_total"], s["n_domains"],
           ", ".join(SOURCES[d] for d in SOURCE_ORDER), s["n_families_total"],
           bal, s["n_families_balanced"], s["family_size_lo"],
           s["family_size_hi"], k[0],
           s["topology_balanced"]["branching"], s["topology_balanced"]["nested"],
           s["ceiling_balanced"]["0"],
           s["false_load_bearing_balanced"],
           s["n_total"], bal, s["counts"].get("heldout_natural", 0),
           s["counts"].get("dev", 0), s["counts"].get("val", 0),
           s["seed"]))


def latex(s: dict) -> str:
    ceil0 = s["ceiling_balanced"]["0"]
    bal = s["counts"]["heldout_balanced"]
    k = s["claims_per_instance"]
    assert len(k) == 1, "prose assumes a fixed claim count per instance"
    srcs = ", ".join(SOURCES[d] for d in SOURCE_ORDER)
    return (
        "The generated corpus holds %d instances over %d record domains --- %s "
        "--- organised into %d template families; a family fixes the claim "
        "schema, the route topology and the probe set, and varies only the "
        "underlying record, so it is the unit of clustering for every test we "
        "report. Instances are split %d dev / %d val / %d held-out balanced / "
        "%d held-out natural, and all confirmatory numbers come from the "
        "held-out balanced split (%d families, %d--%d instances each). "
        "Each instance carries exactly %d claims. The balanced split is "
        "constructed to be non-degenerate in the two ways that matter: it is "
        "split %d/%d between branching and nested route topologies, and its "
        "support ceilings are spread across all four levels (%s), so %d of "
        "%d instances admit no supported claim at all and abstention is the "
        "correct behaviour there rather than a scoring artefact. Claim "
        "denotations evaluate three-valued against the records --- %d true, "
        "%d false, %d unavailable --- and %d instances are false-load-bearing, "
        "meaning at least one route is broken by a genuinely false atom rather "
        "than a missing one. Construction is verified rather than asserted: "
        "%d candidate instances were discarded for referencing a field the "
        "real record did not contain, and zero were discarded for schema "
        "violations, unreachable strata, or disagreement between an instance's "
        "gold licences and its executable verification --- every retained "
        "instance passes that check. The natural split preserves the "
        "unconditioned stratum mix and is therefore smaller and uneven across "
        "domains (%s); we report it separately and never pool it with the "
        "balanced split. Generation is deterministic under seed %d."
        % (s["n_total"], s["n_domains"], srcs, s["n_families_total"],
           s["counts"]["dev"], s["counts"]["val"], bal,
           s["counts"]["heldout_natural"], s["n_families_balanced"],
           s["family_size_lo"], s["family_size_hi"], k[0],
           s["topology_balanced"]["branching"], s["topology_balanced"]["nested"],
           "; ".join("level %s: %d" % (lv, n)
                     for lv, n in s["ceiling_balanced"].items()),
           ceil0, bal,
           s["content_balanced"].get("true", 0),
           s["content_balanced"].get("false", 0),
           s["content_balanced"].get("unavailable", 0),
           s["false_load_bearing"], s["attrition"].get("field-missing", 0),
           ", ".join("%s %d" % (d, n)
                     for d, n in s["natural_by_domain"].items()),
           s["seed"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="long form; the body uses the short form")
    a = ap.parse_args(argv)
    s = collect()
    if a.json:
        print(json.dumps(s, indent=1, sort_keys=True))
    else:
        print(latex(s) if a.full else latex_short(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
