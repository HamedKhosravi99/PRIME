#!/usr/bin/env python3
"""Regenerate the paper's tables from the episode logs and compare with the shipped JSON.

Runs every script of the README's map that writes a JSON file under results/, each
with the arguments the paper used, and checks that the file it writes is identical,
byte for byte, to the one shipped in the logs archive (which is restored if not).
Then checks the two items whose script output is not a stored file: the other-model
rows of Table 8 (build_w2a_tables.py --models gptoss-20b mistral-small-24b, which
rewrites tableT1/T2.json for those models; the shipped files are put back
afterwards) and the sampled-decoding sentence of Appendix E.1. No model is called.
Takes about five minutes.

Usage: python3 experiments/analysis/regenerate_tables.py            # check
       python3 experiments/analysis/regenerate_tables.py --update   # keep the regenerated files
       python3 experiments/analysis/regenerate_tables.py --update --changes new_cells_changes.json

--update is for after the logs change (for example when rerun episodes replace
logged ones): it keeps every regenerated file and writes results/rerun_changes.json,
which lists every number that changed, file by file, so the paper can follow
(--changes names another file under results/, so that a later update does not
overwrite the record of an earlier one).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")

# (script, arguments, files it writes under results/)
RUNS = [
    ("experiments/analysis/build_w2a_tables.py", [], ["tableT1.json", "tableT2.json"]),
    ("experiments/analysis/build_complete_table.py", [], ["tableCOMPLETE.json", "discharge_replay.json"]),
    ("experiments/analysis/discharge_replay.py", ["--split", "locator"], ["discharge_replay_locator.json"]),
    ("experiments/analysis/locator_first_reads.py", [], ["locator_first_reads.json"]),
    ("experiments/analysis/build_text_table.py", [], ["tableTEXT.json"]),
    ("experiments/analysis/text_attestation_counts.py", [], ["text_attestation_counts.json"]),
    ("experiments/analysis/build_extsys_table.py", [], ["tableEXTSYS.json"]),
    ("experiments/analysis/build_flat_table.py", [], ["tableFLAT.json"]),
    ("experiments/analysis/build_multimodel_table.py", [], ["tableMULTI.json"]),
    ("experiments/analysis/build_locator_table.py", [], ["tableLOC.json"]),
    ("experiments/analysis/build_locator_extra.py", [], ["tableLOC_extra.json"]),
    ("experiments/analysis/build_controller_table.py", [], ["tableCONTROLLER.json"]),
    ("experiments/analysis/build_locator_hesp_table.py", [], ["tableLOC_HESP.json"]),
    ("experiments/analysis/established_utility.py", [], ["established_utility_hesp.json"]),
    ("experiments/analysis/hesp_prior_sensitivity.py", [], ["hesp_prior_sensitivity.json"]),
    ("experiments/analysis/stall_cap_replay.py", [], ["stall_cap_replay.json"]),
    ("experiments/analysis/build_fdaaa_table.py", [], ["tableFDAAA.json", "fdaaa_confusion.json"]),
    ("experiments/analysis/rejection_bound.py", [], ["rejection_bound.json"]),
    ("experiments/analysis/episode_accounting.py", [], ["excluded_episodes.json"]),
    ("experiments/analysis/budget_accounting_audit.py", [], ["budget_accounting_audit.json"]),
]

# Table 8, other models: (key in tableT1.json, false authorization %, utility, emission %)
TABLE8_T1 = {
    "gptoss-20b|react_program_attested|balanced": (17.7, 30.5, 100.0),
    "gptoss-20b|masked_reactive_v|balanced": (0.0, 26.9, 98.0),
    "gptoss-20b|prime_guided|balanced": (0.0, 58.5, 100.0),   # 58.7 before the 2026-10-12 rerun
    "mistral-small-24b|react_program_attested|balanced": (19.4, 35.2, 96.7),
    "mistral-small-24b|masked_reactive_v|balanced": (0.0, 29.3, 100.0),
    "mistral-small-24b|prime_guided|balanced": (0.0, 36.6, 100.0),
}
# Table 8, last column: (contrast in tableT2.json's ladder, reduction, 95% CI)
TABLE8_T2 = {
    "gptoss-20b|react_program_attested->masked_reactive_v": (18.0, [13.5, 22.7]),
    "mistral-small-24b|react_program_attested->masked_reactive_v": (19.5, [14.9, 24.2]),
}


def changed_numbers(old, new, path="", out=None):
    """Every leaf that differs between two JSON values, as (path, old, new)."""
    if out is None:
        out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new), key=str):
            changed_numbers(old.get(k), new.get(k), "%s/%s" % (path, k), out)
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (x, y) in enumerate(zip(old, new)):
            changed_numbers(x, y, "%s/%d" % (path, i), out)
    elif old != new:
        out.append((path, old, new))
    return out


def same_within_float(old, new, rel=1e-9) -> bool:
    """Equal up to the last digits of floats: Python 3.12 sums floats with
    compensated summation, so an unrounded mean can differ in its last digit
    from the one Python 3.9 wrote; every rounded value is unaffected."""
    for _p, a, b in changed_numbers(old, new):
        if not (isinstance(a, float) and isinstance(b, (int, float))
                and abs(a - b) <= rel * max(1.0, abs(a))):
            return False
    return True


def run(script, args):
    t = time.time()
    p = subprocess.run([sys.executable, script] + args, cwd=ROOT, capture_output=True, text=True)
    return p, time.time() - t


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    update = "--update" in args
    changes_name = args[args.index("--changes") + 1] if "--changes" in args else "rerun_changes.json"
    changes = {}
    if not os.path.isdir(os.path.join(RESULTS, "w2a")):
        print("results/ holds no episode logs; extract data/episode_logs/prime-episode-logs.tar.gz first (make logs)")
        return 2
    bad = []
    for script, args, outs in RUNS:
        shipped = {}
        for o in outs:
            with open(os.path.join(RESULTS, o), "rb") as fh:
                shipped[o] = fh.read()
        p, dt = run(script, args)
        notes = []
        if p.returncode != 0:
            bad.append("%s failed: %s" % (script, (p.stderr or p.stdout)[-500:]))
            notes.append("FAILED")
        for o, before in shipped.items():
            path = os.path.join(RESULTS, o)
            with open(path, "rb") as fh:
                after = fh.read()
            if after == before:
                notes.append("%s identical" % o)
            elif not update and same_within_float(json.loads(before), json.loads(after)):
                notes.append("%s identical up to float last digits" % o)
                with open(path, "wb") as fh:
                    fh.write(before)
            elif update:
                notes.append("%s UPDATED" % o)
                diffs = changed_numbers(json.loads(before), json.loads(after))
                changes[o] = [{"path": p_, "old": a_, "new": b_} for p_, a_, b_ in diffs]
            else:
                notes.append("%s DIFFERS" % o)
                bad.append("%s: %s differs from the shipped file (restored)" % (script, o))
                with open(path, "wb") as fh:
                    fh.write(before)
        print("%-40s %4.0fs  %s" % (script, dt, "; ".join(notes)))

    # Table 8, other models
    keep = {}
    for o in ("tableT1.json", "tableT2.json"):
        with open(os.path.join(RESULTS, o), "rb") as fh:
            keep[o] = fh.read()
    try:
        p, dt = run("experiments/analysis/build_w2a_tables.py", ["--models", "gptoss-20b", "mistral-small-24b"])
        if p.returncode != 0:
            bad.append("Table 8 run failed: %s" % (p.stderr or p.stdout)[-500:])
        else:
            with open(os.path.join(RESULTS, "tableT1.json"), encoding="utf-8") as fh:
                t1 = json.load(fh)
            with open(os.path.join(RESULTS, "tableT2.json"), encoding="utf-8") as fh:
                ladder = json.load(fh)["ladder"]
            for key, exp in TABLE8_T1.items():
                got = tuple(t1[key][k] for k in ("ofar_a", "utility", "emit_rate_avail"))
                if any(abs(g - x) > 0.05 for g, x in zip(got, exp)):
                    if update:
                        changes.setdefault("Table 8 (values the paper prints)", []).append(
                            {"path": key + " (FA, utility, emission)", "old": list(exp), "new": list(got)})
                    else:
                        bad.append("Table 8 %s: got %s, expected %s" % (key, got, exp))
            for key, (red, ci) in TABLE8_T2.items():
                c = ladder[key]
                if abs(c["ofar_reduction_pp"] - red) > 0.05 or any(abs(a - b) > 0.05 for a, b in zip(c["ci"], ci)):
                    if update:
                        changes.setdefault("Table 8 (values the paper prints)", []).append(
                            {"path": key + " (FA removed, CI)", "old": [red, ci], "new": [c["ofar_reduction_pp"], c["ci"]]})
                    else:
                        bad.append("Table 8 %s: got %s %s, expected %s %s" % (key, c["ofar_reduction_pp"], c["ci"], red, ci))
        print("%-40s %4.0fs  Table 8 other-model rows checked" % ("experiments/analysis/build_w2a_tables.py --models ...", dt))
    finally:
        for o, b in keep.items():
            with open(os.path.join(RESULTS, o), "wb") as fh:
                fh.write(b)

    # Appendix E.1, sampled decoding
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "prime_agents")]
    import fill_v2_tables as fv
    note = fv.fill_SeedNote(None)
    if "25.4--26.9" not in note or "0.0" not in note:
        if update:
            changes["sampled decoding (Appendix E.1 sentence)"] = [{"path": "fill_SeedNote", "old": "25.4--26.9, 0.0", "new": note}]
        else:
            bad.append("sampled decoding: %s" % note)
    print("%-40s        sampled decoding: 25.4 to 26.9%% unenforced, 0.0%% under the mask" % "fill_SeedNote (experiments/analysis/fill_v2_tables.py)")

    if update:
        with open(os.path.join(RESULTS, changes_name), "w", encoding="utf-8") as fh:
            json.dump(changes, fh, indent=1, sort_keys=True, default=str)
        print("\nupdated %d files, %d numbers changed; see results/%s"
              % (len(changes), sum(len(v) for v in changes.values()), changes_name))
    if bad:
        print("\nDIFFERENCES:\n  " + "\n  ".join(bad))
        return 1
    if update:
        print("\nregenerated tables written; differences from the paper's Table 8 and sampled-decoding values, if any, "
              "are in results/%s" % changes_name)
    else:
        print("\nevery regenerated table is identical to the shipped one; Table 8 and the sampled-decoding values match the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
