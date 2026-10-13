#!/usr/bin/env python3
"""Rerun the episodes of results/rerun_manifest.csv with this (corrected) runner.

Every listed episode is run again exactly as its cell was run (same arm, split,
budget anchor, decoding, and model), each into its own directory
<out>/<folder>/<modeltag>-<domain>/<run>/<instance>/, so that the merge step
(experiments/analysis/merge_reruns.py) can replace exactly these episodes in the logs. A
finished episode is skipped on resubmission, so the job can be requeued.

GLEAN's rows run last: GLEAN verifies the fully informed agent's trajectory, so
its verdict is recomputed (src/prime_agents/glean_one.py) on the rerun trajectory of the
same episode.

Usage (on the cluster, with the model served at PRIME_LLM_BASE):
  python3 src/prime_agents/rerun_episodes.py --modeltag qwen2.5-32b --tiers 1 2 3 --out rerun
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def done(ep_dir: str, iid: str) -> bool:
    side = os.path.join(ep_dir, "_episodes", iid + ".json")
    if not os.path.exists(side):
        return False
    try:
        with open(side, encoding="utf-8") as fh:
            return not json.load(fh).get("error")
    except (OSError, ValueError):
        return False


DRY_RUN = False


def task_path(p: str) -> str:
    """A manifest written before the repository was reorganized names bench/instances_v2/."""
    old = "bench/instances_v2/"
    return "data/suffice/" + p[len(old):] if p.startswith(old) else p


def command(r, ep_dir: str):
    py = sys.executable
    if r["entry"] == "run_batch":
        cmd = [py, "src/prime_agents/run_batch.py", "--instances", task_path(r["instance_path"]), "--arm", r["arm"],
               "--license-source", "gold", "--model", r["model"], "--parallel", "1",
               "--max-tool-calls", "20", "--resume", "--out", ep_dir]
        if r["anchor"] != "B20":
            cmd += ["--budget-anchor", r["anchor"]]
        if r["temperature"] not in ("", "0"):
            cmd += ["--temperature", r["temperature"], "--seed", r["seed"]]
        if DRY_RUN:
            cmd += ["--dry-run"]
        return cmd
    if r["entry"] == "blind_verification":
        return [py, "experiments/blind_verification/run_live.py", "--domain", r["domain"],
                "--instances", r["instance"], "--parallel", "1", "--out", ep_dir]
    raise ValueError(r["entry"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=os.path.join(ROOT, "results", "rerun_manifest.csv"))
    ap.add_argument("--modeltag", required=True)
    ap.add_argument("--tiers", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--out", default=os.path.join(ROOT, "rerun"))
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="scripted model, no API calls (tests the driver)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    global DRY_RUN
    DRY_RUN = a.dry_run
    with open(a.manifest, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["modeltag"] == a.modeltag and r["tier"] in a.tiers]
    os.environ.setdefault("PRIME_LLM_MODEL", rows[0]["model"] if rows else "")
    work = [r for r in rows if r["entry"] != "glean" and not (a.dry_run and r["entry"] != "run_batch")]
    if a.limit:
        work = work[: a.limit]
    glean_rows = [] if a.dry_run else [r for r in rows if r["entry"] == "glean"]

    def ep_dir(r, run=None):
        return os.path.join(a.out, r["folder"], "%s-%s" % (r["modeltag"], r["domain"]), run or r["run"], r["instance"])

    def one(r):
        d = ep_dir(r)
        if done(d, r["instance"]):
            return r, "skipped (done)"
        os.makedirs(d, exist_ok=True)
        t = time.time()
        p = subprocess.run(command(r, d), cwd=ROOT, capture_output=True, text=True)
        with open(os.path.join(d, "driver.log"), "w", encoding="utf-8") as fh:
            fh.write(" ".join(command(r, d)) + "\n" + p.stdout[-20000:] + "\n" + p.stderr[-20000:])
        ok = p.returncode == 0 and done(d, r["instance"])
        return r, "%s in %.0fs" % ("ok" if ok else "FAILED rc=%d" % p.returncode, time.time() - t)

    print("[rerun] %d episodes for %s, tiers %s" % (len(work), a.modeltag, " ".join(a.tiers)), flush=True)
    failed = 0
    for tier in sorted({r["tier"] for r in work}):
        batch = [r for r in work if r["tier"] == tier]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as pool:
            for r, status in pool.map(one, batch):
                failed += status.startswith("FAILED")
                print("[rerun] tier %s %s %s %s: %s" % (tier, r["folder"], r["run"], r["instance"], status), flush=True)
    for r in glean_rows:
        base = ep_dir(r, run="react_program_attested__%s" % r["anchor"])
        if not done(base, r["instance"]):
            print("[rerun] GLEAN %s: the agent's rerun trajectory is missing" % r["instance"], flush=True)
            failed += 1
            continue
        d = ep_dir(r)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, "glean_verdict.json")
        if os.path.exists(out):
            print("[rerun] GLEAN %s: skipped (done)" % r["instance"], flush=True)
            continue
        p = subprocess.run([sys.executable, "src/prime_agents/glean_one.py", "--model", r["model"], "--modeltag", r["modeltag"],
                            "--anchor", r["anchor"], "--domain", r["domain"], "--episode-dir", base, "--out", out],
                           cwd=ROOT, capture_output=True, text=True)
        with open(os.path.join(d, "driver.log"), "w", encoding="utf-8") as fh:
            fh.write(p.stdout[-20000:] + "\n" + p.stderr[-20000:])
        ok = p.returncode == 0 and os.path.exists(out)
        failed += not ok
        print("[rerun] GLEAN %s %s: %s" % (r["anchor"], r["instance"], "ok" if ok else "FAILED rc=%d" % p.returncode), flush=True)
    print("RERUN_DONE modeltag=%s failed=%d" % (a.modeltag, failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
