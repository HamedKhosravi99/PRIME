#!/usr/bin/env python3
"""Regenerate ObligationBench from the harvest and compare with the published tasks.

    python3 src/obligationbench/construction/reproduce_corpus.py [--harvest data/harvest] [--keep]

Copies the two generators into a temporary copy of the repository layout (so the
published files in data/obligationbench/ are never touched), runs them in the order the corpus was made
(the primary corpus; the one-branch, locator-uncertain, and text-evidence variants),
and compares every regenerated task file with the published one byte for byte. Exits
nonzero unless all 1,898 tasks are identical. The released package ships the harvest
of 2026-08-01 reduced to the fields the generators read (src/obligationbench/construction/reduce_harvest.py), on
which this takes a few seconds; on the full harvest it takes about two minutes.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DOMAINS = ("security", "clinical", "drugsafety", "financial")
SPLITS = ("dev", "val", "heldout_balanced", "heldout_natural",
          "heldout_onebranch", "heldout_locator", "heldout_text")
RUNS = [
    ("generate_record_instances.py", []),
    ("generate_record_instances.py", ["--one-branch-per-domain", "60", "--seq-base", "9000"]),
    ("generate_record_instances.py", ["--locator-per-domain", "120"]),
    ("generate_text_instances.py", []),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harvest", default=os.path.join(ROOT, "data", "harvest"))
    ap.add_argument("--keep", action="store_true", help="keep the regenerated files")
    a = ap.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="obligationbench-regen-")
    try:
        src = os.path.join(tmp, "src")
        gen = os.path.join(src, "obligationbench", "construction")
        os.makedirs(gen)
        for d in ("prime_runtime", "prime_agents"):
            os.symlink(os.path.join(ROOT, "src", d), os.path.join(src, d))
        for f in ("generate_record_instances.py", "generate_text_instances.py"):
            shutil.copy2(os.path.join(HERE, f), os.path.join(gen, f))
        os.makedirs(os.path.join(tmp, "data"))
        os.symlink(os.path.abspath(a.harvest), os.path.join(tmp, "data", "harvest"))
        out = os.path.join(tmp, "data", "obligationbench")
        for script, args in RUNS:
            p = subprocess.run([sys.executable, script] + args, cwd=gen, capture_output=True, text=True)
            if p.returncode != 0:
                print(p.stdout[-2000:] + p.stderr[-3000:])
                return 1
        ok = True
        print("| Split | Published | Regenerated identical | Missing | Extra |\n|---|---|---|---|---|")
        for split in SPLITS:
            pub = same = missing = extra = 0
            for dom in DOMAINS:
                ref = os.path.join(ROOT, "data", "obligationbench", dom, split)
                new = os.path.join(out, dom, split)
                fr = {f for f in os.listdir(ref) if f.endswith(".json")} if os.path.isdir(ref) else set()
                fn = {f for f in os.listdir(new) if f.endswith(".json")} if os.path.isdir(new) else set()
                pub += len(fr)
                missing += len(fr - fn)
                extra += len(fn - fr)
                for f in fr & fn:
                    with open(os.path.join(ref, f), "rb") as x, open(os.path.join(new, f), "rb") as y:
                        same += x.read() == y.read()
            print("| %s | %d | %d | %d | %d |" % (split, pub, same, missing, extra))
            ok = ok and same == pub and not extra
        print("\nall published tasks regenerated identically" if ok else "\nDIFFERENCES")
        if a.keep:
            print("regenerated files are in", out)
        return 0 if ok else 1
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
