#!/usr/bin/env python3
"""Where each model's reads went on the locator-uncertain split.

For every model of Experiment C, counts PRIME's `inspect_record` reads that hit
an atom's true field, a listed decoy, or no candidate, and how often the first
read made for an atom hit its true field (a first try in listed order hits it
for one atom in three). Reads the logged episodes only; no model is called.
Reuses `build_locator_extra.reads_at`, which produced the 71.2% for Qwen.

Writes: results/locator_first_reads.json
Usage:  python3 experiments/analysis/locator_first_reads.py
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_locator_extra as bl  # noqa: E402

MODELS = ("qwen2.5-32b", "gptoss-20b", "mistral-small-24b")
ANCHORS = ("bmin", "2bmin")


def main() -> int:
    out = {}
    for model in MODELS:
        bl.MODEL = model
        for anchor in ANCHORS:
            r = bl.reads_at(anchor)
            reads = r["true"] + r["decoy"] + r["other"]
            r["reads"] = reads
            r["wasted_pct"] = round(100.0 * (r["decoy"] + r["other"]) / reads, 1) if reads else None
            out["%s|%s" % (model, anchor)] = r
            print("%-18s %-6s first read true %5s%% of %4d atoms; %4.1f%% of %d reads wasted"
                  % (model, anchor, r["first_read_true_pct"], r["first_reads"], r["wasted_pct"], reads))
    path = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "locator_first_reads.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
