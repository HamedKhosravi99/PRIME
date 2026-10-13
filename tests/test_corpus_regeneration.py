"""SUFFICE regenerates byte for byte from the harvest.

Runs src/suffice/construction/reproduce_corpus.py, which copies the generators into a temporary
directory, regenerates the primary corpus and the one-branch, locator-uncertain,
and text-evidence variants, and compares all 1,898 task files with the published
ones. It runs in a fresh interpreter under two string-hash seeds: an earlier
generator seeded its family enumeration with hash(domain), which Python randomizes
per process, so separate runs enumerated different families.

The released package ships the harvest reduced to the fields the generators read
(seconds per run). The research tree keeps the full harvest (2.2 GB), on which a run
takes about two minutes; there the test is skipped and the script is run directly.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
HARVEST = os.path.join(_P, "data", "harvest")
FINANCIAL = os.path.join(HARVEST, "financial", "records.jsonl")


@pytest.mark.skipif(not os.path.exists(FINANCIAL), reason="no harvest in data/harvest")
@pytest.mark.skipif(os.path.exists(FINANCIAL) and os.path.getsize(FINANCIAL) > 100_000_000,
                    reason="full harvest: run python3 src/suffice/construction/reproduce_corpus.py directly (about two minutes)")
@pytest.mark.parametrize("hash_seed", ["1", "2"])
def test_corpus_regenerates_byte_for_byte(hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    p = subprocess.run([sys.executable, os.path.join(_P, "src", "suffice", "construction", "reproduce_corpus.py")],
                       env=env, capture_output=True, text=True, cwd=_P)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "all published tasks regenerated identically" in p.stdout
