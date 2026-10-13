"""Every v2 result must come from the v2 benchmark.

Four workstreams this session were built and frozen but still pointed at the
legacy data/pilot/instances layout or at nothing at all, and each would have
produced normal-looking output about the WRONG instance set. These are cheap
structural guards against the next one.
"""
from __future__ import annotations

import os
import re

_PILOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SBATCH = os.path.join(_PILOT, "experiments", "slurm", "serve_and_run_guided.sbatch")
_ANALYSIS = os.path.join(_PILOT, "experiments", "analysis")
_V2 = "data/suffice"  # the v2 benchmark directory


def _sbatch():
    return open(_SBATCH, encoding="utf-8").read()


def test_v2_configsets_read_the_v2_benchmark():
    """w2alad / w2apol / w2bind drive the confirmatory results."""
    s = _sbatch()
    for tag in ("w2alad", "w2apol", "w2bind"):
        i = s.index('CONFIGSET" = "%s"' % tag)
        block = s[i:i + 2500]
        assert _V2 in block, "%s does not read data/suffice" % tag


def test_w2b_configset_forces_the_conservative_induction():
    """--induction-method defaults to the published one-shot condition, so
    omitting it in the W2b block would measure the wrong induction."""
    s = _sbatch()
    i = s.index('CONFIGSET" = "w2bind"')
    block = s[i:i + 2000]
    assert "--license-source induced" in block
    assert "--induction-method state_first" in block


def test_vocabulary_panel_can_target_the_v2_split():
    s = _sbatch()
    i = s.index("vocabulary_conditions.py")
    block = s[max(0, i - 900):i + 400]
    assert _V2 in block, \
        "T4's vocabulary panel is pinned to the legacy benchmark"


def test_s1_script_can_target_the_v2_split():
    src = open(os.path.join(_ANALYSIS, "openvocab_endtoend.py"),
               encoding="utf-8").read()
    assert "--v2-split" in src and _V2 in src


def test_w2a_and_w2b_builders_read_the_v2_benchmark():
    """Directly, or transitively through build_w2a_tables, which is the
    single reader of the instance files -- but never the legacy layout."""
    for name in ("build_w2a_tables.py", "build_w2b_table.py"):
        src = open(os.path.join(_ANALYSIS, name), encoding="utf-8").read()
        assert _V2 in src or "build_w2a_tables" in src, name
        assert not re.search(r'"pilot",\s*"instances"', src), \
            "%s also reads the legacy layout" % name


def test_the_w2b_builder_sources_facts_from_the_single_reader():
    import sys
    if _ANALYSIS not in sys.path:
        sys.path.insert(0, _ANALYSIS)
    import build_w2b_table as w2b
    import build_w2a_tables as bw
    assert w2b.bw is bw, "W2b must score against the same instance facts as T1"
    assert bw.BENCH.endswith(_V2)


def test_every_configset_driver_has_a_case_arm():
    """The CONFIGS case statement runs BEFORE the driver blocks and exits on
    an unrecognised value. Adding a driver without its case arm produces a
    job that brings up vLLM, passes the smoke check, and then dies with
    'unknown CONFIGSET' -- which is exactly how the first four W2b jobs
    were lost."""
    s = _sbatch()
    drivers = set(re.findall(r'CONFIGSET"\s*=\s*"(\w+)"', s))
    arms = set(re.findall(r'^\s{2}(\w+)\)\s*$', s, re.M))
    arms |= set(re.findall(r'^\s{2}(\w+)\)', s, re.M))
    missing = sorted(d for d in drivers if d not in arms)
    assert not missing, "driver blocks with no case arm: %s" % missing


def test_driver_configsets_also_skip_the_legacy_loop():
    """Each own-loop configset must break out of the legacy per-domain loop,
    whose NINST line fails under set -e for v2 domains."""
    s = _sbatch()
    m = re.search(r'case "\$CONFIGSET" in ([a-z0-9|]+)\) break', s)
    assert m, "legacy-loop break guard not found"
    guarded = set(m.group(1).split("|"))
    for tag in ("v2dev", "w2alad", "w2apol", "w2bind"):
        assert tag in guarded, "%s does not skip the legacy loop" % tag
