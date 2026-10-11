#!/usr/bin/env python3
"""Scaffold a real-repo artifact and verify atom gold-values by construction.

Two responsibilities, both about INTEGRITY (AUTHORING.md rule: never assert a
fact about the artifact you did not observe):

  scaffold  <url> <name> [--branch tag]
      clone --depth 1, pin HEAD, strip .git, measure size, read the LICENSE
      first line, run `python3 -m pytest -q` capturing the TRUE exit code
      (not clobbered by a pipe), and write artifacts/<name>@<short>/PROVENANCE.

  verify    <instance.json>
      For every atom, execute its resolver tool through the SAME ToolExecutor
      the runner uses, apply the instance's attestation rule, and print the
      OBSERVED value next to the atom's declared ``gold_value``. Any mismatch
      is a hard error -- this is what makes authored gold-values trustworthy.

Usage:
    python3 bench/scaffold_repo.py scaffold https://github.com/x/y y --branch v1
    python3 bench/scaffold_repo.py verify bench/instances/swe/swe-0007.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
if PILOT not in sys.path:
    sys.path.insert(0, PILOT)


def _run(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def scaffold(url, name, branch=None):
    tmp = os.path.join(HERE, "_tmp_clone")
    subprocess.run(["rm", "-rf", tmp], check=True)
    clone = ["git", "clone", "--depth", "1"]
    if branch:
        clone += ["--branch", branch]
    clone += [url, tmp]
    r = _run(clone)
    if not os.path.isdir(tmp):
        print("clone failed:\n%s" % r.stderr[-500:]); return 1
    sha = _run(["git", "-C", tmp, "rev-parse", "HEAD"]).stdout.strip()
    short = _run(["git", "-C", tmp, "rev-parse", "--short=8", "HEAD"]).stdout.strip()
    subprocess.run(["rm", "-rf", os.path.join(tmp, ".git")], check=True)
    dest = os.path.join(HERE, "artifacts", "%s@%s" % (name, short))
    subprocess.run(["rm", "-rf", dest], check=True)
    os.rename(tmp, dest)

    size = _run(["du", "-sh", dest]).stdout.split()[0]
    lic_line, lic_file = "", ""
    for cand in ("LICENSE", "LICENSE.txt", "LICENSE-MIT", "LICENSE.md", "COPYING"):
        p = os.path.join(dest, cand)
        if os.path.isfile(p):
            lic_file = cand
            lic_line = open(p, errors="replace").readline().strip()
            break
    # true pytest exit code (subshell, not piped)
    proc = _run([sys.executable, "-m", "pytest", "-q"], cwd=dest, timeout=300)
    last = (proc.stdout.strip().splitlines() or ["<no output>"])[-1]

    prov = os.path.join(dest, "PROVENANCE")
    with open(prov, "w") as f:
        f.write("name: %s\nsource_url: %s\ncommit: %s\nref_cloned: %s\n"
                % (name, url, sha, branch or "default HEAD"))
        f.write("clone_date: 2026-07-28\nlicense_file: %s\nlicense_first_line: %s\n"
                % (lic_file, lic_line))
        f.write("size: %s\ntest_command: python3 -m pytest -q (cwd=repo root)\n" % size)
        f.write("environment: %s, pytest\n" % sys.version.split()[0])
        f.write("test_exit_code: %d\ntest_result_line: %s\n" % (proc.returncode, last))
    print("scaffolded %s@%s  size=%s  license=%s(%s)  pytest_exit=%d  -> %s"
          % (name, short, size, lic_file, lic_line[:30], proc.returncode, last))
    print("  dir: %s" % dest)
    return 0


def verify(instance_path):
    from prime_runtime.schema import load_instance
    from prime_runtime import goldledger
    from prime_runtime.ledger import Ledger
    from runner.arms import ToolExecutor

    inst = load_instance(instance_path)
    raw_gold = {a["id"]: a.get("gold_value") for a in json.load(open(instance_path)).get("atoms", [])}
    ex = ToolExecutor(inst)
    ledger = Ledger(inst.atom_ids, inst.phi_patterns)
    ok = True
    print("verifying %s (artifact: %s)" % (inst.id, inst.artifact.get("path")))
    for atom in inst.atoms:
        declared = raw_gold.get(atom.id)
        if atom.resolver is None:
            print("  %-4s resolver=null (terminal-unavailable) declared=%s" % (atom.id, declared))
            continue
        params = dict(atom.resolver.params)
        obs = ex.execute(atom.resolver.tool, params)
        applied = goldledger.delta(inst, ledger, atom.resolver.tool, params, obs, 1)
        got = next((a["value"] for a in applied if a["atom"] == atom.id), None)
        mark = "ok " if (declared is None or got == declared) else "MISMATCH"
        if declared is not None and got != declared:
            ok = False
        print("  %-4s tool=%-16s attest=%-14s declared=%s observed=%s  %s"
              % (atom.id, atom.resolver.tool, atom.resolver.attest, declared, got, mark))
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "scaffold":
        br = None
        if "--branch" in sys.argv:
            br = sys.argv[sys.argv.index("--branch") + 1]
        args = [a for a in sys.argv[2:] if a != "--branch" and a != br]
        sys.exit(scaffold(args[0], args[1], br))
    elif sys.argv[1] == "verify":
        sys.exit(verify(sys.argv[2]))
    else:
        print(__doc__); sys.exit(1)
