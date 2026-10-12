"""src/obligationbench/__init__.py, the standalone scorer, agrees with the runtime.

The scorer re-implements the runtime's handling of record reads and its
authorization check, so that scoring needs nothing but the standard library.
src/obligationbench/construction/conformance.py computes the runtime's outcomes; this test recomputes each
with src/obligationbench/__init__.py and compares:

* every resolver field, locator candidate, a missing field, and an unknown record
  of the 1,658 record tasks: the observation and the checks the read settles;
* the reference state and 12 seeded check-value states of each task: the
  conclusions established, weaker ones included;
* when the episode logs are present, every completed episode of the 15 reference
  agents: the checks its reads settle, the conclusions they establish, and that
  the runtime's replay matches the logged final ledger.

Text attestations, which src/obligationbench/__init__.py does not handle, are compared by the
standalone benchmark's tests/test_conformance.py against a fixture written by
``python3 src/obligationbench/construction/conformance.py``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "src", "prime_agents"), os.path.join(_P, "src", "obligationbench", "construction"),
           os.path.join(_P, "experiments")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import conformance as cf  # noqa: E402

_SCORER = os.path.join(_P, "src", "obligationbench", "__init__.py")
_spec = importlib.util.spec_from_file_location("obligationbench_single", _SCORER)
ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ob)

RESULTS = os.path.join(_P, "results")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("split", cf.RECORD_SPLITS)
def test_reads_agree(split):
    paths = cf.task_paths((split,))
    assert paths
    for path in paths:
        task = _load(path)
        for rid, fp, dig, settled in cf.read_outcomes(path):
            env = ob.RecordEnv(task)
            assert cf.digest(env.inspect_record(rid, fp)) == dig, (task["id"], rid, fp)
            assert env.ledger == settled, (task["id"], rid, fp, env.ledger, settled)


@pytest.mark.parametrize("split", cf.RECORD_SPLITS + (cf.TEXT_SPLIT,))
def test_authorization_agrees(split):
    for path in cf.task_paths((split,)):
        task = _load(path)
        for state, auth in cf.auth_outcomes(path):
            assert sorted(ob.established(task, state)) == auth, (task["id"], state)


@pytest.mark.skipif(not os.path.isdir(os.path.join(RESULTS, "w2a")),
                    reason="episode logs not extracted (prime-episode-logs.tar.gz)")
def test_logged_episodes_agree():
    import reproduce_tables as rt
    tasks = rt.heldout_tasks()
    prime = cf.episode_outcomes(RESULTS)
    n = 0
    for agent, (_name, folder, _exp) in rt.AGENTS.items():
        preds, _logged, _failed = rt.predictions(agent, folder, tasks)
        assert {p["task_id"] for p in preds} == set(prime[agent]), agent
        for p in preds:
            settled, auth, _charged, logged = prime[agent][p["task_id"]]
            assert settled == logged, (agent, p["task_id"])
            env = ob.RecordEnv(tasks[p["task_id"]])
            for rid, fp in p["reads"]:
                env.inspect_record(rid, fp)
            assert env.ledger == settled, (agent, p["task_id"])
            assert sorted(ob.established(tasks[p["task_id"]], env.ledger)) == auth, (agent, p["task_id"])
            n += 1
    assert n == 7176
