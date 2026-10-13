"""The model-call count recovered from a trace equals the requests the agent made.

experiments/analysis/model_calls.py counts the model requests of the PRIME runs and the
fully informed agent from their traces (Table 10). Here a counting model that
mixes reads, repeats, unauthorized and authorized conclusions, and silent stops
drives the real agent loop, and the count recovered from the written trace and
summary must equal the number of requests the model received.
"""
from __future__ import annotations

import json
import os
import random
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "src", "prime_agents"), os.path.join(_P, "experiments", "analysis")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import pytest  # noqa: E402

import model_calls as mc  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from run_episode import run_episode  # noqa: E402

V2 = os.path.join(_P, "data", "suffice")
INSTANCES = [
    os.path.join(V2, "security", "heldout_balanced", "sec-v2-0012.json"),
    os.path.join(V2, "clinical", "heldout_locator", sorted(os.listdir(os.path.join(V2, "clinical", "heldout_locator")))[0]),
]


class CountingModel:
    """Random tool calls, conclusions and silent stops; counts every request."""

    def __init__(self, instance, seed):
        self.rng = random.Random(seed)
        self.n = 0
        self.reads = []
        for a in instance.atoms:
            if a.resolver is not None and a.resolver.tool == "inspect_record":
                p = dict(a.resolver.params)
                self.reads.append(p)
                for c in (getattr(a, "hint", None) or {}).get("candidates", []):
                    self.reads.append(dict(p, field_path=c))
        rid = sorted(instance.records)[0]
        self.reads.append({"record_id": rid, "field_path": "no.such.field"})
        self.claims = [c.id for c in instance.claims]

    def chat(self, messages, tools=None):
        self.n += 1
        usage = {"prompt_tokens": 1, "completion_tokens": 1}
        if not tools:
            pick = self.rng.sample(self.claims, self.rng.randint(0, 2))
            return {"content": json.dumps({"claims": pick}) if self.rng.random() < 0.8 else "no answer",
                    "tool_calls": [], "usage": usage}
        u = self.rng.random()
        if u < 0.15:
            return {"content": "I will stop here.", "tool_calls": [], "usage": usage}
        if u < 0.35:
            return {"content": "", "usage": usage,
                    "tool_calls": [{"name": "emit_claim", "arguments": {"claim_id": self.rng.choice(self.claims)}}]}
        return {"content": "", "usage": usage,
                "tool_calls": [{"name": "inspect_record", "arguments": dict(self.rng.choice(self.reads))}]}


@pytest.mark.parametrize("arm", ["react_program_attested", "masked_reactive_v", "prime_guided",
                                 "prime_complete", "prime_flat"])
def test_recovered_count_equals_requests(arm, tmp_path):
    checked = 0
    for path in INSTANCES:
        inst = load_instance(path)
        for budget in (2, 5, 20):
            for seed in range(12):
                model = CountingModel(inst, seed)
                out = tmp_path / ("%s-%d-%d" % (inst.id, budget, seed))
                entry = run_episode(path, arm, "gold", model, str(out), max_tool_calls=budget)
                assert not entry["error"], entry["error"]
                got = mc.model_calls(entry, mc.trace_records(entry["trace"]))
                assert got == model.n, (arm, inst.id, budget, seed, got, model.n)
                checked += 1
    assert checked == 72
