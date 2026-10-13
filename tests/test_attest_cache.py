"""An attestation refused because its source text was unread is evaluated afresh
once the text has been read: the call class of attest_atom includes whether the
source text has been read, so the cached refusal is not served again."""
from __future__ import annotations

import glob
import json
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(_TESTS)
for _x in (_P, os.path.join(_P, "src"), os.path.join(_P, "src", "prime_agents")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import pytest  # noqa: E402

import arms as arms_mod  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402

TEXT = sorted(glob.glob(os.path.join(_P, "data", "suffice", "clinical", "heldout_text", "*.json")))


@pytest.mark.skipif(not TEXT, reason="no text-evidence tasks")
def test_attestation_after_reading_is_not_served_the_cached_refusal():
    path = TEXT[0]
    inst = load_instance(path)
    raw = json.load(open(path, encoding="utf-8"))
    atom = next(a for a in raw["atoms"] if (a.get("resolver") or {}).get("tool") == "attest_atom")
    src = atom["spec_hint"]["source"]
    value, _ = arms_mod._resolve_field_path(raw["records"][src["record_id"]], src["field_path"])
    text = " ".join(str(value).split())
    args = {"atom_id": atom["id"], "value": "T", "quote": text[:40].strip()}
    ex = arms_mod.ToolExecutor(inst)
    assert ex.execute("attest_atom", dict(args))["accepted"] is False          # text not read yet
    assert not ex.charges("attest_atom", dict(args))                          # a repeat now is free ...
    ex.execute("read_text", {"record_id": src["record_id"], "field_path": src["field_path"]})
    assert ex.charges("attest_atom", dict(args))                              # ... but after the read it is a new call
    obs = ex.execute("attest_atom", dict(args))
    assert obs["accepted"] is True and ex.last_was_cached is False
    assert ex.execute("attest_atom", dict(args))["accepted"] is True and ex.last_was_cached is True
