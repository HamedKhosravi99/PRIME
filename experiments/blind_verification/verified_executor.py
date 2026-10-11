"""A ToolExecutor whose attest_atom accepts an attestation only when the
blind verifier agrees. Installed by run_live.py through
``arms_mod.ToolExecutor = VerifiedToolExecutor`` (the runtime files are not
edited; run_arm looks the class up in its module globals at call time).

What changes: ``_tool_attest_atom`` runs the ORIGINAL acceptance checks
(super()), and if they pass consults the verifier (one fresh context per
atom per episode) before returning ``accepted``. A disagreement, an
UNCLEAR, or a parse failure returns accepted=False with a reason that does
not reveal the verifier's answer, so goldledger.delta records nothing and
the atom stays open. Everything else in the executor is inherited.

What the verifier sees: the atom's condition text (spec_hint.what) and the
source text already read in this episode. Nothing else.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List

_PILOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PILOT, os.path.join(_PILOT, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import blind_verifier  # noqa: E402

REJECT_REASON = ("not confirmed by an independent check of the source text; "
                 "the atom remains unresolved")

# instance id -> list of verification events (read by run_live.py after
# the episode; guarded for the parallel driver)
REGISTRY: Dict[str, List[Dict[str, Any]]] = {}
_LOCK = threading.Lock()
VERIFIER_LLM = None  # set by run_live.py: a runner.llm.LLMClient


class VerifiedToolExecutor(arms_mod.ToolExecutor):
    def __init__(self, instance) -> None:
        super().__init__(instance)
        self._verdicts: Dict[str, Dict[str, Any]] = {}  # atom id -> verifier result
        self._n_attempt = 0

    def _tool_attest_atom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        obs = super()._tool_attest_atom(params)
        if not obs.get("accepted"):
            return obs  # the original checks already rejected it; unchanged
        aid = str(params.get("atom_id", ""))
        atom = next((a for a in self.instance.atoms if a.id == aid), None)
        hint = getattr(atom, "hint", None) or {}
        src = hint.get("source") or {}
        key = (str(src.get("record_id", "")), str(src.get("field_path", "")))
        text = self._texts_read.get(key)
        condition = hint.get("what", "")
        if VERIFIER_LLM is None or text is None or not condition:
            raise RuntimeError("verifier unavailable for atom %s: refusing to accept unverified" % aid)
        self._n_attempt += 1
        cached = aid in self._verdicts
        if not cached:
            self._verdicts[aid] = blind_verifier.verify(VERIFIER_LLM, condition, text)
        v = self._verdicts[aid]
        agree = v["verdict"] == obs["value"]
        event = {"attempt": self._n_attempt, "atom": aid, "attested": obs["value"],
                 "quote": obs.get("quote", ""), "verdict": v["verdict"], "agree": agree,
                 "invoked": not cached, "prompt_tokens": 0 if cached else v["prompt_tokens"],
                 "completion_tokens": 0 if cached else v["completion_tokens"],
                 "latency_s": 0.0 if cached else v["latency_s"], "raw": v["raw"]}
        with _LOCK:
            REGISTRY.setdefault(self.instance.id, []).append(event)
        if agree:
            return obs
        return {"atom_id": aid, "accepted": False, "reason": REJECT_REASON}
