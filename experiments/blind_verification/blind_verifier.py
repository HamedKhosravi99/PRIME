"""The blind verifier: a fresh-context second judgment of a text atom.

The prompt is frozen (verifier_spec.md) and byte-identical to the `blind`
prompt of the risk-aware study's experiments/risk_aware_prime/
verify_attestations.py (not part of this package); the
SHA-256 of the two prompt strings is asserted at import so a silent edit
cannot go unnoticed.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, Optional

SYSTEM = ("You are a careful reviewer. You judge whether a document establishes a stated "
          "condition. End your reply with a line of the form 'ANSWER: T', 'ANSWER: F' or "
          "'ANSWER: UNCLEAR'.")

USER_TEMPLATE = ("Condition: %s\n\nDocument:\n%s\n\nDoes the document establish the condition? "
                 "T if the document establishes the condition, F if it establishes the opposite, "
                 "UNCLEAR if it establishes neither. Reply with at most one sentence of justification, "
                 "then the line 'ANSWER: T', 'ANSWER: F' or 'ANSWER: UNCLEAR'.")

# sha256 of SYSTEM + "\n" + USER_TEMPLATE, recorded when the spec was frozen
FROZEN_SHA256 = "dd992e7312ce1cb8dd21d682fa168cfaeb34d31431df1fbe0c3633cb9f8e0879"

ANSWER_RE = re.compile(r"ANSWER:\s*(T|F|UNCLEAR)\b", re.I)
TOKEN_RE = re.compile(r"\b(T|F|UNCLEAR)\b")
MAX_TOKENS = 400


def prompt_digest() -> str:
    return hashlib.sha256((SYSTEM + "\n" + USER_TEMPLATE).encode("utf-8")).hexdigest()


def check_frozen() -> None:
    d = prompt_digest()
    if FROZEN_SHA256 != "__FROZEN__" and d != FROZEN_SHA256:
        raise RuntimeError("verifier prompt changed after freezing: %s != %s" % (d, FROZEN_SHA256))


def parse(content: str) -> str:
    if content:
        found = ANSWER_RE.findall(content)
        if found:
            return found[-1].upper()
        if len(content.split()) <= 2:
            m = TOKEN_RE.search(content.upper())
            if m:
                return m.group(1)
    return "PARSE_FAIL"


def verify(llm, condition: str, text: str) -> Dict[str, Any]:
    """One fresh-context verifier call. ``llm`` is a runner.llm.LLMClient
    (or any object with .chat(messages, tools=None) -> {content, usage}).
    Returns the verdict, the raw reply, usage and latency. Receives ONLY the
    condition and the document."""
    check_frozen()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE % (condition, text)}]
    t0 = time.time()
    resp = llm.chat(messages, tools=None)
    usage = resp.get("usage", {}) or {}
    content = (resp.get("content") or "").strip()
    return {"verdict": parse(content), "raw": content[:400],
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": round(time.time() - t0, 3)}
