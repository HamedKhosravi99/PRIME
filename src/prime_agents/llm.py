"""LLM clients for the PRIME runner.

``LLMClient`` speaks the OpenAI-compatible chat-completions protocol.  It
uses the ``openai`` python package when installed, otherwise raw HTTPS via
urllib against an OpenAI-compatible endpoint — the same code hits the OpenAI
API locally or a vLLM server on a cluster.

Environment variables:
    PRIME_LLM_BASE   base URL (default https://api.openai.com/v1)
    PRIME_LLM_KEY    API key (default: env OPENAI_API_KEY)
    PRIME_LLM_MODEL  model name

``ScriptedLLM`` is a deterministic fake used by ``run_batch.py --dry-run``:
it returns canned tool calls so the whole pipeline is testable without API
calls.

Both clients return the same normalized response shape::

    {"content": str, "tool_calls": [{"name": str, "arguments": dict}],
     "usage": {"prompt_tokens": int, "completion_tokens": int}}
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

_PILOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PILOT not in sys.path:
    sys.path.insert(0, _PILOT)
    sys.path.insert(0, os.path.join(_PILOT, "src"))

from prime_runtime.schema import Instance  # noqa: E402

DEFAULT_BASE = "https://api.openai.com/v1"


def _normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for tc in raw_calls or []:
        if isinstance(tc, Mapping):
            fn = tc.get("function", {})
            name = fn.get("name", tc.get("name", ""))
            args = fn.get("arguments", tc.get("arguments", "{}"))
        else:  # openai SDK object
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args = getattr(fn, "arguments", "{}") if fn else "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": name, "arguments": dict(args)})
    return calls


class LLMClient:
    """OpenAI-compatible chat client with retries; temperature 0."""

    def __init__(
        self,
        model: Optional[str] = None,
        base: Optional[str] = None,
        key: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("PRIME_LLM_MODEL", "gpt-4o-mini")
        self.base = (base or os.environ.get("PRIME_LLM_BASE", DEFAULT_BASE)).rstrip("/")
        self.key = key or os.environ.get("PRIME_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        # vLLM honours `seed` per request; it makes a sampled run
        # reproducible, which is what lets a decoding-variance estimate be
        # re-derived rather than merely asserted.
        self.seed = seed
        self.max_retries = max_retries
        self.timeout = timeout
        self._guard_paid_endpoint()
        self._openai_client = None
        try:
            from openai import OpenAI  # type: ignore

            # timeout MUST be passed: without it the SDK default (very long)
            # applies and a server-side hang blocks the episode forever. A
            # malformed tool call can crash vLLM's tool parser, leaving the
            # request in "Running" state with zero throughput -- observed as
            # a 70-minute stall on job 11621570. max_retries=0 keeps retry
            # policy in OUR loop (chat()), which is budget- and log-aware.
            self._openai_client = OpenAI(base_url=self.base,
                                         api_key=self.key or "EMPTY",
                                         timeout=self.timeout,
                                         max_retries=0)
        except Exception:
            self._openai_client = None

    # Every run in this project targets a locally served open-weight model
    # via PRIME_LLM_BASE. OPENAI_API_KEY is nonetheless present in the
    # environment, and the client's default base is api.openai.com, so any
    # code path that forgets to set the base -- a smoke test, a new analysis
    # script, a default argument -- silently issues an authenticated,
    # BILLABLE request. That has happened. The endpoint is now opt-in.
    PAID_HOSTS = ("api.openai.com", "api.anthropic.com")
    OPT_IN = "PRIME_ALLOW_PAID_API"

    def _guard_paid_endpoint(self) -> None:
        host = self.base.split("//", 1)[-1].split("/", 1)[0].lower()
        if not any(host == h or host.endswith("." + h) for h in self.PAID_HOSTS):
            return
        if os.environ.get(self.OPT_IN) == "1":
            return
        raise RuntimeError(
            "refusing to call the paid endpoint %s: set PRIME_LLM_BASE to a "
            "locally served model, or export %s=1 to override deliberately. "
            "(An API key is present in this environment, so an unguarded "
            "call would be authenticated and billable.)"
            % (host, self.OPT_IN))

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self._openai_client is not None:
                    return self._chat_openai(messages, tools)
                return self._chat_urllib(messages, tools)
            except Exception as exc:  # network / rate-limit / parse errors
                last_exc = exc
                time.sleep(2.0 ** attempt)
        raise RuntimeError("LLM request failed after retries: %r" % (last_exc,))

    def _restricted_temp(self) -> bool:
        """gpt-5 and o-series reasoning models reject a custom temperature
        (only the default 1 is allowed) and use max_completion_tokens."""
        m = self.model.lower()
        return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")

    def _chat_openai(self, messages, tools) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
        }
        if not self._restricted_temp():
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if tools:
            kwargs["tools"] = list(tools)
        resp = self._openai_client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return {
            "content": msg.content or "",
            "tool_calls": _normalize_tool_calls(getattr(msg, "tool_calls", None)),
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
        }

    def _chat_urllib(self, messages, tools) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
        }
        if not self._restricted_temp():
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        if tools:
            payload["tools"] = list(tools)
        req = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % (self.key or "EMPTY"),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        usage = body.get("usage", {})
        return {
            "content": msg.get("content") or "",
            "tool_calls": _normalize_tool_calls(msg.get("tool_calls")),
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
        }


class ScriptedLLM:
    """Deterministic fake LLM for --dry-run: replays canned tool calls.

    ``script`` is a list of {"name": ..., "arguments": {...}} tool calls,
    returned one per chat() call that offers tools.  Tool-less calls return
    canned prose (or ``program_json`` when a license program is requested,
    for the induced-license condition).
    """

    def __init__(
        self,
        script: Sequence[Mapping[str, Any]],
        program_json: Optional[str] = None,
    ) -> None:
        self._script: List[Dict[str, Any]] = [dict(c) for c in script]
        self._cursor = 0
        self.program_json = program_json

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if not tools:
            last = str(messages[-1].get("content", "")) if messages else ""
            if self.program_json and "license program" in last.lower():
                return {"content": self.program_json, "tool_calls": [], "usage": usage}
            return {
                "content": "Criteria: correctness, tests, configuration, hygiene.",
                "tool_calls": [],
                "usage": usage,
            }
        allowed = set()
        for t in tools:
            fn = t.get("function", {}) if isinstance(t, Mapping) else {}
            allowed.add(fn.get("name", t.get("name", "")) if isinstance(t, Mapping) else "")
        # Skip scripted calls whose tool is not offered (e.g. the direct arm
        # offers emit_claim only).
        while (
            self._cursor < len(self._script)
            and self._script[self._cursor].get("name") not in allowed
        ):
            self._cursor += 1
        if self._cursor >= len(self._script):
            return {"content": "done", "tool_calls": [], "usage": usage}
        call = self._script[self._cursor]
        self._cursor += 1
        return {"content": "", "tool_calls": [dict(call)], "usage": usage}


def build_dry_run_script(instance: Instance) -> List[Dict[str, Any]]:
    """Canned tool-call sequence discharging the gold route of the strongest
    claim, then emitting it — derived deterministically from the instance's
    resolver definitions."""
    top = max(instance.claims, key=lambda c: (c.level, c.id))
    routes = instance.gold_licenses.get(top.id, ())
    route = routes[0] if routes else ()
    calls: List[Dict[str, Any]] = []
    seen = set()
    for lit in route:
        atom = instance.atom(lit.lstrip("!"))
        r = atom.resolver
        if r is None:
            continue  # unresolvable atom: the scripted agent cannot establish it
        params: Dict[str, Any] = dict(r.params)
        if not params:
            params = {
                "run_tests": {"suite": "default"},
                "inspect_config": {},
                "search_repo": {"pattern": "TODO"},
                "inspect_file": {"path": ""},
                "ask_user": {"question": "What is the deployment context?"},
            }.get(r.tool, {})
        key = (r.tool, json.dumps(params, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append({"name": r.tool, "arguments": params})
    calls.append({"name": "emit_claim", "arguments": {"claim_id": top.id}})
    return calls
