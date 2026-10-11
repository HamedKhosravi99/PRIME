"""No code path may reach a paid API endpoint by accident.

Every run in this project targets a locally served open-weight model via
PRIME_LLM_BASE, and the user's standing constraint is that the paid OpenAI
key is not to be used. But OPENAI_API_KEY is present in the environment and
LLMClient's default base is api.openai.com, so any path that forgets to set
the base issues an authenticated, billable request. That happened: a smoke
test of the S1 script with a placeholder model name reached OpenAI and came
back 404 model_not_found. No completion was produced, so nothing was billed,
but the call should not have been possible.
"""
from __future__ import annotations

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(os.path.dirname(_TESTS))
for _x in (_P, os.path.join(_P, "runner")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import llm as llm_mod  # noqa: E402

import pytest  # noqa: E402


def _clean(monkeypatch):
    for v in ("PRIME_LLM_BASE", "PRIME_ALLOW_PAID_API"):
        monkeypatch.delenv(v, raising=False)


def test_default_base_is_refused(monkeypatch):
    _clean(monkeypatch)
    with pytest.raises(RuntimeError) as e:
        llm_mod.LLMClient(model="gpt-4o-mini")
    assert "refusing to call the paid endpoint" in str(e.value)


def test_error_names_both_ways_out(monkeypatch):
    _clean(monkeypatch)
    with pytest.raises(RuntimeError) as e:
        llm_mod.LLMClient(model="gpt-4o-mini")
    msg = str(e.value)
    assert "PRIME_LLM_BASE" in msg and "PRIME_ALLOW_PAID_API" in msg


def test_locally_served_base_is_allowed(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PRIME_LLM_BASE", "http://localhost:8115/v1")
    assert llm_mod.LLMClient(model="qwen2.5-32b").base.endswith(":8115/v1")


def test_cluster_style_node_base_is_allowed(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PRIME_LLM_BASE", "http://atl1-1-03-010-1-0:8115/v1")
    assert llm_mod.LLMClient(model="qwen2.5-32b")


def test_explicit_override_still_works(monkeypatch):
    """The guard is a tripwire, not a prohibition -- an intentional paid run
    stays possible, but only deliberately."""
    _clean(monkeypatch)
    monkeypatch.setenv("PRIME_ALLOW_PAID_API", "1")
    assert "openai.com" in llm_mod.LLMClient(model="gpt-4o-mini").base


def test_subdomains_of_paid_hosts_are_also_refused(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PRIME_LLM_BASE", "https://eu.api.openai.com/v1")
    with pytest.raises(RuntimeError):
        llm_mod.LLMClient(model="gpt-4o-mini")


def test_lookalike_host_is_not_wrongly_allowed(monkeypatch):
    """endswith() on a bare host would let 'notapi.openai.com' through only
    if the dot were missing; pin the boundary."""
    _clean(monkeypatch)
    monkeypatch.setenv("PRIME_LLM_BASE", "https://myapi.openai.com.evil.test/v1")
    llm_mod.LLMClient(model="x")      # not a paid host: must NOT raise
