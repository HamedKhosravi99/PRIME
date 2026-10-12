# Blind verification: frozen specification

Frozen before the live run (2026-10-10). Nothing here may change in
response to live results. The prompt below is byte-identical to the
`blind` prompt of `pilot/experiments/risk_aware_prime/verify_attestations.py`,
the one whose verdicts produced the offline replay (REPORT.md there,
Section 6.4); `blind_verifier.py` asserts its SHA-256 on every call:
`dd992e7312ce1cb8dd21d682fa168cfaeb34d31431df1fbe0c3633cb9f8e0879`
(SHA-256 of the system message, a newline, and the user template).

## Verifier

- Model: the episode's own model (`PRIME_LLM_MODEL`, Qwen/Qwen2.5-32B-Instruct),
  served by the same vLLM instance, called through `runner.llm.LLMClient`
  with `tools=None`, temperature 0, `max_tokens` 400.
- Context: a fresh two-message conversation per call (system + user). The
  verifier receives the atom's condition (the `spec_hint.what` text of the
  instance file) and the source text the agent read (the `read_text`
  field). It receives nothing else: not the agent's value, quote,
  reasoning, conversation, authorized conclusions, or any label.
- System message:

```
You are a careful reviewer. You judge whether a document establishes a stated condition. End your reply with a line of the form 'ANSWER: T', 'ANSWER: F' or 'ANSWER: UNCLEAR'.
```

- User message (`{condition}` and `{text}` substituted):

```
Condition: {condition}

Document:
{text}

Does the document establish the condition? T if the document establishes the condition, F if it establishes the opposite, UNCLEAR if it establishes neither. Reply with at most one sentence of justification, then the line 'ANSWER: T', 'ANSWER: F' or 'ANSWER: UNCLEAR'.
```

- Parse: the last `ANSWER: X` line (case-insensitive); a bare one- or
  two-token reply is accepted; anything else is `PARSE_FAIL`.

## Agreement rule

An `attest_atom` call first passes the runtime's existing acceptance
checks unchanged (the atom is a text atom, the value is T or F, its source
text was read in this episode, the quote is a verbatim span of at least
eight characters). Only then is the verifier consulted. The attestation
is **accepted into the ledger iff the verifier's answer equals the
agent's value**. `F`, `UNCLEAR` and `PARSE_FAIL` against an attested `T`
(and symmetrically for `F`) all leave the atom unresolved: the executor
returns `accepted: false` with the reason "not confirmed by an independent
check of the source text; the atom remains unresolved", and
`goldledger.delta` records nothing, exactly as for a quote that is not
verbatim. The reason never states the verifier's answer. The rule is the
same for attested `T` and attested `F`.

The verifier's verdict for an atom is computed once per episode and reused
for later attestations of the same atom (the document and condition are
unchanged, the model is deterministic at temperature 0); each actual
request is counted as an invocation, each use as a consultation.

## Budget and accounting

- The agent's 20-call budget is unchanged: every `attest_atom` attempt is
  charged as before, whether or not it is confirmed.
- Verifier calls are not charged to the budget. They are counted
  separately (invocations, consultations, prompt and completion tokens,
  latency) in `_verify/<instance>.json` and in `summary.json`.
- Everything else (arm `prime_guided`, gold program, 20 calls, guidance,
  holds, stall nudges, MaxAuth fallback, forced selection, parallel 6,
  vLLM recipe of the second cluster's job script) is as in the original text
  run, which was produced on that cluster.

## What the verifier and agent never see

Hidden structured fields (the gold twins), gold values, whether any
earlier attestation was right, the scorer, any live-run score. The
wrapper imports no scoring code; `verified_executor.py` reads only the
`Instance` object the runtime already exposes to the executor.
