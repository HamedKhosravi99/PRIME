# PRIME with blind same-model verification: live experiment

Status: COMPLETE (2026-10-10). Live run on an HTCondor cluster 85839
(one NVIDIA A100 80GB, vLLM 0.10.1.1, Qwen2.5-32B-Instruct, the same
serving stack, model, budget, arm and parallelism as the original
Experiment D cells), 240 episodes, 0 errors. Every number below is read
from `results/analysis.json`, produced by `analyze_live.py` from the run's
own traces and the original Experiment D logs; nothing is estimated by
hand. The manuscript, the runtime, the datasets, the original logs and the
reported results were not modified (Section 2).

Scope, as authorized: one live condition, `prime_guided` with blind
verification of text attestations, on the 240 text-evidence tasks. No
other arm, seed, model or benchmark was run; the gpt-oss cross-family
verifier was not run.

## 1. Implementation

**Procedure.** The agent is the paper's `prime_guided` arm, unchanged:
gold program, 20-call budget, guidance, holds, stall nudges, MaxAuth
fallback, forced selection. The one change is the acceptance of a text
attestation into the ledger. `verified_executor.VerifiedToolExecutor`
subclasses the runtime's `ToolExecutor`; its `attest_atom` first runs the
original acceptance checks by calling the parent (text atom, value T or
F, source text read in this episode, quote a verbatim span of at least
eight characters). Only if those pass does it consult the verifier: a
second request to the same served model, in a fresh two-message context,
with the atom's condition text and the source text and nothing else. The
attestation is accepted iff the verifier's answer equals the agent's
value; otherwise the executor returns `accepted: false` with the reason
"not confirmed by an independent check of the source text; the atom
remains unresolved", which never states the verifier's answer, and
`goldledger.delta` records nothing, as for a non-verbatim quote. The rule
is the same for attested T and F; UNCLEAR and parse failures count as
disagreement. The verdict for an atom is computed once per episode and
reused for later attestations of the same atom (deterministic model at
temperature 0); every actual request is an invocation.

**Prompt.** Frozen in `verifier_spec.md` before the run and asserted by
SHA-256 on every call
(`dd992e7312ce1cb8dd21d682fa168cfaeb34d31431df1fbe0c3633cb9f8e0879`),
byte-identical to the `blind` prompt of the offline replay (checked by
string comparison before freezing; recorded in the run's `summary.json`).
System message: "You are a careful reviewer. You judge whether a document
establishes a stated condition. End your reply with a line of the form
'ANSWER: T', 'ANSWER: F' or 'ANSWER: UNCLEAR'." User message: the
condition, the document, and the instruction to answer T if the document
establishes the condition, F if it establishes the opposite, UNCLEAR if
neither, with at most one sentence of justification. `max_tokens` 400,
temperature 0.

**Installation without editing the runtime.** `run_live.py` sets
`arms_mod.ToolExecutor = VerifiedToolExecutor` in its own process;
`run_arm` looks the class up in its module globals at call time, so the
episode loop in `runner/arms.py` runs unmodified. The driver mirrors
`runner/run_batch.py` (same `LLMClient` construction, temperature 0, no
seed, per-episode sidecars and resume, same `summary.json` shape) and adds
a `verification` block per episode and a `_verify/<instance>.json` sidecar
with every verification event (atom, attested value, verdict, agreement,
tokens, latency, the verifier's one-sentence justification).

**Budget and accounting.** The agent's 20-call budget is charged exactly
as before (every `attest_atom` and `emit_claim` attempt). Verifier
requests are not charged to that budget and are counted separately.

**Files**: `verifier_spec.md`, `blind_verifier.py`, `verified_executor.py`,
`run_live.py`, `test_wrapper.py`, `analyze_live.py`;
`results/qwen2.5-32b-{clinical,security}/prime_guided_verified__B20/`
(summary, traces, `_episodes/`, `_verify/`), `results/analysis.json`.

## 2. Experimental integrity

- **Runtime unchanged.** `git status` on `pilot/runner`, `pilot/prime_runtime`,
  `pilot/bench` and `pilot/results/w2text` is clean after the run; the
  SHA-256 prefixes of `arms.py` (35bd91b5f9c2a3aa), `run_episode.py`
  (0bdf01bbab0db278), `ledger.py` (abbfabdd2c090804) and `goldledger.py`
  (45e04d4957396d80) are identical locally and on the HTCondor copy that ran
  the job; the three wrapper files on the HTCondor cluster hash to the same values as the
  committed ones (0ece6125b4f44885, fef466115bf7e99e, 0afd7e87b500dbc3).
- **Verifier blind.** `test_wrapper.py` (passed locally and on the job node
  before the episodes started; the job aborts on failure) records what the
  verifier is shown: one system and one user message, containing the
  atom's condition and the document, not the agent's value, quote or
  conversation. The wrapper imports no scoring code and reads no gold
  value; the reject reason carries no verdict.
- **No tuning on outcomes.** Prompt and rule frozen before the run; the run
  was launched once (an earlier submission to the same cluster was removed
  before it started; three the cluster twins were cancelled the moment this
  job started, before any episode ran there).
- **No label access at inference.** The gold twins are not in the embedded
  records; `analyze_live.py` reads gold after the run, for scoring and the
  audit only.
- **Same environment as the baseline.** The original text cells were
  produced on the HTCondor pool with vLLM 0.10.1.1 on A100s; so was this run.

## 3. Results

Scored with the paper's scorer. "Original PRIME" is the historical
Experiment D cell (one greedy run per task); this is a new greedy run with
the verifier, not a simultaneous paired rerun. The offline replay is the
risk-aware study's verify-all replay of the same logs
(`pilot/experiments/risk_aware_prime/`), listed for comparison only.

| metric | original PRIME (logs) | offline replay | live PRIME + blind verification |
|---|---|---|---|
| false authorization | 11 / 240 (4.6% [2.6, 8.0]) | 2 / 240 (0.8%) | **2 / 240 (0.8% [0.2, 3.0])** |
| supported utility | 57.6 [55.1, 60.0] | 55.7 | **55.3 [53.2, 57.4]** |
| answer coverage | 100% | 100% | 100% (no abstention) |
| false authorization among emitted | 4.6% | 0.8% | 0.8% |
| accepted attestations; wrong | 299; 30 (10.0%) | | 222; 6 (2.7%) |
| episodes with a wrong attestation | 11.7% | | 2.5% |
| agent tool calls per episode | 3.72 | 3.99 (replay accounting) | 3.61 |
| verifier invocations per episode | 0 | 0.98 (unique items) | 1.17 (280 total; 287 consultations) |
| verifier tokens per episode | 0 | | 533 prompt + 33 completion |
| agent prompt tokens per episode | 15,657 | | 15,829 |

Paired contrasts over the 240 tasks (record-level bootstrap): utility
−2.4 [−4.3, −0.3]; false authorization −3.8 points [−6.2, −1.7]. Discordant
pairs: 9 tasks where only the original over-authorized, 0 where only the
live run did.

By domain. Clinical: 0 / 120 (0.0% [0.0, 3.1]), utility 53.6 [50.8, 56.7],
111 accepted attestations with 0 wrong. Security: 2 / 120 (1.7% [0.5,
5.9]), utility 56.9 [53.9, 59.7], 111 accepted attestations with 6 wrong
(5.4%). Original: clinical 6 / 120 at 56.1, security 5 / 120 at 59.2.

The live run reproduces the offline replay: the same two errors remain,
utility is within 0.4 points of the replay's figure, and the verifier
cost is of the same order (the replay counted unique items; live
episodes re-attest and re-consult).

## 4. Error analysis

**The two remaining unsupported conclusions** are the two the replay
predicted, both security, both level 2 on the first text atom, "the
details text names the affected package (by its registered name or a
distinctive part of it, not only the vendor or the ecosystem)":

- `sec-v2-7127`. Text: "The AWS API MCP Server is an open source Model
  Context Protocol (MCP) server that enables AI assistants to interact with
  AWS services ...". Agent: T, quoting that sentence. Verifier: T ("names
  the affected package as 'AWS API MCP Server'"). Hidden reference: F; the
  structured package name's distinctive token (the registry's slug) does
  not occur in the text under the benchmark's token rule. Type: a mismatch
  between the textual and the structured definition of "names the
  package"; the text does name the product.
- `sec-v2-7145`. Text: "AccessControl/AuthEncoding.py in Zope before
  2.13.19, as used in Plone before 4.2.3 and 4.3 before beta 1 ...". Agent:
  T, quoting the version clause. Verifier: T ("mentions 'Zope' and 'Plone'
  ... by their registered names"). Hidden reference: F; the registered name
  differs from the product name as written. Type: the same
  textual-versus-structured definition mismatch.

In both, the reference standard's token rule is what disagrees with a
reasonable reading; neither is a model misreading, and no text-only
verifier can remove them. The ground truth was not modified. Whether the
benchmark's definition of "names the package" should accept a product
name as written is a definitional question for the benchmark, recorded
here and not resolved by this experiment.

**New live-run failures.** None: no episode errored, no parse failure
(0 of 287 consultations), and no unsupported conclusion that the original
run did not also make.

**Where the verifier is wrong.** Of 72 rejections, 20 rejected a wrong
value and 52 a right one. The 52 are the utility cost (Section 5); they
concentrate on the second and third text atoms (fixed version, first
affected version; one-sex and healthy-volunteer conditions), where the
verifier returned UNCLEAR 35 times in total and the agreement rate is
low, as the replay found. The 6 wrong values the verifier accepted are
all security first-atom cases of the definitional kind above (the two
that became false authorizations and four whose episodes ended at a
supported level anyway).

## 5. Behaviour analysis

What the replay could not show is what the agent does after a rejection.
60 episodes (25%) had at least one verifier rejection (72 rejections in
all). After the first rejection: the agent re-attested the same atom in
13 episodes, flipped its value in 4 (3 of these flips were then accepted,
all on atoms where the verifier's answer matched the flipped value; the
agent was never told the verifier's answer), retried the same value 0
times successfully, and moved to another atom in 3; in every one of the 60
the episode ended with the runtime's MaxAuth statement, never abstention.
Rejections prevented 9 of the original run's 11 false authorizations
directly (the other two originals did not recur for ordinary run-to-run
reasons). Against the original run, task by task: the same final level in
191 tasks, lower in 43, higher in 6. Of the 43 lowered, 9 are prevented
errors and the rest are the price: a right attestation refused, the
episode ending one level down. The verifier did not cause more checking
(3.61 calls against 3.72) and did not change route selection, which on a
chain it cannot.

Interpretation: the improvement is correction of evidence judgments, not
suppression. Coverage is 100% in both runs; the live run states a weaker
conclusion in 43 tasks and a wrong one in 9 fewer; it never abstains where
the original answered.

## 6. Cost

Per episode: 1.17 verifier invocations (1.20 consultations), 533 prompt
and 33 completion verifier tokens against 15,829 agent prompt tokens
(+3.6%), 2.55 s of verifier latency inside a 44 s episode; agent tool calls
3.61 against 3.72. Total: 280 verifier requests, 128,015 prompt tokens, 613 s
of latency over 240 episodes; the whole run took about 50 minutes on one
A100 including model load.

## 7. Statistical uncertainty

With 240 tasks the 95% interval on 2 events is [0.2%, 3.0%]; the result
excludes the original 4.6% but does not establish any sub-percent
population rate. The paired utility loss is 2.4 points with interval
[0.3, 4.3], so the cost is real but small; the false-authorization
reduction interval [1.7, 6.2] points excludes zero. The comparison is
against a historical baseline (one greedy run each, same model and
serving stack, different days), not a simultaneous paired rerun; the
nine discordant pairs are all in one direction. One model, one prompt,
two domains, chains of three text atoms: the result is a measurement of
this resolver on this split, not a general calibration claim, and no
population guarantee is asserted.

## 8. Interpretation

The improvement comes from correcting evidence judgments, as the paper's
decomposition predicts: PRIME converts resolver errors into false
authorizations without adding to them, so a better resolver lowers the
term, and a second blind judgment by the same model is a better resolver
for the first text atom (where it caught every misreading) and a poor one
for the third (where it rejects right and wrong alike). It is a resolver
change, not a planning method, and it is not conformal or risk-controlled
in any formal sense.

## 9. Paper recommendation

**Include, as one row and two sentences in Experiment D, accurately
qualified.** The live result is valid (frozen protocol, blind verifier,
same environment, runtime unchanged, labels read only after the run),
reproduces the offline replay, and is a direct measurement of the claim
the paper already makes, that only a better resolver lowers the resolver
term. It should not be presented as a new mechanism of PRIME, as a
guarantee, or as a general fix for text evidence.

## 10. Proposed wording (applied on the user's standing authorization)

Section 6, Experiment D, after the sentence ending "only a better resolver
lowers the term (Table 3)": "Requiring a second, blind judgment by the same
model before an attestation enters the ledger left 2 unsupported
conclusions (0.8%) at utility 55.3 (loss 2.4, 95% CI [0.3, 4.3]) for 1.2
extra model calls per episode; the two that remain are cases where the
structured field and the text disagree." Table 3 gains the row "PRIME,
verified attestations: 0.8 / 55.3 / 2.7 / 2.5". Appendix D's text-split
table gains the same row with 222 accepted attestations, and the appendix
paragraph "Verified attestations (live)" states the protocol, the
behaviour after rejections, the costs, the environment, and the two
surviving errors.
