# Closest external systems on ObligationBench

Five systems from Appendix H are instantiated on the 480 balanced held-out tasks
with the same model (Qwen2.5-32B-Instruct, greedy), tools, resolvers, budget
accounting, trace format, and scorer as PRIME. Code: `src/prime_agents/extsys.py`
(agentic arms), `src/prime_agents/glean.py` (offline verifier). Jobs: a cluster
launcher that is not included (`experiments/slurm/serve_and_run_guided.sbatch` is the recipe). Table: `experiments/analysis/build_extsys_table.py` ->
`results/tableEXTSYS.json` (20 calls) and `results/tableEXTSYS_bmin.json`.

## What is matched, and what is not

Matched for every system: the task prompt (scope, request, ordered conclusions),
the requirement content (the committed program: each conclusion's alternative
sufficient routes over checks, and each check's tool and field), the evidence
(check values come from the same deterministic resolvers applied to the same
tool observations), and the budget (agent actions counted exactly as in
`run_arm`; cache hits free). Each system receives the requirements in its own
specification format, compiled mechanically from the same program.

Not matched, by design: each system's control mechanism (what decides the next
check, when to stop, and which conclusion is reported), and each system's own
model calls outside the tool budget (planning turns, re-plans, judge calls),
which are counted and reported separately.

`ask_user` is not offered to the external arms: none of the systems has a human
in the loop here, and the benchmark's `ask_user` returns the task's annotator
notes (which include the target level). In the 480 W2a episodes per arm,
`ask_user` was called 0 times by PRIME and once by PRIME without guidance.

## Code availability (checked 2026-10-11)

| System | Paper | Released code | Used here |
|---|---|---|---|
| AgentSpec | ICSE 2026, arXiv:2503.18666 v3 | github.com/haoyuwang99/AgentSpec, no license; LangChain text-ReAct only; `invoke_action` unimplemented; reflection path and finish triggers have bugs | rule semantics and the released self-reflection text, reimplemented on the shared tool-calling loop |
| ResearchLoop | arXiv:2605.28282 v1 | github.com/plan-lab-szu/ResearchLoop, no license; the controller and gate evaluator the docs invoke are not in the repo; the LLM studies use a prompt (`full` condition) | the `full` condition prompt (domain nouns swapped) and the gated runtime of Algorithm 1 |
| Proof-or-Stop | arXiv:2607.14890 v1 | none public (github.com/Proof-or-Stop holds only a profile README; the code repo returns 404) | the gate condition (Eqs. 2-3), the block/repair/stop semantics, and the pilot's two-round recovery cap; no prompts are published |
| HESP | arXiv:2609.33446 v1 | github.com/lzwhehe/HESP, MIT | released system prompt and request layout, guard, controller stop, EIG/cost selection; ledger made exact (see below) |
| GLEAN | ICML 2026 (PMLR 306), arXiv:2603.02798 | none public | Appendix B.1 judge prompt (domain nouns swapped), Eqs. 3-7, the paper's constants |

## Turns without an action

With this model and harness, agents often answer in prose (or write a tool call
as text) instead of calling a tool: PRIME's own agent hit its three stall nudges
in 243 of 480 episodes, and the fully informed agent ended 152 of 480 episodes
in prose. Each system handles such a turn by its own rule. AgentSpec adds
nothing (the turn ends the agent's run, and forced claim selection passes
through the rule, as for the fully informed agent). Proof-or-Stop's controller,
not the agent, ends the loop: a turn with no evidence and no advance request is
answered with the lifecycle state (no hint about which checks), and more than
three consecutive such turns end the loop as no progress (the cap PRIME's
stall nudges use). (This is more lenient than PRIME's runtime, which answers at most three such
turns per episode in total; the counter here resets after every real action.)
ResearchLoop's runtime treats such a turn as a failed
attempt of the active task and blocks a task that fails twice (the RUNBOOK's
same-cause rule), then moves down the queue. HESP's model cannot end the
episode except by an explicit stop.

## Run log (2026-10-11)

Three scaffolding problems were found in the first live episodes and fixed
before any result was scored. (1) The first Proof-or-Stop runs let a turn
without an action end the loop. (2) The first ResearchLoop runs left the
planning turn as the last chat message with no dispatch to execute. (3) With a
planning exchange in the chat history (a JSON plan for Proof-or-Stop, a prose
plan for ResearchLoop), the model kept writing its later tool calls as text,
which no other arm's history induces; both systems keep the plan as persisted
state (Proof-or-Stop's story files, ResearchLoop's state files), so the plan
is now recorded in the system prompt's state block and the execution turns
start from the same [system, task] history as every other arm. Runs under (1)
to (3) were stopped mid-run and their partial outputs moved to
`results/_void_pos_v1/` and `results/_void_extsys_v2/` on the cluster; none is
scored. Every reported run uses the code in this commit.

To finish in time, two jobs ran clinical Proof-or-Stop and clinical ResearchLoop at the same time (the runner skips finished episodes). Four Proof-or-Stop episodes (cli-v2-0130, 0144, 0148, 0173) ran in both jobs and their two greedy runs diverged, so the summary entry and the trace came from different runs; their files were moved to `results/_void_extsys_v2/clinical_pos_twins/` on the cluster and each episode was rerun once. Every reported episode's summary matches its trace (AgentSpec traces also record the prose answers its rule then dropped, by design).

## Per-system instantiation

**AgentSpec** (`ext_agentspec`, `ext_agentspec_examine`). AgentSpec adds nothing
to the agent's prompt, so the protected agent is the fully informed agent of
Experiment A (the base agent PRIME's mask is added to). One rule:
`rule @no_unsupported_conclusion trigger emit_claim check !conclusion_established enforce stop end`,
where the predicate holds when the claimed conclusion has a route whose checks
are all recorded true. `stop` (the enforcement the paper evaluates online):
the conclusion action is not executed and the episode ends with no conclusion.
`llm_self_examine` (described in the paper, never evaluated there): the action
is not executed, the released (output, comment) block with the violated rule as
the critique is appended to the task message, the model re-plans, the new action
is re-checked, and after four re-plans in one step the action is skipped with
the released skip observation. Forced claim selection (prose answers) passes
through the same predicate. Choices: trigger on the conclusion action (the
paper's `agent_finish` cannot see the answer in the released code);
`user_inspection` omitted (no human; a denying inspector equals `stop`).

**ResearchLoop** (`ext_researchloop`). The gated runtime the paper specifies
(Algorithm 1): the request is the research question; C1-C3 are registered in a
claim ledger; one task contract per claim queued in ascending gate order; every
tool call yields an evidence object; after every call code evaluates the gates
(route gates are content-match `all` over their checks; a claim gate is `any`
over its route gates, which keeps alternative routes disjunctive); a contract is
done when its claim is admitted or no route is live; closeout when all contracts
are done, the budget is spent, or the agent reports; binding reports the
admitted claims (the strongest is the stated conclusion; workers cannot promote
a claim). The `full` condition prompt is used with domain nouns swapped; one
planning turn (no budget cost). The prompt-only condition (what the paper's own
LLM studies measured) has no gate and is not run.

**Proof-or-Stop** (`ext_prooforstop`). Lifecycle NONE -> C1 -> C2 -> C3 (DONE).
An `emit_claim` is an advance request; a code gate admits a stage only when
every check of one of its routes is established by a resolver receipt from this
episode (Eq. 2's freshness, integrity, and attestation conjuncts hold by
construction: receipts are written only by the harness and records do not
change within an episode). A refused request returns a block report (per-route,
per-check status: established, failed, missing). At most two repairs (the
pilot's two-round cap; three blocks stop the loop); a re-request without new
evidence stops the loop (no progress). At stop the gate runs once more and the
highest admitted stage is reported. One planning turn (no budget cost) with a
mechanical check that the planned route exists. No published prompts: the
lifecycle text is ours.

**HESP** (`ext_hesp`). Full configuration (repo: `hesp_guard_autostop`): blind
EIG/cost selection, finish guard, controller stop, tau = 0.8. Hypotheses: the
strongest supported conclusion is C0..C3, plus `other`. The paper's
fixed-table naive-Bayes ledger cannot represent route conjunctions (it leaves C3
near 0.56 after every check is read, abstaining on 69% of tasks), so the ledger
is exact Bayesian elimination over the values of the checks on the routes, with
the paper's update rule and eps = 0.01, a prior uniform over the feasible
hypotheses (as in HESP) and, within a hypothesis, check base rates counted on
the development split by topology and check role (HESP's counted development
tables). Probes are field reads (cost 1). The model receives the released
system prompt and request layout, and can only abstain (stop) or have its
finishes checked by the guard; the controller picks every probe and concludes
when the leading hypothesis reaches 0.8 with a supporting observation.
Deviations: greedy decoding (paper: 0.2 with per-call seeds); bf16 instead of
AWQ; ties rounded to 1e-12. Model-free check of this implementation on the 480
tasks (a planner that always asks for an action): false authorization 0.0%,
premature 11.7%, utility 59.2 at 20 calls; utility 58.0 at b_min; identical to
an independent simulation written from the paper.

**GLEAN** (`ext_glean`). A post-hoc verifier of the fully informed agent's
logged held-out trajectories (the Table 1 runs). Steps are (observation, action)
pairs; guidelines are the stated conclusion's routes, one document per route
(K = min(3, routes)); the judge is the same model with the Appendix B.1 prompt
(domain nouns swapped), one YES/NO token with top-10 log-probabilities;
Phi = [min, mean]; beta = 0.5 discounted sum of logits; Bayesian logistic
regression with prior N(0, 1) fitted by random-walk Metropolis (2,000 draws) on
100 development trajectories of the same agent (25 per domain, seeded), labeled
by whether the stated conclusion is supported; trigger when the binary entropy
(nats) exceeds 0.5, then differential checks against two other conclusions'
route documents with alpha = 0.2 and the calibrator not refit. Guideline
expansion is unavailable (every route of the stated conclusion is already a
guideline). Accept when the calibrated probability is at least 0.5; otherwise
state nothing (the paper defines no accept rule; this is the Bayes rule for a
calibrated probability). At b_min the calibrator is the 20-call one (no
development runs at b_min).

## Analyses beyond the main table

- `experiments/analysis/hesp_controller_runs.py`: HESP's controller as a scripted policy
  (a planner that always asks for a probe) at the four budget anchors, on the
  balanced split and (`--split heldout_locator`) its locator-uncertain copy;
  `--prior counted_lodo|uniform|bernoulli` for the prior sensitivity. In the
  full configuration the model can only abstain, and at 20 calls the live
  runs equal the controller's on every task.
- `experiments/analysis/build_controller_table.py`: PRIME's schedule (the ceiling-
  targeting policy) against HESP's controller, paired by task, utility,
  premature conclusions, reads, abstention -> `results/tableCONTROLLER.json`.
- `experiments/analysis/build_locator_hesp_table.py`: the same on the locator-uncertain
  copy, with PRIME's live runs -> `results/tableLOC_HESP.json`.
- `experiments/analysis/stall_cap_replay.py`: Proof-or-Stop and ResearchLoop truncated at
  their fourth turn without an action (PRIME's cap), from the logged traces ->
  `results/stall_cap_replay.json`.
- `experiments/analysis/hesp_prior_sensitivity.py`: the HESP controller under three priors
  -> `results/hesp_prior_sensitivity.json`.
