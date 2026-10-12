# From the code's names to the paper's terms

The code was written before the paper's wording settled, so it often uses older names. This file maps the names that appear in the code, the logs, and the plans to the terms of the paper.

## Concepts

| In the code | In the paper |
|---|---|
| claim, claim `level` (1 to 3) | conclusion, and its strength |
| atom (`E1`, `E2`, ...) | check |
| resolver, `attest` rule | the executable predicate that settles a check on the record |
| license, route, `gold_licenses` | route: a minimal set of checks that suffices for a conclusion |
| program | the review program $P_0$: the conclusions, their order, and their routes |
| `license_source=gold`, gold program | the committed program supplied with the task (every experiment of the paper) |
| `license_source=induced`, induced program, induction | routes learned by the model (an earlier study, not in the paper) |
| open vocabulary, `openvocab` | rules constructed by the model (an earlier study, not in the paper) |
| commitment, `CommitmentGate` | precommitment: the program is checked, hashed, and frozen before any resolver may run |
| ledger, attestation | the record of check results; an attestation is one settled check |
| `auth_ids`, authorized claims | the conclusions the ledger authorizes, closed downward under the strength order |
| `max_auth` | the strongest authorized conclusions (MaxAuth) |
| mask, refused emission | the runtime's refusal of a conclusion outside the authorized set |
| reachable ceiling, active obligation, termination rule, stall | guidance: the strongest conclusion still reachable within the budget, the open checks on its cheapest live route, and the reminder given when the agent stops early |
| `emit_claim`, emission | stating a conclusion |
| `phi` | the constraint formula $\Phi_x$, empty in the benchmark |
| instance | task |
| seed | the record a task is built from |
| family | template family |
| topology `nested` / `branching` | one route per conclusion / alternative routes for some conclusion |
| `__noev` variant, `resolver: null` | a check whose evidence is unavailable |

## Endpoints

| In the code | In the paper |
|---|---|
| OFAR-A, `ofar_a` | false authorization |
| OFAR-B, `ofar_b`, $F^*_B$ | false authorization when every check of every route is required (a checklist) |
| `complete_auth_gold == 0` | premature conclusion |
| `utility` | supported utility |
| `rejection_ceiling` | evidence value: the utility of the strongest conclusion the agent's final ledger supports |
| `emit_rate_avail` | emission rate on answerable tasks |
| content-false | a stated conclusion whose content is false on the record |
| degenerate cell | an agent that states anything on fewer than 25% of answerable tasks, so its low false authorization reflects silence |

## Agents

| Arm | In the paper |
|---|---|
| `react` | ReAct |
| `react_verify` | ReAct +verify |
| `react_spec` | ReAct +spec: told the conclusions and their checks |
| `react_spec_verify` | ReAct +spec+verify |
| `react_program` | ReAct +program: told the rule, routes included |
| `react_program_verify` | ReAct +program+verify |
| `react_program_attested` | fully informed, unenforced: told the rule and shown its own ledger |
| `masked_reactive_v` | PRIME without guidance |
| `prime_guided` | PRIME |
| `prime_complete` | PRIME with completion |
| `prime_flat` | the flat gate of Appendix E.2: one conclusion, no grading |
| `ext_agentspec`, `ext_agentspec_examine`, `ext_researchloop`, `ext_prooforstop`, `ext_hesp` | the closest control systems of Section 6.4 (AgentSpec with stop and with self-examination, ResearchLoop, Proof-or-Stop, HESP); GLEAN is `runner/glean.py` |
| `direct`, `rubric`, `cove`, `reflexion`, `conformal`, `static_contract` | Direct, Rubric, CoVe, Reflexion, Conformal, Static checklist |
| `prime`, `masked_reactive` | the unguided PRIME agents of the hand-built pilot |
| `prime_guided_gated` | guided PRIME restricted to conclusions whose learned routes passed validation |
| `prime_nonminimal` | PRIME with redundant checks added to its routes |
| `shortest_route` | the ceiling-targeting policy (no LLM) |
| `random_masked` | the random-order policy (no LLM) |
| `cheapest_literal`, `masked_reactive_prefix`, `informed_route_planner`, `clairvoyant_ceiling` | other scripted policies, not in the paper |

## Budgets

| In the code | In the paper |
|---|---|
| `B20` | 20 tool calls |
| `bmin`, `bmin1`, `2bmin` | the per-task budget anchors $b_{\min}$, $b_{\min}+1$, and $2b_{\min}$ |

## Folders of `results/`

| Folder | Contents |
|---|---|
| `w2a/` | the balanced split: Sections 6.1 and 6.2 with Qwen2.5-32B-Instruct (`qwen2.5-32b-<domain>/<arm>__<budget>`) and with the other models (`gptoss-20b-*`, `mistral-small-24b-*`, `qwen2.5-7b-*`); PRIME with completion (`prime_complete__*`); the flat gate (`prime_flat__B20`); the closest control systems of Section 6.4 (`ext_*`); the natural split (`natural__*`, not reported); the scripted policies (`policy-*`) |
| `w2loc/` | evidence location (Section 6.2), the locator-uncertain copy |
| `w2text/` | judging textual evidence (Section 6.3), the text-evidence split; the verified condition is in `experiments/blind_verification/results/` |
| `w2seed/` | sampled decoding at temperature 0.7, three seeds (Appendix E.1) |
| `glean/` | the GLEAN verifier's diagnostics (Section 6.4) |
| `w2ext/` | six further baselines (Direct, Rubric, CoVe, Reflexion, Conformal, Static checklist), checked by `reproduce_tables.py`; not reported in the paper |
| `fdaaa/`, `specbuild_e2e/` | the regulation study (Appendix F) |
| `v2dev/` | development-split runs that set the minimum effect for guidance (Appendix D) |
| `w2ob/` | the one-branch variant (not reported in the paper) |
| `w2b/`, `induction_*.json`, `openvocab_*` | learned routes and constructed rules (an earlier study, not in the paper) |
| `qwen2.5-*-<domain>/`, `gpt-*`, `tableC_*.json` | the exploratory pilot on hand-built tasks (Appendix G) |
| `w1v5/` | early runs of the ladder agents on the hand-built tasks, not used in the paper |
| `*.json` at the top | the output of each analysis script; README.md maps them to the paper |

## Experiment letters

Earlier drafts lettered the experiments, and some reports and comments still do: A is Section 6.1 (information versus enforcement), B and C are Section 6.2 (guidance, and evidence location), D is Section 6.3 (judging textual evidence), and E is Section 6.4 (the closest control systems).

## Plans and notes

| In the code | Meaning |
|---|---|
| FREEZE F1 to F9 | sections of `docs/ANALYSIS_PLAN.md` |
| FB-01, FB-02, ... | entries of the deviations log, `docs/EXECUTION_PLAN.md` Part 6b |
| W0, W1, W2a, W2b, W3 | stages of `docs/EXECUTION_PLAN.md`: preconditions on development data, agent additions, confirmatory runs, learned routes, and validation gates |
| GPT Q..., Claude N... (in `docs/EXECUTION_PLAN.md`) | items raised by language-model reviews of the execution plan, with their dispositions |
| `PRIME-master.md` and "master §...", `SPEC.md`, `*_REPORT.md`, "the design notes" | internal design documents, not included; the paper and its appendices state everything the results rely on |

## Scripts kept from an earlier draft

`analysis/fill_v2_tables.py` and `analysis/aggregate.py` were written for an earlier draft of the paper and read or write its LaTeX source, which is not included. They are kept because `figures/make_figures.py`, `analysis/end_to_end_tables.py`, `analysis/drift_table.py`, and one test import functions from them. Scripts that only filled that draft are not included.
