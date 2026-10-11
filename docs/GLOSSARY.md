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
| `license_source=gold`, gold program | the supplied rule (S3) |
| `license_source=induced`, induced program, induction | routes learned by the model (S2) |
| open vocabulary, `openvocab` | the rule constructed by the model (S1) |
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
| `direct`, `rubric`, `cove`, `reflexion`, `conformal`, `static_contract` | Direct, Rubric, CoVe, Reflexion, Conformal, Static checklist |
| `prime`, `masked_reactive` | the unguided PRIME agents of the hand-built pilot |
| `prime_guided_gated` | guided PRIME restricted to conclusions whose learned routes passed validation |
| `prime_nonminimal` | PRIME with redundant checks added to its routes |
| `shortest_route` | the ceiling-targeting policy (no LLM) |
| `cheapest_literal`, `random_masked`, `masked_reactive_prefix`, `informed_route_planner`, `clairvoyant_ceiling` | the other fixed policies of Experiment 3 |

## Budgets

| In the code | In the paper |
|---|---|
| `B20` | 20 tool calls |
| `bmin`, `bmin1`, `2bmin` | the per-task budget anchors $b_{\min}$, $b_{\min}+1$, and $2b_{\min}$ |

## Folders of `results/`

| Folder | Contents |
|---|---|
| `w2a/` | confirmatory runs with the supplied rule (Experiments 1, 2, 3, and 6). `natural__` runs use the held-out natural split, `policy-*` holds the fixed policies, and `gptoss-20b-*` and `qwen2.5-7b-*` hold the other models |
| `w2ext/` | the comparators (Experiment 5) |
| `w2seed/` | reruns sampled at temperature 0.7 (Experiment 6) |
| `w2b/`, `induction_*.json` | learned routes (Experiment 7) |
| `openvocab_endtoend_*.json`, `*-openvocab/` | constructed rules (Experiment 8) |
| `v2dev/`, `w0_pilots_*.json` | development-split runs that set the pre-registered margin |
| `fdaaa/`, `tableFDAAA.json` | the FDAAA set |
| `qwen2.5-*-<domain>/`, `gpt-*`, `tableC_*.json` | the pilot on hand-built tasks (e-companion) |
| `w1v5/` | early runs of the ladder agents on the hand-built tasks, not used in the paper |

## Plans and notes

| In the code | Meaning |
|---|---|
| FREEZE F1 to F9 | sections of `docs/ANALYSIS_PLAN.md` |
| FB-01, FB-02, ... | entries of the deviations log, `docs/EXECUTION_PLAN.md` Part 6b |
| W0, W1, W2a, W2b, W3 | stages of `docs/EXECUTION_PLAN.md`: preconditions on development data, agent additions, confirmatory runs, learned routes, and validation gates |
| GPT Q..., Claude N... | comments from language-model reviewers of the execution plan |
| `PRIME-master.md` and "master §...", `SPEC.md`, `*_REPORT.md`, "the design notes" | internal design documents, not included; the paper and its e-companion state everything the results rely on |

## Scripts kept from an earlier draft

`analysis/fill_v2_tables.py` and `analysis/aggregate.py` were written for an earlier draft of the paper and read or write its LaTeX source, which is not included. They are kept because `figures/make_figures.py`, `analysis/end_to_end_tables.py`, `analysis/drift_table.py`, and one test import functions from them. Scripts that only filled that draft are not included.
