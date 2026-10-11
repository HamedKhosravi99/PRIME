# Specification construction from an independently authored regulation, tested end to end

Run 3, the run reported here: the cluster job 13948660, 2026-10-10 23:39:59 to
2026-10-11 01:18:13 EDT, one RTX PRO 6000 Blackwell, vLLM 0.24.0,
Qwen2.5-32B-Instruct. 98 minutes of GPU time. Two earlier runs were discarded
before any test label was read; together they used about 30 GPU minutes (see
PROTOCOL.md, deviation log). The protocol and code were committed before any
run (commits 9c8573799 and 17f8bd10f). The harness fixes that followed
runs 1 and 2 are in 39fc17d2e and 5a290e501.

## Design (PROTOCOL.md)

- **Requirements:** 42 CFR Part 11. The creator got the selected definitions of 11.10(a) and the text of 11.22(b), 11.42, and 11.44(a) to (c), as fetched from the eCFR API.
- **Reference:** the FDAAA TrialsTracker export of 3 August 2026. Level 3 means reported late. Level 2 means results due: reported on time, overdue, or overdue-cancelled. Level 1 means not yet due. Every tracker trial counts as applicable.
- **Creation:** Qwen2.5-32B-Instruct, given the regulation, a field catalog, three unlabelled development-pool registrations, and the program format. Each round drew one greedy and eight sampled candidates.
- **Validation:** 600 development-pool registrations, re-fetched in full. Candidates were scored on full-evidence level agreement with the tracker. One repair round showed the creator 12 disagreements. Selection used both rounds, and the selected program was frozen and hashed.
- **Test:** 1,999 held-out registrations from sponsors outside the development pool. They ran under two programs: (A) the paper's assistant-built program and (B) the frozen protocolized program. The fixed ceiling-targeting policy ran on all 1,999. The fully informed agent, PRIME, and PRIME with completion ran on a random subset of 600, drawn before any label was read. Tracker labels for the test set were read only by `score_e2e.py`, after every episode.

## Creation and validation

| round | candidate | valid | agreement % | over | under | atoms | derive errors |
|---|---|---|---|---|---|---|---|
| 0 | 0 | yes | 38.8 | 190 | 177 | 5 | 77 |
| 0 | 1 | yes | 30.7 | 181 | 235 | 7 | 0 |
| 0 | 2 | no: dominated route | | | | | |
| 0 | 3 | yes | 33.2 | 198 | 203 | 8 | 0 |
| 0 | 4 | yes | 30.7 | 181 | 235 | 6 | 59 |
| 0 | 5 | yes | 30.0 | 178 | 242 | 9 | 0 |
| 0 | 6 | yes | **42.0** (selected for repair) | 215 | 133 | 6 | 0 |
| 0 | 7 | yes | 30.3 | 176 | 242 | 8 | 77 |
| 0 | 8 | yes | 33.2 | 198 | 203 | 6 | 59 |
| 1 | 0 | yes | 42.0 | 215 | 133 | 6 | 0 |
| 1 | 1 | yes | 46.8 | 240 | 79 | 6 | 0 |
| 1 | 2 | yes | 42.0 | 215 | 133 | 6 | 0 |
| 1 | 3 | yes | **71.0** (frozen) | 41 | 133 | 5 | 0 |
| 1 | 4 | yes | 30.0 | 178 | 242 | 6 | 0 |
| 1 | 5 | yes | 38.8 | 192 | 175 | 5 | 0 |
| 1 | 6 | yes | 42.0 | 215 | 133 | 5 | 0 |
| 1 | 7 | yes | 42.0 | 215 | 133 | 5 | 0 |
| 1 | 8 | yes | 46.8 | 240 | 79 | 5 | 0 |

Derive errors are records on which the candidate's own `derive` raised, mostly when it applied a day-precision date format to month-precision dates. Per the protocol, those records fail closed.

**Frozen program** (round 1, candidate 3; sha256 7cbae0727ca28b8c...; `candidates/FROZEN_PROGRAM.json`):

| atom | check |
|---|---|
| A1 | studyType equals INTERVENTIONAL |
| A2 | derived applicability is true: interventional, and an FDA-regulated device or drug |
| A3 | primary completion date after 2017-01-17 |
| A4 | derived deadline, primary completion plus 365 days, before 2026-08-03 |
| A5 | derived late submission is true: first results submission after that deadline |

The routes are nested: C1 needs A1 and A2, C2 adds A3 and A4, and C3 adds A5.

The program omits the phase-1 and feasibility exclusions of 11.22(b). It also omits the U.S.-connection condition: a U.S. site, a U.S.-exported product, or an IND or IDE number. It ignores the certification extensions of 11.44(b) and (c). Its date rule, primary completion on or after 18 January 2017, follows 11.42.

## Test results

Reference levels over the 1,999 test registrations:

| level | registrations |
|---|---|
| 1, not yet due | 887 |
| 2, due | 755 |
| 3, reported late | 357 |

Program A built 526 instances. It excluded 370 registrations as not applicable, 750 as not yet due, and 353 as undeterminable. Program B built 1,999.

| program | policy | n | unsupported % [95% CI] | utility | coverage % |
|---|---|---|---|---|---|
| A, assistant-built | fixed, no LLM | 1,999 | 1.8 [1.3, 2.5] | 18.4 | 26.3 |
| | fixed, no LLM | 600 | 1.7 [0.9, 3.0] | 19.0 | 26.8 |
| | fully informed agent | 600 | 0.2 [0.0, 0.9] | 13.1 | 26.8 |
| | PRIME | 600 | 0.2 [0.0, 0.9] | 9.9 | 26.8 |
| | PRIME, completion | 600 | 1.7 [0.9, 3.0] | 19.0 | 26.8 |
| B, protocolized | fixed, no LLM | 1,999 | 3.5 [2.8, 4.4] | 47.5 | 92.4 |
| | fixed, no LLM | 600 | 3.2 [2.0, 4.9] | 47.7 | 92.5 |
| | fully informed agent | 600 | 0.0 [0.0, 0.6] | 31.2 | 92.5 |
| | PRIME | 600 | 0.3 [0.1, 1.2] | 32.3 | 92.5 |
| | PRIME, completion | 600 | 3.2 [2.0, 4.9] | 47.7 | 92.5 |

Secondary numbers come from `analyze_e2e.py`, saved in `reported/analysis.json`:

| measure | A | B |
|---|---|---|
| full-evidence level agreement, all 1,999 | 22.8% | 81.6% |
| level agreement, the 1,112 due | 40.9% | 71.6% |
| level agreement, the 887 not due | 0.0% | 94.3% |
| fixed-policy coverage, the 1,112 due | 44.4% | 87.1% |
| unsupported conclusions, fixed policy, all | 36 | 70 |
| ... of which a certification was filed | 16 | 55 |
| adverse-against-adverse agreement | 518/526 (98.5%) | 843/922 (91.4%) |

- **Where the unsupported conclusions come from.** Program A states C2 for 25 registrations the tracker keeps ongoing. It also states C2 for 7 that the tracker lists as reported but not yet due, and C3 for 4. Program B states C3 for 31 the tracker counts as reported on time, and C2 for 31 and C3 for 5 that the tracker keeps ongoing. Most of these turn on the certification extension that program B omits. Every unsupported conclusion is C2 or C3, because every test registration is tracker-applicable.
- **Classification agreement.** For program A it comes from its builder's verdicts and reproduces the paper's held-out figure. Program B has no compliant or overdue verdict. For it, a due-but-not-late registration counts as compliant when the registry records a results submission, and overdue otherwise.
- **Validation agreement is lower than test agreement.** The validation sample is almost all due (587 of 600; 262 reported, 221 reported late, 117 overdue). The test sample is 56% due. On due test registrations the frozen program agrees on 71.6%, its validation figure. The not-yet-due registrations, which validation lacks, are easy for it. Certifications are 9.7% of validation registrations and 4.8% of test registrations.

## Runtime check and cost (agent subset)

| program | arm | episodes | errors | emissions | without route | calls/ep | completion calls/ep | tokens/ep |
|---|---|---|---|---|---|---|---|---|
| A | fully informed agent | 161 | 0 | 237 | 0 | 3.99 | 0 | 6,785 |
| A | PRIME | 161 | 0 | 161 | 0 | 3.20 | 0 | 10,791 |
| A | PRIME, completion | 161 | 0 | 161 | 0 | 5.04 | 1.84 | 10,768 |
| B | fully informed agent | 600 | 0 | 561 | 0 | 2.95 | 0 | 3,722 |
| B | PRIME | 600 | 0 | 555 | 0 | 2.84 | 0 | 9,425 |
| B | PRIME, completion | 600 | 0 | 555 | 0 | 4.54 | 1.68 | 9,441 |

Program A has 161 agent episodes because only 161 of the 600 subset registrations have an A instance. The other 439 count as abstentions.

Phase times, run 3:

| phase | time |
|---|---|
| server start | 75 s |
| creation, round 0 | 62 s |
| validation, round 0 | 2 s |
| creation, round 1 | 49 s |
| validation and freezing | 3 s |
| test instances | 36 s |
| B arms: PRIME with completion, fully informed agent, PRIME | 27.9, 17.6, and 27.8 min |
| A arms, same order | 7.2, 6.9, and 6.1 min |
| scoring | 1 min |

## What the results say

- **Coverage.** The protocolized open-model workflow produced a program that covers nearly every registration. It states a conclusion for 92.4% of the 1,999, with 3.5% unsupported against the tracker and utility 47.5. The assistant-built program states one for 26.3%, with 1.8% unsupported and utility 18.4, because its builder only makes instances for registrations it classifies as applicable and due.
- **The errors are the program's.** Program B's errors come mostly from the certification extension it omits. Program A's come mostly from registrations it counts as due that the tracker keeps ongoing.
- **The runtime adds no error of its own.** No arm emitted a conclusion without an established route. PRIME with completion matched each program's fixed policy on the same 600 registrations exactly.
- **Stopping early.** PRIME without completion and the unenforced fully informed agent stopped at weaker conclusions. They lost a third of the utility and avoided most of the program's errors, which sit at C2 and C3.
- **Scope.** This covers one regulation, one creator model, and one greedy run per policy. The reference is a tracker's reading of the rule, not a legal adjudication. Every test registration is one the tracker follows, so the test cannot penalize a program for calling an inapplicable trial applicable. That is the error the omitted 11.22(b) conditions would cause.
