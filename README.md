# PRIME and ObligationBench

Code, benchmark, and analysis for the paper "PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents" (under review).

PRIME fixes, before an LLM agent examines a case, which checks suffice for each conclusion. It then records what the agent's checks establish, points the agent at the checks that remain, and refuses any conclusion that the completed checks do not justify. ObligationBench, the paper's benchmark, has 938 tasks built from public security advisories, clinical-trial registrations, drug recalls, and company filings, plus derived variants, scored by executable rules with no LLM judge. The benchmark is also distributed on its own, with a dataset card and an installable scorer.

## Contents

| Path | What it is |
|---|---|
| `prime_runtime/` | the runtime: commitment, ledger, authorization, guidance, and the renderer; its tests are in `prime_runtime/tests/` |
| `runner/` | the agents (the ReAct ladder, the comparators, the closest external systems, and PRIME) and the batch driver |
| `bench/` | record fetchers, the task generators, the validator, and the tasks: `instances_v2/` (ObligationBench), `instances_fdaaa/` (the regulation set), and `instances/` (the hand-built tasks of the exploratory pilot) |
| `bench/harvest/` | the harvested records of 2026-08-01 that the generators read, reduced to the fields they use (10 MB) |
| `bench/artifacts/` | the sources the hand-built tasks read: open-source Python projects at pinned commits, arXiv titles and abstracts, and public records |
| `analysis/` | the scripts that compute every table and quoted number, and the statistics (`stats_v5.py`) |
| `figures/`, `paper/` | the scripts that draw the figures and write the LaTeX tables from `results/` |
| `experiments/` | the regulation study (`specbuild/`, `fdaaa_heldout/`) and the verified-attestation runs (`blind_verification/`) |
| `obligationbench.py` | a standard-library scorer: loads tasks, lets an agent read a record, scores its conclusions |
| `reproduce_tables.py` | rebuilds the per-agent endpoints of the information-versus-enforcement comparison (Section 6.1, Table 1, Figure 3) from the logs and checks them against the paper |
| `logs/prime-episode-logs.tar.gz` | every episode log behind the paper, and the JSON each analysis script writes (`results/`) |
| `cluster/` | the SLURM scripts that served each model with vLLM and ran the agents |
| `docs/` | `ANALYSIS_PLAN.md` (frozen before any confirmatory episode was scored), `EXECUTION_PLAN.md` (with the dated log of deviations, Part 6b), `GLOSSARY.md` (the code's names, run folders, and the paper's terms), `EXTSYS.md` (how the closest external systems were instantiated), `ENVIRONMENT.md` (the software and hardware of the runs), `KNOWN_ISSUES.md` |

## Setup

Python 3.9 or later. The runtime, the scorer, the generators, and most analysis scripts use only the standard library; the figures need numpy and matplotlib, the tests pytest, and running agents the `openai` client and an OpenAI-compatible endpoint.

```bash
pip install -r requirements.txt        # or: pip install -e ".[figures,dev]"
make test                              # the runtime's tests
```

## Reproducing the paper

Three levels, with different needs. `make` runs the first two.

**1. Tables and figures from the logs (CPU, minutes; no model).** Extract the logs, then rebuild and check:

```bash
make logs                              # extracts logs/prime-episode-logs.tar.gz into results/
make tables                            # reproduce_tables.py, then every table regenerated and compared
make figures                           # recomputes every plotted value and checks it against the paper
make conformance                       # the standalone scorer against the runtime, on every task and episode
```

`reproduce_tables.py` rebuilds each agent's predictions from its trace, using the stated conclusions and the record reads only. It scores them with `obligationbench.py` and checks false authorization, premature conclusions, supported utility, emission, and evidence value against the paper for 15 agents. It also accounts for every episode: an episode whose model calls still timed out after the client's retries carries no measurement and is left out of the rates. There are 24 such episodes, declared in the script, and the script fails if any other episode is missing. `analysis/episode_accounting.py` lists the failed episodes of every run.

**2. The benchmark from its sources (CPU, seconds).**

```bash
make corpus                            # python3 bench/reproduce_corpus.py
```

This regenerates all 1,898 tasks from `bench/harvest/` and compares each with the published file byte for byte: 938 primary, 240 one-branch, 480 locator-uncertain, and 240 text-evidence tasks. The generators are deterministic: their family enumeration uses pinned seed offsets (`bench/generate_record_instances.py`). `bench/reduce_harvest.py` documents how the shipped harvest was reduced from the full one, and `docs/KNOWN_ISSUES.md` gives the history. The regulation set is built by `bench/fdaaa/build_fdaaa_instances.py`; one of its inputs is a third-party table that is not included (see Data provenance).

**3. Rerunning the agents (GPU).** See Running an agent. Greedy decoding on a served model is close to deterministic but not exactly so, and `docs/KNOWN_ISSUES.md` lists the episodes in which the released runner's budget accounting differs from the logged runs; the 139 episodes it affected were rerun.

## From the paper to the code

Numbers in the paper are computed by the scripts below from the episode logs; each writes the JSON file named in the last column (under `results/`, which `make logs` extracts). `docs/GLOSSARY.md` maps the code's names (arms, run folders such as `w2a`, budget anchors) to the paper's terms.

| Paper item | Code | Output in `results/` |
|---|---|---|
| Table 1, information versus enforcement (Section 6.1) | `analysis/build_w2a_tables.py`; `reproduce_tables.py` checks every row | `tableT1.json`, `tableT2.json` |
| Figure 2, utility by budget with named and with uncertain evidence fields (Section 6.2) | `paper/make_fig_budget.py`, which recomputes the curves with `analysis/build_w2a_tables.py` and `analysis/policy_frontier.py` | `tableLOC.json`, `tableLOC_extra.json`, `tableCOMPLETE.json`, `frontier_policies.json` |
| Section 6.2: share of utility from the runtime's statement; live completion | `analysis/build_w2a_tables.py`, `analysis/build_complete_table.py` | `utility_attribution.json`, `tableCOMPLETE.json` |
| Section 6.2: first-read accuracy per model | `analysis/locator_first_reads.py` | `locator_first_reads.json` |
| Table 2, judging textual evidence (Section 6.3) | `analysis/build_text_table.py`; the verified condition: `experiments/blind_verification/` | `tableTEXT.json` |
| Table 3, the closest control systems (Section 6.4) | `runner/extsys.py` and `runner/glean.py` (instantiations: `docs/EXTSYS.md`); `analysis/build_extsys_table.py`; `paper/make_tab_extsys.py` | `tableEXTSYS.json`, `glean/diagnostics_qwen2.5-32b_B20.json` |
| Table 4, conformance tests (Appendix B) | `prime_runtime/tests/` | |
| Table 6, atoms of the running example `sec-v2-0114` (Appendix C) | `bench/generate_record_instances.py` | |
| Figure 3, the full information ladder (Appendix E.1) | `paper/make_naacl_appendix_figures.py`, through `figures/make_figures.py` | `tableT1.json`, `tableT2.json` |
| Table 8, information versus enforcement with other models; sampled decoding (Appendix E.1) | `analysis/build_w2a_tables.py --models <tags>`; sampled decoding: `fill_SeedNote` in `analysis/fill_v2_tables.py` (reads the `w2seed` logs) | `tableT1.json`, `tableT2.json` |
| Figure 4, guidance gain in each registered cell (Appendix E.2) | `figures/make_figures.py` | `tableT2.json` (`control_law`) |
| Figure 5, after-the-fact filter (Appendix E.2) | `paper/make_naacl_appendix_figures.py` | `rejection_bound.json`, `frontier_policies.json`, `tableEXT.json` |
| Appendix E.2, flat gate | `analysis/build_flat_table.py` | `tableFLAT.json` |
| Table 9, information versus enforcement, guidance, and evidence location on three models (Appendix E.3) | `analysis/build_multimodel_table.py`; `paper/make_tab_multimodel.py` | `tableMULTI.json` |
| Appendix E.3, live completion on the locator-uncertain split | `analysis/build_locator_table.py`, `analysis/build_locator_extra.py` | `tableLOC.json`, `tableLOC_extra.json` |
| Table 10, the closest systems in full (Appendix E.5) | `analysis/build_extsys_table.py`; `paper/make_tab_extsys.py` | `tableEXTSYS.json` |
| Table 11, PRIME against HESP's controller (Appendix E.5) | `analysis/hesp_controller_runs.py` (`--split heldout_locator` for the right half), `analysis/build_controller_table.py`, `analysis/build_locator_hesp_table.py`; `paper/make_tab_extsys.py` | `tableCONTROLLER.json`, `tableLOC_HESP.json` |
| Appendix E.5, utility crediting only established conclusions | `analysis/established_utility.py` | `established_utility_hesp.json` |
| Appendix E.5, sensitivity of HESP's prior and threshold; stall cap | `analysis/hesp_prior_sensitivity.py`, `analysis/hesp_controller_runs.py --prior counted_lodo`, `analysis/stall_cap_replay.py` | `hesp_prior_sensitivity.json`, `hesp_tau_sensitivity.json`, `stall_cap_replay.json` |
| Tables 12 and 14, the regulation and the two construction workflows (Appendix F) | `experiments/specbuild/` (`create.py`, `validate.py`, `build_test.py`, `score_e2e.py`, `analyze_e2e.py`) | `experiments/specbuild/reported/e2e.json`, `analysis.json`; agent episodes in `results/specbuild_e2e/` |
| Tables 13, 15, and 16, held-out attrition, the assistant-built program, and its confusion table (Appendix F) | `analysis/build_fdaaa_table.py`; `bench/fdaaa/` | `tableFDAAA.json`, `fdaaa_confusion.json` |
| Table 17, the exploratory pilot on hand-built tasks (Appendix G) | `analysis/end_to_end_tables.py` | |
| Family-level intervals and tests, everywhere | `analysis/stats_v5.py` | |
| Episodes left out of the rates | `analysis/episode_accounting.py` | `excluded_episodes.json` |

## Running an agent

Serve an open-weight model with vLLM; the tool-call parser flags are required, because every agent acts through the OpenAI tools API.

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

Then point the runner at it:

```bash
export PRIME_LLM_BASE=http://127.0.0.1:8000/v1 PRIME_LLM_KEY=local
python3 runner/run_batch.py --instances "bench/instances_v2/security/heldout_balanced/*.json" \
  --arm prime_guided --license-source gold --model Qwen/Qwen2.5-32B-Instruct \
  --max-tool-calls 20 --out results/mine/prime_guided__B20
```

`--dry-run` replaces the model with a scripted one and makes no API calls. The runner refuses paid commercial endpoints unless `PRIME_ALLOW_PAID_API=1` is set. `--budget-anchor bmin|bmin1|2bmin` sets the per-task budgets of Section 6.2.

The arms of the paper are:
- the information ladder of Section 6.1: `react`, `react_verify`, `react_spec`, `react_spec_verify`, `react_program`, `react_program_verify`, and `react_program_attested` (the fully informed agent);
- PRIME: `masked_reactive_v` (PRIME without guidance), `prime_guided` (PRIME), and `prime_complete` (PRIME with completion);
- the closest control systems of Section 6.4: `ext_agentspec`, `ext_agentspec_examine`, `ext_researchloop`, `ext_prooforstop`, `ext_hesp`, and GLEAN as an offline verifier (`runner/glean.py`).

The scripted policies run without a model through `runner/policies.py`.

## Data provenance

The records come from public APIs and are turned into tasks as follows:
- `bench/fetch_osv.py`, `bench/fetch_ctgov.py`, `bench/fetch_openfda.py`, and `bench/fetch_edgar.py` retrieve the records;
- `bench/harvest_records.py` gathers them into the harvest;
- `bench/generate_record_instances.py` and `bench/generate_text_instances.py` turn them into tasks.

Records change, so a new harvest yields a different corpus. The published tasks embed the snapshots retrieved on 1 August 2026, 3 August 2026 for the regulation set, and `bench/harvest/` holds the harvest they were generated from. The SEC requires a contact address in the User-Agent of every request; set `SEC_USER_AGENT` before running `fetch_edgar.py`.

`bench/fdaaa/build_fdaaa_instances.py` reads two inputs:
- a ClinicalTrials.gov snapshot, which is included;
- the FDAAA TrialsTracker's trial table of 3 August 2026, which belongs to the Bennett Institute for Applied Data Science and is not included. The tracker publishes its current table at https://fdaaa.trialstracker.net.

`bench/fdaaa/external/SHA256SUMS.txt` pins both.

The hand-built tasks read `bench/artifacts/`:
- open-source Python projects at pinned commits, redistributed under their own licenses (each project directory keeps its license file);
- arXiv titles and abstracts, which arXiv releases under CC0;
- public records from the same sources as the benchmark.

Each artifact's `PROVENANCE` file records where and when it was retrieved.

## Known issues

`docs/KNOWN_ISSUES.md` lists where the released code differs from the code that produced the logs:
- budget accounting after a repeated call: `analysis/budget_accounting_audit.py` found the 139 episodes whose outcome it could change, and all were rerun with the corrected runner (the originals ship in `results/superseded_by_rerun/`); the paper's numbers include the reruns;
- attestations refused before a read, which affects no logged episode.

It also explains the excluded episodes, and says what the one-branch variant's family ids mean.

## License

The code is released under the MIT License (`LICENSE`). The benchmark data are released under CC BY 4.0, and the records remain subject to their sources' terms (`NOTICE.md`).

## Citation

The citation will be added when the paper is published.
