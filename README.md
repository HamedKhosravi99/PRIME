# PRIME and SUFFICE

Code, benchmark, and analysis for the paper "PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents" (under review).

PRIME fixes, before an LLM agent examines a case, which checks suffice for each conclusion. It then records what the agent's checks establish, points the agent at the checks that remain, and refuses any conclusion that the completed checks do not justify. SUFFICE, the paper's benchmark, has 938 tasks built from public security advisories, clinical-trial registrations, drug recalls, and company filings, plus derived variants, scored by executable rules with no LLM judge. The benchmark is also distributed on its own, with a dataset card and an installable scorer.

## Contents

```
src/                    code you import
  prime_runtime/        the runtime: commitment, ledger, authorization, guidance, and the renderer
  prime_agents/         the agents (the ReAct ladder, the comparators, the closest external systems, and PRIME) and the batch driver
  suffice/              the standard-library scorer: loads tasks, lets an agent read a record, scores its conclusions; the task schema and validator
    construction/       record fetchers, the harvest, the task generators, the corpus check, and the regulation-set builder (fdaaa/)
data/                   everything shipped as data (see data/README.md and data/NOTICE.md)
  suffice/              SUFFICE: 1,898 tasks in four domains and seven splits
  fdaaa/                the regulation set (instances/) and its ClinicalTrials.gov snapshot (external/)
  harvest/              the harvested records of 2026-08-01 that the generators read, reduced to the fields they use (10 MB)
  pilot/                the hand-built tasks of the exploratory pilot (instances/) and the sources they read (artifacts/)
  episode_logs/         every episode log behind the paper, and the JSON each analysis script writes
experiments/            everything that turns the data into the paper's numbers
  reproduce_tables.py   rebuilds the per-agent endpoints of Section 7.1 (Table 2, Figure 3) from the logs and checks them
  analysis/             the scripts that compute every table and quoted number, and the statistics (stats_v5.py); pilot/ holds the Appendix G scripts
  paper/                the scripts that draw the figures (into figures/) and write the LaTeX tables
  specbuild/, fdaaa_heldout/   the regulation study
  blind_verification/   the verified-attestation runs
  slurm/                the SLURM scripts that served each model with vLLM and ran the agents
tests/                  the conformance and regression tests (Table 7)
docs/                   the frozen analysis and execution plans, the glossary, and the notes below
```

`docs/` holds `ANALYSIS_PLAN.md` (frozen before any confirmatory episode was scored), `EXECUTION_PLAN.md` (with the dated log of deviations, Part 6b), `GLOSSARY.md` (the code's names, run folders, and the paper's terms), `EXTSYS.md` (how the closest external systems were instantiated), `ENVIRONMENT.md` (the software and hardware of the runs), `KNOWN_ISSUES.md`, and `PATH_MAP.md`. The frozen plans, the episode logs, and the pilot tasks name files by the paths they had when they were written; `PATH_MAP.md` maps those to the current layout.

## Setup

Python 3.9 or later. The runtime, the scorer, the generators, and most analysis scripts use only the standard library; the figures need numpy and matplotlib, the tests pytest, and running agents the `openai` client and an OpenAI-compatible endpoint.

```bash
pip install -r requirements.txt        # or: pip install -e ".[figures,dev]"
make test                              # the tests
```

Every script runs from a fresh checkout without installing anything: it puts `src/` on its own path. `pip install -e .` also makes `prime_runtime`, `prime_agents`, and `suffice` importable from anywhere; with it (or with `PYTHONPATH=src`), `python -m suffice score --tasks ... --predictions ...` runs the scorer.

## Reproducing the paper

Three levels, with different needs. `make` runs the first two.

**1. Tables and figures from the logs (CPU, minutes; no model).** Extract the logs, then rebuild and check:

```bash
make logs                              # extracts data/episode_logs/prime-episode-logs.tar.gz into results/
make tables                            # experiments/reproduce_tables.py, then every table regenerated and compared
make figures                           # recomputes every plotted value and checks it against the paper
make conformance                       # the standalone scorer against the runtime, on every task and episode
```

`experiments/reproduce_tables.py` rebuilds each agent's predictions from its trace, using the stated conclusions and the record reads only. It scores them with `src/suffice/` and checks false authorization, premature conclusions, supported utility, emission, and evidence value against the paper for 15 agents. It also accounts for every episode: an episode whose model calls still timed out after the client's retries carries no measurement and is left out of the rates. There are 24 such episodes, declared in the script, and the script fails if any other episode is missing. `experiments/analysis/episode_accounting.py` lists the failed episodes of every run.

**2. The benchmark from its sources (CPU, seconds).**

```bash
make corpus                            # python3 src/suffice/construction/reproduce_corpus.py
```

This regenerates all 1,898 tasks from `data/harvest/` and compares each with the published file byte for byte: 938 primary, 240 one-branch, 480 locator-uncertain, and 240 text-evidence tasks. The generators are deterministic: their family enumeration uses pinned seed offsets (`src/suffice/construction/generate_record_instances.py`). `src/suffice/construction/reduce_harvest.py` documents how the shipped harvest was reduced from the full one, and `docs/KNOWN_ISSUES.md` gives the history. The regulation set is built by `src/suffice/construction/fdaaa/build_fdaaa_instances.py`; one of its inputs is a third-party table that is not included (see Data provenance).

**3. Rerunning the agents (GPU).** See Running an agent. Greedy decoding on a served model is close to deterministic but not exactly so, and `docs/KNOWN_ISSUES.md` lists the episodes in which the released runner's budget accounting differs from the logged runs; the 139 episodes it affected were rerun. Four cells were run afterwards with the corrected runner, as new cells rather than reruns: the fully informed agent on the locator-uncertain split at $b_{\min}$ for the three models (Table 11) and PRIME at $b_{\min}{+}1$ on that split with Qwen (Table 13); `results/new_cells.md` lists them.

## From the paper to the code

Numbers in the paper are computed by the scripts below from the episode logs; each writes the JSON file named in the last column (under `results/`, which `make logs` extracts). `docs/GLOSSARY.md` maps the code's names (arms, run folders such as `w2a`, budget anchors) to the paper's terms.

| Paper item | Code | Output in `results/` |
|---|---|---|
| Table 2, information versus enforcement (Section 7.1) | `experiments/analysis/build_w2a_tables.py`; `experiments/reproduce_tables.py` checks every row | `tableT1.json`, `tableT2.json` |
| Table 3, other models (Section 7.1); sampled decoding (Section 7.1, Appendix E.1) | `experiments/analysis/build_w2a_tables.py --models <tags>`; sampled decoding: `fill_SeedNote` in `experiments/analysis/fill_v2_tables.py` (reads the `w2seed` logs) | `tableT1.json`, `tableT2.json` |
| Figure 2, utility by budget with known and with uncertain evidence locations (Section 7.2) | `experiments/paper/make_fig_budget.py`, which recomputes the curves with `experiments/analysis/build_w2a_tables.py` and `experiments/analysis/policy_frontier.py` | `tableLOC.json`, `tableLOC_extra.json`, `tableCOMPLETE.json`, `frontier_policies.json` |
| Section 7.2, completion on the balanced split | `experiments/analysis/build_complete_table.py` | `tableCOMPLETE.json`, `discharge_replay.json` |
| Table 4, text judgments (Section 7.3) | `experiments/analysis/build_text_table.py`; the verified condition: `experiments/blind_verification/` | `tableTEXT.json` |
| Table 5, the closest control systems (Section 7.4) | `src/prime_agents/extsys.py` and `src/prime_agents/glean.py` (instantiations: `docs/EXTSYS.md`); `experiments/analysis/build_extsys_table.py`; `experiments/paper/make_tab_extsys.py`; utility crediting only established conclusions: `experiments/analysis/established_utility.py` | `tableEXTSYS.json`, `glean/diagnostics_qwen2.5-32b_B20.json`, `established_utility_hesp.json` |
| Table 6, held-out regulatory evaluation (Section 7.5) | `experiments/specbuild/` (`build_test.py`, `score_e2e.py`, `analyze_e2e.py`) | `experiments/specbuild/reported/e2e.json`, `analysis.json`; agent episodes in `results/specbuild_e2e/` |
| Table 7, runtime and renderer tests (Appendix A) | `tests/` |  |
| Table 9, routes of the running example `sec-v2-0114` (Appendix C) | `src/suffice/construction/generate_record_instances.py` |  |
| Table 10, what each agent of the information ladder sees (Appendix C) | `src/prime_agents/arms.py` |  |
| Figure 3, the full information ladder (Appendix E.1) | `experiments/paper/make_appendix_figures.py`, through `experiments/paper/make_figures.py` | `tableT1.json`, `tableT2.json` |
| Figure 4, guidance gains by budget and route structure (Appendix E.2) | `experiments/paper/make_figures.py` | `tableT2.json` (`control_law`) |
| Figure 5, the ideal after-the-fact filter (Appendix E.2) | `experiments/paper/make_appendix_figures.py` | `rejection_bound.json`, `frontier_policies.json`, `tableEXT.json` |
| Appendix E.2, the flat gate | `experiments/analysis/build_flat_table.py` | `tableFLAT.json` |
| Table 11, three-model results on the balanced and locator-uncertain splits (Appendix E.3) | `experiments/analysis/build_multimodel_table.py`; `experiments/paper/make_tab_multimodel.py` | `tableMULTI.json` (the fully informed agent's locator-split cell is a new cell, `results/new_cells.md`) |
| Appendix E.3, share of utility from the runtime's statement; first-read accuracy per model; live completion on the locator-uncertain split | `experiments/analysis/build_w2a_tables.py`; `experiments/analysis/locator_first_reads.py`; `experiments/analysis/build_locator_table.py`, `experiments/analysis/build_locator_extra.py` | `utility_attribution.json`, `locator_first_reads.json`, `tableLOC.json`, `tableLOC_extra.json` |
| Appendix E.4, rejected and wrong attestations; verification | `experiments/analysis/text_attestation_counts.py`; `experiments/blind_verification/` (`analyze_live.py`) | `text_attestation_counts.json`, `tableTEXT.json` |
| Table 12, closest-system results in full (Appendix E.5) | `experiments/analysis/build_extsys_table.py`; `experiments/paper/make_tab_extsys.py`; model calls of the PRIME runs and the fully informed agent: `experiments/analysis/model_calls.py`, which counts them from the traces (checked by `tests/test_model_calls.py`) | `tableEXTSYS.json` |
| Table 13, budgeted PRIME–HESP comparison (Appendix E.5) | `experiments/analysis/hesp_controller_runs.py` (`--split heldout_locator` for the locator-uncertain copy), `experiments/analysis/build_controller_table.py`, `experiments/analysis/build_locator_hesp_table.py`, `experiments/analysis/discharge_replay.py --split locator` (completion replayed on the locator split); `experiments/paper/make_tab_extsys.py` | `tableCONTROLLER.json`, `tableLOC_HESP.json`, `discharge_replay_locator.json` (PRIME at $b_{\min}{+}1$ is a new cell, `results/new_cells.md`) |
| Appendix E.5, utility crediting only established conclusions; HESP's prior and threshold; Proof-or-Stop under PRIME's turn cap | `experiments/analysis/established_utility.py`; `experiments/analysis/hesp_prior_sensitivity.py`, `experiments/analysis/hesp_controller_runs.py --prior counted_lodo`; `experiments/analysis/stall_cap_replay.py` | `established_utility_hesp.json`, `hesp_prior_sensitivity.json`, `hesp_tau_sensitivity.json`, `stall_cap_replay.json` |
| Appendix F, the two construction workflows, validation and freezing | `experiments/specbuild/` (`create.py`, `validate.py`; protocol in `PROTOCOL.md`); `src/suffice/construction/fdaaa/` (the assistant-built program) | `experiments/specbuild/reported/analysis.json` |
| Table 14, the assistant-built held-out sample (Appendix F) | `experiments/fdaaa_heldout/heldout_eval.py` | `experiments/fdaaa_heldout/results/heldout_eval.json` |
| Table 15, the assistant-built program on development and held-out registrations (Appendix F) | `experiments/analysis/build_fdaaa_table.py`; `experiments/fdaaa_heldout/heldout_eval.py` | `tableFDAAA.json`, `experiments/fdaaa_heldout/results/heldout_eval.json` |
| Table 16, development registrations: program verdicts against tracker status (Appendix F) | `experiments/analysis/build_fdaaa_table.py` | `tableFDAAA.json`, `fdaaa_confusion.json` |
| Table 17, the exploratory pilot on hand-built tasks (Appendix G) | `experiments/analysis/pilot/end_to_end_tables.py` |  |
| Family-level intervals and tests, everywhere (Appendix D) | `experiments/analysis/stats_v5.py` |  |
| Episodes left out of the rates | `experiments/analysis/episode_accounting.py` | `excluded_episodes.json` |

## Running an agent

Serve an open-weight model with vLLM; the tool-call parser flags are required, because every agent acts through the OpenAI tools API.

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

Then point the runner at it:

```bash
export PRIME_LLM_BASE=http://127.0.0.1:8000/v1 PRIME_LLM_KEY=local
python3 src/prime_agents/run_batch.py --instances "data/suffice/security/heldout_balanced/*.json" \
  --arm prime_guided --license-source gold --model Qwen/Qwen2.5-32B-Instruct \
  --max-tool-calls 20 --out results/mine/prime_guided__B20
```

`--dry-run` replaces the model with a scripted one and makes no API calls. The runner refuses paid commercial endpoints unless `PRIME_ALLOW_PAID_API=1` is set. `--budget-anchor bmin|bmin1|2bmin` sets the per-task budgets of Section 7.2.

The arms of the paper are:
- the information ladder of Section 7.1: `react`, `react_verify`, `react_spec`, `react_spec_verify`, `react_program`, `react_program_verify`, and `react_program_attested` (the fully informed agent);
- PRIME: `masked_reactive_v` (PRIME without guidance), `prime_guided` (PRIME), and `prime_complete` (PRIME with completion);
- the closest control systems of Section 7.4: `ext_agentspec`, `ext_agentspec_examine`, `ext_researchloop`, `ext_prooforstop`, `ext_hesp`, and GLEAN as an offline verifier (`src/prime_agents/glean.py`).

The scripted policies run without a model through `src/prime_agents/policies.py`.

## Data provenance

The records come from public APIs and are turned into tasks as follows:
- `src/suffice/construction/fetch_osv.py`, `src/suffice/construction/fetch_ctgov.py`, `src/suffice/construction/fetch_openfda.py`, and `src/suffice/construction/fetch_edgar.py` retrieve the records;
- `src/suffice/construction/harvest_records.py` gathers them into the harvest;
- `src/suffice/construction/generate_record_instances.py` and `src/suffice/construction/generate_text_instances.py` turn them into tasks.

Records change, so a new harvest yields a different corpus. The published tasks embed the snapshots retrieved on 1 August 2026, 3 August 2026 for the regulation set, and `data/harvest/` holds the harvest they were generated from. The SEC requires a contact address in the User-Agent of every request; set `SEC_USER_AGENT` before running `fetch_edgar.py`.

`src/suffice/construction/fdaaa/build_fdaaa_instances.py` reads two inputs:
- a ClinicalTrials.gov snapshot, which is included;
- the FDAAA TrialsTracker's trial table of 3 August 2026, which belongs to the Bennett Institute for Applied Data Science and is not included. The tracker publishes its current table at https://fdaaa.trialstracker.net.

`data/fdaaa/external/SHA256SUMS.txt` pins both.

The hand-built tasks read `data/pilot/artifacts/`:
- open-source Python projects at pinned commits, redistributed under their own licenses (each project directory keeps its license file);
- arXiv titles and abstracts, which arXiv releases under CC0;
- public records from the same sources as the benchmark.

Each artifact's `PROVENANCE` file records where and when it was retrieved.

## Known issues

`docs/KNOWN_ISSUES.md` lists where the released code differs from the code that produced the logs:
- budget accounting after a repeated call: `experiments/analysis/budget_accounting_audit.py` found the 139 episodes whose outcome it could change, and all were rerun with the corrected runner (the originals ship in `results/superseded_by_rerun/`); the paper's numbers include the reruns; the four cells run afterwards used the corrected runner, and the audit checks them under its rule;
- attestations refused before a read, which affects no logged episode.

It also explains the excluded episodes, and says what the one-branch variant's family ids mean.

## License

The code is released under the MIT License (`LICENSE`). The benchmark data are released under CC BY 4.0, and the records remain subject to their sources' terms (`data/NOTICE.md`).

## Citation

The citation will be added when the paper is published.
