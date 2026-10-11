# PRIME and ObligationBench

Code, benchmark, and analysis for the paper "PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents" (under review).

PRIME fixes, before an LLM agent examines a case, which checks suffice for each conclusion. It then records what the agent's checks establish, points the agent at the checks that remain, and refuses any conclusion that the completed checks do not justify. ObligationBench, the paper's benchmark, has 938 tasks built from public security advisories, clinical-trial registrations, drug recalls, and company filings, scored by executable rules with no LLM judge. The benchmark is also distributed on its own, with a dataset card, as `obligationbench/`.

## Contents

| Path | What it is |
|---|---|
| `obligationbench.py` | loads tasks, gives an agent the record to read, and scores its conclusions; standard library only |
| `reproduce_tables.py` | rebuilds the per-agent endpoints of Experiments 1, 2, and 5 from the episode logs and checks them against the paper |
| `prime_runtime/` | the runtime: commitment, ledger, authorization, guidance, and the renderer, with its tests in `tests/` |
| `runner/` | the agents (the ReAct ladder, the comparators, and PRIME) and the batch driver |
| `bench/` | record fetchers, the task generator, the validator, and the tasks: `instances_v2/` (ObligationBench), `instances_fdaaa/` (the FDAAA set), and `instances/` (the hand-built tasks of the e-companion pilot and of Experiment 8) |
| `bench/artifacts/` | the sources the hand-built tasks read: open-source Python projects at pinned commits, arXiv titles and abstracts, and public records |
| `analysis/` | the table generators and the statistics (`stats_v5.py`) |
| `figures/make_figures.py` | regenerates the figures from the logs and checks every plotted value against the paper |
| `cluster/` | the SLURM scripts that served each open-weight model with vLLM and ran the agents; set the account, partition, and paths for your cluster |
| `docs/ANALYSIS_PLAN.md` | the analysis plan, frozen before any confirmatory episode was scored |
| `docs/EXECUTION_PLAN.md` | the execution plan, including the dated log of deviations from the analysis plan (Part 6b) |
| `docs/GLOSSARY.md` | how the code's names (claims, atoms, licenses, arms, result folders) map to the paper's terms |

## Setup

Python 3.9 or later. The runtime, the scorer, and the analysis use only the standard library. The figures need numpy and matplotlib, the tests need pytest, and running agents needs the `openai` client and an OpenAI-compatible endpoint.

```bash
pip install -r requirements.txt
```

## Reproducing the paper without running a model

The episode logs are distributed separately as `prime-episode-logs.tar.gz`. Extract the archive at the repository root, so that `results/` sits next to `bench/`, then run:

```bash
python3 reproduce_tables.py
python3 analysis/build_w2a_tables.py
python3 figures/make_figures.py
python3 -m pytest -q prime_runtime/tests
```

`reproduce_tables.py` scores the stated conclusions and record reads of 15 agents with the standalone scorer and checks false authorization, premature conclusions, supported utility, the emission rate, and evidence value against the paper. `make_figures.py` recomputes every plotted value and exits nonzero if any differs from the paper.

| Paper | Scripts |
|---|---|
| Experiment 1 (information versus enforcement) | `analysis/build_w2a_tables.py`, `analysis/failure_mode.py`, `reproduce_tables.py` |
| Experiment 2 (cost of enforcement) | `analysis/build_w2a_tables.py`, `analysis/rejection_bound.py`, `reproduce_tables.py` |
| Experiment 3 (guidance and budgets) | `analysis/build_w2a_tables.py`, `analysis/policy_frontier.py` |
| Experiment 4 (filtering afterwards) | `analysis/rejection_bound.py`, `analysis/build_ext_table.py` |
| Experiment 5 (verification mechanisms) | `analysis/build_ext_table.py`, `analysis/build_conformal_confirmatory.py`, `reproduce_tables.py` |
| Experiment 6 (other models, natural split, sampling) | `analysis/build_w2a_tables.py --models qwen2.5-7b gptoss-20b`, `analysis/mcnemar_split.py` |
| Experiment 7 (learned routes) | `analysis/build_w2b_table.py`, `analysis/induction_anatomy.py`, `analysis/induction_quality.py` |
| Experiment 8 (constructed rules) | `analysis/openvocab_endtoend.py`, `analysis/vocabulary_conditions.py` |
| FDAAA (external regulation) | `bench/fdaaa/build_fdaaa_instances.py`, `analysis/build_fdaaa_table.py`, `bench/fdaaa/contract/notices_check.py` |
| Pre-registered margin and protocols | `analysis/w0_pilots.py`, `analysis/protocols_v2.py`, `analysis/stats_v5.py` |
| Figures | `figures/make_figures.py` |

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

`--dry-run` replaces the model with a scripted one and makes no API calls. The runner refuses paid commercial endpoints unless `PRIME_ALLOW_PAID_API=1` is set. The arms of the paper are `react`, `react_verify`, `react_spec`, `react_spec_verify`, `react_program`, `react_program_verify`, `react_program_attested`, `masked_reactive_v` (PRIME without guidance), `prime_guided` (PRIME), and the comparators `direct`, `rubric`, `cove`, `reflexion`, `conformal`, and `static_contract`.

## Data provenance

`bench/fetch_osv.py`, `bench/fetch_ctgov.py`, `bench/fetch_openfda.py`, and `bench/fetch_edgar.py` retrieve records from public APIs, and `bench/harvest_records.py` and `bench/generate_record_instances.py` turn them into tasks. Records change, so regenerating from the live sources yields a different corpus; the released tasks embed the snapshots retrieved on 1 August 2026 (3 August 2026 for the FDAAA set). The SEC requires a contact address in the User-Agent of every request; set `SEC_USER_AGENT` before running `fetch_edgar.py`. `bench/fdaaa/build_fdaaa_instances.py` reads a ClinicalTrials.gov snapshot, which is included, and the FDAAA TrialsTracker's trial table of 3 August 2026, which belongs to the Bennett Institute for Applied Data Science and is not included; `bench/fdaaa/external/SHA256SUMS.txt` pins both. The tracker publishes its current table at https://fdaaa.trialstracker.net.

The hand-built tasks read `bench/artifacts/`: open-source Python projects at pinned commits, redistributed under their own licenses (each project directory keeps its license file); arXiv titles and abstracts, which arXiv releases under CC0; and public records from the same sources as the benchmark. Each artifact's `PROVENANCE` file records where and when it was retrieved.

## License

The code is released under the MIT License (`LICENSE`). The benchmark data are released under CC BY 4.0, and the records remain subject to their sources' terms (`NOTICE.md`).

## Citation

The citation will be added when the paper is published.
