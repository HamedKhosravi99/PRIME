# Old paths and where they are now

The repository was reorganized after the runs. The frozen plans (`ANALYSIS_PLAN.md`, `EXECUTION_PLAN.md`), the experiment reports, the episode logs (`instance_path` fields and notes), and the hand-built tasks' notes keep the paths they were written with, and are not edited. This table maps those paths to the current layout; the first matching row applies. File contents did not change in the move.

| Old path | Current path |
|---|---|
| `prime_runtime/tests/` | `tests/` |
| `prime_runtime/` | `src/prime_runtime/` |
| `runner/` | `src/prime_agents/` |
| `obligationbench.py` | `src/obligationbench/__init__.py` (also `python -m obligationbench`) |
| `bench/schema.json`, `bench/validate.py` | `src/obligationbench/` |
| `bench/instances_v2/` | `data/obligationbench/` |
| `bench/instances_fdaaa/` | `data/fdaaa/instances/` |
| `bench/fdaaa/external/` | `data/fdaaa/external/` |
| `bench/fdaaa/build_fdaaa_instances.py`, `bench/fdaaa/contract/` | `src/obligationbench/construction/fdaaa/` |
| `bench/harvest/` | `data/harvest/` |
| `bench/instances/` | `data/pilot/instances/` |
| `bench/artifacts/` | `data/pilot/artifacts/` |
| `bench/*.py` (fetchers, harvest, generators, `reproduce_corpus.py`, `conformance.py`, `scaffold_repo.py`, ...) | `src/obligationbench/construction/` |
| `analysis/end_to_end_tables.py`, `analysis/build_tableC_*.py`, `analysis/tableC_guided_common.py`, `analysis/build_financial_table.py`, `analysis/drift_table.py`, `analysis/mcnemar_split.py` (the pilot, Appendix G) | `experiments/analysis/pilot/` |
| `analysis/` | `experiments/analysis/` |
| `figures/*.py`, `paper/*.py` | `experiments/paper/` |
| `figures/*.pdf`, `paper/*.pdf` | `experiments/paper/figures/` |
| `paper/make_naacl_appendix_figures.py`, `paper/fig_*_naacl.pdf` | `experiments/paper/make_appendix_figures.py`, `experiments/paper/figures/fig_*_appendix.pdf` |
| `reproduce_tables.py` | `experiments/reproduce_tables.py` |
| `cluster/` | `experiments/slurm/` |
| `logs/` | `data/episode_logs/` |
| `NOTICE.md` | `data/NOTICE.md` |
| `requirements-lock.txt` | the pinned versions in `docs/ENVIRONMENT.md` |

Unchanged: `experiments/specbuild/`, `experiments/fdaaa_heldout/`, `experiments/blind_verification/`, `docs/`, and `results/` (extracted from the logs by `make logs`).

Absolute paths in the logs (`/home/anonymous/prime_pilot/...`, `/home/anonymous/prime_rerun/...`) name the machines' checkouts; the part after the checkout follows the table above.
