# Checks that need no model; see README.md, "Reproducing the paper".
# `make` runs all of them (about ten minutes on a laptop).
PYTHON ?= python3
LOGS := logs/prime-episode-logs.tar.gz

.PHONY: all test corpus logs tables figures conformance lint

all: test corpus tables figures conformance

# the runtime's tests (with the logs extracted, also the episode conformance test)
test:
	$(PYTHON) -m pytest -q prime_runtime/tests

# regenerate all 1,898 ObligationBench tasks from bench/harvest/ and compare them byte for byte
corpus:
	$(PYTHON) bench/reproduce_corpus.py

# extract the episode logs and the JSON the analysis scripts wrote into results/
logs: results/.extracted
results/.extracted: $(LOGS)
	tar -xzf $(LOGS)
	touch $@

# the per-agent endpoints of Section 6.1 from the traces, then every table JSON regenerated and compared
tables: logs
	$(PYTHON) reproduce_tables.py
	$(PYTHON) analysis/regenerate_tables.py

# recompute every plotted value and check it against the paper
figures: logs
	$(PYTHON) figures/make_figures.py
	$(PYTHON) paper/make_fig_budget.py
	$(PYTHON) paper/make_naacl_appendix_figures.py

# obligationbench.py against the runtime, on every task and every logged reference episode
conformance: logs
	$(PYTHON) -m pytest -q prime_runtime/tests/test_obligationbench_conformance.py

# syntax errors and undefined names only; the code is not reformatted
lint:
	$(PYTHON) -m ruff check --select E9,F63,F7,F82 .
