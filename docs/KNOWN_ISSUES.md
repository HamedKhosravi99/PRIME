# Known issues and differences from the logged runs

The paper's numbers are computed from the logged episodes in `prime-episode-logs.tar.gz`. The released code differs from the code that produced those logs in the ways listed here; each item says what changed and what it means for the logged results.

## Budget accounting after a repeated call (fixed; affected episodes rerun)

An agent's call that repeats an executed call class (same tool, same parameters) is served from the executor's cache and is meant to consume no budget. In the logged runs, every model-driven loop decided whether a call consumed budget from the cache status of the previously executed call, not of the call itself. This covers the agent loops in `runner/arms.py` and the closest-system arms in `runner/extsys.py`. A repeat was therefore charged when the call before it was new, and the calls after a repeat, up to the next executed one, were free. Runtime completion and the scripted policies charged correctly. The released runner decides from the call's own class (`ToolExecutor.charges`), and `prime_runtime/tests/test_budget_accounting.py` checks it.

The two rules charge every call identically unless the agent repeats a call, so only such episodes can differ. `analysis/budget_accounting_audit.py` replays every logged trace of every reported cell under both rules, with no model call. It covers 113 cells (model or program, split, agent, and budget, pooled over domains, the regulation study and the verified condition of Table 2 included) and 49,297 completed episodes.

It found 139 episodes whose outcome the old accounting could have changed:
- for an agent that sees or uses the remaining budget (guided PRIME, PRIME with completion, HESP), every episode whose charged count differed;
- for any other agent, which receives identical messages under both rules until the cap ends its episode, the episodes that would have ended at a different call;
- the 21 episodes of the next section.

All 139 were rerun with the corrected runner on 2026-10-12 (`analysis/rerun_manifest.py`, `runner/rerun_episodes.py`, `cluster/rerun_episodes.sbatch`). The reruns replaced them in the logs (`analysis/merge_reruns.py`); the logged originals ship in `results/superseded_by_rerun/`. The audit now:
- reproduces the call count of every rerun episode under the corrected rule;
- finds no episode left whose outcome the old accounting could change, so its bound is 0.00 points in every cell.

Every other episode was charged identically under both rules, so its logged run is what the corrected runner produces.

The rerun changed:
- the stated conclusions of 6 of the 138 agent episodes;
- GLEAN's verdict on one trajectory, which it now rejects.

The reported numbers moved by at most 0.2 points: Table 2, Table 8's gpt-oss-20b row, Tables 9 and 10, and two percentages of Appendix E.3. Table 1, Figure 2, Table 3, the regulation study, and every false-authorization rate are unchanged. `results/rerun_changes.md` lists each change, and the paper reports the rerun numbers.

Four cells were run after the rerun with the corrected runner (`cluster/new_cells.sbatch`). They fill cells that had not been run, so they replace nothing: the fully informed agent on the locator-uncertain split at $b_{\min}$ for the three models (Table 9), and PRIME at $b_{\min}{+}1$ on that split with Qwen (Table 11). Their episodes carry a `new_cell` marker (`analysis/import_new_cells.py`), and the audit checks their call counts under the corrected rule. `results/new_cells.md` lists the jobs and the values they added.

## Episodes whose trace and summary came from different runs (rerun)

In 21 logged episodes the trace and the summary entry came from different runs of the same episode, so the summary's call count differed from the one the trace replays to:
- 11 of the flat gate on the drug-safety tasks, which was rerun;
- 8 of Proof-or-Stop and 2 of ResearchLoop on the clinical tasks, two jobs that ran the same cells at once (`docs/EXTSYS.md` describes four further such episodes, rerun at the time).

They are among the 139 rerun episodes above, so each now has one trace and one summary entry from the same run.

## An attestation refused before reading stayed refused (fixed)

The executor's cache keyed a call by tool and arguments only. An `attest_atom` call refused because the check's source text had not been read yet was therefore served from the cache when the agent read the text and repeated it word for word. This contradicts the rule the paper states: an attestation is admitted if the agent read the text and the quote occurs in it verbatim.

The call class of `attest_atom` now also records whether the source text has been read (`ToolExecutor._class_key`), and `prime_runtime/tests/test_attest_cache.py` checks the case. No logged episode is affected: the text-evidence runs contain 812 attestations, and none attested before reading.

| Outcome | Attestations |
|---|---|
| Accepted | 660 |
| Quote not in the text | 86 |
| Not a text check | 66 |

## Conformance of the standalone scorer

`prime_runtime/tests/test_obligationbench_conformance.py` compares `obligationbench.py` with the runtime on:

- every read probe of the 1,658 record tasks;
- seeded check-value states of all 1,898 tasks;
- the 7,176 completed reference episodes.

The standalone benchmark's `tests/test_conformance.py` compares its environment with the same runtime outcomes, text attestations included, through a fixture written by `python3 bench/conformance.py`.

Two rules were aligned in the standalone benchmark when these tests were added:

- A check that receives both values is conflicted and settles nothing, as in the runtime's four-valued ledger; before, the last value counted.
- Its scorer now also reports the distinct call classes an agent used (what the runtime charges).

Neither change affects the reference predictions, which contain record reads only.

## Failed episodes are excluded from the rates

An episode whose model calls still timed out after the client's retries carries no measurement and is left out of every rate. An agent's denominator is therefore its completed episodes.

- `analysis/episode_accounting.py` lists every such episode of every run in `results/excluded_episodes.json`.
- `reproduce_tables.py` checks that, for the 15 agents it reproduces, the failed episodes are exactly the declared ones: 24 of 7,200, at most 6 for one agent. It fails if any other episode is absent.

## One-branch family ids

The one-branch variant (`bench/instances_v2/<domain>/heldout_onebranch`, 240 tasks) was generated on 2026-10-08 in a separate run. That run enumerated its families with its own seed offsets, so its family ids (for example `clinical-branching-hom-17`) name other probe sets than the same ids in the primary corpus.

None of its 40 families matches any primary-corpus family, by routes or by probe set, so the variant is held out from every primary split. Its family ids are local to the variant; do not join them with primary-corpus families. `bench/generate_record_instances.py` pins both runs' offsets (`FAMILY_SEED_OFFSET`, `ONE_BRANCH_FAMILY_SEED_OFFSET`), and `python3 bench/reproduce_corpus.py` regenerates all 1,898 tasks byte for byte.
