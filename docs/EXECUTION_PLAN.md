# PRIME → 8: Unified Execution Plan (v5 — FROZEN, 2026-08-01)

Round 4: **Claude confirmed the freeze** (3 non-blocking W0 notes, all
adopted below). **GPT objected on 4 protocol points** (Q1–Q4 below), all
corrections within scope; v5 resolves them and is the frozen plan. No new
scope was added in this round, per both reviewers' instruction. Execution
begins at W0; changes after this version require a named freeze-break with
justification logged in this file.

## Part R4 — Round-4 dispositions

| # | Point | Disposition |
|---|---|---|
| GPT Q1 | **F*_B collapse makes "protocol-robust unsupported" mathematically redundant** (Support_B ⊆ Support_A ⟹ ¬A∧¬B ≡ ¬A) — a real error in v4 | **Adopted in full.** The term "protocol-robust unsupported" is deleted. Report instead: OFAR-A, OFAR-B, protocol disagreement rate, protocol sensitivity (OFAR_B − OFAR_A), and content-false separately. Add the mechanical realizability gate: Φ ∧ p_B must be SAT; if the collapsed route is unrealizable, F*_B is reported as undefined for that claim (preregistered fallback), never silently claim-rejecting |
| GPT Q2 | Topology confounded with cost heterogeneity | **Adopted.** Full 2×2 cross: {nested, branching} × {homogeneous, heterogeneous call-class costs}, with support mix, bad-artifact rate, signed-literal rate, route widths, and alternative-route counts matched across cells. Preregistered: primary effect within branching; topology×policy and cost×policy interactions; three-way interaction descriptive unless powered. If the factorial is underpowered, the allowed headline narrows to *"obligation-directed selection improves utility in branching, heterogeneous-cost evidence programs"* — never attributed to branching alone |
| GPT Q3 | Template-only randomization invalid if records cross families | **Adopted, preferred design:** the generator assigns each held-out record to exactly one template family within each confirmatory benchmark (a construction constraint, checked mechanically). Test frozen precisely: per-record paired differences → aggregate within template family → joint sign-flip at the family-cluster level → permutations stratified by domain and topology → statistic = mean cluster-weighted paired difference. If record reuse ever becomes necessary, the two-way wild-cluster bootstrap becomes primary and template-only randomization demotes to sensitivity — but the default design makes this moot |
| GPT Q4 + Claude N3 | Conditional `masked_reactive_prefix` = selected secondary analysis | **Adopted, GPT option 1 (both reviewers converged):** `masked_reactive_prefix` runs **upfront on all branching confirmatory instances** as a standard arm. The attribution question (obligation-direction vs ordering heuristic) is answered unconditionally, whatever E2 shows — including the inconclusive branch Claude flagged |
| GPT small | Natural-prevalence subset frozen | {`react`, `react_program_attested`, `masked_reactive`, `prime_guided`} (replaces v4's `react_program_verify` slot with the cleaner enforcement comparator) |
| GPT small | Multiplicity families separated | Three Holm families: instructed information/enforcement ladder; control-law contrasts; induction-method contrasts. E2 is never in the ladder family |
| GPT small | Cross-stratum difficulty balancing | Support-level mix, bad-artifact prevalence, signed-literal rate, route-width and alternative-route distributions frozen ≈equal within each domain × topology × cost cell |
| GPT small | Report both n | Every table reports evaluated-instance n AND independent template-family cluster n |
| Claude N1 | E1 may go degenerate (attested arm near 0% OFAR → no discordant clusters) | **Adopted:** W0(d) dev run checks it; E1 always reports the discordant-cluster count beside the p-value, so a null reads "no room to differ" vs "no effect"; a floor-effect E1 routes to the P7 headline branch explicitly |
| Claude N2 | δ_min episode-fraction gloss assumes level-linear U | **Adopted:** the interpretation sentence is re-derived from the actual frozen U in W0(b); the (0,33,67,100) illustration never enters the paper |
| Claude process | Abstract/intro drafted against frozen slots before W2a; W0 is all-or-nothing | **Adopted:** abstract + intro drafts move to the W0→W1 boundary (not W6); W0 gains an explicit exit checklist — all eight preconditions green before any W2a episode; "mostly done" is defined as not done |

---

## Part R — Round-3 dispositions

| # | Point | Disposition |
|---|---|---|
| L1 + P1 | Ladder confounded (instruction × information; ledger × mask) | **Adopted, both halves.** New arm `react_verify` (bare ReAct + compliance instruction) anchors the instructed chain at the bottom; `react_program_attested` promoted to **mandatory**; **primary E1 = `react_program_attested` vs `masked_reactive_v`** (GPT's boxed contrast); the primary chain is the instructed row only (Part 1); uninstructed arms form a parallel row estimating the instruction effect at each information level; **E1 primary condition = gold** (pure compliance-vs-enforcement), induced secondary (L) |
| L2 | `B_min` may floor the densest cell; anchors differ across strata | **Adopted:** W0 dev pilot of the sweep is a **freeze precondition**: report realized `B_min ÷ cost(cheapest all-true route to best supported claim)` per stratum; verify policy divergence at `B_min`, `B_min+1`; if floored, shift sweep to `{B_min+1, B_min+2, 2·B_min, 20}`. Sweep anchor chosen per stratum finding, then frozen |
| L3 | Effective n = template clusters; TOST likely inconclusive; missing Part 5B rows | **Adopted:** E2's n is stated **in clusters** everywhere (manifest included); budget = within-instance factor; **branching template families raised to ≥16/domain (target 20), nested ≥8/domain**; W0 computes TOST power from dev variance — if <0.8, the paper pre-commits to "E2 yields a CI, not a verdict." Two Part 5B rows added: *inconclusive E2* and *control law wins in the nested stratum contra prediction* (= stopping behavior, not planning — its own sentence) |
| L4 | F*_B should be generated, not authored | **Adopted:** F*_B = pre-registered mechanical transform of F*_A — the **conjunctive collapse** (every atom optional across A's alternative routes becomes required). "Protocol-robust unsupported" now has stated semantics: *unsupported even under the strictest reading of the same protocol*. One function, no authoring objection |
| P2 | `informed_route_planner` is outcome-informed | **Adopted:** renamed **`outcome_informed_planner`**; its gap is "outcome-informed headroom," never "planning gap"; MDE uses it only with the δ_min floor (below) and the absolute utility effect is always reported independently; synthetic suite uses the information-matched exact planner as the clean planning comparator |
| P3 | T4b scope ambiguity | **Adopted verbatim:** T4b preconditions = (1) chain order; (2) **exactly one live route per claim**; (3) weaker routes ⊂ stronger routes; (4) prefix-first resolution of shared classes; (5) cached outcomes; (6) deterministic known costs; (7) each call class charged once; (8) no premature termination. Route multiplicity and all branching live entirely in T4c + E2 |
| P4 + L-Q3 | δ_min arithmetic wrong (3 pts ≈ 10× off) | **Adopted:** freeze U first (W0). δ_min = **0.15 × median adjacent-level utility gap under the frozen U** (≈5 pts under (0,33,67,100)), reported with Claude's interpretation: ≈ upgrading one episode in seven by one claim level. MDE: δ = max(δ_min, 0.2·G_dev) with G_dev labeled outcome-informed headroom |
| P5 + L-Q1 | Prefix-first scope | **Merged:** prefix-first activates **only for detected single-chain ladder programs** (deterministic, tested topology classifier — GPT); within a route it is a **tie-break inside the ceiling's minimal route, never overriding ceiling selection** (Claude) — so `prime_guided` semantics on branching programs are unchanged from the landed version (no re-tag needed there; ladder programs get old/new tagging). **Pre-committed follow-up:** if the control law wins E2, run `masked_reactive_prefix` (reactive selection + prefix-first ordering) in the branching stratum — if it captures most of the gain, the finding is the ordering heuristic, and the paper says so |
| P6 + L-Q2 | One primary decision rule | **Resolved — the reviews converge via GPT's own small-cluster clause:** at 24–40 template clusters, the **template-cluster randomization test is the sole inferential primary**; the two-way cluster bootstrap CI is the **effect-size estimate**; McNemar is sensitivity-under-independence. One decision rule, stated in the preregistration |
| P7 | Unexpected-null headline overstated | **Adopted, GPT's wording verbatim:** "the deterministic mask retains its policy-independent structural guarantee, but provides no measurable incremental empirical benefit over an attested, explicitly instructed policy on this benchmark" |
| P8 | Which guided arm is primary where | **Frozen:** gold → `masked_reactive_v` vs `prime_guided` (no gate; program is gold). Induced deployable → `masked_reactive_v(induced)` vs `prime_guided_gated(induced)`; ungated induced guided reported **only** as the error-amplification diagnostic. No post-hoc selection |
| P9 | Template split counts | **Frozen allocation per domain and stratum** at GPT's ratio: ~40% development / ~10% validation-freeze / ~50% held-out (e.g., branching 20 → 8/2/10). Manifest records: generator sharing (one generator, family = fixed lattice topology × predicate composition; a parameter re-binding is the SAME family), overlap rules (field paths may repeat across families; topology×composition may not), and replacement policy (a family failing verification is dropped, never replaced post-freeze) |
| P10 | Two attrition types | **Adopted:** natural-prevalence accounting separates **data-source attrition** (missing/malformed/dup/retrieval) from **benchmark-construction attrition** (resolver incompatibility, no denotation, verification failure), each with counts; the construction-attrition rate is reported as the representation-selection bias bound |
| L-Q4 + P-Q4 | `masked_reactive_v` naming + continuity | **Merged:** paper tables say `masked_reactive` with a prompt footnote; `_v` lives in the manifest only. Old uninstructed results: appendix continuity table, labeled previous-baseline-version, never pooled, never in confirmatory statistics. **Pre-commitment (L):** both `react_program` prompts run on dev, and the measured instruction effect goes in the body regardless of direction — if instruction closes most of the old gate gap, that is a finding, not a footnote |

---

## Part 1 — Final ladder (two rows; primary chain = instructed row)

**Instructed row (primary chain, Holm family):**
`react_verify` → `react_spec_verify` → `react_program_verify` →
`react_program_attested` → `masked_reactive_v` → `prime_guided`

| Adjacent contrast | Isolates |
|---|---|
| `react_verify` → `react_spec_verify` | vocabulary information |
| `react_spec_verify` → `react_program_verify` | support-logic information |
| `react_program_verify` → `react_program_attested` | ledger/attestation information |
| **`react_program_attested` → `masked_reactive_v`** | **deterministic enforcement (E1 primary, gold condition)** |
| `masked_reactive_v` → `prime_guided` | control law (E2, branching stratum) |

**Uninstructed row (secondary, instruction-effect estimates):** `react`,
`react_spec` (unpadded primary / padded sensitivity), `react_program`.
Every arm differs from its instructed twin by exactly the compliance
instruction.

## Part 2 — Theorem program (frozen scope)

T4a₁ (architectural invariant) / T4a₂ (conditional replanning, per-state,
no initial-state promise) — separate statements. **T4b** with P3's eight
explicit preconditions; certifies the termination rule; ladder-detected
programs only. **T4c** ($C_{\mathrm{waste}}$ accounting corollary) + E2
carry everything branching. Enumeration suite holds all three recorded
counterexamples (both K-examples + GPT's nested-order example) and
validates; proofs are separate deliverables.

## Part 3 — Statistics (frozen)

- Units: clusters (record/entity × template family); **all reported n are
  cluster counts**; budget is a within-instance factor.
- E1 (gold; `react_program_attested` vs `masked_reactive_v`): primary =
  template-cluster randomization test; effect size = two-way cluster
  bootstrap CI; sensitivity = exact McNemar. Ladder family Holm-corrected.
  Induced E1 secondary.
- E2 (branching stratum; `masked_reactive_v` vs `prime_guided` on gold;
  vs `prime_guided_gated` on induced): utility-unit regret; randomization
  primary, bootstrap CI estimate, Wilcoxon sensitivity; TOST with
  δ = max(δ_min, 0.2·G_dev); W0 power calculation gates the "verdict vs
  CI-only" language. Nested stratum = predicted-null control.
- E3 unchanged. Endpoints everywhere: OFAR-A; OFAR-B (F*_B = mechanical
  conjunctive collapse of F*_A with the Φ∧p_B realizability gate, both
  hashed pre-held-out); protocol disagreement rate and sensitivity
  (OFAR_B − OFAR_A); content-false (three-value rule). The term
  "protocol-robust unsupported" is retired (GPT Q1: it reduces to OFAR-A).

## Part 4 — Benchmark (frozen numbers)

Balanced 480 (120/domain; full 2×2 topology × cost-heterogeneity cross
with matched difficulty covariates per GPT Q2; 30/30/25/15 mix; 40% false
load-bearing atom). Natural 240 (60/domain; frozen 4-arm subset {react, react_program_attested,
masked_reactive, prime_guided}; two-type attrition accounting). Synthetic stress ~200 (prevalence sweep +
information-matched planner). Template families per domain: **≥16 branching
(target 20) + ≥8 nested**, split ~40/10/50 dev/val/held-out, families held
out whole, replacement forbidden post-freeze; **each held-out record
belongs to exactly one template family** (GPT Q3 constraint, checked
mechanically at generation).

## Part 5 — Workstreams (delta from v3 only)

**W0 (freeze preconditions, all must pass before any confirmatory run):**
(a) budget-sweep dev pilot per L2 — anchor set frozen from its result;
(b) freeze U → derive δ_min → freeze MDE; (c) TOST power from dev variance
→ freeze "verdict vs CI-only" language; (d) both `react_program` prompts on
dev → instruction effect destined for the body; (e) F*_A authored, F*_B
generated by the frozen collapse rule, both hashed; (f) template split
allocation frozen per P9; (g) topology classifier implemented + tested;
(h) everything from v3's W0 (git baseline snapshot incl. results, bib
audit, rebuttal purge, skeleton + abstract number slots, anonymization,
manifest, sampling frame).
**W1:** adds `react_verify`, `react_program_attested` (now standard),
prefix-first tie-break gated by the topology classifier (+ tests; ladder
programs old/new tagged).
**W2a–W6:** unchanged from v3, except `masked_reactive_prefix` is a
**standard upfront arm on all branching confirmatory instances** (GPT Q4 /
Claude N3), and abstract+intro drafts are due at the W0→W1 boundary.

## Part 5B — Outcome → headline ledger (v4 additions in bold)

| Outcome | Headline |
|---|---|
| Information gaps large, enforcement gap small | Information was the missing ingredient; spec-carrying agents + auditability; decomposition first |
| Enforcement gap large (E1 rejects) | Enforcement is the separately validated contribution |
| `react_program_attested` ≈ `masked_reactive_v` | **P7 wording verbatim:** structural guarantee retained; no measurable incremental empirical benefit over an attested, instructed policy on this benchmark |
| Control law recovers ≥δ in branching stratum | First separated validation of obligation-directed selection (branching topologies; `masked_reactive_prefix`, run upfront, decides attribution — if it captures the gain, the headline is the ordering heuristic) |
| Control law null in branching stratum (powered) | Informative null at the abstract level |
| **E2 inconclusive (neither rejection nor equivalence)** | **Pre-committed: report the CI as the finding; no verdict language; the benchmark's cluster count is stated as the limiting factor** |
| **Control law wins in the NESTED stratum, contra prediction** | **Effect is stopping behavior, not planning — reported as such, distinct claim from obligation-direction** |
| S1 outcomes | unchanged from v3 (record-scoped / two-constructed-families / floor claim) |

## Part 6 — Compute

Unchanged envelope (35–55k episodes) + dev pilots (≈2k). Triage order
unchanged; the W0 pilots are never skipped — they are freeze preconditions.

## Part 6b — Freeze-break log

**FB-01 (2026-08-02): W2a episode collection may queue behind the dev
pilot via SLURM dependency, rather than after the W0_GREEN sentinel.**
Justification: seven of the eight W0 preconditions are green and hashed;
the two outstanding values (G_dev, TOST power) are computed from the DEV
split, which is instance-disjoint from every W2a heldout instance, and
they enter only E2's equivalence ANALYSIS -- never episode generation,
prompts, budgets, or arm definitions. The cluster's free queue is
the binding resource, and serialising submission costs ~1 day of wall
clock for no methodological gain. Mitigations, all binding: (i) shards are
submitted `--dependency=afterok:<pilot>` so they cannot start unless the
pilot completes successfully, preserving the pilot's defect-detection
function (it has caught two real defects already); (ii) shards are
cancelled if the pilot analysis reveals any defect; (iii) NO confirmatory
endpoint is computed or inspected until G_dev/TOST are frozen into
FREEZE.md F3/F6 from dev data. The analysis-order discipline that
preregistration exists to protect is unchanged.

**FB-02 (2026-08-02): T1 gains two derived columns, `emit_rate_avail` and
a `degenerate` flag, and the T1 prose excludes flagged arms from its
safety ranking.** FREEZE F9 fixes T1's endpoints and forbids new tables;
this adds no table and no new measurement, only a ratio of two quantities
F9 already reports (abstention, and the ceiling stratum recorded in the
frozen manifest). Justification: the first 7B confirmatory shards returned
OFAR-A of 0.0 for all three `react_program*` rungs, which is the same
number `masked_reactive_v` and `prime_guided` return -- but the program
rungs reach it by emitting on 0.3--4.0% of answerable instances while the
enforcement arms emit on 88.9--94.6%. Ranked by OFAR alone the most silent
arm is the safest arm, which inverts the paper's finding. The R4 Claude-N1
rule already guards this floor effect on the INFERENCE side (discordant
cluster counts beside every p); FB-02 is its table-side counterpart. The
threshold (emit on fewer than 25% of answerable instances) is registered
here BEFORE any 32B confirmatory data exists, and separates the observed
7B cells by more than an order of magnitude (0.3--8.9% vs 41.9--94.6%), so
it is not tuned to a boundary case. Degenerate cells are still printed in
full; they are excluded only from the "lowest false authorization"
comparison, and the prose states why.

**OPERATIONAL NOTE (2026-08-02): SLURM snapshots the batch script at
submission.** The eight 32B confirmatory shards were submitted alongside the
7B ones, i.e. BEFORE the launch bug was fixed, and `scontrol write
batch_script` confirmed they still carried the pre-fix ordering (legacy loop
at line 198, the fatal `NINST=` at 201, the w2a driver stranded at 224).
Editing the file on disk does not touch queued jobs, so all eight would have
died in ~60s the moment the pilot released them, exactly as the 7B batch
did. They were cancelled and resubmitted against the current script
(11624731-38, same `afterok` dependency), verified to contain the guard and
the corrected ordering. Any future fix to the sbatch must be followed by
re-submitting anything already queued.

**FB-06 (2026-08-02): a failure-mode table is added to the appendix.** F9
forbids ADDING tables without a freeze-break, so this is logged. It reports
no new endpoint: every column is a decomposition of emissions the existing
endpoints already count, computed from the same gold licences.

Justification: false authorization conflates defects that the confirmatory
data shows are distinct. `react` loses 76.6% of its tool calls to
field-not-found and fails at ADDRESSING the record, not judging it, because
schemas differ in guessability (OSV exposes `published`; ClinicalTrials.gov
nests `protocolSection.identificationModule.briefTitle`). Arms with the
specification view locate evidence reliably and then OVER-REACH, with their
unsupported mass at the strongest claim attempted. And the informed arms
emit against their own attested ledger -- every route to the claim
containing an atom they had themselves established false -- 19 times for
`react_program_attested` against 2 for `react`. Reported as one rate, those
three look alike; separated, they say what the ladder actually shows.

The table goes to the appendix rather than the body: T1 is already eight
columns and the 9-page body is at its limit. The body's T1 prose states the
addressing confound, because a reader must not credit the vocabulary rung
with fixing judgement when it fixed addressing.

**FB-05 (2026-08-02): W2b induces ONE program per template family rather
than one per instance.** Not a mechanism change -- the runtime, the mask and
the commitment path are untouched -- but it changes the number of
independent inductions from 480 to 60, so it is logged.

Feasibility: state-first induction asks the model about every realizable
evidence state, which is 1,152 calls for a median instance (128 states x 3
claims x 3 draws) and 2,304 at the maximum. Per instance that is 552,960
calls per arm and 1.66M across the three W2b arms, roughly 154 GPU-hours on
a preemptible free queue. W2b was not runnable as specified.

Correctness: within a template family the user prompt and the claim texts
are byte-identical and ONLY the record identifier differs. That identifier
is a locator -- it says which advisory to fetch, not which evidence
licenses which claim -- so the elicited support logic is the same object
4-15 times over. Caching it per family removes redundant work rather than
approximating anything.

It also makes the inference honest. The family is already the clustering
unit for every test in the study, so per-instance induction would have put
4-15 noisy draws of the SAME induction inside each cluster and invited
treating them as independent evidence about induction quality. One
committed program per family is what the design means.

Reported consequence: W2b's induction-quality endpoints are per-family
(n=60 programs), and the episode-level endpoints remain per-instance
(n=480) under those 60 committed programs. Both n's appear in T4.

**FB-04 (2026-08-02): T3 (cross-family transfer) and T4 (induction and
vocabulary conditions) move from the 9-page body to an appendix, with a
two-sentence pointer in Experiments.** F9 froze WHICH tables exist and what
they contain; it did not fix their placement, and it forbids ADDING tables,
not relocating them. Nothing is dropped: both tables, both slots and both
subsections move verbatim, and the limitations section still states the S1
result as the paper's weakest point. Justification: measured against the
correct text block (5.5in x 9.0in, not the 6.5in the inherited
approximation used), the assembled body ran to 13 pages against a 9-page
limit. After relocating 42 regions and 5,919 words losslessly it is 10. T1
(the ladder) and T2 (enforcement and the control law) carry the paper's two
claims; T3 and T4 are secondary, and a body that cannot show T1 and T2
properly is worse than one that cites T3 and T4 to an appendix.

**FB-03 (2026-08-02): `compile_routes_exact` width cap made opt-in.** Not a
freeze change to any endpoint, arm, or statistic -- recorded here because it
changes what W2b can produce. The compiler defaulted to `max_width=6` and
`induce_state_first` inherited that default, while 189 gold routes in the
corpus are 7 atoms wide: those claims compiled to an EMPTY route family, so
an induced program could not license them at all. Left in place, W2b would
have reported a configuration cap as an induction-quality result. The cap is
now opt-in for vocabularies at or below 16 atoms and still applies above.
Caught by `analysis/w3_freeze.py`, which round-trips the compiler against
perfect labels derived from gold and requires exact agreement on every
realizable state; it now PASSES on all 480 confirmatory instances (1,440
claims, 74,784 states, zero disagreements) and is the W3 exit gate.

**W3 EXIT NOTE (2026-08-02).** 129 of the 480 balanced instances carry a
resolver-less atom on a gold route (146 atoms, `gold_value` F, producing
exactly the 235 `content=unavailable` claims). Nothing can establish those
atoms, so `unresolvable_atom_in_route` fires and `eligibility` refuses to
guide toward those claims. That is the fail-closed design working, not a
defect -- but W2b MUST report those claims as guidance-ineligible for a
STRUCTURAL reason, or the induced-guidance arm will look worse than it is.

## Part 7 — Status

**FROZEN.** Claude confirmed round 4; GPT's four objections are resolved
above exactly as specified. Any further change is a named freeze-break
logged in this section with justification. Execution begins at W0.
