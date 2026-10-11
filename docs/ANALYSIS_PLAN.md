# W0 Freeze Document (EXECUTION_PLAN v5) — v0.9, items frozen as completed

Every value here is frozen before any confirmatory run on held-out v2
instances. Slots marked PENDING-PILOT are filled by the named dev pilot and
then hashed; nothing else may change without a named freeze-break in
EXECUTION_PLAN.md.

## F1. Utility function U (FROZEN)
Claims sit on 3-level ladders (levels 1..3; abstention = 0).
U(c) = level(c) / 3, reported on the [0, 100] scale (x100).
Adjacent-level gaps: uniform 33.33 points; median adjacent gap = 33.33.

## F2. delta_min (FROZEN, derived from F1 per P4/N2)
delta_min = 0.15 x median adjacent-level utility gap = 0.15 x 33.33 = 5.0
points. Interpretation derived from the actual frozen U (not an
illustration): a 5.0-point mean shift equals upgrading one episode in 6.67
by exactly one adjacent level.

## F3. MDE for E2 (FROZEN 2026-08-02: delta = 6.16)

NOTE (2026-08-02). G_dev must come from the DEV split -- computing it on
heldout would let the confirmatory data choose the equivalence margin. The
planner side is now staged on dev for every budget the pilot uses (branching
stratum, n=96): informed_route_planner utility 50.69 at B=3, 65.62 at B=4
and B=20. Only masked_reactive_v(32B, dev, branching) is outstanding, and
the pilot is producing it; delta is then one subtraction away.

MEASURED SEPARATELY, and NOT a margin input: on the held-out balanced split
the deployable ceiling-targeting policy (shortest_route) exactly matches the
nondeployable clairvoyant_ceiling at every anchor, so the planning headroom
above a simple heuristic is ZERO. The maximum effect any control law can
show on this benchmark is therefore the ordering-to-direction gap, which is
+9.58 points in the branching stratum at B_min and B_min+1 and +0.00 at
2*B_min and B=20 (analysis/policy_frontier.py, LLM-free). An E2 null must be
read against that bound rather than as an unbounded failure to separate.
delta = max(5.0, 0.20 x G_dev), where G_dev = mean utility-regret gap
masked_reactive_v -> informed_route_planner on the DEV branching split at
the frozen budget set. The clairvoyant_ceiling planner is never the MDE
denominator (it is the nondeployable regret ceiling).
ONE delta is frozen for the study, computed from the PRIMARY panel
(qwen2.5-32b per F7); the secondary panel is reported against the same
delta so that no per-model margin can be selected after the fact.
  qwen2.5-7b (secondary, dev, 288 obs / 32 clusters): G_dev = 35.65 ->
    delta would be 7.13; TOST power at zero effect 0.736 (< 0.8), so for
    the 7B panel the paper reports an interval, NOT a verdict (the
    pre-committed contingency).
  qwen2.5-32b (PRIMARY, dev branching, 288 obs / 32 clusters, job
    11623115 COMPLETED 02:50:03): G_dev = 30.79 -> **delta = 6.16 is the
    ONE frozen equivalence margin for the study**, and the 7B secondary
    panel is reported against this same 6.16, not against its own 7.13.
    TOST power at zero effect = 0.708 with cluster SD 12.91 (< 0.8), so the
    PRIMARY panel also reports an interval and declines an equivalence
    verdict. That contingency was pre-committed before any confirmatory
    episode was scored, and it is enforced in code (F6).

  Two further W0-exit readings from the same pilot, both destined for the
  body either way:
  - Instruction effect (react_program vs react_program_verify, 32B dev,
    n=48/cell): OFAR moves 0.208 vs 0.250 clinical, 0.208 vs 0.167
    drugsafety, 0.208 vs 0.292 financial -- small and sign-inconsistent, so
    the compliance instruction is not what carries the ladder.
  - E1 degeneracy watch (R4 Claude-N1): 31 of 48 dev clusters are
    discordant on react_program_attested vs masked_reactive_v at 32B, so
    E1 is NOT floored for the primary model -- unlike the 7B confirmatory
    split, where it was 0/45. The contrast has room to move at 32B.

## F4. Budget sweep (FROZEN 2026-08-02: anchors unshifted)
B_min = specification-only: min over routes of C_max of call-class cost,
computed from the committed program + call-class schema (never gold-aware).
FROZEN SET: {B_min, B_min+1, 2*B_min, 20}. Final dev pilot (2026-08-02, on
the corrected 50/50-topology dev split, 192 instances, policy arms +
planners, absolute budgets 3/4/20 bracketing per-instance B_min): policies
diverge sharply in the B_min region AND the divergence is topology-
moderated exactly as K1 pre-registers -- at B=4, shortest_route reaches
0.615 in the BRANCHING stratum vs 0.493 for cheapest_literal/prefix (a
12.2-point route-selection gap), while in the NESTED stratum all non-random
policies are identical (0.465) at every budget; random_masked collapses to
0.163-0.295 at tight budgets everywhere. Anchors unshifted. Regenerable via
runner/policies.py on bench/instances_v2/*/dev.

## F5. Protocols (FROZEN at generation; re-hashed after financial regen)
F*_A = generated gold licenses. F*_B = mechanical conjunctive collapse with
SAT realizability gate (analysis/protocols_v2.py). Unrealizable collapsed
routes: 0 so far. Hashes: results/protocols_v2.json (per split). The
retired endpoint "protocol-robust unsupported" is never reported; the
reported set is OFAR-A, OFAR-B, disagreement rate, sensitivity
(OFAR_B - OFAR_A), content-false (three-value rule: false if any evaluable
conjunct false; true only if fully evaluated true; else unavailable).

## F6. Statistical decision rules (FROZEN)
- Units: two-way clusters (record x template family); every table reports
  instance n AND cluster n.
- E1 (gold; react_program_attested vs masked_reactive_v; episode OFAR):
  inferential primary = template-cluster randomization test (per-record
  paired differences -> family aggregate -> family-level sign-flip ->
  stratified by domain and topology; statistic = mean cluster-weighted
  paired difference). Effect size = two-way cluster bootstrap CI.
  Sensitivity = exact McNemar. E1 always reports the discordant-cluster
  count beside p (degenerate-floor detection per R4 Claude-N1).
- E2 (branching stratum; utility-unit regret): paired permutation primary;
  cluster bootstrap CI; Wilcoxon sensitivity; TOST vs F3's delta. TOST
  power from dev variance decides "verdict vs CI-only" language.
  COMPUTED for the secondary panel (analysis/w0_pilots.py, 2026-08-02,
  qwen2.5-7b, dev branching, 288 obs / 32 clusters): delta = 7.13, cluster
  SD 14.6, power at zero effect = 0.736 -> BELOW 0.8, so the secondary
  panel gets interval language and no equivalence verdict. The PRIMARY
  (qwen2.5-32b) values remain pending its dev pilot, and the ONE frozen
  delta comes from that panel per F3. The gate is enforced in code:
  analysis/fill_v2_tables.py reads w0_pilots_<model>.json and refuses
  verdict wording whenever verdict_language is false, so a powered-looking
  confirmatory cell cannot grant permission the dev pilot withheld.
- Three separate Holm families: ladder contrasts; control-law contrasts;
  induction-method contrasts.

## F7. Arm matrix (FROZEN; run manifest in bench/instances_v2/manifest.json
+ results/run_manifest_v2.json at W2a launch)
Instructed primary chain: react_verify -> react_spec_verify ->
react_program_verify -> react_program_attested -> masked_reactive_v ->
prime_guided; uninstructed twins react / react_spec (unpadded primary,
padded sensitivity) / react_program; policy family adds shortest_route,
cheapest_literal, random_masked, masked_reactive_prefix (UPFRONT on all
branching confirmatory instances); computed planners: clairvoyant_ceiling,
informed_route_planner (+ information-matched exact planner on the
synthetic suite only). Guided arms: gold -> prime_guided; induced ->
prime_guided_gated primary, ungated = amplification diagnostic only.
Models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-32B-Instruct (exact HF
revisions recorded at run launch); optional gpt-oss-20b only under slack.

## F8. Natural-prevalence set (FROZEN)
60/domain (financial: 40 -- its record pool exhausted after the
50/50-balanced fill; shortfall recorded in instances_v2/manifest.json
rather than replaced, per the no-post-freeze-replacement rule), 4-arm
subset {react, react_program_attested, masked_reactive_v,
prime_guided}; sampling frames + two-type attrition in
bench/harvest/*/sampling_frame.json and instances_v2/manifest.json.

## F9. Table shells for the 9-page body (FROZEN slots; numbers only from
generators)
T1 main record-family result (OFAR-A/B, sensitivity, content-false,
utility, cost, both n's); T2 ladder decomposition + powered control law;
T3 cross-family transfer (software n=21, research n=8, record family);
T4 induction-under-spec + vocabulary panel. Figure 2: ladder waterfall.
Abstract number slots: [record-family OFAR headline], [enforcement gap +
p], [control-law verdict/CI], [S1 outcome sentence]. W6 may not add
tables beyond these without a freeze-break.
