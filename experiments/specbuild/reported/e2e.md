# Specification construction: end-to-end evaluation on held-out registrations

Reference: FDAAA TrialsTracker (level 3 reported late, 2 results due, 1 not yet due). Test registrations: 1999; agent subset: 600.

## (A) paper's program (coding assistant, readings chosen on the development pool)

Instances built: 526 of 1999; full-evidence level agreement with the tracker: 22.8%.

| policy | registrations | unsupported % [95% CI] | utility | coverage % |
|---|---|---|---|---|
| ceiling-targeting, no LLM (all) | 1999 | 1.8 [1.3, 2.5] | 18.4 | 26.3 |
| ceiling-targeting, no LLM (subset) | 600 | 1.7 [0.9, 3.0] | 19.0 | 26.8 |
| fully informed agent (unenforced) | 600 | 0.2 [0.0, 0.9] | 13.1 | 26.8 |
| PRIME | 600 | 0.2 [0.0, 0.9] | 9.9 | 26.8 |
| PRIME with completion | 600 | 1.7 [0.9, 3.0] | 19.0 | 26.8 |

By tracker status, ceiling-targeting on all registrations (unsupported % / coverage % / utility):

- reported (n=273): 3.3 / 54.2 / 33.9
- reported-late (n=357): 0.0 / 44.0 / 40.7
- overdue (n=491): 0.4 / 39.9 / 26.3
- ongoing (n=878): 2.8 / 2.8 / 0.0

## (B) protocolized program (open model, validation-selected, frozen)

Instances built: 1999 of 1999; full-evidence level agreement with the tracker: 81.6%.

| policy | registrations | unsupported % [95% CI] | utility | coverage % |
|---|---|---|---|---|
| ceiling-targeting, no LLM (all) | 1999 | 3.5 [2.8, 4.4] | 47.5 | 92.4 |
| ceiling-targeting, no LLM (subset) | 600 | 3.2 [2.0, 4.9] | 47.7 | 92.5 |
| fully informed agent (unenforced) | 600 | 0.0 [0.0, 0.6] | 31.2 | 92.5 |
| PRIME | 600 | 0.3 [0.1, 1.2] | 32.3 | 92.5 |
| PRIME with completion | 600 | 3.2 [2.0, 4.9] | 47.7 | 92.5 |

By tracker status, ceiling-targeting on all registrations (unsupported % / coverage % / utility):

- reported (n=273): 11.7 / 89.7 / 50.7
- reported-late (n=357): 0.0 / 80.4 / 72.9
- overdue (n=491): 0.4 / 90.8 / 55.6
- ongoing (n=878): 4.1 / 99.0 / 31.6

