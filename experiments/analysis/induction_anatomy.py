#!/usr/bin/env python3
"""Why state-enumeration induction fails, as a mechanism rather than a verdict.

The paper reports that inducing the support logic rather than authoring it
costs utility and breaks the zero. That is a verdict; the obvious next
question is a diagnosis that generalises -- over-conjunction?
unresolvable atoms? schema violation? -- with evidence that it is the binding
constraint. This computes that diagnosis.

Four measured facts, each independently checkable from the artifacts:

  1. ADMISSION IS AN EXPONENTIAL CONJUNCTION OF NOISY LABELS.
     compile_routes_exact admits a route r only if EVERY enumerated evidence
     state containing r was accepted, and acceptance under the `unanimous`
     policy needs every draw to agree. Admitting a width-w route over an
     n-atom vocabulary therefore requires draws * 2^(n-w) labels to come back
     "yes" together. The requirement falls exponentially in w, so the
     procedure is structurally biased AGAINST narrow routes -- and narrow
     routes are the only ones a budgeted agent can finish gathering. This is
     a property of the method, not of the model.

  2. THE BIAS IS VISIBLE IN THE OUTPUT.
     Gold routes are narrow and bimodal; induced routes are wide and spread.

  3. WIDTH IS THE BINDING END-TO-END CONSTRAINT.
     The agent resolves ~3 atoms per episode whatever its budget, so the
     probability a route ever fires collapses with its width.

  4. CONSERVATISM CANNOT REPAIR IT.
     Committing the whole vocabulary as a single route is soundness-preserving
     by construction -- every gold route lies inside the vocabulary and no
     claim has an empty gold licence -- so it attains zero false authorization.
     It is also worthless, because a route that wide almost never fires. Every
     sound repair available without new elicitation makes routes wider, which
     is precisely what the agent cannot pay for.

Writes: results/induction_anatomy.json
Usage:  python3 experiments/analysis/induction_anatomy.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "src"), HERE, os.path.join(PILOT, "src", "prime_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import induction_policy_lab as lab   # noqa: E402

RESULTS = os.path.join(PILOT, "results")
DRAWS = 3                            # the committed induction's draw count


def resolved_per_episode(model="qwen2.5-32b", arm="prime_guided"):
    """Atoms the agent actually establishes, by budget anchor."""
    out = collections.defaultdict(list)
    for p in glob.glob(os.path.join(RESULTS, "w2a", "%s-*" % model,
                                    "%s__*" % arm, "summary.json")):
        tag = os.path.basename(os.path.dirname(p)).split("__")[-1]
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            led = e.get("final_ledger") or {}
            out[tag].append(sum(1 for v in led.values() if v in ("T", "F")))
    return {k: round(statistics.mean(v), 2) for k, v in out.items() if v}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    insts = lab.load_instances()
    ind = lab.load_cached()

    # (1) the admission requirement
    vocab = {}
    for i in insts:
        vocab.setdefault(i["family"], len(i["atoms"]))
    n = int(statistics.median(vocab.values()))
    labels = {w: DRAWS * 2 ** (n - w) for w in range(1, n + 1)}
    survival = {w: round(100 * (0.98 ** L), 1) for w, L in labels.items()}

    # (2) the width distributions
    iw, gw = collections.Counter(), collections.Counter()
    for prog in ind.values():
        for routes in prog.values():
            for r in (routes or []):
                iw[len(r)] += 1
    seen = set()
    for i in insts:
        if i["family"] in seen:
            continue
        seen.add(i["family"])
        for routes in i["gold"].values():
            for r in (routes or []):
                gw[len(r)] += 1

    # (3) firing probability by width, measured
    K = []
    for p in glob.glob(os.path.join(RESULTS, "w2a", "qwen2.5-32b-*",
                                    "prime_guided__B20", "summary.json")):
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            led = e.get("final_ledger") or {}
            K.append(sum(1 for v in led.values() if v in ("T", "F")))
    fire = {w: round(sum(1 for k in K if k >= w) / len(K), 3)
            for w in range(1, 9)} if K else {}

    # (4) the backstop: sound by construction, worthless in practice
    back = {}
    for i in insts:
        back.setdefault(i["family"], {c: [sorted(i["atoms"])] for c in i["gold"]})
    viol = sum(1 for i in insts for c, rs in i["gold"].items()
               for r in rs if not set(r) <= set(i["atoms"]))
    empty = sum(1 for i in insts for rs in i["gold"].values() if not rs)

    out = {
        "vocab_median": n, "draws": DRAWS,
        "labels_required": labels, "survival_at_2pct": survival,
        "gold_widths": dict(sorted(gw.items())),
        "induced_widths": dict(sorted(iw.items())),
        "resolved_per_episode": resolved_per_episode(),
        "fire_prob_by_width": fire,
        "backstop_sound": viol == 0 and empty == 0,
        "backstop_lab": lab.score(back, insts, "backstop"),
        "backstop_width_aware": lab.score_width_aware(back, insts, "backstop"),
        "induced_lab": lab.score(ind, insts, "induced"),
        "induced_width_aware": lab.score_width_aware(ind, insts, "induced"),
    }
    with open(os.path.join(RESULTS, "induction_anatomy.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print("induction anatomy (median vocabulary %d atoms, %d draws)" % (n, DRAWS))
    print("  admitting a width-w route needs draws*2^(n-w) labels to agree:")
    for w in sorted(labels):
        print("     w=%d  %4d labels   survives at 2%%%% label error: %5.1f%%"
              % (w, labels[w], survival[w]))
    print("  route widths   gold: %s" % out["gold_widths"])
    print("                 ind.: %s" % out["induced_widths"])
    print("  atoms resolved per episode by budget: %s" % out["resolved_per_episode"])
    print("  P(route of width w fires): %s" % fire)
    print("  fail-closed backstop sound by construction: %s" % out["backstop_sound"])
    print("     lab %.1f util -> width-aware %.1f util"
          % (out["backstop_lab"]["utility"], out["backstop_width_aware"]["utility"]))
    print("wrote results/induction_anatomy.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
