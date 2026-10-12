"""Batch driver: run one arm over a set of benchmark instances.

For every instance matched by ``--instances``, runs one episode of ``--arm``
(src/prime_agents/run_episode.py) and scores its emissions against the reconstructed
gold ledger. Writes to ``--out`` (default
``results/<arm>__<license-source>__<model>/``) one ``<instance>.jsonl`` trace
per episode plus a ``summary.json`` with per-episode emitted claims and
emission steps, CompleteAuth_gold per emission, tool calls, and tokens.

Usage (from the package root):
    python3 src/prime_agents/run_batch.py --instances 'data/pilot/instances/swe/*.json' \
        --arm prime --license-source gold --model gpt-4o-mini --parallel 4

    python3 src/prime_agents/run_batch.py --dry-run --arm prime --license-source gold

``--dry-run`` runs the full loop with a scripted deterministic fake LLM
(no API calls) on the fixture instance unless ``--instances`` is given.
``--budget-anchor`` derives per-instance budgets from the committed program
(see ``instance_budget``); ``--resume`` skips episodes whose sidecar in
``<out>/_episodes/`` recorded no error.
"""

import argparse
import concurrent.futures
import glob
import json
import os
import sys
from typing import Any, Dict, List

_PILOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUNNER = os.path.dirname(os.path.abspath(__file__))
for _p in (_PILOT, os.path.join(_PILOT, "src"), _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime.schema import load_instance  # noqa: E402

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from run_episode import run_episode  # noqa: E402

FIXTURE_INSTANCE = os.path.join(
    _PILOT, "tests", "fixtures", "instance_toy.json"
)


def _make_llm(instance_path: str, args: argparse.Namespace):
    if args.dry_run:
        instance = load_instance(instance_path)
        program_json = json.dumps({"licenses": {
            cid: [list(r) for r in routes]
            for cid, routes in instance.gold_licenses.items()
        }})
        return llm_mod.ScriptedLLM(
            llm_mod.build_dry_run_script(instance), program_json=program_json
        )
    return llm_mod.LLMClient(model=args.model,
                             temperature=getattr(args, "temperature", 0.0),
                             seed=getattr(args, "seed", None))



def instance_budget(path: str, anchor: str, fixed: int) -> int:
    """Spec-only per-instance budget anchors (docs/ANALYSIS_PLAN.md, F4).

    Shared by run_batch and src/prime_agents/policies.py so the LLM arms and the
    scripted/computed policy arms are budgeted by IDENTICAL code. They were
    not: policies.py only accepted a fixed integer, so the whole policy
    family could only be measured at B=20 -- the regime the W0 dev pilot had
    already shown is converged (every arm reaches the ceiling, so all
    utilities coincide). G_dev and the pre-committed masked_reactive_prefix
    attribution both live at the CONSTRAINED anchors, where the arms differ.

    bmin = min call-class cost over the committed gold program's routes to
    the strongest schema claim, computed from the program alone, never
    gold-aware.
    """
    if anchor == "fixed":
        return fixed
    from prime_runtime import guidance, status
    from prime_runtime.ledger import Ledger
    inst = load_instance(path)
    program = inst.gold_program()
    resolvers = status.resolver_table(inst)
    ledger = Ledger(inst.atom_ids, inst.phi_patterns)
    top = max(inst.claims, key=lambda c: (c.level, c.id))
    g = guidance.reachable_ceiling(program, ledger, resolvers,
                                   {c.id: c.level for c in inst.claims}, 10**6)
    bmin = g["per_claim"][top.id]["cost"]
    if bmin is None:
        return fixed
    return {"bmin": bmin, "bmin1": bmin + 1, "2bmin": 2 * bmin}[anchor]


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default="", help="glob of instance JSON files")
    parser.add_argument(
        "--arm",
        required=True,
        # NOT a second hand-maintained list: it drifted from arms.ARMS, so a
        # newly registered arm was silently unrunnable from the CLI.
        choices=arms_mod.ARMS,
    )
    parser.add_argument("--license-source", default="gold", choices=("gold", "induced"))
    parser.add_argument(
        "--induction-method", default="oneshot",
        choices=("oneshot", "state_first"),
        help="HOW an induced program is obtained (W2b). 'oneshot' asks the "
        "model for routes directly and is the published condition, so it "
        "stays the default and no existing result changes. 'state_first' "
        "runs the W3 conservative stack: enumerate realizable evidence "
        "states, label each from a spec-only description, aggregate over "
        "draws, compile minimal routes mechanically, then gate on "
        "counterfactual validation and the typed linter. The stack existed "
        "but was NOT reachable from the runner, so a W2b run would have "
        "measured the one-shot induction the paper does not describe.")
    parser.add_argument(
        "--revision-mode",
        default="fixed",
        choices=("fixed", "strengthen_only", "unfrozen_lattice", "arbitrary", "posthoc"),
        help="revision-freedom ablation mode (prime arm only; 'fixed' is the "
        "deployed full-freeze default)",
    )
    parser.add_argument("--model", default=os.environ.get("PRIME_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--parallel", type=int, default=1)
    # Confirmatory runs are greedy and single-sample. These exist so the
    # decoding-variance question can be ANSWERED rather than assumed away:
    # at temperature 0 a "seed" changes nothing, so measuring sampling
    # sensitivity requires actually sampling.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-tool-calls", type=int, default=20,
        help="matched action budget (default 20, Table C; E3 frontier uses 4/8/16)",
    )
    parser.add_argument(
        "--budget-anchor", default="fixed",
        choices=("fixed", "bmin", "bmin1", "2bmin"),
        help="v5 spec-only per-instance budgets (docs/ANALYSIS_PLAN.md, F4): bmin = min "
        "call-class cost over the committed gold program's routes to the "
        "strongest schema claim, computed from the program alone (never "
        "gold-aware); bmin1 = bmin+1; 2bmin = 2*bmin. 'fixed' uses "
        "--max-tool-calls unchanged (default; all published behavior).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use a scripted deterministic fake LLM (no API calls)",
    )
    parser.add_argument(
        "--induction-view", default="ids", choices=("ids", "spec"),
        help="how much of the specification view the license-INDUCTION policy "
        "is conditioned on (induced license-source only). 'ids' is the "
        "originally-run condition and the default, so published induced "
        "numbers stay reproducible; 'spec' additionally supplies claim text "
        "and each atom's type/resolver/locator, i.e. the full specification "
        "view Definition 2 grants (still no atom proposition, gold value, or "
        "gold licenses).",
    )
    parser.add_argument(
        "--comparator-spec-view", action="store_true",
        help="information-matched comparators: give direct/rubric/cove/"
        "reflexion/conformal/static_contract the specification-view block "
        "(claim text, atom types, resolver tools and locators) that the "
        "ladder's +spec rung carries. Sets PRIME_COMPARATOR_SPEC=1; off by "
        "default, so every published comparator prompt is unchanged.")
    parser.add_argument("--out", default="", help="output dir (default results/<run-id>/)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip instances whose per-episode summary sidecar already exists "
        "in <out>/_episodes/ and recorded no error (for preemptible queues, "
        "e.g. a free, preemptible cluster queue). Off by default: without it the "
        "batch re-runs every instance exactly as before.",
    )
    args = parser.parse_args(argv)

    if args.comparator_spec_view:
        os.environ["PRIME_COMPARATOR_SPEC"] = "1"
    if args.dry_run:
        args.model = "scripted"
    pattern = args.instances or (FIXTURE_INSTANCE if args.dry_run else "")
    if not pattern:
        parser.error("--instances is required unless --dry-run")
    paths = sorted(glob.glob(pattern))
    if not paths:
        parser.error("no instances match %r" % pattern)

    run_id = "%s__%s__%s" % (args.arm, args.license_source, args.model)
    out_dir = args.out or os.path.join(_PILOT, "results", run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Per-episode sidecars: written as each episode finishes so a preempted job
    # (preemptible queue) can be resubmitted with --resume and only re-run what
    # is actually missing. summary.json is still written whole at the end and
    # remains the single artifact experiments/analysis/ reads.
    ep_dir = os.path.join(out_dir, "_episodes")
    if args.resume:
        os.makedirs(ep_dir, exist_ok=True)

    def _sidecar(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(ep_dir, "%s.json" % stem)

    def _instance_budget(path: str) -> int:
        return instance_budget(path, args.budget_anchor, args.max_tool_calls)

    def _one(path: str) -> Dict[str, Any]:
        if args.resume:
            side = _sidecar(path)
            if os.path.exists(side):
                try:
                    with open(side, encoding="utf-8") as fh:
                        done = json.load(fh)
                    if not done.get("error"):
                        print("resume: skipping %s (already complete)" % done.get("instance", path))
                        return done
                except (OSError, json.JSONDecodeError):
                    pass  # unreadable/truncated sidecar -> re-run this episode
        client = _make_llm(path, args)
        summary = run_episode(
            path, args.arm, args.license_source, client, out_dir,
            revision_mode=args.revision_mode, max_tool_calls=_instance_budget(path),
            induction_view=args.induction_view,
            induction_method=args.induction_method,
        )
        if args.resume:
            with open(_sidecar(path), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
        return summary

    episodes: List[Dict[str, Any]] = []
    if args.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            episodes = list(pool.map(_one, paths))
    else:
        episodes = [_one(p) for p in paths]

    summary = {
        "run": {
            "arm": args.arm,
            "license_source": args.license_source,
            "model": args.model,
            "dry_run": bool(args.dry_run),
            "instances": paths,
            "max_tool_calls": args.max_tool_calls,
            "induction_view": args.induction_view,
            "induction_method": args.induction_method,
        },
        "episodes": episodes,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("wrote %s (%d episode(s))" % (summary_path, len(episodes)))
    errors = [e for e in episodes if e.get("error")]
    if errors:
        print("episodes with errors: %d" % len(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
