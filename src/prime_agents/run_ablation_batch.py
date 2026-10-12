"""Batch runner for the 2^4 mechanism-interaction ablation (Table E): one
process per fractional-factorial cell (src/prime_agents/arms.py's ``ABLATION_CELLS`` /
``run_prime_ablation``).

Usage (from the package root):
    python3 src/prime_agents/run_ablation_batch.py \
        --instances 'data/pilot/instances/swe/*.json' \
        --model gpt-4.1 --parallel 4

    python3 src/prime_agents/run_ablation_batch.py --dry-run

Writes ``results/<model>/ablation__cell<i>__m<minimal><f><frozen>...
/summary.json`` per cell, in the SAME run-summary schema run_batch.py uses
(a ``run`` block + an ``episodes`` list), so aggregate.py's loaders and
experiments/analysis/pilot/end_to_end_tables.py read it uninterpreted.
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

from prime_runtime import goldledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402

FIXTURE_INSTANCE = os.path.join(
    _PILOT, "tests", "fixtures", "instance_toy.json"
)


def _cell_tag(cell: Dict[str, bool]) -> str:
    return "m%d_f%d_k%d_g%d" % (
        int(cell["minimal"]), int(cell["frozen"]), int(cell["masked"]), int(cell["graded"])
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
    return llm_mod.LLMClient(model=args.model)


def _run_one_episode(
    instance_path: str, cell: Dict[str, bool], client, out_dir: str, max_tool_calls: int
) -> Dict[str, Any]:
    instance = load_instance(instance_path)
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "%s.jsonl" % instance.id)
    if os.path.exists(trace_path):
        os.remove(trace_path)

    summary: Dict[str, Any] = {
        "instance": instance.id,
        "instance_path": os.path.abspath(instance_path),
        "arm": "prime_ablation",
        "license_source": "gold",
        "revision_mode": "fixed" if cell["frozen"] else "posthoc",
        "cell": dict(cell),
        "max_tool_calls": max_tool_calls,
        "trace": trace_path,
        "emitted": [],
        "tool_calls": 0,
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
        "error": "",
    }
    try:
        with TraceLogger(trace_path) as trace:
            result = arms_mod.run_prime_ablation(
                instance, client, trace, license_source="gold",
                max_tool_calls=max_tool_calls, **cell
            )
    except Exception as exc:
        summary["error"] = "%s: %s" % (type(exc).__name__, exc)
        return summary

    replay = goldledger.reconstruct(trace_path, instance)
    for emission in result.emitted:
        try:
            level = instance.claim(emission["claim"]).level
        except KeyError:
            level = None
        summary["emitted"].append({
            "claim": emission["claim"],
            "level": level,
            "step": emission["step"],
            "ledger_version": emission["ledger_version"],
            "complete_auth_gold": replay.complete_auth(emission["claim"], emission["step"]),
        })
    summary["seed"] = getattr(instance, "seed", None) or instance.id.split("__")[0]
    summary["induced_program_valid"] = getattr(result, "induced_program_valid", None)
    summary["refused_emissions"] = getattr(result, "refused", 0)
    summary["max_level"] = max((c.level for c in instance.claims), default=0)
    try:
        summary["max_completion"] = replay.max_completion_final()
    except AttributeError:
        summary["max_completion"] = replay.completion_at_final()
    summary["tool_calls"] = result.tool_calls
    summary["tokens"] = result.tokens
    summary["final_ledger"] = result.ledger_states
    summary["revisions_proposed"] = getattr(result, "revisions_proposed", 0)
    summary["revisions_accepted"] = getattr(result, "revisions_accepted", 0)
    summary["drift_events"] = getattr(result, "drift_events", 0)
    summary["revision_log"] = getattr(result, "revision_log", [])
    if result.error:
        summary["error"] = result.error
    return summary


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default="", help="glob of instance JSON files")
    parser.add_argument("--model", default=os.environ.get("PRIME_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-root", default="", help="default: results/<model>/")
    parser.add_argument(
        "--cells", default="", help="comma-separated cell tags to run (default: all 8)"
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        args.model = "scripted"
    pattern = args.instances or (FIXTURE_INSTANCE if args.dry_run else "")
    if not pattern:
        parser.error("--instances is required unless --dry-run")
    paths = sorted(glob.glob(pattern))
    if not paths:
        parser.error("no instances match %r" % pattern)

    out_root = args.out_root or os.path.join(_PILOT, "results", args.model)
    cells = arms_mod.ABLATION_CELLS
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = tuple(c for c in cells if _cell_tag(c) in wanted)

    overall_rc = 0
    for cell in cells:
        tag = _cell_tag(cell)
        out_dir = os.path.join(out_root, "ablation__%s" % tag)
        os.makedirs(out_dir, exist_ok=True)

        def _one(path: str, cell=cell, out_dir=out_dir) -> Dict[str, Any]:
            client = _make_llm(path, args)
            return _run_one_episode(path, cell, client, out_dir, args.max_tool_calls)

        if args.parallel > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
                episodes = list(pool.map(_one, paths))
        else:
            episodes = [_one(p) for p in paths]

        summary = {
            "run": {
                "arm": "prime_ablation",
                "license_source": "gold",
                "model": args.model,
                "dry_run": bool(args.dry_run),
                "instances": paths,
                "max_tool_calls": args.max_tool_calls,
                "cell": dict(cell),
                "cell_tag": tag,
            },
            "episodes": episodes,
        }
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        errors = [e for e in episodes if e.get("error")]
        print(
            "wrote %s (%d episode(s), %d error(s)) cell=%s"
            % (summary_path, len(episodes), len(errors), cell)
        )
        if errors:
            overall_rc = 1
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
