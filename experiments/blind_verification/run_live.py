#!/usr/bin/env python3
"""Live PRIME (prime_guided, gold program, 20 calls) with blind same-model
verification of text attestations, on the text-evidence split.

Mirrors runner/run_batch.py (same LLM client construction, same per-episode
sidecars and --resume, same summary.json shape) and installs
VerifiedToolExecutor in place of the runtime's ToolExecutor for the
duration of the process. Nothing under runner/ or prime_runtime/ is
edited. Each episode's verification events are written to
<out>/_verify/<instance>.json and summarized into the episode record under
"verification".

Usage (from the pilot root, with PRIME_LLM_BASE/MODEL set to the served model):
  python3 experiments/blind_verification/run_live.py --domain security --parallel 6
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))
for _p in (PILOT, os.path.join(PILOT, "runner"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as arms_mod  # noqa: E402
import llm as llm_mod  # noqa: E402
from run_episode import run_episode  # noqa: E402

import blind_verifier  # noqa: E402
import verified_executor as VE  # noqa: E402

ARM = "prime_guided"
MODELTAG = os.environ.get("MODELTAG", "qwen2.5-32b")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=("security", "clinical"))
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--max-tool-calls", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--instances", nargs="*", default=None,
                    help="run only these instance ids (for reruns of single episodes)")
    a = ap.parse_args(argv)
    blind_verifier.check_frozen()
    model = os.environ.get("PRIME_LLM_MODEL", "")
    if not model:
        print("PRIME_LLM_MODEL is not set", file=sys.stderr)
        return 2
    paths = sorted(glob.glob(os.path.join(PILOT, "bench", "instances_v2", a.domain, "heldout_text", "*.json")))
    if a.instances:
        keep = set(a.instances)
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in keep]
    if a.limit:
        paths = paths[: a.limit]
    out_dir = a.out or os.path.join(HERE, "results", "%s-%s" % (MODELTAG, a.domain), "%s_verified__B20" % ARM)
    ep_dir = os.path.join(out_dir, "_episodes"); ver_dir = os.path.join(out_dir, "_verify")
    os.makedirs(ep_dir, exist_ok=True); os.makedirs(ver_dir, exist_ok=True)

    # the verifier: same model, same endpoint, its own client (tools=None,
    # temperature 0, fresh messages per call)
    VE.VERIFIER_LLM = llm_mod.LLMClient(model=model, temperature=0.0, seed=None)
    # install the verifying executor for this process
    arms_mod.ToolExecutor = VE.VerifiedToolExecutor
    print("[live] arm=%s domain=%s n=%d model=%s prompt_sha256=%s parallel=%d" % (
        ARM, a.domain, len(paths), model, blind_verifier.prompt_digest(), a.parallel), flush=True)

    def sidecar(path):
        return os.path.join(ep_dir, os.path.splitext(os.path.basename(path))[0] + ".json")

    def one(path: str) -> Dict[str, Any]:
        side = sidecar(path)
        if os.path.exists(side):
            try:
                done = json.load(open(side, encoding="utf-8"))
                if not done.get("error"):
                    return done
            except (OSError, json.JSONDecodeError):
                pass
        iid = os.path.splitext(os.path.basename(path))[0]
        with VE._LOCK:
            VE.REGISTRY.pop(iid, None)
        client = llm_mod.LLMClient(model=model, temperature=0.0, seed=None)
        t0 = time.time()
        summary = run_episode(path, ARM, "gold", client, out_dir, max_tool_calls=a.max_tool_calls)
        with VE._LOCK:
            events = list(VE.REGISTRY.pop(iid, []))
        summary["verification"] = {
            "consultations": len(events),
            "invocations": sum(1 for e in events if e["invoked"]),
            "agreed": sum(1 for e in events if e["agree"]),
            "rejected": sum(1 for e in events if not e["agree"]),
            "verdicts": {v: sum(1 for e in events if e["verdict"] == v) for v in ("T", "F", "UNCLEAR", "PARSE_FAIL")},
            "prompt_tokens": sum(e["prompt_tokens"] for e in events),
            "completion_tokens": sum(e["completion_tokens"] for e in events),
            "latency_s": round(sum(e["latency_s"] for e in events), 3),
            "wall_s": round(time.time() - t0, 1),
        }
        with open(os.path.join(ver_dir, iid + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"instance": iid, "events": events}, fh, indent=1)
        with open(side, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print("[live] %s calls=%s emitted=%s verif=%d/%d err=%r" % (
            iid, summary.get("tool_calls"), [x["claim"] for x in summary.get("emitted", [])],
            summary["verification"]["agreed"], summary["verification"]["consultations"], summary.get("error", "")), flush=True)
        return summary

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as pool:
        episodes: List[Dict[str, Any]] = list(pool.map(one, paths))
    summary = {"run": {"arm": ARM + "_verified", "license_source": "gold", "model": model, "dry_run": False,
                       "instances": [os.path.relpath(p, PILOT) for p in paths], "max_tool_calls": a.max_tool_calls,
                       "induction_view": "ids", "induction_method": "oneshot",
                       "verifier": {"model": model, "prompt_sha256": blind_verifier.prompt_digest(),
                                    "spec": "verifier_spec.md"}},
               "episodes": episodes}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    errors = [e for e in episodes if e.get("error")]
    print("[live] wrote %s (%d episodes, %d errors)" % (os.path.join(out_dir, "summary.json"), len(episodes), len(errors)), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
