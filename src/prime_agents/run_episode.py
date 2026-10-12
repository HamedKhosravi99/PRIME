"""Single-episode driver: run an arm on an instance, write the trace, and
score emissions against the reconstructed method-independent gold ledger."""

import os
import sys
from typing import Any, Dict, Optional

_PILOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUNNER = os.path.dirname(os.path.abspath(__file__))
for _p in (_PILOT, os.path.join(_PILOT, "src"), _RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prime_runtime import goldledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402

import arms as arms_mod  # noqa: E402


def run_episode(
    instance_path: str,
    arm: str,
    license_source: str,
    llm,
    out_dir: str,
    revision_mode: str = "fixed",
    max_tool_calls: int = arms_mod.MAX_TOOL_CALLS,
    induction_view: str = "ids",
    induction_method: str = "oneshot",
) -> Dict[str, Any]:
    """Run one episode; returns the per-episode summary record.

    Emission scoring: CompleteAuth_gold is computed by replaying the logged
    (action, observation) pairs through the instance's resolver definitions —
    never from the arm's self-reports.
    """
    instance = load_instance(instance_path)
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "%s.jsonl" % instance.id)
    if os.path.exists(trace_path):
        os.remove(trace_path)  # one fresh trace per episode run

    summary: Dict[str, Any] = {
        "instance": instance.id,
        "instance_path": os.path.abspath(instance_path),
        "arm": arm,
        "license_source": license_source,
        "revision_mode": revision_mode,
        "max_tool_calls": max_tool_calls,
        "induction_method": induction_method,
        "induction_view": induction_view,
        "trace": trace_path,
        "emitted": [],
        "tool_calls": 0,
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
        "error": "",
    }
    try:
        with TraceLogger(trace_path) as trace:
            result = arms_mod.run_arm(
                arm, instance, llm, trace, license_source, revision_mode,
                max_tool_calls=max_tool_calls, induction_view=induction_view,
                induction_method=induction_method,
            )
    except Exception as exc:
        summary["error"] = "%s: %s" % (type(exc).__name__, exc)
        return summary

    replay = goldledger.reconstruct(trace_path, instance)
    for emission in result.emitted:
        try:
            level = instance.claim(emission["claim"]).level
        except KeyError:
            # A claim the agent added in the unfrozen-lattice ablation: not a
            # benchmark claim, so it has no gold level; the gold ledger by
            # construction never authorizes it (complete_auth_gold = 0).
            level = None
        summary["emitted"].append(
            {
                "claim": emission["claim"],
                "level": level,
                "step": emission["step"],
                "ledger_version": emission["ledger_version"],
                "complete_auth_gold": replay.complete_auth(
                    emission["claim"], emission["step"]
                ),
            }
        )
    # canonical extras for experiments/analysis/aggregate.py
    summary["seed"] = getattr(instance, "seed", None) or instance.id.split("__")[0]
    summary["induced_program_valid"] = getattr(result, "induced_program_valid", None)
    summary["induction_eligibility"] = getattr(
        result, "induction_eligibility", None)
    summary["refused_emissions"] = getattr(result, "refused", 0)
    summary["max_level"] = max((c.level for c in instance.claims), default=0)
    try:
        summary["max_completion"] = replay.max_completion_final()
    except AttributeError:
        # per-claim max completion over the gold program at the final step
        summary["max_completion"] = replay.completion_at_final()
    summary["tool_calls"] = result.tool_calls
    summary["tokens"] = result.tokens
    summary["final_ledger"] = result.ledger_states
    # Revision-freedom ablation metrics:
    # all zero on the deployed fixed path.
    summary["revisions_proposed"] = getattr(result, "revisions_proposed", 0)
    summary["revisions_accepted"] = getattr(result, "revisions_accepted", 0)
    summary["drift_events"] = getattr(result, "drift_events", 0)
    summary["revision_log"] = getattr(result, "revision_log", [])
    # prime_guided only (obligation-guided completion): zero for every other
    # arm. See src/prime_runtime/guidance.py.
    summary["guided_holds"] = getattr(result, "guided_holds", 0)
    summary["guided_stalls"] = getattr(result, "guided_stalls", 0)
    summary["discharged_calls"] = getattr(result, "discharged_calls", 0)
    # conformal arm only ({} elsewhere): analysis-time cross-conformal
    # thresholding reads these (experiments/analysis/build_conformal.py).
    summary["claim_confidences"] = dict(getattr(result, "claim_confidences", {}) or {})
    # external-system arms only ({} elsewhere): mechanism counters (extsys.py)
    ext = dict(getattr(result, "ext", {}) or {})
    if ext:
        summary["ext"] = ext
    if result.error:
        summary["error"] = result.error
    return summary
