#!/usr/bin/env python3
"""Adversarial-reviser drift ablation (revision freedom, freedom-permits row).

This is a DETERMINISTIC policy, not a language model. It measures what each
revision language *permits*, isolating the claim about revision freedom from
the temperament of any particular model: instruction-tuned models rarely weaken
their own standard spontaneously (the behavioral drift runs show ~0), but the
theorem is about what the freedom allows. The adversarial reviser exercises
exactly that freedom.

Policy (per instance):
  1. Establish only the CHEAP evidence a lazy agent would: run the honest
     tools that satisfy the weakest claim's gold route (via the real
     ToolExecutor + gold resolvers -- real repos, real observations).
  2. Target the STRONGEST claim C_top.
  3. If C_top is already authorized, emit it (no drift -- it earned it).
  4. Otherwise exercise the granted freedom to license C_top anyway:
       fixed / strengthen_only / unfrozen_lattice : cannot manufacture
         authorization for C_top (freeze, or Auth-shrinking edits only), so
         the policy falls back to the maximal *honestly* authorized claim
         (no drift, no false license -- the safe modes hold);
       arbitrary : remove literals from C_top's route until the current
         (cheap) evidence satisfies it -> authorization-expanding -> DRIFT,
         then emit C_top -> false license on the gold ledger;
       posthoc   : supply a program licensing C_top with exactly the atoms
         currently established -> DRIFT -> false license.

Every acceptance and its exact auth-expanding flag come from the SAME
interpreter.propose_revision path the model arms use; drift and gold-FLR are
scored identically. Output matches the model drift runs' summary schema so
drift_table.py / fill_paper.py consume it unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_PILOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PILOT not in sys.path:
    sys.path.insert(0, _PILOT)

from prime_runtime import goldledger  # noqa: E402
from prime_runtime import auth as prime_auth  # noqa: E402
from prime_runtime.interpreter import Interpreter  # noqa: E402
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import load_instance  # noqa: E402
from prime_runtime.trace import TraceLogger  # noqa: E402
from arms import ToolExecutor  # noqa: E402


def _cheap_atoms(instance):
    """Atoms in the weakest claim's first gold route (what a lazy agent does)."""
    weak = min(instance.claims, key=lambda c: c.level)
    routes = instance.gold_program().licenses.get(weak.id, ())
    return [a.lstrip("!") for a in (routes[0] if routes else ())]


def _establish(instance, ledger, executor, atom_ids, trace, step0=1):
    """Run the real tools whose attestations establish the given atoms."""
    step = step0
    for atom in instance.atoms:
        if atom.id not in atom_ids or atom.resolver is None:
            continue
        params = dict(atom.resolver.params)
        # fill minimal params for the standard tools
        tool = atom.resolver.tool
        if tool == "run_tests":
            params.setdefault("suite", "default")
        elif tool in ("inspect_file", "inspect_config"):
            params.setdefault("path", params.get("path", ""))
        obs = executor.execute(tool, params)
        trace.log_action(step, tool, params)
        trace.log_observation(step, tool, params, obs)
        for att in goldledger.delta(instance, ledger, tool, params, obs, step):
            trace.log_attestation(step, att, ledger.version)
        step += 1
    return step


def run_one(instance_path, mode, out_dir):
    instance = load_instance(instance_path)
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "%s.jsonl" % instance.id)
    if os.path.exists(trace_path):
        os.remove(trace_path)
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    interp = Interpreter(instance, revision_mode=mode)
    executor = ToolExecutor(instance)
    top = max(instance.claims, key=lambda c: (c.level, c.id))

    summary = {
        "instance": instance.id, "instance_path": os.path.abspath(instance_path),
        "arm": "prime_adv", "license_source": "gold", "revision_mode": mode,
        "trace": trace_path, "emitted": [], "error": "",
        "seed": getattr(instance, "seed", None) or instance.id.split("__")[0],
        "max_level": max((c.level for c in instance.claims), default=0),
    }
    try:
        with TraceLogger(trace_path) as trace:
            trace.log_meta(instance=instance.id, arm="prime_adv", license_source="gold")
            step = _establish(instance, ledger, executor, _cheap_atoms(instance), trace)

            emitted = None
            if top.id in prime_auth.auth_ids(interp.program, ledger):
                emitted = top.id  # earned honestly
            else:
                # exercise the granted freedom to license C_top
                if mode == "arbitrary":
                    _adv_weaken_route(interp, top, ledger, step)
                elif mode == "posthoc":
                    _adv_supply_program(interp, top, ledger, step)
                # (fixed / strengthen_only / unfrozen_lattice: no auth-expanding
                #  move is available; fall through to honest maximal claim)
                if top.id in prime_auth.auth_ids(interp.program, ledger):
                    emitted = top.id
                else:
                    maxa = prime_auth.max_auth(interp.program, ledger)
                    emitted = maxa[0] if maxa else None

            if emitted is not None:
                try:
                    obj = interp.emit(emitted, ledger)
                    trace.log_emission(step, obj, ledger.version)
                    summary["emitted"].append(
                        {"claim": emitted, "step": step, "ledger_version": ledger.version}
                    )
                except Exception:
                    pass

        replay = goldledger.reconstruct(trace_path, instance)
        for em in summary["emitted"]:
            try:
                em["level"] = instance.claim(em["claim"]).level
            except KeyError:
                em["level"] = None
            em["complete_auth_gold"] = replay.complete_auth(em["claim"], em["step"])
        summary["max_completion"] = replay.completion_at_final()
        summary["revisions_proposed"] = len(interp.revision_log)
        summary["revisions_accepted"] = sum(1 for e in interp.revision_log if e["accepted"])
        summary["drift_events"] = interp.drift_events
        summary["revision_log"] = interp.revision_log
        summary["tool_calls"] = len(_cheap_atoms(instance))
        summary["tokens"] = {"prompt_tokens": 0, "completion_tokens": 0}
        summary["final_ledger"] = ledger.states()
    except Exception as exc:  # noqa: BLE001
        summary["error"] = "%s: %s" % (type(exc).__name__, exc)
    return summary


def _adv_weaken_route(interp, top, ledger, step):
    """Remove literals from C_top's routes until the current ledger satisfies
    one (minimal weakening to license the unearned claim)."""
    from prime_runtime.closure import cl, entails
    closed = cl(ledger)
    routes = list(interp.program.licenses.get(top.id, ()))
    for ridx, route in enumerate(routes):
        for lit in list(route):
            atom = lit.lstrip("!")
            if not entails(closed, lit):
                interp.propose_revision(
                    "remove_literal", {"claim": top.id, "route_index": ridx, "literal": lit}, step=step
                )
        if top.id in prime_auth.auth_ids(interp.program, ledger):
            return


def _adv_supply_program(interp, top, ledger, step):
    """Supply a post-hoc program licensing C_top with exactly the atoms the
    cheap evidence has established (the post-hoc selector)."""
    from prime_runtime.closure import cl
    closed = cl(ledger)
    established = [a for a in interp.instance.atom_ids if closed.get(a) == "T"]
    licenses = {top.id: [established or []]}
    # keep weaker claims licensable too, so the program is well-formed
    for c in interp.instance.claims:
        if c.id != top.id:
            r = interp.program0.licenses.get(c.id, ())
            licenses[c.id] = [list(x) for x in r] if r else [established or []]
    interp.propose_revision("replace_program", {"licenses": licenses}, step=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=os.path.join(_PILOT, "bench/instances/swe/*.json"))
    ap.add_argument("--modes", nargs="+",
                    default=["fixed", "strengthen_only", "unfrozen_lattice", "arbitrary", "posthoc"])
    ap.add_argument("--out-root", default=os.path.join(_PILOT, "results/adversarial"))
    args = ap.parse_args()
    paths = sorted(glob.glob(args.instances))
    for mode in args.modes:
        out_dir = os.path.join(args.out_root, "advdrift__%s" % mode)
        episodes = [run_one(p, mode, out_dir) for p in paths]
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump({"run": {"arm": "prime_adv", "license_source": "gold",
                               "model": "adversarial-reviser", "revision_mode": mode},
                       "episodes": episodes}, fh, indent=2)
        drift = sum(e.get("drift_events", 0) for e in episodes)
        flr = sum(1 for e in episodes for x in e.get("emitted", []) if not x.get("complete_auth_gold"))
        print("%-16s episodes=%d drift_events=%d gold_false_licenses=%d"
              % (mode, len(episodes), drift, flr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
