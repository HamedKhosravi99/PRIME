#!/usr/bin/env python3
"""Generate normalized blind-judge evaluation packets from logged traces.

Master §11.7: packets are produced from the logged environment trace by a
METHOD-INDEPENDENT procedure — never assembled by the evaluated agent — with
identical format across methods: artifact + declared scope, emitted claim,
every logged tool observation bearing on that claim, explicit conflicts, and
unavailable observations. No method name, no PRIME certificate, no
method-specific prose, and NO gold-license-derived "required evidence" hints
(judges decide what evidence the claim required).

Output: packets/ directory with shuffled packet_XXX.md files plus a hidden
key.json (packet -> arm/model/instance/claim) kept OUT of the judges' folder
for later unblinding. Judges record verdicts in verdicts_template.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PILOT)

from prime_runtime import schema  # noqa: E402


def load_trace(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def packet_body(instance, claim_id, emission_step, trace_entries):
    claim = instance.claim(claim_id)
    lines = []
    lines.append("# Evaluation packet")
    lines.append("")
    lines.append("## Task given to the system")
    lines.append("> %s" % instance.user_prompt)
    lines.append("")
    lines.append("**Artifact:** `%s` (%s)" % (instance.artifact.get("path", "?"), instance.artifact.get("kind", "?")))
    lines.append("**Declared scope:** %s" % instance.scope)
    lines.append("")
    lines.append("## Claim asserted by the system")
    lines.append("> %s" % claim.text)
    lines.append("")
    lines.append("## Complete record of the system's evidence-gathering actions")
    lines.append("(every tool action and its observation, in order, up to the moment of assertion; nothing omitted)")
    lines.append("")
    n_obs = 0
    for e in trace_entries:
        if e.get("type") != "observation" or int(e.get("step", 0)) > emission_step:
            continue
        n_obs += 1
        data = json.dumps(e.get("data", {}), default=str)
        if len(data) > 1200:
            data = data[:1200] + " …[truncated]"
        lines.append("**%d. %s** `%s`" % (n_obs, e.get("action", "?"), json.dumps(e.get("params", {}), default=str)))
        lines.append("")
        lines.append("```\n%s\n```" % data)
        lines.append("")
    if n_obs == 0:
        lines.append("*The system performed no evidence-gathering actions before asserting this claim.*")
        lines.append("")
    lines.append("## Question for you")
    lines.append(
        "Based only on the artifact scope, the claim, and the evidence record above: "
        "is this claim adequately supported by the work the system actually did? "
        "Answer SUPPORTED / NOT SUPPORTED / CANNOT JUDGE, with one sentence of reasoning. "
        "You decide what evidence the claim required.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(PILOT, "results"))
    ap.add_argument("--out", default=os.path.join(PILOT, "packets"))
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    entries = []
    for spath in glob.glob(os.path.join(args.results, "**", "summary.json"), recursive=True):
        # Blind-judge set = the DEPLOYED system's emissions only. Exclude the
        # drift-ablation and adversarial-reviser runs (they revise their own
        # program), which are scored in Table~tab:drift, not by human judges.
        if "/drift__" in spath or "/adversarial/" in spath or "__scripted/" in spath:
            continue
        s = json.load(open(spath))
        run = s.get("run", {})
        model = str(run.get("model") or "?")
        if "scripted" in model:
            continue
        for ep in s.get("episodes", []):
            if ep.get("error") or ep.get("revision_mode", "fixed") not in (None, "fixed"):
                continue
            if ep.get("arm") == "prime_adv":
                continue
            trace_path = ep.get("trace")
            if trace_path and not os.path.isabs(trace_path):
                trace_path = os.path.join(PILOT, trace_path)
            if not trace_path or not os.path.exists(trace_path):
                continue
            for em in ep.get("emitted", []):
                entries.append({
                    "instance_path": ep["instance_path"],
                    "trace": trace_path,
                    "claim": em["claim"],
                    "step": em["step"],
                    "arm": run.get("arm"),
                    "lsrc": run.get("license_source"),
                    "model": model,
                    "instance": ep["instance"],
                })
    if not entries:
        print("REFUSING: no completed emissions found.")
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    judge_dir = os.path.join(args.out, "for_judges")
    os.makedirs(judge_dir, exist_ok=True)
    key = {}
    for i, ent in enumerate(entries, 1):
        pid = "packet_%03d" % i
        inst = schema.load_instance(ent["instance_path"])
        body = packet_body(inst, ent["claim"], ent["step"], load_trace(ent["trace"]))
        with open(os.path.join(judge_dir, pid + ".md"), "w", encoding="utf-8") as fh:
            fh.write(body + "\n")
        key[pid] = {k: ent[k] for k in ("arm", "lsrc", "model", "instance", "claim", "step")}
    with open(os.path.join(args.out, "key.json"), "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2, sort_keys=True)
    with open(os.path.join(judge_dir, "verdicts_template.csv"), "w", encoding="utf-8") as fh:
        fh.write("packet,verdict(SUPPORTED|NOT_SUPPORTED|CANNOT_JUDGE),reason\n")
        for i in range(1, len(entries) + 1):
            fh.write("packet_%03d,,\n" % i)
    print("wrote %d packets -> %s (key withheld in %s)" % (len(entries), judge_dir, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
