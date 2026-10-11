#!/usr/bin/env python3
"""ObligationBench: load tasks, run an agent's checks, and score its conclusions.

Each task gives an agent one public record (a security advisory, a clinical-trial
registration, a drug recall, or a company filing), a short list of conclusions of
increasing strength, and the checks each conclusion requires. The agent reads fields
of the record and states the strongest conclusion it can justify. Which checks
suffice for each conclusion (its routes) is fixed in the task before any record is
read, and the true value of every check is computed from the record, so scoring is
executable: no model or human judges an answer.

This file needs only the Python standard library and the predicate grammar in
``predicates.py`` (shipped next to it, or importable from ``prime_runtime``).

Load tasks
    tasks = load_tasks("data/heldout_balanced.jsonl")    # {task_id: task}

Let any agent read the record (the only evidence action)
    env = RecordEnv(tasks["sec-v2-0012"])
    env.inspect_record("GHSA-2wx6-wc87-rmjm", "published")
    # -> {"record_id": ..., "field_path": ..., "value": "2020-03-19T17:29:58Z"}
    env.reads     # the reads so far, to store with the prediction
    env.ledger    # checks the reads established, e.g. {"E1": "T"}

Score predictions, one JSON object per line
    {"task_id": "sec-v2-0012", "stated": ["C2"],
     "reads": [["GHSA-2wx6-wc87-rmjm", "published"], ...]}

    python obligationbench.py score --tasks data/heldout_balanced.jsonl \\
        --predictions my_agent.jsonl [--by domain] [--json out.json]

``stated`` lists the conclusion ids the agent stated (empty means it abstained).
``reads`` lists the (record_id, field_path) reads made before the statement. It is
optional; without it, the premature-conclusion rate and evidence value are not
computed. The scorer replays the reads against the record and never trusts an
agent's own account of what it checked.

Endpoints, as defined in the paper
    false authorization   share of tasks on which some stated conclusion is not
                          supported by the record: none of its routes holds
    premature             share of tasks on which some stated conclusion was not
                          established by the agent's own reads, directly or
                          through a stronger established conclusion
    utility               100 x level of the strongest supported stated
                          conclusion / top level (levels 1-3: 33.3, 66.7, 100);
                          0 when nothing supported was stated
    evidence value        100 x level of the strongest conclusion the reads
                          establish / top level
    emission rate         share of answerable tasks (some conclusion is
                          supported) on which the agent stated anything
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # standalone copy shipped next to this file
    from predicates import evaluate_predicate, parse_predicate_rule
except ImportError:  # inside the PRIME code repository
    from prime_runtime.predicates import evaluate_predicate, parse_predicate_rule

Task = Dict[str, Any]


# ---------------------------------------------------------------- loading

def _task_from_row(row: Dict[str, Any]) -> Task:
    """A dataset row stores the verbatim task as JSON in ``task_json``; a task
    file is the task itself."""
    if "task_json" in row:
        return json.loads(row["task_json"])
    return row


def load_tasks(path: str) -> Dict[str, Task]:
    """Load tasks from a .jsonl data file, a single task .json file, or a
    directory searched recursively for task .json files."""
    tasks: Dict[str, Task] = {}
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        for f in files:
            if os.path.basename(f) == "manifest.json":
                continue
            with open(f, encoding="utf-8") as fh:
                t = json.load(fh)
            if isinstance(t, dict) and "gold_licenses" in t:
                tasks[t["id"]] = t
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = _task_from_row(json.loads(line))
                    tasks[t["id"]] = t
    else:
        with open(path, encoding="utf-8") as fh:
            t = _task_from_row(json.load(fh))
        tasks[t["id"]] = t
    if not tasks:
        raise SystemExit("no tasks found under %s" % path)
    return tasks


def domain_of(task: Task) -> str:
    """Task domain without the generator's version suffix (security_v2 -> security)."""
    return str(task.get("domain", "")).split("_v")[0]


# ------------------------------------------------------------ environment

def resolve_field_path(record: Any, field_path: str) -> Tuple[Any, bool]:
    """Walk a dotted path such as ``severity.0.score`` through nested JSON.
    Returns (value, found). Integer segments index lists."""
    if not field_path:
        return record, True
    cur = record
    for part in field_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.lstrip("-").isdigit():
            idx = int(part)
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return None, False
        else:
            return None, False
    return cur, True


class RecordEnv:
    """The record a task is about, and the one evidence action on it.

    ``inspect_record`` returns the raw field value, never a check's verdict. A
    check is established when a read of its field returns a value its predicate
    can evaluate; a failed read establishes nothing. Checks whose evidence is
    unavailable in the task (no resolver) can never be established.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.ledger: Dict[str, str] = {}
        self.reads: List[Tuple[str, str]] = []

    def inspect_record(self, record_id: str, field_path: str) -> Dict[str, Any]:
        record_id, field_path = str(record_id), str(field_path)
        self.reads.append((record_id, field_path))
        obs: Dict[str, Any] = {"record_id": record_id, "field_path": field_path}
        records = self.task.get("records") or {}
        if record_id not in records:
            obs["error"] = "record not found"
            return obs
        value, found = resolve_field_path(records[record_id], field_path)
        if not found:
            obs["error"] = "field not found"
            return obs
        obs["value"] = value
        params = {"record_id": record_id, "field_path": field_path}
        for atom in self.task.get("atoms", []):
            r = atom.get("resolver")
            if not r or r.get("tool") != "inspect_record":
                continue
            if r.get("params") and not all(params.get(k) == v for k, v in r["params"].items()):
                continue
            parsed = parse_predicate_rule(r.get("attest", ""))
            if parsed is None:
                continue
            result = evaluate_predicate(parsed[0], parsed[1], value)
            if result is not None:
                self.ledger[atom["id"]] = "T" if result else "F"
        return obs


# ---------------------------------------------------------------- scoring

def _levels(task: Task) -> Dict[str, int]:
    return {c["id"]: int(c["level"]) for c in task["claims"]}


def _holds(route: Sequence[str], values: Dict[str, str]) -> bool:
    """A route holds when each literal does: an atom id needs the value T,
    its negation ("!E3", text-evidence tasks) needs F."""
    return all((values.get(a[1:]) == "F") if a.startswith("!") else (values.get(a) == "T")
               for a in route)


def supported(task: Task) -> Dict[str, bool]:
    """Conclusions the record supports: some route whose checks are all true."""
    gold = {a["id"]: a["gold_value"] for a in task["atoms"]}
    return {c: any(_holds(r, gold) for r in routes)
            for c, routes in task["gold_licenses"].items()}


def established(task: Task, ledger: Dict[str, str]) -> set:
    """Conclusions the ledger establishes: those with a route all of whose
    checks the reads found true, plus every weaker conclusion (the task's
    ``order`` lists [weaker, stronger] pairs)."""
    licensed = {c for c, routes in task["gold_licenses"].items()
                if any(_holds(r, ledger) for r in routes)}
    weaker = defaultdict(set)
    for w, s in task.get("order", []):
        weaker[s].add(w)
    out: set = set()
    stack = list(licensed)
    while stack:
        c = stack.pop()
        if c not in out:
            out.add(c)
            stack.extend(weaker[c])
    return out


def score_task(task: Task, stated: Sequence[str],
               reads: Optional[Iterable[Sequence[str]]] = None) -> Dict[str, Any]:
    levels = _levels(task)
    top = max(levels.values())
    supp = supported(task)
    ceiling = max([levels[c] for c, ok in supp.items() if ok] + [0])
    best = max([levels.get(c, 0) for c in stated if supp.get(c, False)] + [0])
    row: Dict[str, Any] = {
        "task_id": task["id"],
        "domain": domain_of(task),
        "false_authorization": int(any(not supp.get(c, False) for c in stated)),
        "utility": 100.0 * best / top,
        "stated": int(bool(stated)),
        "answerable": int(ceiling > 0),
    }
    if reads is not None:
        env = RecordEnv(task)
        for rid, fp in reads:
            env.inspect_record(rid, fp)
        est = established(task, env.ledger)
        row["premature"] = int(any(c not in est for c in stated))
        row["evidence_value"] = 100.0 * max([levels[c] for c in est] + [0]) / top
        row["reads"] = len(env.reads)
    return row


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    answerable = [r for r in rows if r["answerable"]]
    out: Dict[str, Any] = {
        "n": n,
        "false_authorization_pct": 100.0 * sum(r["false_authorization"] for r in rows) / n,
        "utility": sum(r["utility"] for r in rows) / n,
        "emission_rate_pct": (100.0 * sum(r["stated"] for r in answerable) / len(answerable)
                              if answerable else float("nan")),
        "abstention_pct": 100.0 * sum(1 - r["stated"] for r in rows) / n,
    }
    with_reads = [r for r in rows if "premature" in r]
    if with_reads:
        m = len(with_reads)
        out["premature_pct"] = 100.0 * sum(r["premature"] for r in with_reads) / m
        out["evidence_value"] = sum(r["evidence_value"] for r in with_reads) / m
        out["mean_reads"] = sum(r["reads"] for r in with_reads) / m
    return out


def score(tasks: Dict[str, Task], predictions: Iterable[Dict[str, Any]],
          by: Optional[str] = None, missing: str = "abstain") -> Dict[str, Any]:
    """Score predictions against tasks. A task without a prediction counts as an
    abstention (``missing="abstain"``) or raises (``missing="error"``)."""
    preds = {p["task_id"]: p for p in predictions}
    unknown = sorted(set(preds) - set(tasks))
    if unknown:
        raise SystemExit("predictions for unknown task ids, e.g. %s" % unknown[:3])
    rows = []
    for tid, task in sorted(tasks.items()):
        p = preds.get(tid)
        if p is None:
            if missing == "error":
                raise SystemExit("no prediction for task %s" % tid)
            p = {"stated": []}
        rows.append(score_task(task, list(p.get("stated") or []), p.get("reads")))
    result: Dict[str, Any] = {"overall": summarize(rows)}
    if by == "domain":
        groups = defaultdict(list)
        for r in rows:
            groups[r["domain"]].append(r)
        result["by_domain"] = {d: summarize(g) for d, g in sorted(groups.items())}
    return result


def _print(result: Dict[str, Any]) -> None:
    cols = [("n", "%d"), ("false_authorization_pct", "%.1f"), ("premature_pct", "%.1f"),
            ("utility", "%.1f"), ("evidence_value", "%.1f"), ("emission_rate_pct", "%.1f"),
            ("abstention_pct", "%.1f"), ("mean_reads", "%.2f")]
    head = ["group"] + [c for c, _ in cols]
    lines = [head]
    groups = [("overall", result["overall"])] + sorted((result.get("by_domain") or {}).items())
    for name, s in groups:
        lines.append([name] + [(fmt % s[c]) if c in s else "-" for c, fmt in cols])
    widths = [max(len(str(l[i])) for l in lines) for i in range(len(head))]
    for l in lines:
        print("  ".join(str(v).rjust(w) for v, w in zip(l, widths)))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ObligationBench scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score a predictions file")
    s.add_argument("--tasks", required=True,
                   help="a .jsonl data file, a task .json file, or a directory of tasks")
    s.add_argument("--predictions", required=True, help="predictions .jsonl")
    s.add_argument("--by", choices=["domain"], default=None)
    s.add_argument("--missing", choices=["abstain", "error"], default="abstain")
    s.add_argument("--json", default=None, help="also write the result here")
    a = ap.parse_args(argv)
    tasks = load_tasks(a.tasks)
    with open(a.predictions, encoding="utf-8") as fh:
        preds = [json.loads(l) for l in fh if l.strip()]
    result = score(tasks, preds, by=a.by, missing=a.missing)
    _print(result)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
