#!/usr/bin/env python3
"""Creation phase: an LLM turns 42 CFR Part 11 into PRIME programs (PROTOCOL.md).

Round 0: the regulation excerpt, the field catalog, three unlabelled example
registrations from the development pool, and the program format; one greedy
and N sampled candidates. Round 1 (repair): the selected round-0 program and
validation disagreements written by validate.py (feedback.json). The creator
never sees a test registration or a test label.

Usage: python3 create.py --round 0|1 [--samples 8]
Writes candidates/round<r>/cand_<k>.json (raw reply, parsed program or parse error).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.environ.get("SPECBUILD_PILOT") or os.path.dirname(os.path.dirname(HERE))
for p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "src", "prime_agents"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import common  # noqa: E402
import llm as llm_mod  # noqa: E402

PRED = """equals              field value == argument (JSON equality)
in_set              field value is one of the values in argument (a JSON list)
numeric_eq          float(value) == float(argument)
numeric_ge          float(value) >= float(argument)
numeric_le          float(value) <= float(argument)
in_version_range    version comparison (not useful here)
date_before         date(value) <  date(argument)   (ISO dates YYYY-MM-DD; a month-precision value YYYY-MM cannot be compared and leaves the atom unresolved)
date_after          date(value) >  date(argument)   (same date rules)
substring_contains  str(argument) occurs in str(value) (a list value is converted to its JSON text)
exists              the field is present (argument null)"""

SYSTEM = ("You turn a published regulation into an executable evidence program for an automated review runtime. "
          "Follow the regulation's text. Reply with exactly two fenced code blocks: a ```python block defining derive(record), then a ```json block.")


def base_prompt():
    reg = open(os.path.join(HERE, "inputs", "regulation_for_creator.txt")).read()
    cat = json.load(open(os.path.join(HERE, "inputs", "field_catalog.json")))
    ex = json.load(open(os.path.join(HERE, "inputs", "creation_examples.json")))
    claims = "\n".join("%s (level %d): %s" % (c["id"], c["level"], c["text"]) for c in common.CLAIMS)
    return f"""A review runtime examines one ClinicalTrials.gov registration at a time against the results-reporting rules of 42 CFR Part 11. The date of review is 3 August 2026. For each registration the runtime must establish which of three graded conclusions hold:
{claims}

You write the evidence program the runtime will commit to before it sees any registration. The program has three parts.

1. derive(record): a Python function that receives the embedded registration (a dict shaped like the examples below) and returns a dict of derived values (strings, dates as "YYYY-MM-DD", booleans, numbers, or None). Use it for anything that needs more than one field, date arithmetic, or iteration over lists. Only the modules datetime, math, re and json may be imported. Derived value d is readable by atoms as field "derived.d".

2. atoms: each atom applies ONE predicate to ONE field, either a dotted path into the registration (list items by 0-based index: the catalog writes an index as N, so "protocolSection.contactsLocationsModule.locations.N.country" means locations.0.country, locations.1.country, ...) or "derived.<name>". The argument is a constant; to compare one field with another field or with a computed date, compute the comparison in derive() and test the derived value. The runtime reads the field and evaluates the predicate; an absent field or a value the predicate cannot evaluate leaves the atom unresolved, never true or false. Predicates (closed grammar):
{PRED}

3. routes: for each of C1, C2, C3, a list of routes; a route is a list of atom ids, an id prefixed with "!" requiring that atom to be false. A conclusion is established when every literal of one of its routes holds. Every C2 route must contain all literals of some C1 route, and every C3 route all literals of some C2 route.

The regulation (excerpt from the electronic Code of Federal Regulations):
{reg}

Fields of the embedded registration (path, type, observed values, description):
{json.dumps(cat, indent=0)}

Three example registrations as embedded (no verdicts are given):
{json.dumps(ex, indent=0)}

Write the program. Reply with a ```python block containing derive(record), then a ```json block of the form
{{"atoms": [{{"id": "A1", "field": "...", "predicate": "...", "argument": ..., "meaning": "..."}}], "routes": {{"C1": [["A1", ...]], "C2": [[...]], "C3": [[...]]}}}}
Use at most 12 atoms. Do not refer to any specific registration."""


def repair_prompt(prev, feedback):
    return base_prompt() + f"""

A previous version of the program is below. On a validation sample of {feedback['n']} registrations, an independently written tracker of FDAAA compliance agreed with this program's strongest established conclusion on {feedback['agree']} ({feedback['agree_pct']:.1f}%). Disagreements (registration as embedded, the program's derived values, the program's strongest established level, and the tracker's level: 3 results submitted late, 2 results due and the deadline passed, 1 applicable but results not yet due):
{json.dumps(feedback['examples'], indent=0)}

Previous program:
```python
{prev.get('derive', '')}
```
```json
{json.dumps({'atoms': prev['atoms'], 'routes': prev['routes']}, indent=0)}
```
derive(record) raised an exception on {feedback.get('derive_errors', 0)} validation registrations{(': ' + '; '.join(feedback.get('derive_error_examples', []))) if feedback.get('derive_errors') else ''}.
The tracker is an independent implementation and may itself depart from the regulation. Revise the program where the regulation supports a change; do not special-case individual registrations. Reply with the full revised program in the same two-block format."""


def parse(reply):
    code = re.search(r"```python\s*(.*?)```", reply, re.S)
    js = re.search(r"```json\s*(.*?)```", reply, re.S)
    if not js:
        raise ValueError("no json block")
    prog = json.loads(js.group(1))
    prog["derive"] = code.group(1).strip() if code else ""
    common.validate_program_shape(prog)
    if prog["derive"]:
        common.check_code(prog["derive"])
    return prog


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=int, required=True); ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()
    model = os.environ.get("PRIME_LLM_MODEL", "")
    if not model:
        print("PRIME_LLM_MODEL not set", file=sys.stderr); return 2
    if a.round == 0:
        prompt = base_prompt()
    else:
        sel = json.load(open(os.path.join(HERE, "candidates", "round0", "selected.json")))
        fb = json.load(open(os.path.join(HERE, "candidates", "round0", "feedback.json")))
        prompt = repair_prompt(sel["program"], fb)
    out = os.path.join(HERE, "candidates", "round%d" % a.round); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "prompt.txt"), "w").write(SYSTEM + "\n\n" + prompt)
    settings = [(0, 0.0, None)] + [(k, 0.7, k) for k in range(1, a.samples + 1)]

    def one(s):
        k, temp, seed = s
        client = llm_mod.LLMClient(model=model, temperature=temp, seed=seed, timeout=900.0)
        rec = {"k": k, "temperature": temp, "seed": seed}
        try:
            resp = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], tools=None)
            rec["reply"] = resp.get("content", ""); rec["usage"] = resp.get("usage")
            rec["program"] = parse(rec["reply"])
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        json.dump(rec, open(os.path.join(out, "cand_%d.json" % k), "w"), indent=1)
        print("[create r%d] cand %d %s" % (a.round, k, "ok" if "program" in rec else rec.get("error", "")[:160]), flush=True)
        return rec
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(settings)) as pool:
        recs = list(pool.map(one, settings))
    print("[create r%d] parsed %d of %d" % (a.round, sum(1 for r in recs if "program" in r), len(recs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
