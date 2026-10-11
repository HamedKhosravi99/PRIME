#!/usr/bin/env python3
"""Sample the validation set (and three unlabelled creation examples) from the
development pool and fetch the full registrations from the public API.
Sampling uses only registry ids: development-pool ids that the tracker lists,
sorted, then drawn with seed 20261011. No tracker status is read here."""
import csv, json, os, random, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PILOT = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(PILOT, "bench", "fdaaa", "external")
API = "https://clinicaltrials.gov/api/v2/studies/%s?format=json"
OUT = os.path.join(HERE, "data", "validation_records")
pool = set(json.load(open(os.path.join(EXT, "ctgov_v2_6000.json"))))
listed = {r["registry_id"] for r in csv.DictReader(open(os.path.join(EXT, "trialstracker_2026-08-03.csv")))}
ids = sorted(pool & listed)
rng = random.Random(20261011); pick = rng.sample(ids, 603)
val, examples = pick[:600], pick[600:]
json.dump({"seed": 20261011, "population": len(ids), "validation": val, "creation_examples": examples},
          open(os.path.join(HERE, "data", "split.json"), "w"), indent=1)
done = failed = 0; t0 = time.time()
for i, nct in enumerate(val + examples, 1):
    p = os.path.join(OUT, nct + ".json")
    if os.path.exists(p):
        done += 1; continue
    r = subprocess.run(["curl", "-s", "--max-time", "40", API % nct], capture_output=True, text=True)
    try:
        doc = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except json.JSONDecodeError:
        doc = None
    if doc is None or "protocolSection" not in doc:
        failed += 1; print("failed", nct, flush=True)
    else:
        json.dump(doc, open(p, "w")); done += 1
    time.sleep(0.12)
    if i % 100 == 0:
        print("[fetch] %d/%d ok=%d failed=%d %.0fs" % (i, len(val) + len(examples), done, failed, time.time() - t0), flush=True)
print("done: ok=%d failed=%d population=%d" % (done, failed, len(ids)))
