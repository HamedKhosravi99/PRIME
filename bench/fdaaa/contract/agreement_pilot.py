"""Reading-by-reading comparison of the statute's deadline logic with the
FDAAA TrialsTracker on the pinned 6,000-registration snapshot.

Each reading adds one clause of 42 CFR 11.44 to the previous one; the last is
the faithful reading the instance builder uses. Inputs are the pinned files in
../external/ (SHA256SUMS.txt); no network access. The tracker table
(trialstracker_2026-08-03.csv) is third-party and is not part of the public
release; place a copy in ../external/ to run.

Usage: python3 bench/fdaaa/contract/agreement_pilot.py
"""
import csv, json, os
from datetime import date, timedelta
from collections import Counter

EXTERNAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external")
SNAP = date(2026, 8, 3)
api = json.load(open(os.path.join(EXTERNAL, "ctgov_v2_6000.json")))
rows = list(csv.DictReader(open(os.path.join(EXTERNAL, "trialstracker_2026-08-03.csv"))))
tr = {r["registry_id"]: r for r in rows}

def parse(d, eom):
    if not d: return None
    p = d.split("-")
    if len(p) == 3: return date(int(p[0]), int(p[1]), int(p[2]))
    if len(p) == 2:
        y, m = int(p[0]), int(p[1])
        return (date(y + m // 12, m % 12 + 1, 1) - timedelta(days=1)) if eom else date(y, m, 1)
    return date(int(p[0]), 12, 31) if eom else date(int(p[0]), 1, 1)

def plus_year(d, n=1):
    try: return d.replace(year=d.year + n)
    except ValueError: return d.replace(year=d.year + n, day=28)

def sub_of(s):
    st = s["protocolSection"]["statusModule"]
    d = parse(st.get("resultsFirstSubmitDate"), False)
    if d: return d
    tk = s.get("derivedSection", {}).get("miscInfoModule", {}).get("submissionTracking", {})
    return parse(tk.get("estimatedResultsFirstSubmitDate"), False)

def verdict(pcd, disp, sub, grace, cal_year, cert_timeliness):
    base = plus_year(pcd, 1) if cal_year else pcd + timedelta(days=365)
    due = base
    if disp is not None and (not cert_timeliness or disp <= base):
        due = plus_year(disp, 2) if cal_year else disp + timedelta(days=730)
    eff = due + timedelta(days=grace)
    if sub is not None: return "reported" if sub <= eff else "reported-late"
    return "overdue" if eff < SNAP else "ongoing"

V = {
 "A tracker emulation      (365d, +30d grace, cert always 2y)": dict(grace=30, cal_year=False, cert_timeliness=False),
 "B  + calendar year       (11.44(a) '1 year')":                dict(grace=30, cal_year=True,  cert_timeliness=False),
 "C  + cert timeliness     (11.44(b)(1)/(c)(1) 'prior to')":    dict(grace=30, cal_year=True,  cert_timeliness=True),
 "D  + drop admin grace    (fully faithful contract)":          dict(grace=0,  cal_year=True,  cert_timeliness=True),
}
print("=== Cumulative ablation: each row adds ONE named clause (n=6000, QC-pending signal on) ===")
res = {}
for name, kw in V.items():
    ag = 0; tot = 0; conf = Counter()
    for nct, s in api.items():
        st = s["protocolSection"]["statusModule"]
        raw = st.get("primaryCompletionDateStruct", {}).get("date") or st.get("completionDateStruct", {}).get("date")
        if not raw: continue
        v = verdict(parse(raw, True), parse(st.get("dispFirstSubmitDate"), False), sub_of(s), **kw)
        truth = tr[nct]["status"];  truth = "overdue" if truth == "overdue-cancelled" else truth
        tot += 1
        if v == truth: ag += 1
        else: conf[(truth, v)] += 1
    res[name] = (ag, tot, conf)
    print(f"  {name:58s} {ag}/{tot} = {100*ag/tot:6.2f}%   residual modes: {dict(conf.most_common(3))}")

# The check against FDA's eight Notices of Noncompliance queries the live registry;
# it is notices_check.py in this folder.
