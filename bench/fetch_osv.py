#!/usr/bin/env python3
"""Fetch REAL vulnerability advisory records from the OSV API (security/OSV
domain; see prime_runtime/RECORD_ORACLE_REPORT.md SS6 for the field-mapping
blueprint this fetcher's output is authored against). No auth needed.

    python3 bench/fetch_osv.py <vuln_id> [<vuln_id> ...] [--nvd]

For each ``<vuln_id>`` (an OSV/GHSA id, e.g. ``GHSA-x4qr-2fvf-3mr5``, or any
other id the OSV API recognizes -- PYSEC-..., RUSTSEC-..., etc.):

  1. GET https://api.osv.dev/v1/vulns/<vuln_id> -- the exact JSON OSV serves,
     written verbatim (no edits) to bench/artifacts/osv/raw/<vuln_id>.json.
  2. With ``--nvd``: also look up the record's first ``CVE-*`` alias against
     the public NVD API (https://services.nvd.nist.gov/rest/json/cves/2.0),
     to obtain a genuine NUMERIC CVSS v3.1 base score -- OSV's own
     ``severity[].score`` field for GHSA-sourced records is a CVSS *vector
     string* (e.g. ``"CVSS:3.1/AV:N/AC:H/.../A:H"``), not a bare float, so it
     cannot directly drive the closed grammar's ``numeric_ge`` predicate
     (prime_runtime/predicates.py); the NVD lookup is a second REAL fetch
     (cross-referenced via the CVE alias OSV itself reports), never a value
     computed/guessed by this script. When found, the numeric base score is
     merged into the cached record as a clearly-provenanced
     ``nvd_cvss_v3`` block (source URL, cve id, retrieval date) alongside
     the untouched OSV fields -- it never replaces or edits any OSV field.
  3. Appends one line to bench/artifacts/osv/PROVENANCE recording the id,
     source URL, HTTP status, and UTC retrieval timestamp for every fetch
     (OSV and, when requested, NVD) -- so every embedded record's origin is
     independently auditable, per AUTHORING.md's "never assert a fact you
     did not observe" rule.

Never fabricates: a 404/network failure is printed and skipped (exit code
reflects any failures), never silently replaced with invented content.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OSV_DIR = os.path.join(HERE, "artifacts", "osv")
RAW_DIR = os.path.join(OSV_DIR, "raw")
PROVENANCE = os.path.join(OSV_DIR, "PROVENANCE")

OSV_VULN_URL = "https://api.osv.dev/v1/vulns/%s"
NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=%s"


def _get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "prime-pilot-osv-fetcher/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as exc:
        return None, exc.code
    except (urllib.error.URLError, TimeoutError) as exc:
        return None, str(exc)
    try:
        return json.loads(body.decode("utf-8")), status
    except (ValueError, UnicodeDecodeError):
        return None, status


def _log(line: str) -> None:
    os.makedirs(OSV_DIR, exist_ok=True)
    with open(PROVENANCE, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
    print(line)


def fetch_osv_vuln(vuln_id: str, want_nvd: bool = False) -> bool:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = OSV_VULN_URL % vuln_id
    doc, status = _get_json(url)
    if doc is None:
        _log("%s  FETCH FAILED id=%s url=%s status=%s" % (now, vuln_id, url, status))
        return False
    _log("%s  OSV FETCH OK id=%s url=%s http_status=%s summary=%r"
         % (now, vuln_id, url, status, doc.get("summary", "")[:80]))

    if want_nvd:
        cve_id = next((a for a in doc.get("aliases", []) if a.startswith("CVE-")), None)
        if cve_id:
            time.sleep(1.0)  # be polite to NVD's unauthenticated rate limit
            nurl = NVD_CVE_URL % cve_id
            ndoc, nstatus = _get_json(nurl)
            base_score = base_sev = None
            if ndoc and ndoc.get("vulnerabilities"):
                cve = ndoc["vulnerabilities"][0]["cve"]
                metrics = cve.get("metrics", {})
                for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                    entries = metrics.get(key)
                    if entries:
                        cvss_data = entries[0].get("cvssData", {})
                        if "baseScore" in cvss_data:
                            base_score = cvss_data["baseScore"]
                            base_sev = cvss_data.get("baseSeverity")
                            break
            if base_score is not None:
                doc["nvd_cvss_v3"] = {
                    "source": "NVD API v2.0 (services.nvd.nist.gov), "
                    "cvssMetricV31[0].cvssData",
                    "cve_id": cve_id,
                    "retrieved": now,
                    "base_score": base_score,
                    "base_severity": base_sev,
                }
                _log("%s  NVD FETCH OK cve=%s url=%s http_status=%s base_score=%s base_severity=%s"
                     % (now, cve_id, nurl, nstatus, base_score, base_sev))
            else:
                _log("%s  NVD FETCH: no numeric CVSS metric found cve=%s url=%s http_status=%s"
                     % (now, cve_id, nurl, nstatus))
        else:
            _log("%s  NVD SKIPPED id=%s: no CVE-* alias in OSV record" % (now, vuln_id))

    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, "%s.json" % vuln_id)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s" % out_path)
    return True


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    want_nvd = "--nvd" in argv
    ids = [a for a in argv if not a.startswith("--")]
    ok = 0
    for vid in ids:
        if fetch_osv_vuln(vid, want_nvd=want_nvd):
            ok += 1
        time.sleep(0.5)  # be polite to the OSV API
    print("%d/%d fetched OK" % (ok, len(ids)))
    return 0 if ok == len(ids) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
