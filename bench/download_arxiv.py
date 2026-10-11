#!/usr/bin/env python3
"""Download real arXiv title+abstract into a local text file (research domain
artifacts). Uses the abs-page citation_* meta tags (the export API is flaky
from here). Records provenance. Integrity: we store exactly what arXiv serves.

    python3 bench/download_arxiv.py <arxiv_id> <out.txt> ["role: target|corpus"]
"""
from __future__ import annotations

import html
import re
import subprocess
import sys


def fetch(arxiv_id):
    url = "https://arxiv.org/abs/%s" % arxiv_id
    r = subprocess.run(["curl", "-s", "--max-time", "40", "-A", "Mozilla/5.0", url],
                       capture_output=True, text=True)
    page = r.stdout
    def meta(name):
        m = re.search(r'name="%s"\s+content="(.*?)"' % re.escape(name), page, re.S)
        return html.unescape(m.group(1).strip()) if m else ""
    title = meta("citation_title")
    abstract = meta("citation_abstract")
    authors = re.findall(r'name="citation_author"\s+content="(.*?)"', page)
    date = meta("citation_date") or meta("citation_online_date")
    if not title or not abstract:
        return None
    return {"id": arxiv_id, "title": title, "authors": [html.unescape(a) for a in authors[:8]],
            "date": date, "abstract": abstract, "url": url}


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    aid, out = sys.argv[1], sys.argv[2]
    role = sys.argv[3] if len(sys.argv) > 3 else ""
    d = fetch(aid)
    if d is None:
        print("FAILED to fetch %s" % aid); return 2
    with open(out, "w", encoding="utf-8") as f:
        f.write("arXiv:%s  %s\n" % (d["id"], role))
        f.write("TITLE: %s\n" % d["title"])
        f.write("AUTHORS: %s\n" % ", ".join(d["authors"]))
        f.write("DATE: %s\nURL: %s\n\n" % (d["date"], d["url"]))
        f.write("ABSTRACT:\n%s\n" % d["abstract"])
    print("wrote %s  (%s, %d-char abstract)" % (out, d["title"][:50], len(d["abstract"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
