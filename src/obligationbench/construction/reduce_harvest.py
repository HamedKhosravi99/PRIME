#!/usr/bin/env python3
"""Reduce a harvest to the record fields the generators read.

    python3 src/obligationbench/construction/reduce_harvest.py --src /path/to/full/harvest --dst /path/to/reduced/harvest

The full harvest of 2026-08-01 (``harvest/<domain>/records.jsonl``, 2.3 GB, almost all
of it SEC companyfacts) keeps every field the sources returned. The generators read 55
record paths (``READ_PATHS`` below: the probe fields, the locator decoys, the text
sources and their structured twins, and the ``affected`` list the security text checks
iterate over). This script keeps exactly those paths of every record, in the original
record order, with the original ``record_id`` and ``provenance``, and copies each
domain's ``sampling_frame.json``. Credential-like strings that some security advisories
quote in their proofs of concept (AWS's documented example key, a key id inside a signed
URL, a mock API key, an OpenSSH key header) are replaced by ``[REDACTED]`` so that
secret scanners do not flag the repository; none of the four records is in a published
task. The released package ships the output as ``data/harvest/``; every published task
regenerates byte for byte from it (``src/obligationbench/construction/reproduce_corpus.py``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_record_instances import slice_into  # noqa: E402
DOMAINS = ("security", "clinical", "drugsafety", "financial")

# Every path the record generator (all modes) and the text generator read, logged by
# instrumenting get_path and slice_into over a full regeneration, plus "affected",
# which the security text checks walk without get_path.
READ_PATHS = (
    "affected",
    "affected.0.package.ecosystem",
    "affected.0.package.name",
    "aliases.0",
    "cik",
    "classification",
    "code_info",
    "country",
    "database_specific.cwe_ids.0",
    "database_specific.github_reviewed",
    "database_specific.severity",
    "details",
    "distribution_pattern",
    "entityName",
    "event_id",
    "facts.us-gaap.AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment.units.USD.0.val",
    "facts.us-gaap.AccumulatedOtherComprehensiveIncomeLossNetOfTax.units.USD.0.val",
    "facts.us-gaap.Assets.units.USD.0.val",
    "facts.us-gaap.CashAndCashEquivalentsAtCarryingValue.units.USD.0.val",
    "facts.us-gaap.CommonStockSharesIssued.units.shares.0.val",
    "facts.us-gaap.EarningsPerShareBasic.units.USD/shares.0.val",
    "facts.us-gaap.LiabilitiesAndStockholdersEquity.units.USD.0.val",
    "facts.us-gaap.NetIncomeLoss.units.USD.0.val",
    "facts.us-gaap.PropertyPlantAndEquipmentNet.units.USD.0.val",
    "facts.us-gaap.RetainedEarningsAccumulatedDeficit.units.USD.0.val",
    "facts.us-gaap.StockholdersEquity.units.USD.0.val",
    "modified",
    "product_description",
    "product_quantity",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.designModule.studyType",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.eligibilityModule.healthyVolunteers",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.statusModule.lastUpdateSubmitDate",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.statusVerifiedDate",
    "protocolSection.statusModule.studyFirstSubmitDate",
    "published",
    "reason_for_recall",
    "recall_initiation_date",
    "recall_number",
    "recalling_firm",
    "references.0.type",
    "references.0.url",
    "report_date",
    "schema_version",
    "severity.0.score",
    "status",
    "summary",
    "voluntary_mandated",
)


# Credential patterns, those of the release's anonymity check.
CREDENTIAL = re.compile(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}"
                        r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I)


def _mask(value):
    if isinstance(value, str):
        return CREDENTIAL.sub("[REDACTED]", value)
    if isinstance(value, list):
        return [_mask(v) for v in value]
    if isinstance(value, dict):
        return {k: _mask(v) for k, v in value.items()}
    return value


def reduce_record(record):
    out = {}
    for path in READ_PATHS:
        slice_into(out, record, path)
    return _mask(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="full harvest: <src>/<domain>/records.jsonl")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    for dom in DOMAINS:
        src, dst = os.path.join(a.src, dom), os.path.join(a.dst, dom)
        os.makedirs(dst, exist_ok=True)
        n = 0
        with open(os.path.join(src, "records.jsonl"), encoding="utf-8") as fi, \
                open(os.path.join(dst, "records.jsonl"), "w", encoding="utf-8") as fo:
            for line in fi:
                row = json.loads(line)
                row["record"] = reduce_record(row["record"])
                fo.write(json.dumps(row) + "\n")
                n += 1
        shutil.copyfile(os.path.join(src, "sampling_frame.json"), os.path.join(dst, "sampling_frame.json"))
        print("%-10s %5d records -> %s" % (dom, n, os.path.relpath(dst)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
