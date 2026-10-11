# Data sources and attribution

ObligationBench embeds snapshots of fields from public records. The records of the four ObligationBench domains were retrieved on 1 August 2026 and those of the FDAAA set on 3 August 2026; they are redistributed under their sources' terms, listed below. Check each source's current terms before reusing its records outside this dataset. No source endorses this dataset.

| Domain | Source | Terms |
|---|---|---|
| `security` | GitHub Advisory Database (https://github.com/github/advisory-database), retrieved through the OSV API (https://osv.dev) | Licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Attribution: "Contains advisories from the GitHub Advisory Database, licensed under CC BY 4.0." |
| `clinical`, `fdaaa` | ClinicalTrials.gov, U.S. National Library of Medicine (https://clinicaltrials.gov) | Public registry data used under the ClinicalTrials.gov terms and conditions (https://clinicaltrials.gov/about-site/terms-conditions). The registry's records change over time; the snapshots here may differ from the current records. |
| `drugsafety` | openFDA drug enforcement reports, U.S. Food and Drug Administration (https://open.fda.gov) | Public domain, dedicated under CC0 1.0 (https://open.fda.gov/license/). openFDA data must not be relied on for decisions about medical care. |
| `financial` | SEC EDGAR XBRL company facts, U.S. Securities and Exchange Commission (https://www.sec.gov/edgar) | Public information that may be copied and redistributed (https://www.sec.gov/privacy). Retrieved under the SEC's fair access policy (https://www.sec.gov/os/accessing-edgar-data). |

The FDAAA rule was compiled from 42 CFR Part 11, which implements Section 801 of the Food and Drug Administration Amendments Act; U.S. federal regulations are in the public domain.

The paper compares the compiled FDAAA rule with the FDAAA TrialsTracker of the Bennett Institute for Applied Data Science, University of Oxford (https://fdaaa.trialstracker.net). The tracker's data are not part of this release.
