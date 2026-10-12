#!/usr/bin/env python3
"""Appendix E figures, labelled with the paper's agent names.

Reuses make_figures.py (same folder): the same episodes, the same numbers,
and the same checks against its expected values. Writes fig_ladder_appendix.pdf
and fig_rejection_appendix.pdf into figures/, beside the shared figures, which
stay unchanged. The control-law figure is used as is from
figures/fig_controllaw.pdf.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE,                                              # main repository
           os.path.join(HERE, "code", "figures")):          # the paper's source tree
    if os.path.exists(os.path.join(_d, "make_figures.py")):
        sys.path.insert(0, _d)
        break
import make_figures as mf  # noqa: E402

mf.LADDER_LABELS[6] = "Fully informed"
mf.LEDGER_NOTE = "adding the ledger display raised it"
mf.REJECTION_FLOOR_LABEL = "Answer without evidence"
mf.HERE = HERE
_save = mf.save
mf.save = lambda fig, name: _save(fig, name.replace(".pdf", "_appendix.pdf"))


def main() -> int:
    mf.setup_style()
    rates, mask, ledger = mf.ladder_data()
    rej = mf.rejection_data()
    for path in (mf.fig_ladder(rates, mask, ledger), mf.fig_rejection(rej)):
        print("wrote", os.path.relpath(path, HERE))
    keys = sorted(k for k in mf.EXPECTED if k.startswith(("ladder/", "rejection/")))
    bad = []
    for k in keys:
        dp = mf.DECIMALS.get(k, 1)
        if k not in mf.PLOTTED:
            bad.append("%s: expected %s but never plotted" % (k, mf.EXPECTED[k]))
        elif mf._fmt(mf.PLOTTED[k], dp) != mf._fmt(mf.EXPECTED[k], dp):
            bad.append("%s: plotted %s, expected %s" % (k, mf._fmt(mf.PLOTTED[k], dp), mf._fmt(mf.EXPECTED[k], dp)))
    bad += ["[json cross-check] " + m for m in mf.XCHECK_FAIL]
    if bad:
        print("FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print("verified %d plotted numbers against make_figures.EXPECTED: all match" % len(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
