#!/usr/bin/env python3
"""Regenerate the four PRIME publication figures from results on disk.

Every plotted number is RECOMPUTED from the logged episodes under
results by the same scoring and inference functions the table
generators use (analysis/build_w2a_tables.py, build_w2b_table.py,
build_ext_table.py, rejection_bound.py, policy_frontier.py, stats_v5.py).
No model or GPU job is run and no file outside figures/ is written
(bytecode caching is switched off so importing the generators leaves no
__pycache__ behind).

Each recomputed value is then checked twice:
  1. against the JSON the corresponding generator last wrote
     (results/tableT1.json, tableT2.json, tableT4.json, tableEXT.json,
     rejection_bound.json, frontier_policies.json, w0_pilots_*.json), so a
     stale table and a fresh figure cannot silently disagree; and
  2. at the end, against EXPECTED below (the values the manuscript prints),
     to one decimal unless stated. Either failure exits nonzero AFTER the
     PDFs are written, so the offending figure can be inspected.

Outputs (vector PDF, next to this script):
  fig_ladder.pdf      information/enforcement ladder (Table T2 panel a/b)
  fig_asymmetry.pdf   supplied vs learned vs constructed standard (T1, T4,
                      openvocab)
  fig_controllaw.pdf  pre-registered control-law forest plot (T2 panel c)
  fig_rejection.pdf   reject-and-regenerate bound vs PRIME (rejection_bound,
                      EXT, F2)

Usage (from anywhere):  python3 figures/make_figures.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing analysis must not write files

import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PILOT = REPO
ANALYSIS = os.path.join(PILOT, "analysis")
RESULTS = os.path.join(PILOT, "results")
for _p in (ANALYSIS, os.path.join(PILOT, "runner"), PILOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import build_ext_table as ext  # noqa: E402
import build_w2a_tables as bw  # noqa: E402
import build_w2b_table as w2b  # noqa: E402
import fill_v2_tables as fv  # noqa: E402
import policy_frontier as pf  # noqa: E402
import rejection_bound as rb  # noqa: E402
import stats_v5 as st  # noqa: E402

MODEL = "qwen2.5-32b"     # confirmatory model (docs/ANALYSIS_PLAN.md, F7)
BUDGET = "B20"            # confirmatory budget for every pooled rate

# ---------------------------------------------------------------------------
# Hard-coded values. NONE are needed: every plotted number below is
# recomputed from episodes on disk. The dict is kept (empty) so that any
# future value that cannot be recomputed has an obvious, labelled home --
# each entry must name the table in the paper's source it comes from.
# ---------------------------------------------------------------------------
HARDCODED: dict = {}

# ---------------------------------------------------------------------------
# Expected values (from the paper's source). Every plotted
# number must appear here and every entry here must be plotted. Default
# comparison is at 1 decimal; DECIMALS overrides it per key.
# ---------------------------------------------------------------------------
LADDER_ARMS = ["react", "react_verify", "react_spec", "react_spec_verify",
               "react_program", "react_program_verify",
               "react_program_attested", "masked_reactive_v", "prime_guided"]
LADDER_LABELS = ["ReAct", "+verify", "+spec", "+spec+verify", "+program",
                 "+program+verify", "+attested ledger",
                 "PRIME without\nguidance", "PRIME"]
# Wording that a script importing this module may override; the defaults
# are the wording these figures use.
LEDGER_NOTE = "adding the attested ledger raised it"
REJECTION_FLOOR_LABEL = "Answer without evidence (Direct)"
_LADDER_EXPECTED = {  # arm: (OFAR-A %, supported utility)   [Table T2(a)]
    "react": (13.0, 17.9), "react_verify": (19.9, 10.0),
    "react_spec": (21.9, 28.3), "react_spec_verify": (22.3, 26.9),
    "react_program": (22.9, 31.9), "react_program_verify": (22.3, 32.7),
    "react_program_attested": (26.7, 38.0), "masked_reactive_v": (0.0, 30.1),
    "prime_guided": (0.0, 45.5)}
ANCHORS = [("bmin", r"$b_{\min}$"), ("bmin1", r"$b_{\min}+1$"),
           ("2bmin", r"$2\,b_{\min}$"), ("B20", r"$B=20$")]
_CONTROL_EXPECTED = {  # (anchor, topology): (delta, lo, hi)   [Table T2(c)]
    ("bmin", "branching"): (24.0, 19.6, 28.5),
    ("bmin", "nested"): (20.9, 16.7, 25.3),
    ("bmin1", "branching"): (21.5, 17.1, 25.9),
    ("bmin1", "nested"): (19.6, 15.5, 24.1),
    ("2bmin", "branching"): (18.0, 13.9, 22.3),
    ("2bmin", "nested"): (16.4, 12.4, 20.3),
    ("B20", "branching"): (16.2, 12.3, 20.2),
    ("B20", "nested"): (14.4, 10.1, 18.6)}

EXPECTED: dict = {}
for _arm, (_o, _u) in _LADDER_EXPECTED.items():
    EXPECTED["ladder/ofar_a/%s" % _arm] = _o
    EXPECTED["ladder/utility/%s" % _arm] = _u
EXPECTED.update({
    # mask-rung contrast, Table T2(b) row program_attested->masked
    "ladder/mask_rung/delta": 28.0,
    "ladder/mask_rung/ci_lo": 22.7,
    "ladder/mask_rung/ci_hi": 33.5,
    # attested-ledger rung, Table T2(b) row program_verify->program_attested
    # (printed there as -4.8 [-8.6, -1.2] on the "reduction" scale; plotted
    # here as the increase it is). Added because the figure notes that this
    # rung raised false authorization.
    "ladder/ledger_rung/increase": 4.8,
    "ladder/ledger_rung/ci_lo": 1.2,
    "ladder/ledger_rung/ci_hi": 8.6,
    # Fig 2 -- S3: Table T2(a); S2: Table T4 / T2(d) source cells;
    # S1: Table openvocab
    "asymmetry/S3/mask/utility": 30.1, "asymmetry/S3/mask/ofar_a": 0.0,
    "asymmetry/S3/guided/utility": 45.5, "asymmetry/S3/guided/ofar_a": 0.0,
    "asymmetry/S2/mask/utility": 12.8, "asymmetry/S2/mask/ofar_a": 2.8,
    "asymmetry/S2/guided/utility": 16.0, "asymmetry/S2/guided/ofar_a": 5.5,
    "asymmetry/S1/mask/utility": 0.0, "asymmetry/S1/mask/ofar_a": 0.0,
    "asymmetry/S1/n_valid": 14, "asymmetry/S1/n_total": 31,
    # Fig 4 -- rejection_bound / EXT / F2
    "rejection/direct": 30.3, "rejection/mask_only": 30.1,
    "rejection/reject_regenerate": 43.5, "rejection/prime": 45.5,
    "rejection/ceiling": 59.2,
    # Fig 3 margin: docs/ANALYSIS_PLAN.md, F3 frozen equivalence margin,
    # printed to 2 dp
    "controllaw/margin": 6.16,
})
for (_a, _t), (_d, _lo, _hi) in _CONTROL_EXPECTED.items():
    EXPECTED["controllaw/%s/%s/delta" % (_a, _t)] = _d
    EXPECTED["controllaw/%s/%s/ci_lo" % (_a, _t)] = _lo
    EXPECTED["controllaw/%s/%s/ci_hi" % (_a, _t)] = _hi
DECIMALS = {"controllaw/margin": 2, "asymmetry/S1/n_valid": 0,
            "asymmetry/S1/n_total": 0}

PLOTTED: dict = {}         # key -> value actually drawn / printed
SOURCES: dict = {}         # key -> provenance string
XCHECK_FAIL: list = []     # recomputed value disagrees with stored JSON
_N_XCHECK = [0]            # cross-checks performed
ROUNDING_NOTES: list = []  # generator double rounding differs from single


def reg(key, value, source):
    """Register a number that is drawn or printed; return it unchanged."""
    PLOTTED[key] = float(value)
    SOURCES[key] = source
    return float(value)


def _fmt(x, dp=1):
    return "%.*f" % (dp, x)


def xcheck(label, recomputed, stored, dp=1):
    """Recomputed-from-episodes vs the generator's last JSON output."""
    _N_XCHECK[0] += 1
    if stored is None:
        XCHECK_FAIL.append("%s: stored value missing" % label)
        return
    if _fmt(recomputed, dp) != _fmt(stored, dp):
        XCHECK_FAIL.append("%s: recomputed %s vs stored JSON %s"
                           % (label, _fmt(recomputed, dp), _fmt(stored, dp)))


def _json(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# data: every function recomputes from episodes, then cross-checks the JSON
# ---------------------------------------------------------------------------

def ladder_data():
    """Pooled instance-level rates per rung (Table T2a / T1) plus the two
    cluster-level contrasts the figure annotates (Table T2b)."""
    assert bw.LADDER == LADDER_ARMS, "ladder order changed in build_w2a_tables"
    T1 = _json("tableT1.json")
    T2 = _json("tableT2.json")
    rates = {}
    for arm in LADDER_ARMS:
        c = bw.cell(MODEL, "%s__%s" % (arm, BUDGET))
        if c.get("status"):
            raise SystemExit("ladder cell %s is %s" % (arm, c["status"]))
        rates[arm] = c
        stored = T1.get("%s|%s|balanced" % (MODEL, arm), {})
        xcheck("T1 %s ofar_a" % arm, c["ofar_a"], stored.get("ofar_a"))
        xcheck("T1 %s utility" % arm, c["utility"], stored.get("utility"))

    def contrast(lo, hi):
        rows = bw.paired(MODEL, "%s__%s" % (lo, BUDGET),
                         "%s__%s" % (hi, BUDGET), "ofar_a")
        ci = st.hierarchical_cluster_bootstrap(rows)
        out = {"delta": 100 * st.observed_statistic(rows),
               "lo": 100 * ci["lo"], "hi": 100 * ci["hi"],
               "n_clusters": ci["n_clusters"], "n": ci["n_records"]}
        s = T2["ladder"].get("%s|%s->%s" % (MODEL, lo, hi), {})
        xcheck("T2b %s->%s delta" % (lo, hi), out["delta"],
               s.get("ofar_reduction_pp"))
        xcheck("T2b %s->%s ci_lo" % (lo, hi), out["lo"], (s.get("ci") or [None])[0])
        xcheck("T2b %s->%s ci_hi" % (lo, hi), out["hi"], (s.get("ci") or [None, None])[1])
        return out

    mask = contrast("react_program_attested", "masked_reactive_v")
    ledger = contrast("react_program_verify", "react_program_attested")
    return rates, mask, ledger


def asymmetry_data():
    """S3 (authored programme, Table T2a), S2 (induced programme, Table T4),
    S1 (open-vocabulary end to end, Table openvocab)."""
    out = {"S3": {}, "S2": {}}
    T1 = _json("tableT1.json")
    for arm, key in (("masked_reactive_v", "mask"), ("prime_guided", "guided")):
        c = bw.cell(MODEL, "%s__%s" % (arm, BUDGET))
        out["S3"][key] = c
        stored = T1.get("%s|%s|balanced" % (MODEL, arm), {})
        xcheck("T1 %s ofar_a (S3)" % arm, c["ofar_a"], stored.get("ofar_a"))
        xcheck("T1 %s utility (S3)" % arm, c["utility"], stored.get("utility"))
    T4 = _json("tableT4.json")
    for arm, key in (("masked_reactive_v", "mask"), ("prime_guided", "guided")):
        c = w2b.cell(MODEL, arm)
        if c.get("status"):
            raise SystemExit("T4 cell %s is %s" % (arm, c["status"]))
        out["S2"][key] = c
        s = T4["cells"].get("%s|%s" % (MODEL, arm), {})
        xcheck("T4 %s ofar_a" % arm, c["ofar_a"], s.get("ofar_a"))
        xcheck("T4 %s utility" % arm, c["utility"], s.get("utility"))

    # S1: pooled over episodes of every per-domain run file, the same glob
    # fill_v2_tables._s1_facts() reads (the -v2 rerun does not match it).
    eps = []
    for p in sorted(glob.glob(os.path.join(
            RESULTS, "openvocab_endtoend_%s_*.json" % MODEL))):
        with open(p, encoding="utf-8") as fh:
            eps.extend(json.load(fh)["episodes"])
    valid = [e for e in eps if e.get("valid_proposal")]
    s1 = {"n_total": len(eps), "n_valid": len(valid),
          # openvocab utility is a FRACTION of the instance's oracle ceiling;
          # x100 puts it on the same 0-100 axis (it is 0 either way)
          "utility": 100.0 * sum(e["utility"] for e in valid) / len(valid),
          # worst case: an unscoreable emission counts as false
          "ofar_a": 100.0 * sum(e["episode_ofar_worst"] for e in valid) / len(valid),
          "ofar_best": 100.0 * sum(e["episode_ofar_best"] for e in valid) / len(valid),
          "n_emitted": sum(e.get("n_emitted", 0) for e in valid)}
    f = fv._s1_facts() or {}
    xcheck("openvocab n_total", s1["n_total"], f.get("n"), 0)
    xcheck("openvocab n_valid", s1["n_valid"], f.get("valid"), 0)
    xcheck("openvocab ofar_worst (max over domains)", s1["ofar_a"],
           100 * max(f.get("ofar") or [float("nan")]))
    xcheck("openvocab utility (max over domains)", s1["utility"],
           100 * max(f.get("util") or [float("nan")]))
    out["S1"] = s1
    return out


def controllaw_data():
    """Table T2(c): guided minus masked utility, family-cluster paired mean
    with the hierarchical bootstrap CI, per budget anchor x topology."""
    T2 = _json("tableT2.json")
    cells = {}
    for anchor, _ in ANCHORS:
        for topo in ("branching", "nested"):
            rows = bw.paired(MODEL, "prime_guided__%s" % anchor,
                             "masked_reactive_v__%s" % anchor, "utility",
                             topology=topo)
            ci = st.hierarchical_cluster_bootstrap(rows)
            raw = {"delta": st.observed_statistic(rows), "lo": ci["lo"],
                   "hi": ci["hi"]}
            # build_w2a_tables.main() stores these cells rounded to 2 dp and
            # fill_v2_tables prints them with %.1f. Plot exactly what the
            # generator emits, so figure and Table T2(c) cannot disagree --
            # and say so loudly wherever that double rounding differs from
            # rounding the unrounded estimate once.
            c = {k: round(v, 2) for k, v in raw.items()}
            for k in raw:
                if _fmt(raw[k]) != _fmt(c[k]):
                    ROUNDING_NOTES.append(
                        "T2c %s/%s %s: unrounded %.4f prints as %s once "
                        "rounded, but the generator path (round 2 dp -> %%.1f)"
                        " prints %s, as Table T2(c) does; figure follows the "
                        "table" % (anchor, topo, k, raw[k], _fmt(raw[k]),
                                   _fmt(c[k])))
            c.update(n_clusters=ci["n_clusters"], n=ci["n_records"])
            cells[(anchor, topo)] = c
            s = T2["control_law"].get("%s|%s|%s" % (MODEL, anchor, topo), {})
            xcheck("T2c %s/%s delta" % (anchor, topo), c["delta"],
                   s.get("utility_gap_pp"), 2)
            xcheck("T2c %s/%s ci_lo" % (anchor, topo), c["lo"],
                   (s.get("ci") or [None])[0], 2)
            xcheck("T2c %s/%s ci_hi" % (anchor, topo), c["hi"],
                   (s.get("ci") or [None, None])[1], 2)
            xcheck("T2c %s/%s margin" % (anchor, topo), bw.frozen_margin(),
                   s.get("tost_delta"), 2)
    return cells, bw.frozen_margin()


def rejection_data():
    """rejection_bound.main()'s computation, without its file write; Direct
    from build_ext_table.cell(); the ceiling from policy_frontier.collect()."""
    idx = rb.instance_index()
    arms = {k: rb.arm(MODEL, k) for k in
            (rb.UNGATED, "masked_reactive_v", "prime_guided")}
    common = sorted(set.intersection(*(set(v) for v in arms.values())))
    util = {k: [] for k in arms}
    ceil = []
    for i in common:
        got = {k: bw.score(v[i]) for k, v in arms.items()}
        if not all(got.values()):
            continue
        for k, s in got.items():
            util[k].append(s["utility"])
        ceil.append(rb.rejection_ceiling(arms[rb.UNGATED][i], idx))
    n = len(ceil)
    out = {"n": n,
           "reject_regenerate": sum(ceil) / n,
           "mask_only": sum(util["masked_reactive_v"]) / n,
           "prime": sum(util["prime_guided"]) / n}
    d = ext.cell(MODEL, "direct")
    out["direct"], out["direct_n"] = d["utility"], d["n_instances"]
    fr = pf.collect()
    ceil_raw, out["ceiling_n"] = fr[("shortest_route", BUDGET)]
    out["ceiling"] = round(ceil_raw, 2)    # policy_frontier.main() rounding

    R = _json("rejection_bound.json")
    xcheck("rejection_bound n", n, R.get("n_instances"), 0)
    xcheck("rejection_bound rejection_ceiling", out["reject_regenerate"],
           R["rejection_ceiling"]["utility"])
    xcheck("rejection_bound gate_only", out["mask_only"], R["gate_only"]["utility"])
    xcheck("rejection_bound deployed", out["prime"], R["deployed"]["utility"])
    xcheck("tableEXT direct utility", out["direct"],
           _json("tableEXT.json")["cells"]["direct"]["utility"])
    xcheck("frontier shortest_route B20", out["ceiling"],
           _json("frontier_policies.json")["frontier"]["shortest_route"][BUDGET], 2)
    return out


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
BLUE = "#1f4e79"      # PRIME / guided
ORANGE = "#d9822b"    # mask only
GRAY = "#8c8c8c"      # ungated / information-only rungs
DARK = "#4d4d4d"      # reference policies
INK = "#1a1a1a"
MUTED = "#555555"
GRID = "#e4e4e4"
ENFORCED_TINT = "#e9eff6"


def _serif():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "DejaVu Serif"):
        if name in have:
            return name
    return "serif"


def setup_style():
    face = _serif()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [face, "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3a3a3a",
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#3a3a3a",
        "ytick.color": "#3a3a3a",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,          # embed TrueType, never Type 3
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })
    return face


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def fill_canvas(fig, pad=0.03, iters=4):
    """Resize the subplot area so the TIGHT bounding box (labels, side text
    columns, brackets included) is the requested figure size, so that the
    cropped PDF really is e.g. 6.5 in wide instead of the default margins'
    ~5.6 in or a side column's ~7.1 in."""
    W, H = fig.get_size_inches()
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())   # inches
        sp = fig.subplotpars
        over_l = sp.left * W - bb.x0
        over_r = bb.x1 - sp.right * W
        over_b = sp.bottom * H - bb.y0
        over_t = bb.y1 - sp.top * H
        fig.subplots_adjust(left=(over_l + pad) / W,
                            right=1 - (over_r + pad) / W,
                            bottom=(over_b + pad) / H,
                            top=1 - (over_t + pad) / H)


def save(fig, name):
    fill_canvas(fig)
    path = os.path.join(HERE, name)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03,
                metadata={"Creator": None, "Producer": None,
                          "CreationDate": None})
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Fig 1 -- the ladder
# ---------------------------------------------------------------------------

def fig_ladder(rates, mask, ledger):
    src_rate = ("build_w2a_tables.cell('%s', '<arm>__%s') -> ofar_a / utility "
                "(== results/tableT1.json)" % (MODEL, BUDGET))
    ofar = [reg("ladder/ofar_a/%s" % a, rates[a]["ofar_a"], src_rate)
            for a in LADDER_ARMS]
    util = [reg("ladder/utility/%s" % a, rates[a]["utility"], src_rate)
            for a in LADDER_ARMS]
    src_c = ("stats_v5.observed_statistic / hierarchical_cluster_bootstrap "
             "on build_w2a_tables.paired(..., 'ofar_a') (== tableT2.json "
             "ladder)")
    m_d = reg("ladder/mask_rung/delta", mask["delta"], src_c)
    m_lo = reg("ladder/mask_rung/ci_lo", mask["lo"], src_c)
    m_hi = reg("ladder/mask_rung/ci_hi", mask["hi"], src_c)
    # ledger contrast is stored as a reduction (negative = got worse)
    l_d = reg("ladder/ledger_rung/increase", -ledger["delta"], src_c)
    l_lo = reg("ladder/ledger_rung/ci_lo", -ledger["hi"], src_c)
    l_hi = reg("ladder/ledger_rung/ci_hi", -ledger["lo"], src_c)

    x = np.arange(len(LADDER_ARMS))
    colors = [GRAY] * 7 + [ORANGE, BLUE]
    markers = ["o"] * 7 + ["s", "D"]
    sizes = [26] * 7 + [30, 28]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.14})

    for ax, ys in ((ax1, ofar), (ax2, util)):
        ax.axvspan(6.5, 8.6, color=ENFORCED_TINT, lw=0, zorder=0)
        ax.axvline(6.5, color="#b9c7d6", lw=0.6, ls=(0, (2, 2)), zorder=0.5)
        hgrid(ax)
        ax.plot(x, ys, color="#a6a6a6", lw=0.8, zorder=2)
        for xi, yi, c, mk, s in zip(x, ys, colors, markers, sizes):
            ax.scatter([xi], [yi], s=s, marker=mk, color=c, edgecolor="white",
                       linewidth=0.6, zorder=3)
        ax.set_xlim(-0.5, 8.6)

    # -- top panel: false authorization
    ax1.set_ylim(-1.5, 36)
    ax1.set_yticks([0, 10, 20, 30])
    ax1.set_ylabel("False authorization (%)")
    for i, (xi, yi) in enumerate(zip(x, ofar)):
        if i == 6:      # attested ledger: up-right, clear of the ledger note
            ax1.annotate(_fmt(yi), (xi, yi), xytext=(3, 4),
                         textcoords="offset points", ha="left", va="bottom",
                         fontsize=8.5)
        elif i >= 7:    # the two enforced zeros: label to the right
            ax1.annotate(_fmt(yi), (xi, yi), xytext=(6, 1.5),
                         textcoords="offset points", ha="left", va="bottom",
                         fontsize=8.5)
        else:
            ax1.annotate(_fmt(yi), (xi, yi), xytext=(0, 5),
                         textcoords="offset points", ha="center",
                         va="bottom", fontsize=8.5)
    # the single drop: a vertical bracket at the mask rung
    ax1.annotate("", xy=(7, 1.2), xytext=(7, ofar[6]),
                 arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.7,
                             "mutation_scale": 7, "shrinkA": 0,
                             "shrinkB": 0})
    ax1.plot([6.08, 7.0], [ofar[6], ofar[6]], color=INK, lw=0.6,
             ls=(0, (1.5, 1.5)), zorder=1)
    ax1.text(7.12, 17.5,
             "−%s points\n95%% CI [%s, %s]" % (_fmt(m_d), _fmt(m_lo), _fmt(m_hi)),
             ha="left", va="center", fontsize=8.5, color=INK, linespacing=1.25,
             bbox={"facecolor": ENFORCED_TINT, "edgecolor": "none", "pad": 0.6})
    ax1.text(7.12, 9.8, "family-cluster\npaired difference",
             ha="left", va="center", fontsize=7.5, color=MUTED,
             style="italic", linespacing=1.15,
             bbox={"facecolor": ENFORCED_TINT, "edgecolor": "none", "pad": 0.6})
    # the attested-ledger rung raised false authorization
    ax1.annotate(LEDGER_NOTE + ":\n+%s points, 95%% CI [%s, %s], "
                 "family-cluster paired"
                 % (_fmt(l_d), _fmt(l_lo), _fmt(l_hi)),
                 xy=(5.9, ofar[6] + 0.5), xytext=(3.2, 31.8),
                 ha="center", va="center", fontsize=7.5, color=MUTED,
                 style="italic", linespacing=1.15,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
                 arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.6,
                             "mutation_scale": 6, "shrinkA": 1, "shrinkB": 3,
                             "connectionstyle": "arc3,rad=-0.15",
                             "relpos": (1.0, 0.5)})

    # -- bottom panel: supported utility
    ax2.set_ylim(0, 54)
    ax2.set_yticks([0, 10, 20, 30, 40, 50])
    ax2.set_ylabel("Supported utility")
    for xi, yi in zip(x, util):
        ax2.annotate(_fmt(yi), (xi, yi), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(LADDER_LABELS, rotation=28, ha="right",
                        rotation_mode="anchor")
    ax1.tick_params(axis="x", length=0)

    # region brackets above the top panel
    tr = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    for (a, b, label, col) in ((-0.35, 6.35, "information only (no enforcement)",
                                MUTED),
                               (6.65, 8.45, "enforced", BLUE)):
        ax1.plot([a, a, b, b], [1.02, 1.05, 1.05, 1.02], transform=tr,
                 color=col, lw=0.7, clip_on=False)
        ax1.text((a + b) / 2, 1.075, label, transform=tr, ha="center",
                 va="bottom", fontsize=8.5, color=col)
    return save(fig, "fig_ladder.pdf")


# ---------------------------------------------------------------------------
# Fig 2 -- the asymmetry
# ---------------------------------------------------------------------------

def fig_asymmetry(d):
    s3 = ("build_w2a_tables.cell('%s', '<arm>__%s') (== tableT1.json)"
          % (MODEL, BUDGET))
    s2 = ("build_w2b_table.cell('%s', '<arm>') on <arm>__induced__B20 "
          "(== tableT4.json cells)" % MODEL)
    s1 = ("pooled over valid_proposal episodes of results/openvocab_endtoend_"
          "%s_*.json (== fill_v2_tables._s1_facts)" % MODEL)
    vals = {
        "utility": {
            "mask": [reg("asymmetry/S3/mask/utility", d["S3"]["mask"]["utility"], s3),
                     reg("asymmetry/S2/mask/utility", d["S2"]["mask"]["utility"], s2),
                     reg("asymmetry/S1/mask/utility", d["S1"]["utility"], s1)],
            "guided": [reg("asymmetry/S3/guided/utility", d["S3"]["guided"]["utility"], s3),
                       reg("asymmetry/S2/guided/utility", d["S2"]["guided"]["utility"], s2),
                       None]},
        "ofar_a": {
            "mask": [reg("asymmetry/S3/mask/ofar_a", d["S3"]["mask"]["ofar_a"], s3),
                     reg("asymmetry/S2/mask/ofar_a", d["S2"]["mask"]["ofar_a"], s2),
                     reg("asymmetry/S1/mask/ofar_a", d["S1"]["ofar_a"], s1)],
            "guided": [reg("asymmetry/S3/guided/ofar_a", d["S3"]["guided"]["ofar_a"], s3),
                       reg("asymmetry/S2/guided/ofar_a", d["S2"]["guided"]["ofar_a"], s2),
                       None]}}
    nv = int(reg("asymmetry/S1/n_valid", d["S1"]["n_valid"], s1))
    nt = int(reg("asymmetry/S1/n_total", d["S1"]["n_total"], s1))
    # S1 on ObligationBench itself (30 held-out clinical tasks, the -v2 run):
    # every proposal valid, none states a conclusion. Reported beside the
    # pilot run in the annotation; it does not change the plotted zeros.
    with open(os.path.join(RESULTS, "openvocab_endtoend_%s-v2_clinical.json" % MODEL),
              encoding="utf-8") as fh:
        _e2 = json.load(fh)["episodes"]
    s1b = {"n_total": len(_e2),
           "n_valid": sum(1 for e in _e2 if e.get("valid_proposal"))}
    assert sum(e.get("n_emitted", 0) for e in _e2) == 0, "S1 benchmark run emitted"
    assert all((e.get("utility") or 0) == 0 for e in _e2), "S1 benchmark run has utility"
    assert d["S1"]["n_emitted"] == 0, "S1 pilot run emitted"

    groups = ["Supplied\nstandard\n(S3)", "Learned\nsufficiency\n(S2)",
              "Constructed\nstandard\n(S1)"]
    x = np.arange(3)
    w = 0.36
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 3.5),
                                   gridspec_kw={"wspace": 0.30})
    spec = (("mask", -w / 2, ORANGE, INK, "PRIME without guidance"),
            ("guided", +w / 2, BLUE, "white", "PRIME"))
    for ax, key, ylim, ylabel, tag in (
            (axa, "utility", (0, 52), "Supported utility", "(a)"),
            (axb, "ofar_a", (0, 8), "False authorization (%)", "(b)")):
        hgrid(ax)
        top = ylim[1]
        for series, dx, col, txtcol, label in spec:
            for i, v in enumerate(vals[key][series]):
                if v is None:
                    ax.text(x[i] + dx, 0.035 * top, "not run",
                            rotation=90, ha="center", va="bottom",
                            fontsize=8, color=MUTED, style="italic")
                    continue
                ax.bar(x[i] + dx, v, w, color=col, edgecolor="none",
                       zorder=2)
                if v == 0:   # make a zero visible as a stub, not a gap
                    ax.plot([x[i] + dx - w / 2, x[i] + dx + w / 2], [0, 0],
                            color=col, lw=2.4, solid_capstyle="butt",
                            zorder=3, clip_on=False)
                ax.text(x[i] + dx, v + 0.018 * top, _fmt(v), ha="center",
                        va="bottom", fontsize=8.5, color=INK, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.text(-0.02, 1.03, tag, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9, fontweight="bold")

    # direct series labels inside the tall S3 bars of panel (a)
    for series, dx, col, txtcol, label in spec:
        v = vals["utility"][series][0]
        axa.text(x[0] + dx, v / 2, label, rotation=90, ha="center",
                 va="center", fontsize=8, color=txtcol, zorder=5)
    # S1 notes
    axa.text(x[2], 0.20 * 52, "%d of %d rules valid;\nnone stated\na conclusion" % (nv + s1b["n_valid"], nt + s1b["n_total"]),
             ha="center", va="bottom", fontsize=8, color=MUTED,
             linespacing=1.2)
    axb.text(x[2], 0.20 * 8, "%d of %d rules valid;\nnone stated\na conclusion" % (nv + s1b["n_valid"], nt + s1b["n_total"]),
             ha="center", va="bottom", fontsize=8, color=MUTED,
             linespacing=1.2)
    return save(fig, "fig_asymmetry.pdf")


# ---------------------------------------------------------------------------
# Fig 3 -- the control law
# ---------------------------------------------------------------------------

def fig_controllaw(cells, margin):
    src = ("stats_v5.observed_statistic / hierarchical_cluster_bootstrap on "
           "build_w2a_tables.paired('%s', 'prime_guided__<anchor>', "
           "'masked_reactive_v__<anchor>', 'utility', topology=<t>) "
           "(== tableT2.json control_law)" % MODEL)
    margin = reg("controllaw/margin", margin,
                 "build_w2a_tables.frozen_margin() <- results/w0_pilots_%s."
                 "json MDE_delta (== tableT2.json tost_delta)" % MODEL)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    gap = 0.85
    ypos, rows = [], []
    y = 0.0
    for anchor, _ in ANCHORS:
        for topo in ("branching", "nested"):
            ypos.append(y)
            rows.append((anchor, topo))
            y += 1.0
        y += gap
    ypos = np.array(ypos)

    for yy in ypos:
        ax.axhline(yy, color=GRID, lw=0.5, zorder=0)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(margin, color=MUTED, lw=0.8, ls=(0, (4, 2.5)), zorder=1)

    style = {"branching": dict(marker="o", ms=5.2, mfc=BLUE, mec=BLUE),
             "nested": dict(marker="s", ms=4.8, mfc="white", mec=BLUE,
                            mew=1.1)}
    tr_right = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yy, (anchor, topo) in zip(ypos, rows):
        c = cells[(anchor, topo)]
        k = "controllaw/%s/%s/" % (anchor, topo)
        dlt = reg(k + "delta", c["delta"], src)
        lo = reg(k + "ci_lo", c["lo"], src)
        hi = reg(k + "ci_hi", c["hi"], src)
        ax.plot([lo, hi], [yy, yy], color=BLUE, lw=1.3, zorder=2,
                solid_capstyle="butt")
        ax.plot([dlt], [yy], linestyle="none", zorder=3, **style[topo])
        ax.text(1.015, yy, "+%s  [%s, %s]" % (_fmt(dlt), _fmt(lo), _fmt(hi)),
                transform=tr_right, ha="left", va="center", fontsize=8.5,
                clip_on=False)
    ax.text(1.015, -0.95, "Δ utility  [95% CI]", transform=tr_right,
            ha="left", va="center", fontsize=8.5, color=MUTED, clip_on=False)

    ax.set_yticks(ypos)
    ax.set_yticklabels([t for _, t in rows], fontsize=8)
    ax.tick_params(axis="y", length=0, pad=3)
    # anchor group labels, left of the topology labels
    tr_left = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, (anchor, lab) in enumerate(ANCHORS):
        mid = (ypos[2 * i] + ypos[2 * i + 1]) / 2
        ax.text(-0.135, mid, lab, transform=tr_left, ha="right", va="center",
                fontsize=9, clip_on=False)
    ax.text(-0.135, -0.95, "budget", transform=tr_left, ha="right",
            va="center", fontsize=8.5, color=MUTED, clip_on=False)

    ax.set_ylim(ypos[-1] + 0.7, -1.45)
    ax.set_xlim(-1.5, 31)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Utility gain from guidance: PRIME − PRIME without guidance (points)")
    ax.spines["left"].set_visible(False)
    ax.text(margin + 0.35, -0.95,
            "pre-registered margin δ = %s" % _fmt(margin, 2),
            ha="left", va="center", fontsize=8, color=MUTED, style="italic")
    return save(fig, "fig_controllaw.pdf")


# ---------------------------------------------------------------------------
# Fig 4 -- rejection sampling vs PRIME
# ---------------------------------------------------------------------------

def fig_rejection(d):
    s_rb = ("rejection_bound.{instance_index,arm,rejection_ceiling} + "
            "build_w2a_tables.score on the %d paired episodes "
            "(== rejection_bound.json)" % d["n"])
    rows = [
        (REJECTION_FLOOR_LABEL,
         reg("rejection/direct", d["direct"],
             "build_ext_table.cell('%s', 'direct') (== tableEXT.json)" % MODEL),
         dict(marker="o", ms=5.5, mfc=GRAY, mec=GRAY)),
        ("PRIME without guidance",
         reg("rejection/mask_only", d["mask_only"], s_rb + " gate_only"),
         dict(marker="s", ms=5.5, mfc=ORANGE, mec=ORANGE)),
        ("Ideal after-the-fact filter on the\nfully informed agent's evidence",
         reg("rejection/reject_regenerate", d["reject_regenerate"],
             s_rb + " rejection_ceiling"),
         dict(marker="^", ms=6.2, mfc="white", mec=DARK, mew=1.1)),
        ("PRIME",
         reg("rejection/prime", d["prime"], s_rb + " deployed"),
         dict(marker="D", ms=5.2, mfc=BLUE, mec=BLUE)),
        ("Ceiling-targeting policy (no LLM)",
         reg("rejection/ceiling", d["ceiling"],
             "policy_frontier.collect()[('shortest_route', '%s')] "
             "(== frontier_policies.json)" % BUDGET),
         dict(marker="o", ms=5.5, mfc="white", mec=DARK, mew=1.1)),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 2.75))
    y = np.arange(len(rows))
    floor, ceiling = rows[0][1], rows[4][1]
    for yy in y:
        ax.axhline(yy, color=GRID, lw=0.5, zorder=0)
    for xv, lab in ((floor, "floor: no evidence gathered"),
                    (ceiling, "ceiling: no-LLM policy")):
        ax.axvline(xv, color="#bdbdbd", lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(xv, -0.72, lab, ha="center", va="bottom", fontsize=8,
                color=MUTED, style="italic")
    for yy, (lab, v, sty) in zip(y, rows):
        ax.plot([v], [yy], linestyle="none", zorder=3, **sty)
        ax.annotate(_fmt(v), (v, yy), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], linespacing=1.1)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.5, -0.85)
    ax.set_xlim(20, 65)
    ax.set_xticks([20, 30, 40, 50, 60])
    ax.set_xlabel("Supported utility (%d confirmatory episodes)" % d["n"])
    return save(fig, "fig_rejection.pdf")


# ---------------------------------------------------------------------------

def verify():
    """Every plotted number vs EXPECTED; nonzero exit on any disagreement."""
    bad = []
    for k in sorted(set(EXPECTED) | set(PLOTTED)):
        if k not in PLOTTED:
            bad.append("%s: expected %s but never plotted" % (k, EXPECTED[k]))
            continue
        if k not in EXPECTED:
            bad.append("%s: plotted %s with no expected value" % (k, PLOTTED[k]))
            continue
        dp = DECIMALS.get(k, 1)
        if _fmt(PLOTTED[k], dp) != _fmt(EXPECTED[k], dp):
            bad.append("%s: plotted %s, expected %s   [%s]"
                       % (k, _fmt(PLOTTED[k], dp), _fmt(EXPECTED[k], dp),
                          SOURCES.get(k, "?")))
    return bad


def main(argv=None) -> int:
    face = setup_style()
    print("font: %s" % face)
    rates, mask, ledger = ladder_data()
    asym = asymmetry_data()
    cells, margin = controllaw_data()
    rej = rejection_data()

    for p in (fig_ladder(rates, mask, ledger), fig_asymmetry(asym),
              fig_controllaw(cells, margin), fig_rejection(rej)):
        print("wrote %s" % os.path.relpath(p, REPO))

    # caveats that the figures' captions must carry; printed, never hidden
    s2 = asym["S2"]
    print("note: S2 cells score %d (mask) / %d (guided) of %d attempted "
          "episodes; the rest errored (T4 n_errored)."
          % (s2["mask"]["n_instances"], s2["guided"]["n_instances"],
             s2["mask"]["n_attempted"]))
    print("note: S1 is %d records of the earlier instance family, not the "
          "480-instance split; its %d valid proposals emitted %d claims "
          "(OFAR worst=%s, best=%s)."
          % (asym["S1"]["n_total"], asym["S1"]["n_valid"],
             asym["S1"]["n_emitted"], _fmt(asym["S1"]["ofar_a"]),
             _fmt(asym["S1"]["ofar_best"])))
    short = {a: rates[a]["n_instances"] for a in LADDER_ARMS
             if rates[a]["n_instances"] != 480}
    if short:
        print("note: ladder rungs scored on fewer than 480 episodes "
              "(errored): %s" % short)

    for m in ROUNDING_NOTES:
        print("ROUNDING NOTE: " + m)
    fails = ["[json cross-check] " + m for m in XCHECK_FAIL]
    fails += ["[expected value] " + m for m in verify()]
    if fails:
        print("\nFAILED: %d check(s) did not pass" % len(fails), file=sys.stderr)
        for m in fails:
            print("  " + m, file=sys.stderr)
        return 1
    print("verified %d plotted numbers against EXPECTED and %d generator "
          "JSON cross-checks: all match" % (len(PLOTTED), _N_XCHECK[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
