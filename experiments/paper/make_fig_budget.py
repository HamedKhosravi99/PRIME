#!/usr/bin/env python3
"""Figure: supported utility by budget on the balanced split (left) and the
locator-uncertain split (right), LLM agents against the LLM-free policies.
Reads the results JSONs and cells under results/; writes fig_budget.pdf."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))  # the package root
for p in (PILOT, os.path.join(PILOT, "src"), os.path.join(PILOT, "experiments", "analysis"), os.path.join(PILOT, "src", "prime_agents")):
    sys.path.insert(0, p)
import build_w2a_tables as bw  # noqa: E402
import policy_frontier as pf   # noqa: E402

A4 = ["bmin", "bmin1", "2bmin", "B20"]; A3 = ["bmin", "2bmin", "B20"]
def bal(arm): return [bw.cell("qwen2.5-32b", "%s__%s" % (arm, a))["utility"] for a in A4]
fixed = pf.collect("heldout_balanced")
bal_rows = {
    "PRIME, no guidance": bal("masked_reactive_v"),
    "PRIME": bal("prime_guided"),
    "PRIME, completion": bal("prime_complete"),
    "Random open check (no LLM)": [fixed[("random_masked", a)][0] for a in A4],
    "Ceiling-targeting (no LLM)": [fixed[("shortest_route", a)][0] for a in A4],
}
loc = json.load(open(os.path.join(PILOT, "results", "tableLOC.json")))["cells"]
locx = json.load(open(os.path.join(PILOT, "results", "tableLOC_extra.json")))
def lc(arm, a):
    c = loc.get("%s|%s" % (arm, a)) or loc.get("%s__%s" % (arm, a))
    return c.get("utility") if c else None
loc_rows = {
    "PRIME, no guidance": [lc("masked_reactive_v", a) for a in A3],
    "PRIME": [lc("prime_guided", a) for a in A3],
    "PRIME, completion (replayed)": [locx["cells"]["completion"][a] for a in A3],
    "Blind enumeration (no LLM)": [lc("shortest_route_candidates", a) for a in A3],
    "True locators (no LLM)": [lc("shortest_route", a) for a in A3],
}
print("balanced:", bal_rows); print("locator:", loc_rows)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.03, 1.55), sharey=True)
style = {"PRIME, no guidance": dict(color="#7f7f7f", marker="s"), "PRIME": dict(color="#1f77b4", marker="o"),
         "PRIME, completion": dict(color="#d62728", marker="^"), "PRIME, completion (replayed)": dict(color="#d62728", marker="^"),
         "Random open check (no LLM)": dict(color="#2ca02c", marker="x", ls="--"), "Ceiling-targeting (no LLM)": dict(color="#000000", marker="d", ls="--"),
         "Blind enumeration (no LLM)": dict(color="#2ca02c", marker="x", ls="--"), "True locators (no LLM)": dict(color="#000000", marker="d", ls="--")}
for name, ys in bal_rows.items():
    ax1.plot(range(4), ys, label=name, **style[name])
ax1.set_xticks(range(4)); ax1.set_xticklabels([r"$b_{\min}$", r"$b_{\min}{+}1$", r"$2b_{\min}$", "20"])
ax1.set_title("Balanced split", fontsize=7); ax1.set_ylabel("Supported utility")
for name, ys in loc_rows.items():
    xs = [i for i, y in enumerate(ys) if y is not None]; yy = [y for y in ys if y is not None]
    ax2.plot(xs, yy, label=name, **style[name])
ax2.set_xticks(range(3)); ax2.set_xticklabels([r"$b_{\min}$", r"$2b_{\min}$", "20"])
ax2.set_title("Locator-uncertain split", fontsize=7)
for ax in (ax1, ax2):
    ax.set_ylim(15, 63); ax.grid(axis="y", lw=0.3, alpha=0.5); ax.tick_params(length=2, pad=1.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
handles = h1[:3] + [h1[3], h1[4]]; labels = ["PRIME, no guidance", "PRIME", "PRIME, completion", "Random check / blind enum. (no LLM)", "Ceiling-targeting / true locators (no LLM)"]
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, -0.30), handlelength=1.8, columnspacing=1.0)
fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.17, wspace=0.08)
fig.savefig(os.path.join(HERE, "figures", "fig_budget.pdf"), bbox_inches="tight", pad_inches=0.01,
            metadata={"Creator": None, "Producer": None, "CreationDate": None})
print("wrote fig_budget.pdf")
