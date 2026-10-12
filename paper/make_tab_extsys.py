#!/usr/bin/env python3
"""Write the comparison with the closest external systems from its result files:
tab_extsys.tex      main-text table, 20 calls  (results/tableEXTSYS.json)
tab_extsys_app.tex  appendix table, 20 calls and b_min (tableEXTSYS.json, tableEXTSYS_bmin.json)
Usage: python3 make_tab_extsys.py   (after analysis/build_extsys_table.py [--anchor bmin])
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)  # the package root

MAIN = (("react_program_attested", "Fully informed agent"),
        ("ext_agentspec", "AgentSpec, stop"),
        ("ext_agentspec_examine", "AgentSpec, self-examine"),
        ("ext_researchloop", "ResearchLoop"),
        ("ext_prooforstop", "Proof-or-Stop"),
        ("ext_glean", "GLEAN"),
        ("prime_guided", "PRIME"),
        ("prime_complete", "PRIME with completion"))


def f(v, d=1):
    return "--" if v is None else ("%%.%df" % d) % v


def load(anchor):
    p = os.path.join(PILOT, "results", "tableEXTSYS%s.json" % ("" if anchor == "B20" else "_" + anchor))
    return json.load(open(p)) if os.path.exists(p) else None


def row(c, cols):
    if c is None or c.get("status"):
        return " & ".join(["--"] * len(cols))
    return " & ".join(f(c.get(k), d) for k, d in cols)


def main_table():
    t = load("B20")
    cols = (("ofar_a", 1), ("premature", 1), ("utility", 1), ("emit_rate_avail", 1), ("mean_reads", 1))
    rows = []
    for arm, name in MAIN:
        rows.append(r"%s & %s \\" % (name, row(t["cells"].get(arm), cols)))
        if arm in ("react_program_attested", "ext_glean"):
            rows.append(r"\midrule")
    tex = r"""\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lrrrrr}
\toprule
System & \makecell[r]{False\\auth.\,\%%\,$\downarrow$} & \makecell[r]{Prem.\\\%%\,$\downarrow$} & \makecell[r]{Util.\\$\uparrow$} & \makecell[r]{Emit\\\%%\,$\uparrow$} & \makecell[r]{Reads\\$\downarrow$} \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{Experiment E: the closest systems on the 480 balanced tasks (20 calls), with the same model, tools, requirements, and resolvers. The maximum utility is 59.2: the strongest supported conclusion on every task. Reads: record fields read per task. $\uparrow$/$\downarrow$: higher/lower is better.}
\label{tab:extsys}
\end{table}
""" % "\n".join(rows)
    open(os.path.join(HERE, "tab_extsys.tex"), "w").write(tex.replace("%%", "%"))


def app_table():
    t20 = load("B20")
    cols = (("ofar_a", 1), ("premature", 1), ("utility", 1), ("emit_rate_avail", 1),
            ("mean_reads", 2), ("llm_calls", 1))
    rows = []
    for arm, name in MAIN:
        rows.append(r"%s & %s \\" % (name, row(t20["cells"].get(arm), cols)))
        if arm in ("react_program_attested", "ext_glean"):
            rows.append(r"\midrule")
    tex = r"""\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lrrrrrr}
\toprule
System & FA\,$\downarrow$ & Prem.\,$\downarrow$ & Util.\,$\uparrow$ & Emit\,$\uparrow$ & Reads\,$\downarrow$ & Calls\,$\downarrow$ \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{Experiment E in full (480 balanced tasks, 20 calls). FA: false authorization (\%%); Prem.: premature conclusions (\%%); Emit: answerable tasks with a stated conclusion (\%%); Reads: record fields read per task; Calls: model calls per task (agent turns, planning turns, re-plans, and planner decisions; for GLEAN, the judge calls added to the fully informed agent's run; not recorded for the PRIME runs and the fully informed agent).}
\label{app:tab:extsys}
\end{table}
""" % "\n".join(rows)
    open(os.path.join(HERE, "tab_extsys_app.tex"), "w").write(tex.replace("%%", "%"))


def hesp_table():
    """PRIME against HESP's controller by budget, on the balanced split and
    its locator-uncertain copy (tableCONTROLLER.json, tableLOC_HESP.json,
    and the live PRIME cells)."""
    ctl = json.load(open(os.path.join(PILOT, "results", "tableCONTROLLER.json")))
    loc = json.load(open(os.path.join(PILOT, "results", "tableLOC_HESP.json")))
    live = {a: load(a) for a in ("B20",)}
    anchors = ("bmin", "bmin1", "2bmin", "B20")

    def cell(tab, key):
        c = tab["cells"].get(key)
        return None if c is None else c

    def u(tab, arm, a, k="utility"):
        c = cell(tab, "%s|%s" % (arm, a))
        return "--" if c is None else f(c[k])

    rows = [r"\multicolumn{5}{l}{\emph{Balanced split: utility $\uparrow$}} \\"]
    rows.append(r"\quad PRIME schedule (code) & %s \\" % " & ".join(u(ctl, "shortest_route", a) for a in anchors))
    rows.append(r"\quad HESP controller & %s \\" % " & ".join(u(ctl, "hesp_controller", a) for a in anchors))
    prem = (u(ctl, "shortest_route", "B20", "premature"), u(ctl, "hesp_controller", "B20", "premature"))
    rows.append(r"\multicolumn{5}{l}{\emph{Locator-uncertain copy: utility $\uparrow$}} \\")
    replay = json.load(open(os.path.join(PILOT, "results", "discharge_replay_locator.json")))["arms"]["prime_guided"]

    def comp(a):
        live = cell(loc, "prime_complete|%s" % a)
        if live is not None:
            return f(live["utility"])
        r = replay.get(a)
        return (f(r["utility_completed"]) + "$^{r}$") if r else "--"
    for arm, name in (("prime_guided", "PRIME"), ("prime_complete", "PRIME with completion"),
                      ("shortest_route_candidates", "PRIME schedule, listed order"),
                      ("hesp_controller", "HESP controller"), ("shortest_route", "True-locator schedule")):
        vals = [comp(a) for a in anchors] if arm == "prime_complete" else [u(loc, arm, a) for a in anchors]
        rows.append(r"\quad %s & %s \\" % (name, " & ".join(vals)))
    tex = r"""\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
 & $b_{\min}$ & $b_{\min}{+}1$ & $2b_{\min}$ & 20 \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{PRIME against HESP's controller by budget (480 tasks). Both controllers run without a model; PRIME and PRIME with completion are the runs of Experiments B and C ($^{r}$: completion replayed from PRIME's logged stopping points, as in Figure~\ref{fig:budget}; --: not run). On the balanced split, conclusions stated before the system's own reads established them: %s\%% of tasks for PRIME's schedule and %s\%% for HESP's controller at every budget.}
\label{app:tab:hesp}
\end{table}
""" % ("\n".join(rows), prem[0], prem[1])
    open(os.path.join(HERE, "tab_extsys_hesp.tex"), "w").write(tex.replace("%%", "%"))


if __name__ == "__main__":
    main_table()
    app_table()
    hesp_table()
    print("wrote tab_extsys.tex, tab_extsys_app.tex, tab_extsys_hesp.tex")
