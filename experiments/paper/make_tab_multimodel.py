#!/usr/bin/env python3
"""Table: Experiments A to C on more than one model (tab_multimodel.tex).

Reads results/tableMULTI.json (written by
experiments/analysis/build_multimodel_table.py) and writes the tabular body that
sections/appendix.tex inputs. One block of four rows per model: the fully
informed agent at 20 calls, then PRIME without guidance, PRIME, and PRIME
with completion at b_min, 2b_min, and 20 calls on the balanced split, with
the family-mean guidance gain at b_min and, on the locator-uncertain split,
utility at b_min and the gain over blind enumeration. A cell that has not
been run prints as an en dash; nothing is invented."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(os.path.dirname(HERE))  # the package root
SRC = os.path.join(PILOT, "results", "tableMULTI.json")
OUT = os.path.join(HERE, "tab_multimodel.tex")

NAMES = {"qwen2.5-32b": "Qwen2.5-32B", "gptoss-20b": "gpt-oss-20b", "mistral-small-24b": "Mistral-Small-24B"}
ORDER = ["qwen2.5-32b", "gptoss-20b", "mistral-small-24b"]
DASH = "--"

# The whole table environment is written here (a body \input inside a tabular
# leaves LaTeX inside a cell at end of file, and \bottomrule then fails).
HEAD = r"""\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llrrrrrrr}
\toprule
 & & FA (\%) & \multicolumn{3}{c}{Utility, balanced split} & Guidance gain & \multicolumn{2}{c}{Locator split, $b_{\min}$} \\
\cmidrule(lr){4-6}\cmidrule(lr){8-9}
Model & Agent & at 20 & $b_{\min}$ & $2b_{\min}$ & 20 & at $b_{\min}$ [95\% CI] & Utility & vs.\ blind [95\% CI] \\
\midrule
"""
TAIL = r"""\bottomrule
\end{tabular}
\caption{Experiments A to C on three models (480 balanced tasks and their locator-uncertain copies; one greedy run per task). False authorization (FA) and supported utility at 20 calls for the fully informed agent; FA at 20 calls and utility at three budgets for PRIME without guidance, PRIME, and PRIME with completion; the family-mean gain of guidance over the mask alone at $b_{\min}$; and, on the locator-uncertain split, utility at $b_{\min}$ with the family-mean difference from blind enumeration (40.3). Completion on the locator split is a live run for all three models (the replayed value for Qwen is 53.6). A dash marks a cell that was not run or a contrast not defined for that row.}
\label{app:tab:multimodel}
\end{table*}
"""


def f1(x):
    return DASH if x is None else "%.1f" % x


def util(cells, key):
    c = cells.get(key)
    return DASH if not c else "%.1f" % c["utility"]


def fa(cells, key):
    c = cells.get(key)
    return DASH if not c else "%.1f" % c["ofar_a"]


def ci(contrasts, key):
    c = contrasts.get(key)
    if not c:
        return DASH
    return "%.1f [%.1f, %.1f]" % (c["delta"], c["ci"][0], c["ci"][1])


def main():
    data = json.load(open(SRC, encoding="utf-8"))["models"]
    lines = []
    models = [m for m in ORDER if m in data] + [m for m in data if m not in ORDER]
    for i, m in enumerate(models):
        M = data[m]
        b, l, c = M["balanced"], M["locator"], M["contrasts"]
        name = NAMES.get(m, m)
        if i:
            lines.append(r"\midrule")
        rows = [
            (name, "Fully informed agent", fa(b, "fully_informed_B20"), util(b, "fully_informed_bmin"), util(b, "fully_informed_2bmin"), util(b, "fully_informed_B20"), DASH, DASH, DASH),
            ("", "PRIME without guidance", fa(b, "mask_B20"), util(b, "mask_bmin"), util(b, "mask_2bmin"), util(b, "mask_B20"),
             DASH, util(l, "mask_bmin"), ci(c, "loc_bmin|mask-vs-blind")),
            ("", "PRIME", fa(b, "prime_B20"), util(b, "prime_bmin"), util(b, "prime_2bmin"), util(b, "prime_B20"),
             ci(c, "bmin|prime-vs-mask"), util(l, "prime_bmin"), ci(c, "loc_bmin|prime-vs-blind")),
            ("", "PRIME with completion", fa(b, "complete_B20"), util(b, "complete_bmin"), util(b, "complete_2bmin"), util(b, "complete_B20"),
             DASH, util(l, "complete_bmin"), ci(c, "loc_bmin|complete-vs-blind")),
        ]
        for r in rows:
            lines.append(" & ".join(r) + r" \\")
    body = "\n".join(lines) + "\n"
    table = HEAD + body + TAIL
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(table)
    print(body)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
