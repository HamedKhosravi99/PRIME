"""End-to-end integration of the confirmatory table builder.

analysis/build_w2a_tables.py is the single path from episode files to the
paper's T1/T2 numbers. Until the cluster returns W2a data every one of its
cells reads "NOT RUN", so a schema drift between what runner/run_batch.py
writes and what score()/cell()/paired() read would stay invisible until
after the GPU hours were already spent.

This test therefore synthesises a complete W2a results tree from the REAL
benchmark instances -- same directory layout, same summary.json shape as
runner/run_batch.py emits -- with emissions chosen so every endpoint has a
known closed-form value, and asserts the builder reproduces them and that
the frozen inference (Holm family 1, TOST control law) actually fires.
"""
from __future__ import annotations

import json
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(os.path.dirname(_TESTS))
for _x in (_P, os.path.join(_P, "analysis"), os.path.join(_P, "runner")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import build_w2a_tables as bw  # noqa: E402

MODEL = "qwen2.5-fixture"
BAL = ("heldout_balanced",)


# ---------------------------------------------------------------------------
# synthetic-episode construction
# ---------------------------------------------------------------------------

def _best_supported(f):
    ok = [(f["levels"][c], c) for c, s in f["supp_a"].items() if s]
    return max(ok)[1] if ok else None


def _some_unsupported(f):
    bad = sorted(c for c, s in f["supp_a"].items() if not s)
    return bad[0] if bad else None


def episodes(behaviour, split=BAL, domain=None, cost=4.0):
    """behaviour in {clean, dirty, silent}; mirrors run_batch's episode keys."""
    out = []
    for f in bw.FACTS.values():
        if f["split"] not in split or (domain and f["domain"] != domain):
            continue
        good, bad = _best_supported(f), _some_unsupported(f)
        if behaviour == "silent":
            emitted = []
        elif behaviour == "dirty" and bad is not None:
            emitted = [{"claim": bad}] + ([{"claim": good}] if good else [])
        else:
            emitted = [{"claim": good}] if good else []
        out.append({"instance": f["id"], "arm": behaviour, "error": None,
                    "emitted": emitted, "tool_calls": cost})
    return out


def write_run(root, dom, name, eps):
    d = os.path.join(root, "w2a", "%s-%s" % (MODEL, dom), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump({"run": {"arm": name}, "episodes": eps}, fh)


def populate(root, arm_behaviour, cost=4.0):
    """arm_behaviour: {run_name: behaviour}"""
    for dom in bw.DOMAINS:
        for name, beh in arm_behaviour.items():
            write_run(root, dom, name, episodes(beh, domain=dom, cost=cost))


# ---------------------------------------------------------------------------
# T1: endpoint arithmetic
# ---------------------------------------------------------------------------

def test_cell_reports_closed_form_endpoints(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(bw, "RESULTS", root)
    populate(root, {"clean__B20": "clean", "dirty__B20": "dirty",
                    "silent__B20": "silent"}, cost=7.0)

    clean = bw.cell(MODEL, "clean__B20")
    dirty = bw.cell(MODEL, "dirty__B20")
    silent = bw.cell(MODEL, "silent__B20")

    assert clean["n_instances"] == 480, clean          # the balanced split
    assert clean["n_clusters"] == 60
    # emitting only best-supported claims cannot be unsupported under either
    # protocol, and F*_B >= F*_A means B cannot fire where A does not.
    assert clean["ofar_a"] == 0.0 and clean["ofar_b"] == 0.0
    assert clean["sensitivity"] == 0.0
    assert clean["content_false"] == 0.0
    # 84/480 balanced instances have NO A-supported claim: on those the
    # closed-form clean behaviour IS abstention, so a nonzero abstain rate
    # here is the benchmark's designed abstain stratum, not a scoring bug.
    assert clean["abstain"] == 17.5
    assert clean["cost"] == 7.0
    # utility is exactly mean(ceiling)/3*100
    want = sum(100.0 * f["ceiling"] / 3.0 for f in bw.FACTS.values()
               if f["split"] == "heldout_balanced") / 480
    assert abs(clean["utility"] - round(want, 1)) < 0.05

    # 328/480 balanced instances carry at least one unsupported claim
    assert dirty["ofar_a"] == 68.3, dirty
    assert dirty["utility"] == clean["utility"]        # same supported claim
    assert silent["abstain"] == 100.0
    assert silent["utility"] == 0.0 and silent["ofar_a"] == 0.0


def test_not_run_is_reported_rather_than_zero(tmp_path, monkeypatch):
    """A missing arm must read NOT RUN -- never 0.0, which would print as a
    perfect score in T1."""
    monkeypatch.setattr(bw, "RESULTS", str(tmp_path))
    assert bw.cell(MODEL, "never_launched__B20") == {"status": "NOT RUN"}


def test_errored_episodes_are_dropped_not_scored(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(bw, "RESULTS", root)
    eps = episodes("clean", domain="security")
    for e in eps[:10]:
        e["error"] = "timeout"
    write_run(root, "security", "x__B20", eps)
    for dom in ("clinical", "drugsafety", "financial"):
        write_run(root, dom, "x__B20", [])
    c = bw.cell(MODEL, "x__B20")
    assert c["n_instances"] == len(eps) - 10


# ---------------------------------------------------------------------------
# T2: the frozen inference actually fires
# ---------------------------------------------------------------------------

def test_ladder_contrast_detects_and_holm_adjusts(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(bw, "RESULTS", root)
    # every rung dirty except the top one: the last contrast must reject,
    # the interior (dirty->dirty) contrasts must not.
    behav = {"%s__B20" % a: ("clean" if a == "prime_guided" else "dirty")
             for a in bw.LADDER}
    populate(root, behav)
    bw.main(["--models", MODEL])
    T2 = json.load(open(os.path.join(root, "tableT2.json")))

    keys = [k for k in T2["ladder"] if k.startswith(MODEL)]
    assert len(keys) == len(bw.CHAIN) - 1 == 5, keys
    top = "%s|masked_reactive_v->prime_guided" % MODEL
    row = T2["ladder"][top]
    # SIGN: ofar_reduction_pp = OFAR(lower rung) - OFAR(higher rung), so a
    # LARGE POSITIVE value means the top rung removed the violations. Pinning
    # the sign here is the guard against a flipped column in T2.
    #
    # ESTIMAND: this is 69.3, NOT the 68.3 that T1 reports for the identical
    # emissions, because T2 is the pre-registered CLUSTER-level estimand
    # (mean over the 60 families of each family's mean difference) while T1
    # is an instance-level rate, and the balanced families hold 4-15
    # instances each. The two are pinned together here precisely so nobody
    # later "fixes" the apparent discrepancy by conflating the estimands.
    assert bw.cell(MODEL, "react_verify__B20")["ofar_a"] == 68.3
    assert row["ofar_reduction_pp"] == 69.3, row
    assert row["ci"][0] > 0 and row["ci"][1] > 0, row   # CI signed to match
    assert row["n_clusters"] == 60
    assert row["discordant_clusters"] == 60
    assert row["p_randomization"] < 0.01
    assert row["p_holm"] < 0.05 and row["reject_05"] is True
    interior = T2["ladder"]["%s|react_verify->react_spec_verify" % MODEL]
    assert interior["ofar_reduction_pp"] == 0.0
    assert interior["reject_05"] is False


def test_control_law_cells_split_by_topology_and_run_tost(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(bw, "RESULTS", root)
    behav = {}
    for anchor in ("bmin", "bmin1", "2bmin", "B20"):
        behav["prime_guided__%s" % anchor] = "clean"
        behav["masked_reactive_v__%s" % anchor] = "clean"
    populate(root, behav)
    bw.main(["--models", MODEL, "--delta", "5.0"])
    T2 = json.load(open(os.path.join(root, "tableT2.json")))

    assert "_conventions" in T2, "T2 must carry its sign conventions"
    cl = {k: v for k, v in T2["control_law"].items() if k.startswith(MODEL)}
    assert len(cl) == 8, sorted(cl)               # 4 anchors x 2 topologies
    for k, v in cl.items():
        assert v["n_instances"] > 0, k
        # identical behaviour => exact equivalence, and the fixture is large
        # enough that TOST is powered rather than merely inconclusive.
        assert v["utility_gap_pp"] == 0.0, k
        assert v["tost_equivalent"] is True, k
        assert v["power_at_zero"] > 0.8, (k, v["power_at_zero"])
    tot = sum(v["n_instances"] for k, v in cl.items() if "|B20|" in k)
    assert tot == 480, tot                        # topologies partition the split


def test_underpowered_control_law_is_not_called_equivalent(tmp_path, monkeypatch):
    """The pre-committed 'inconclusive' branch: a genuine utility gap must
    not be reported as equivalence."""
    root = str(tmp_path)
    monkeypatch.setattr(bw, "RESULTS", root)
    populate(root, {"prime_guided__B20": "clean",
                    "masked_reactive_v__B20": "silent"})
    bw.main(["--models", MODEL, "--delta", "5.0"])
    T2 = json.load(open(os.path.join(root, "tableT2.json")))
    cl = [v for k, v in T2["control_law"].items() if k.startswith(MODEL)]
    assert cl and all(v["tost_equivalent"] is False for v in cl)
    # SIGN: utility_gap_pp = prime_guided - masked_reactive_v, so PRIME
    # keeping utility while the baseline abstains must read POSITIVE.
    assert all(v["utility_gap_pp"] > 5.0 for v in cl)


def test_paired_join_is_by_instance_and_drops_unmatched(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(bw, "RESULTS", root)
    for dom in bw.DOMAINS:
        full = episodes("clean", domain=dom)
        write_run(root, dom, "a__B20", full)
        write_run(root, dom, "b__B20", full[:5])   # partial second arm
    rows = bw.paired(MODEL, "a__B20", "b__B20", "utility")
    assert len(rows) == 5 * len(bw.DOMAINS)
    assert all(r["a"] == r["b"] for r in rows)
    assert {r["topology"] for r in rows} <= {"branching", "nested"}
