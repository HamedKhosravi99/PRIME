"""Tests for the closed structured-record predicate grammar
(prime_runtime/predicates.py, Tier-M contract).

Exercises every one of the 10 predicate types, the rule-string parse/format
round-trip, and the fail-closed contract (an inapplicable predicate/value
combination returns None, never a guessed True/False).
"""

import pytest

from prime_runtime import predicates as P


# ---------------------------------------------------------------------------
# parse_predicate_rule / format_predicate_rule
# ---------------------------------------------------------------------------


def test_parse_rejects_non_matching_prefix():
    assert P.parse_predicate_rule("span_read") is None
    assert P.parse_predicate_rule("exit_code==0") is None
    assert P.parse_predicate_rule("") is None
    assert P.parse_predicate_rule(None) is None  # type: ignore[arg-type]


def test_parse_rejects_unknown_predicate_type():
    assert P.parse_predicate_rule("record_predicate:frobnicate:1") is None


def test_parse_rejects_malformed_json_argument():
    assert P.parse_predicate_rule("record_predicate:equals:{not json") is None


@pytest.mark.parametrize(
    "ptype,expected",
    [
        ("equals", "CRITICAL"),
        ("equals", 9.8),
        ("in_set", ["LOW", "HIGH"]),
        ("numeric_eq", 1.0),
        ("numeric_ge", 7.0),
        ("numeric_le", 10),
        ("in_version_range", ["1.0.0", "1.4.2"]),
        ("in_version_range", [None, "2.0.0"]),
        ("date_before", "2026-01-01"),
        ("date_after", "2020-01-01"),
        ("substring_contains", "CVE-2023"),
        ("exists", None),
    ],
)
def test_format_then_parse_round_trips(ptype, expected):
    rule = P.format_predicate_rule(ptype, expected)
    assert rule.startswith("record_predicate:%s:" % ptype)
    parsed = P.parse_predicate_rule(rule)
    assert parsed == (ptype, expected)


def test_format_rejects_unknown_type():
    with pytest.raises(ValueError):
        P.format_predicate_rule("not_a_real_type", 1)


def test_predicate_types_is_exactly_the_documented_closed_set():
    assert set(P.PREDICATE_TYPES) == {
        "equals",
        "in_set",
        "numeric_eq",
        "numeric_ge",
        "numeric_le",
        "in_version_range",
        "date_before",
        "date_after",
        "substring_contains",
        "exists",
    }


# ---------------------------------------------------------------------------
# evaluate_predicate: one True and one False case per type
# ---------------------------------------------------------------------------


def test_exists():
    assert P.evaluate_predicate("exists", None, "anything") is True
    assert P.evaluate_predicate("exists", None, 0) is True  # falsy but present
    assert P.evaluate_predicate("exists", None, False) is True
    assert P.evaluate_predicate("exists", None, None) is False


def test_equals():
    assert P.evaluate_predicate("equals", "CRITICAL", "CRITICAL") is True
    assert P.evaluate_predicate("equals", "CRITICAL", "HIGH") is False
    assert P.evaluate_predicate("equals", 9.8, 9.8) is True


def test_in_set():
    assert P.evaluate_predicate("in_set", ["LOW", "HIGH"], "HIGH") is True
    assert P.evaluate_predicate("in_set", ["LOW", "HIGH"], "MEDIUM") is False
    # non-list expected argument: not applicable, fail-closed
    assert P.evaluate_predicate("in_set", "HIGH", "HIGH") is None


def test_numeric_eq_ge_le():
    assert P.evaluate_predicate("numeric_eq", 9.8, 9.8) is True
    assert P.evaluate_predicate("numeric_eq", 9.8, 9.9) is False
    assert P.evaluate_predicate("numeric_ge", 7.0, 9.8) is True
    assert P.evaluate_predicate("numeric_ge", 7.0, 6.9) is False
    assert P.evaluate_predicate("numeric_le", 10.0, 9.8) is True
    assert P.evaluate_predicate("numeric_le", 10.0, 10.1) is False
    # non-numeric value: fail-closed, never a guess
    assert P.evaluate_predicate("numeric_ge", 7.0, "high") is None
    assert P.evaluate_predicate("numeric_ge", 7.0, None) is None


def test_in_version_range_half_open_both_bounds():
    rng = ["1.0.0", "1.4.2"]
    assert P.evaluate_predicate("in_version_range", rng, "1.2.0") is True
    assert P.evaluate_predicate("in_version_range", rng, "1.0.0") is True  # introduced: inclusive
    assert P.evaluate_predicate("in_version_range", rng, "1.4.2") is False  # fixed: EXCLUDED
    assert P.evaluate_predicate("in_version_range", rng, "0.9.9") is False  # below introduced
    assert P.evaluate_predicate("in_version_range", rng, "2.0.0") is False  # at/above fixed


def test_in_version_range_open_bounds():
    assert P.evaluate_predicate("in_version_range", [None, "2.0.0"], "0.0.1") is True
    assert P.evaluate_predicate("in_version_range", [None, "2.0.0"], "2.0.0") is False
    assert P.evaluate_predicate("in_version_range", ["1.0.0", None], "999.0.0") is True
    assert P.evaluate_predicate("in_version_range", ["1.0.0", None], "0.1.0") is False


def test_in_version_range_malformed_arguments_fail_closed():
    assert P.evaluate_predicate("in_version_range", "not-a-list", "1.0.0") is None
    assert P.evaluate_predicate("in_version_range", ["1.0.0"], "1.0.0") is None  # wrong length
    assert P.evaluate_predicate("in_version_range", ["1.0.0", "2.0.0"], "") is None  # unparseable value


def test_in_version_range_incomparable_tokens_fail_closed():
    """Mixed int/str token at the same tuple position is not a guess."""
    assert P.evaluate_predicate("in_version_range", ["1.0.0", "2.0.0"], "rc.0.0") is None


def test_date_before_after():
    assert P.evaluate_predicate("date_before", "2026-01-01", "2025-03-01") is True
    assert P.evaluate_predicate("date_before", "2025-01-01", "2025-03-01") is False
    assert P.evaluate_predicate("date_after", "2020-01-01", "2025-03-01") is True
    assert P.evaluate_predicate("date_after", "2030-01-01", "2025-03-01") is False
    # datetime-with-time and trailing 'Z' both parse
    assert P.evaluate_predicate("date_before", "2026-01-01", "2025-03-01T12:00:00Z") is True


def test_date_unparseable_fails_closed():
    assert P.evaluate_predicate("date_before", "2026-01-01", "not-a-date") is None
    assert P.evaluate_predicate("date_before", "not-a-date", "2025-03-01") is None


def test_substring_contains():
    assert P.evaluate_predicate("substring_contains", "CVE-2023", "see CVE-2023-1234 for detail") is True
    assert P.evaluate_predicate("substring_contains", "CVE-2099", "see CVE-2023-1234 for detail") is False


def test_all_predicates_fail_closed_on_absent_value_except_exists():
    for ptype in P.PREDICATE_TYPES:
        expected = ["x"] if ptype in ("in_set",) else (
            ["1.0.0", "2.0.0"] if ptype == "in_version_range" else "x"
        )
        result = P.evaluate_predicate(ptype, expected, None)
        if ptype == "exists":
            assert result is False
        else:
            assert result is None, ptype
