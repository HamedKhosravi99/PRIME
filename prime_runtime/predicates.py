"""Closed structured-record predicate grammar (Tier-M contract).

Implements the finite, mechanically-decidable predicate-atom grammar shared
by every structured-record evaluation domain this module is infrastructure
for (security/OSV, financial/SEC-XBRL, drug-safety/openFDA, clinical-trials/
CTgov -- authored separately, on top of this module -- and the synthetic
smoke instance under bench/instances/_smoke_record/). See
prime_runtime/RECORD_ORACLE_REPORT.md for the design rationale and the
per-domain field mapping.

A predicate atom resolves an executable Boolean over ONE field of ONE named
JSON record embedded in the instance (``instance.records[record_id]``, SPEC
"records" top-level key): given the raw field value the ``inspect_record``
tool reads (runner/arms.py ``ToolExecutor._tool_inspect_record``), the
atom's ``resolver.attest`` rule string

    "record_predicate:<predicate_type>:<json-encoded expected argument>"

is evaluated against that value by :func:`evaluate_predicate`. This module
owns the grammar; ``goldledger.eval_attest`` dispatches to it exactly like
every other attestation rule (fail-closed: ``None`` on inapplicable/
malformed input, ``"T"``/``"F"`` otherwise) -- the atom's determined value
then flows through the SAME ``Ledger`` / ``closure`` / ``oracle`` machinery
every other atom uses. Nothing downstream of ``eval_attest`` -- ``Ledger``,
``closure``, ``auth``, ``oracle``, ``interpreter`` -- is record-domain
specific or needed any change for this capability.

CLOSED GRAMMAR -- exactly these 10 predicate types (documented once here;
extend this file by adding a case to :func:`evaluate_predicate` plus a new
entry in :data:`PREDICATE_TYPES`, never by inventing an ad hoc rule string
elsewhere):

    equals              value == expected (JSON deep equality)
    in_set              value in expected               (expected: JSON list)
    numeric_eq          float(value) == float(expected)
    numeric_ge          float(value) >= float(expected)
    numeric_le          float(value) <= float(expected)
    in_version_range    expected: [introduced, fixed], either may be null;
                         True iff introduced <= version(value) < fixed
                         (dotted semver/Maven-style comparison; half-open,
                         matching the OSV/Maven "fixed version excluded"
                         convention -- a null bound is unbounded on that side)
    date_before         date(value) < date(expected)     (ISO 8601)
    date_after          date(value) > date(expected)     (ISO 8601)
    substring_contains  str(expected) in str(value)
    exists              value is not None                (no other argument)

Design note -- WHY the predicate type and expected argument live in the
ATOM's ``attest`` string rather than being supplied by the model's tool
call. The trap this design avoids (see RECORD_ORACLE_REPORT.md): if an atom
declared exact evidence content (a predicate/expected/threshold) as part of
``resolver.params``, it would never fire, because ``goldledger.delta``
matches ``resolver.params`` against the model's OWN tool-call arguments by
EXACT equality, and a real model's natural call essentially never
reproduces such content verbatim (this already happened with span-read
atoms that declared exact start/end line numbers). This grammar's atoms
declare only STABLE, tool-call-shaped identifying params (``record_id``,
``field_path`` -- the record-domain analogue of ``inspect_file``'s
``path``): the predicate type and the expected value are never something
the calling model has to guess or reproduce; they are fixed at authoring
time and evaluated here against whatever value the tool call actually
returns.
"""

import json
import re
from datetime import date, datetime
from typing import Any, Optional, Tuple

__all__ = [
    "PREDICATE_TYPES",
    "parse_predicate_rule",
    "format_predicate_rule",
    "evaluate_predicate",
]

PREDICATE_TYPES: Tuple[str, ...] = (
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
)

_PREFIX = "record_predicate:"
_VERSION_TOKEN_RE = re.compile(r"[0-9]+|[A-Za-z]+")


def parse_predicate_rule(rule: str) -> Optional[Tuple[str, Any]]:
    """Parse a ``record_predicate:<type>:<json>`` attest rule string.

    Returns ``(predicate_type, expected)``, or ``None`` if ``rule`` is not a
    well-formed member of the closed grammar (wrong prefix, unknown type, or
    the expected argument does not parse as JSON). Callers must treat
    ``None`` as "cannot attest" (fail-closed) -- never as False.
    """
    if not isinstance(rule, str) or not rule.startswith(_PREFIX):
        return None
    rest = rule[len(_PREFIX):]
    ptype, sep, raw = rest.partition(":")
    if not sep or ptype not in PREDICATE_TYPES:
        return None
    try:
        expected = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return ptype, expected


def format_predicate_rule(predicate_type: str, expected: Any) -> str:
    """Inverse of :func:`parse_predicate_rule`: build a well-formed attest
    rule string. Used by instance authoring and by tests (round-trip)."""
    if predicate_type not in PREDICATE_TYPES:
        raise ValueError("unknown predicate type %r" % (predicate_type,))
    return "%s%s:%s" % (_PREFIX, predicate_type, json.dumps(expected))


def _parse_version(raw: Any) -> Optional[Tuple[Any, ...]]:
    """Dotted semver/Maven-style version -> comparable tuple of int/str
    tokens, e.g. ``'1.2.0'`` -> ``(1, 2, 0)``; ``'2.0.0-rc1'`` ->
    ``(2, 0, 0, 'rc', 1)``. Returns ``None`` if ``raw`` yields no tokens."""
    if raw is None:
        return None
    tokens = _VERSION_TOKEN_RE.findall(str(raw))
    if not tokens:
        return None
    return tuple(int(t) if t.isdigit() else t for t in tokens)


def _version_cmp(a: Tuple[Any, ...], op: str, b: Tuple[Any, ...]) -> Optional[bool]:
    """``a <op> b`` over version tuples; None when the tuples are not
    order-comparable (mixed int/str token at the same position -- e.g.
    comparing '1.2.0' against 'rc.2.0' -- fail-closed, never a guess)."""
    try:
        return (a < b) if op == "<" else (a >= b)
    except TypeError:
        return None


def _parse_date(raw: Any) -> Optional[date]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def evaluate_predicate(predicate_type: str, expected: Any, value: Any) -> Optional[bool]:
    """Evaluate one closed-grammar predicate.

    Returns True/False, or None when the predicate cannot be applied to
    ``value``/``expected`` as given (fail-closed -- e.g. an unparseable
    version or date, a non-list ``in_set``/``in_version_range`` argument, a
    non-numeric value under a numeric predicate). Callers (``goldledger.
    eval_attest``) must treat None as "no attestation", never as False.
    """
    if predicate_type == "exists":
        return value is not None
    if value is None:
        return None  # no other predicate is decidable over an absent field
    if predicate_type == "equals":
        return value == expected
    if predicate_type == "in_set":
        if not isinstance(expected, list):
            return None
        return value in expected
    if predicate_type in ("numeric_eq", "numeric_ge", "numeric_le"):
        try:
            v, e = float(value), float(expected)
        except (TypeError, ValueError):
            return None
        if predicate_type == "numeric_eq":
            return v == e
        if predicate_type == "numeric_ge":
            return v >= e
        return v <= e
    if predicate_type == "in_version_range":
        if not (isinstance(expected, list) and len(expected) == 2):
            return None
        introduced, fixed = expected
        v = _parse_version(value)
        if v is None:
            return None
        if introduced not in (None, ""):
            lo = _parse_version(introduced)
            if lo is None:
                return None
            lt = _version_cmp(v, "<", lo)
            if lt is None:
                return None
            if lt:
                return False
        if fixed not in (None, ""):
            hi = _parse_version(fixed)
            if hi is None:
                return None
            ge = _version_cmp(v, ">=", hi)
            if ge is None:
                return None
            if ge:
                return False
        return True
    if predicate_type in ("date_before", "date_after"):
        v, e = _parse_date(value), _parse_date(expected)
        if v is None or e is None:
            return None
        return v < e if predicate_type == "date_before" else v > e
    if predicate_type == "substring_contains":
        return str(expected) in str(value)
    return None  # pragma: no cover - predicate_type is validated by the parser
