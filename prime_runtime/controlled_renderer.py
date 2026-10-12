"""Deterministic controlled natural-language renderer (additive stronger path).

Implements a controlled natural-language renderer, the renderer-soundness
definition, grammar-level enforcement, and the hooks used by the
mutation/property tests R1-R4 (prime_runtime/tests/test_controlled_renderer.py).

Problem this module removes
----------------------------
The old ``renderer.py`` is already deterministic and template-based, but its
"claim content" is an opaque, benchmark-authored free-text string
(``claim_texts[cid]``).  Nothing in that module checks that the free text
itself cannot say more than the typed object licenses; fidelity there would
have to be checked by a learned entailment verifier, which does not reach
the exact-guarantee bar.

This module instead makes the *sentence itself* a typed, finite object:

    Claim(dimension, strength_level, scope, qualifiers)  -->  ONE fixed
    template string per (dimension, strength_level, polarity, scope-shape),
    whose slots accept ONLY provenance-bearing identifiers, numbers, dates,
    or a whitelisted-character quoted span -- never adjectives or clauses.

``render`` and ``parse`` are built from the SAME finite ``TEMPLATES``
registry (single source of truth: parsing cannot silently drift from
generation), so:

    renderer_sound(O)  <=>  denote(render(O)) subseteq AuthContents(O)

is a property that can be checked exactly, with no learned component and no
human/LLM judge (see :func:`renderer_sound`).

This is an ADDITIVE stronger path.  ``renderer.py`` is untouched; other code
that imports it keeps working unchanged.  This module does not depend on, or
modify, ``interpreter.py`` -- it composes with the interpreter's Auth mask
(``auth.py``) rather than re-deriving it: what counts as "authorized" for a
given typed object is supplied by the caller (typically: the interpreter's
``Auth_P(z)`` output, converted to :class:`Claim` objects), matching the
three-term risk decomposition:

    Pr(E_user_false) <= Pr(E_prog) + Pr(E_mask) + Pr(E_render)

``renderer_sound`` here targets exactly and only ``Pr(E_render) = 0``.
"""

import itertools
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SlotError(ValueError):
    """A claim/evidence/obligation/retraction field is not a valid slot
    value for its declared type -- render refuses (fail-closed at
    construction time, before any string is ever produced)."""


class TemplateNotFoundError(ValueError):
    """No registered template exists for a (dimension, strength, polarity,
    scope-shape) combination -- render refuses rather than improvising
    prose for an unregistered claim shape."""


class RenderGrammarError(ValueError):
    """Candidate text is not a member of the fixed controlled grammar --
    parse refuses (fail-closed): unparseable text can never be treated as an
    authorized rendering."""


# ---------------------------------------------------------------------------
# Slot types and their exact character-level grammars
# ---------------------------------------------------------------------------


class SlotType(Enum):
    IDENT = "ident"      # provenance-bearing identifier (commit, path, DOI, id)
    NUMBER = "number"    # integer or decimal
    DATE = "date"        # ISO-8601 calendar date
    QUOTED = "quoted"    # whitelisted-character quoted span (escaped)
    TVAL = "tval"        # a ledger four-valued state token: T | F | C | U


_IDENT_BODY = r"[A-Za-z0-9_./:-]{0,127}"
_IDENT_PAT = r"[A-Za-z0-9]" + _IDENT_BODY
_NUMBER_PAT = r"-?[0-9]+(?:\.[0-9]+)?"
_DATE_PAT = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
# Whitelisted inner alphabet for quoted spans: letters, digits, space, and a
# small closed punctuation set.  Deliberately EXCLUDES: newlines, quote/
# guillemet characters, backslash, braces, angle brackets, pipe, and "=" --
# so no character available inside a QUOTED slot can ever form a line break,
# a delimiter, or a zone-header token (see R3 in the test suite).
_QUOTED_INNER_PAT = r"[A-Za-z0-9 ,.;:/_'()-]{1,200}"
_QUOTE_OPEN, _QUOTE_CLOSE = "«", "»"  # « »  (not in the inner alphabet)
_TVAL_PAT = r"[TFCU]"

_IDENT_RE = re.compile("^%s$" % _IDENT_PAT)
_NUMBER_RE = re.compile("^%s$" % _NUMBER_PAT)
_DATE_RE = re.compile("^%s$" % _DATE_PAT)
_QUOTED_INNER_RE = re.compile("^%s$" % _QUOTED_INNER_PAT)
_TVAL_RE = re.compile("^%s$" % _TVAL_PAT)

_SLOT_GROUP_PATTERN: Dict[SlotType, str] = {
    SlotType.IDENT: _IDENT_PAT,
    SlotType.NUMBER: _NUMBER_PAT,
    SlotType.DATE: _DATE_PAT,
    SlotType.QUOTED: _QUOTE_OPEN + _QUOTED_INNER_PAT + _QUOTE_CLOSE,
    SlotType.TVAL: _TVAL_PAT,
}


def _format_value(value: Any, slot_type: SlotType) -> str:
    """Validate ``value`` against its slot grammar and return the exact
    substring to place in rendered text.  Raises :class:`SlotError` on any
    character outside the slot's whitelist -- this is where injection is
    actually prevented, at render (construction) time, not merely detected
    later at parse time."""
    s = str(value)
    if slot_type is SlotType.IDENT:
        if not _IDENT_RE.match(s):
            raise SlotError("invalid IDENT slot value %r" % (value,))
        return s
    if slot_type is SlotType.NUMBER:
        if not _NUMBER_RE.match(s):
            raise SlotError("invalid NUMBER slot value %r" % (value,))
        return s
    if slot_type is SlotType.DATE:
        if not _DATE_RE.match(s):
            raise SlotError("invalid DATE slot value %r" % (value,))
        return s
    if slot_type is SlotType.QUOTED:
        if not _QUOTED_INNER_RE.match(s):
            raise SlotError("invalid QUOTED slot value %r" % (value,))
        return _QUOTE_OPEN + s + _QUOTE_CLOSE
    if slot_type is SlotType.TVAL:
        if not _TVAL_RE.match(s):
            raise SlotError("invalid ledger-value slot %r" % (value,))
        return s
    raise SlotError("unknown slot type %r" % (slot_type,))  # pragma: no cover


# ---------------------------------------------------------------------------
# The typed claim AST
# ---------------------------------------------------------------------------


class Dimension(Enum):
    TEST_STATUS = "TEST_STATUS"
    VULNERABILITY_STATUS = "VULNERABILITY_STATUS"
    LICENSE_COMPLIANCE = "LICENSE_COMPLIANCE"
    ARTIFACT_PROVENANCE = "ARTIFACT_PROVENANCE"
    BUDGET_COMPLIANCE = "BUDGET_COMPLIANCE"
    DEADLINE_COMPLIANCE = "DEADLINE_COMPLIANCE"
    CITATION_INTEGRITY = "CITATION_INTEGRITY"


POLARITY_POS = "POS"
POLARITY_NEG = "NEG"


@dataclass(frozen=True)
class Qualifier:
    """A registered qualifier clause, e.g. ``Qualifier("OBSERVED_AT",
    "2026-07-01")``."""

    kind: str
    value: str


@dataclass(frozen=True)
class Claim:
    """Typed claim AST:

        Claim(dimension, strength_level, scope, qualifiers)

    ``claim_id`` and ``polarity`` are additional required/optional fields
    (default-valued so the 4-positional-argument call shape above
    still works); ``claim_id`` is enforced non-empty in ``__post_init__``
    because every rendered finding is addressed by id (for retraction
    cross-referencing, R4)."""

    dimension: Dimension
    strength_level: int
    scope: Mapping[str, Any]
    qualifiers: Tuple[Qualifier, ...] = ()
    claim_id: str = ""
    polarity: str = POLARITY_POS

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("Claim.claim_id is required")


@dataclass(frozen=True)
class EvidenceRecord:
    """One factual, non-evaluative evidence-log entry: a ledger attestation
    record, verbatim -- no evaluative predicate is representable here."""

    record_id: str
    atom: str
    value: str  # "T" | "F" | "C" | "U"
    tool: str
    step: int
    ledger_version: int


@dataclass(frozen=True)
class UnresolvedClaim:
    """A claim not (yet) authorized; rendered only via the fixed
    "NOT AUTHORIZED" template -- this is a statement ABOUT non-authorization,
    never the underlying strong claim itself."""

    claim_id: str
    dimension: Dimension
    strength_level: int
    residual_obligations: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.residual_obligations:
            raise ValueError("residual_obligations must be non-empty")


@dataclass(frozen=True)
class Retraction:
    """R4: a prior claim's retraction, referencing the exact earlier claim
    id and ledger version it supersedes."""

    claim_id: str
    ledger_version: int
    prior_claim_id: str
    prior_ledger_version: int
    conflicting_atom: str


@dataclass(frozen=True)
class RenderObject:
    """The typed object handed to :func:`render`.  ``authorized`` is exactly
    what the caller (normally the interpreter's Auth mask) has determined is
    licensed; the renderer's only job is to never say more than this list."""

    authorized: Tuple[Claim, ...] = ()
    evidence: Tuple[EvidenceRecord, ...] = ()
    unresolved: Tuple[UnresolvedClaim, ...] = ()
    retractions: Tuple[Retraction, ...] = ()


@dataclass(frozen=True)
class ClaimContent:
    """The denotation of one rendered finding: exactly the semantic content
    a controlled sentence can be read as asserting.  Two claims with the
    same content are the same claim regardless of surface claim_id."""

    dimension: Dimension
    strength_level: int
    polarity: str
    scope: FrozenSet[Tuple[str, str]]
    qualifiers: FrozenSet[Tuple[str, str]]


def _claim_content(c: Claim) -> ClaimContent:
    return ClaimContent(
        dimension=c.dimension,
        strength_level=c.strength_level,
        polarity=c.polarity,
        scope=frozenset((str(k), str(v)) for k, v in c.scope.items()),
        qualifiers=frozenset((q.kind, str(q.value)) for q in c.qualifiers),
    )


def auth_contents(obj: RenderObject) -> FrozenSet[ClaimContent]:
    """AuthContents(O): the exact set of claim contents ``O`` authorizes."""
    return frozenset(_claim_content(c) for c in obj.authorized)


# ---------------------------------------------------------------------------
# The fixed template registry (the grammar's single source of truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSpec:
    dimension: Dimension
    strength_level: int
    polarity: str
    scope_keys: Tuple[str, ...]  # sorted tuple = the claim's "scope shape"
    text: str  # fixed sentence, {name} placeholders only, no qualifiers
    slot_types: Mapping[str, SlotType]


def _spec(dimension, strength_level, polarity, slot_types, text):
    scope_keys = tuple(sorted(slot_types))
    placeholders = set(_PLACEHOLDER_RE.findall(text))
    assert placeholders == set(slot_types), (
        "template/slot mismatch for %s/%s/%s: %r vs %r"
        % (dimension, strength_level, polarity, placeholders, set(slot_types))
    )
    return TemplateSpec(dimension, strength_level, polarity, scope_keys, text, slot_types)


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Qualifier registry: kind -> (slot type, fixed appended clause).  Applied in
# canonical (sorted-by-kind) order; at most one clause per kind.
QUALIFIER_REGISTRY: Dict[str, Tuple[SlotType, str]] = {
    "CONFIG": (SlotType.IDENT, " Configuration: {value}."),
    "OBSERVED_AT": (SlotType.DATE, " Observed at {value}."),
}
_QUAL_KINDS_SORTED: Tuple[str, ...] = tuple(sorted(QUALIFIER_REGISTRY))
_QUAL_SUBSETS: Tuple[Tuple[str, ...], ...] = tuple(
    itertools.chain.from_iterable(
        itertools.combinations(_QUAL_KINDS_SORTED, r)
        for r in range(len(_QUAL_KINDS_SORTED) + 1)
    )
)

_TEMPLATE_LIST: List[TemplateSpec] = [
    # -- TEST_STATUS ----------------------------------------------------
    _spec(
        Dimension.TEST_STATUS, 1, POLARITY_POS,
        {"commit": SlotType.IDENT, "repo": SlotType.IDENT, "suite": SlotType.IDENT},
        "At commit {commit} in {repo}, the registered test suite {suite} was inspected.",
    ),
    _spec(
        Dimension.TEST_STATUS, 1, POLARITY_POS,
        {"commit": SlotType.IDENT, "suite": SlotType.IDENT},
        "At commit {commit}, the registered test suite {suite} was inspected.",
    ),
    _spec(
        Dimension.TEST_STATUS, 2, POLARITY_POS,
        {"commit": SlotType.IDENT, "repo": SlotType.IDENT, "suite": SlotType.IDENT},
        "At commit {commit} in {repo}, the registered test suite {suite} passed.",
    ),
    _spec(
        Dimension.TEST_STATUS, 2, POLARITY_POS,
        {"commit": SlotType.IDENT, "suite": SlotType.IDENT},
        "At commit {commit}, the registered test suite {suite} passed.",
    ),
    _spec(
        Dimension.TEST_STATUS, 3, POLARITY_POS,
        {"commit": SlotType.IDENT, "repo": SlotType.IDENT, "suite": SlotType.IDENT},
        "At commit {commit} in {repo}, the registered test suite {suite} passed and "
        "the protected regression set showed zero failures.",
    ),
    # -- VULNERABILITY_STATUS (both polarities are registered claims,
    #    not free negation of one another) -------------------------------
    _spec(
        Dimension.VULNERABILITY_STATUS, 1, POLARITY_NEG,
        {"advisory": SlotType.IDENT, "package": SlotType.IDENT, "version": SlotType.IDENT},
        "Package {package} at version {version} is not listed as affected by advisory {advisory}.",
    ),
    _spec(
        Dimension.VULNERABILITY_STATUS, 1, POLARITY_POS,
        {"advisory": SlotType.IDENT, "package": SlotType.IDENT, "version": SlotType.IDENT},
        "Package {package} at version {version} is listed as affected by advisory {advisory}.",
    ),
    # -- LICENSE_COMPLIANCE ----------------------------------------------
    _spec(
        Dimension.LICENSE_COMPLIANCE, 1, POLARITY_POS,
        {"commit": SlotType.IDENT, "repo": SlotType.IDENT, "spdx": SlotType.IDENT},
        "At commit {commit} in {repo}, the declared SPDX identifier {spdx} is present "
        "in the license manifest.",
    ),
    # -- ARTIFACT_PROVENANCE ----------------------------------------------
    _spec(
        Dimension.ARTIFACT_PROVENANCE, 1, POLARITY_POS,
        {"commit": SlotType.IDENT, "repo": SlotType.IDENT, "snapshot": SlotType.IDENT},
        "The checked revision {commit} of {repo} matches the archived snapshot {snapshot}.",
    ),
    # -- BUDGET_COMPLIANCE -------------------------------------------------
    _spec(
        Dimension.BUDGET_COMPLIANCE, 1, POLARITY_POS,
        {"amount": SlotType.NUMBER, "ceiling": SlotType.NUMBER, "opportunity": SlotType.IDENT},
        "The requested budget {amount} for opportunity {opportunity} is within the "
        "official ceiling {ceiling}.",
    ),
    # -- DEADLINE_COMPLIANCE -----------------------------------------------
    _spec(
        Dimension.DEADLINE_COMPLIANCE, 1, POLARITY_POS,
        {"deadline": SlotType.DATE, "opportunity": SlotType.IDENT, "submitted": SlotType.DATE},
        "The submission for opportunity {opportunity} on {submitted} is before the "
        "listed deadline {deadline}.",
    ),
    # -- CITATION_INTEGRITY -------------------------------------------------
    _spec(
        Dimension.CITATION_INTEGRITY, 1, POLARITY_POS,
        {"doi": SlotType.IDENT},
        "The cited identifier {doi} resolves in the Crossref metadata record.",
    ),
    _spec(
        Dimension.CITATION_INTEGRITY, 2, POLARITY_POS,
        {"doi": SlotType.IDENT, "title": SlotType.QUOTED},
        "The cited title {title} matches the Crossref metadata record for {doi}.",
    ),
]

TEMPLATES: Dict[Tuple[Dimension, int, str, Tuple[str, ...]], TemplateSpec] = {}
for _t in _TEMPLATE_LIST:
    _key = (_t.dimension, _t.strength_level, _t.polarity, _t.scope_keys)
    assert _key not in TEMPLATES, "duplicate template key %r" % (_key,)
    TEMPLATES[_key] = _t
del _t, _key


def iter_templates() -> Tuple[TemplateSpec, ...]:
    return tuple(TEMPLATES.values())


def iter_qualifier_subsets() -> Tuple[Tuple[str, ...], ...]:
    return _QUAL_SUBSETS


# ---------------------------------------------------------------------------
# render: typed object -> controlled string
# ---------------------------------------------------------------------------

_HEADER_FINDINGS = "=== AUTHORIZED FINDINGS ==="
_HEADER_EVIDENCE = "=== EVIDENCE LOG ==="
_HEADER_OBLIGATIONS = "=== REMAINING OBLIGATIONS ==="
_HEADER_RETRACTIONS = "=== RETRACTIONS ==="
_NONE_LINE = "(none)"
_ABSTENTION_LINE = "No claim is authorized under the committed program; abstaining."

_EVIDENCE_TEXT = "Atom {atom} attested {value} by tool {tool} (step {step}, ledger v{ledger_version})."
_OBLIGATION_TEXT = (
    "NOT AUTHORIZED: dimension={dimension}, strength_level={strength_level} -- "
    "residual obligations: {obligations}."
)
_RETRACTION_TEXT = (
    "RETRACTED at ledger v{ledger_version} (was [{prior_claim_id}]@v{prior_ledger_version}); "
    "reason: conflicting attestation on {conflicting_atom}."
)


def _lookup_template(c: Claim) -> TemplateSpec:
    scope_keys = tuple(sorted(c.scope.keys()))
    key = (c.dimension, c.strength_level, c.polarity, scope_keys)
    spec = TEMPLATES.get(key)
    if spec is None:
        raise TemplateNotFoundError(
            "no registered template for dimension=%s strength_level=%s polarity=%s "
            "scope_keys=%s" % (c.dimension, c.strength_level, c.polarity, scope_keys)
        )
    return spec


def _render_qualifiers(qualifiers: Sequence[Qualifier]) -> str:
    ordered = sorted(qualifiers, key=lambda q: q.kind)
    seen = set()
    out = []
    for q in ordered:
        if q.kind in seen:
            raise SlotError("duplicate qualifier kind %r" % q.kind)
        seen.add(q.kind)
        reg = QUALIFIER_REGISTRY.get(q.kind)
        if reg is None:
            raise SlotError("unregistered qualifier kind %r" % q.kind)
        qslot, clause = reg
        out.append(clause.format(value=_format_value(q.value, qslot)))
    return "".join(out)


def _render_claim_sentence(c: Claim) -> str:
    spec = _lookup_template(c)
    fills = {k: _format_value(c.scope[k], spec.slot_types[k]) for k in spec.scope_keys}
    return spec.text.format(**fills) + _render_qualifiers(c.qualifiers)


def _render_finding_line(c: Claim) -> str:
    return "[%s] %s" % (_format_value(c.claim_id, SlotType.IDENT), _render_claim_sentence(c))


def _render_evidence_line(e: EvidenceRecord) -> str:
    body = _EVIDENCE_TEXT.format(
        atom=_format_value(e.atom, SlotType.IDENT),
        value=_format_value(e.value, SlotType.TVAL),
        tool=_format_value(e.tool, SlotType.IDENT),
        step=_format_value(e.step, SlotType.NUMBER),
        ledger_version=_format_value(e.ledger_version, SlotType.NUMBER),
    )
    return "[%s] %s" % (_format_value(e.record_id, SlotType.IDENT), body)


def _render_obligation_line(u: UnresolvedClaim) -> str:
    idents = [_format_value(o, SlotType.IDENT) for o in u.residual_obligations]
    body = _OBLIGATION_TEXT.format(
        dimension=_format_value(u.dimension.value, SlotType.IDENT),
        strength_level=_format_value(u.strength_level, SlotType.NUMBER),
        obligations=", ".join(idents),
    )
    return "[%s] %s" % (_format_value(u.claim_id, SlotType.IDENT), body)


def _render_retraction_line(r: Retraction) -> str:
    body = _RETRACTION_TEXT.format(
        ledger_version=_format_value(r.ledger_version, SlotType.NUMBER),
        prior_claim_id=_format_value(r.prior_claim_id, SlotType.IDENT),
        prior_ledger_version=_format_value(r.prior_ledger_version, SlotType.NUMBER),
        conflicting_atom=_format_value(r.conflicting_atom, SlotType.IDENT),
    )
    return "[%s] %s" % (_format_value(r.claim_id, SlotType.IDENT), body)


def render(obj: RenderObject) -> str:
    """R(O): render a typed object to the controlled grammar.  Every
    evaluative sentence comes from a fixed template; slot values are
    validated (fail-closed) before any text is emitted."""
    lines: List[str] = [_HEADER_FINDINGS]
    if not obj.authorized:
        lines.append(_ABSTENTION_LINE)
    else:
        lines.extend(_render_finding_line(c) for c in obj.authorized)

    lines.append(_HEADER_EVIDENCE)
    if not obj.evidence:
        lines.append(_NONE_LINE)
    else:
        lines.extend(_render_evidence_line(e) for e in obj.evidence)

    lines.append(_HEADER_OBLIGATIONS)
    if not obj.unresolved:
        lines.append(_NONE_LINE)
    else:
        lines.extend(_render_obligation_line(u) for u in obj.unresolved)

    lines.append(_HEADER_RETRACTIONS)
    if not obj.retractions:
        lines.append(_NONE_LINE)
    else:
        lines.extend(_render_retraction_line(r) for r in obj.retractions)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parse: controlled string -> set[ClaimContent]  (deterministic, same grammar)
# ---------------------------------------------------------------------------


def _template_to_regex(text: str, slot_types: Mapping[str, SlotType], prefix: str) -> str:
    out = []
    pos = 0
    for m in _PLACEHOLDER_RE.finditer(text):
        out.append(re.escape(text[pos:m.start()]))
        name = m.group(1)
        out.append("(?P<%s%s>%s)" % (prefix, name, _SLOT_GROUP_PATTERN[slot_types[name]]))
        pos = m.end()
    out.append(re.escape(text[pos:]))
    return "".join(out)


def _compile_finding_regex(spec: TemplateSpec, qual_kinds: Tuple[str, ...]) -> "re.Pattern":
    pattern = r"^\[(?P<claim_id>%s)\] " % _IDENT_PAT
    pattern += _template_to_regex(spec.text, spec.slot_types, prefix="scope__")
    for kind in qual_kinds:
        qslot, clause = QUALIFIER_REGISTRY[kind]
        pattern += _template_to_regex(clause, {"value": qslot}, prefix="qual_%s__" % kind)
    pattern += "$"
    return re.compile(pattern)


_FINDING_PATTERNS: List[Tuple["re.Pattern", TemplateSpec, Tuple[str, ...]]] = []
for _spec_ in TEMPLATES.values():
    for _subset in _QUAL_SUBSETS:
        _FINDING_PATTERNS.append((_compile_finding_regex(_spec_, _subset), _spec_, _subset))
del _spec_, _subset

_EVIDENCE_RX = re.compile(
    r"^\[(?P<record_id>%s)\] " % _IDENT_PAT
    + _template_to_regex(
        _EVIDENCE_TEXT,
        {
            "atom": SlotType.IDENT,
            "value": SlotType.TVAL,
            "tool": SlotType.IDENT,
            "step": SlotType.NUMBER,
            "ledger_version": SlotType.NUMBER,
        },
        prefix="",
    )
    + "$"
)

_OBLIGATIONS_LIST_PAT = r"%s(?:, %s)*" % (_IDENT_PAT, _IDENT_PAT)
_OBLIGATION_RX = re.compile(
    r"^\[(?P<claim_id>%s)\] NOT AUTHORIZED: dimension=(?P<dimension>%s), "
    r"strength_level=(?P<strength_level>%s) -- residual obligations: "
    r"(?P<obligations>%s)\.$"
    % (_IDENT_PAT, _IDENT_PAT, _NUMBER_PAT, _OBLIGATIONS_LIST_PAT)
)

_RETRACTION_RX = re.compile(
    r"^\[(?P<claim_id>%s)\] " % _IDENT_PAT
    + _template_to_regex(
        _RETRACTION_TEXT,
        {
            "ledger_version": SlotType.NUMBER,
            "prior_claim_id": SlotType.IDENT,
            "prior_ledger_version": SlotType.NUMBER,
            "conflicting_atom": SlotType.IDENT,
        },
        prefix="",
    )
    + "$"
)


@dataclass(frozen=True)
class ParsedEvidence:
    record_id: str
    atom: str
    value: str
    tool: str
    step: str
    ledger_version: str


@dataclass(frozen=True)
class ParsedObligation:
    claim_id: str
    dimension: str
    strength_level: str
    obligations: Tuple[str, ...]


@dataclass(frozen=True)
class ParsedRetraction:
    claim_id: str
    ledger_version: str
    prior_claim_id: str
    prior_ledger_version: str
    conflicting_atom: str


@dataclass(frozen=True)
class ParsedDocument:
    findings: FrozenSet[ClaimContent]
    finding_ids: Tuple[str, ...]
    evidence: Tuple[ParsedEvidence, ...]
    obligations: Tuple[ParsedObligation, ...]
    retractions: Tuple[ParsedRetraction, ...]


def _strip_quoted(raw: str, slot_type: SlotType) -> str:
    return raw[1:-1] if slot_type is SlotType.QUOTED else raw


def _parse_finding_line(line: str, lineno: int) -> Tuple[str, ClaimContent]:
    for rx, spec, qual_kinds in _FINDING_PATTERNS:
        m = rx.fullmatch(line)
        if not m:
            continue
        gd = m.groupdict()
        scope = {
            k: _strip_quoted(gd["scope__%s" % k], spec.slot_types[k]) for k in spec.scope_keys
        }
        qualifiers = frozenset((k, gd["qual_%s__value" % k]) for k in qual_kinds)
        content = ClaimContent(
            dimension=spec.dimension,
            strength_level=spec.strength_level,
            polarity=spec.polarity,
            scope=frozenset(scope.items()),
            qualifiers=qualifiers,
        )
        return gd["claim_id"], content
    raise RenderGrammarError(
        "line %d matches no registered finding template: %r" % (lineno, line)
    )


def _parse_evidence_line(line: str, lineno: int) -> ParsedEvidence:
    m = _EVIDENCE_RX.fullmatch(line)
    if not m:
        raise RenderGrammarError("line %d is not a well-formed evidence record: %r" % (lineno, line))
    gd = m.groupdict()
    return ParsedEvidence(**gd)


def _parse_obligation_line(line: str, lineno: int) -> ParsedObligation:
    m = _OBLIGATION_RX.fullmatch(line)
    if not m:
        raise RenderGrammarError("line %d is not a well-formed obligation record: %r" % (lineno, line))
    gd = m.groupdict()
    obligations = tuple(gd["obligations"].split(", "))
    return ParsedObligation(
        claim_id=gd["claim_id"],
        dimension=gd["dimension"],
        strength_level=gd["strength_level"],
        obligations=obligations,
    )


def _parse_retraction_line(line: str, lineno: int) -> ParsedRetraction:
    m = _RETRACTION_RX.fullmatch(line)
    if not m:
        raise RenderGrammarError("line %d is not a well-formed retraction record: %r" % (lineno, line))
    gd = m.groupdict()
    return ParsedRetraction(**gd)


def _expect_header(lines: Sequence[str], i: int, expected: str) -> int:
    if i >= len(lines) or lines[i] != expected:
        got = lines[i] if i < len(lines) else "<end of text>"
        raise RenderGrammarError("expected header %r at line %d, got %r" % (expected, i, got))
    return i + 1


def parse(s: str) -> ParsedDocument:
    """The deterministic parser over the SAME finite grammar ``render``
    generates from.  Fail-closed: any deviation from the fixed skeleton
    (missing/duplicated/reordered zone headers, an unrecognized line, or
    trailing content) raises :class:`RenderGrammarError` rather than
    guessing a denotation."""
    if "\r" in s:
        raise RenderGrammarError("carriage return is not permitted in controlled output")
    lines = s.split("\n")

    i = _expect_header(lines, 0, _HEADER_FINDINGS)
    findings: List[Tuple[str, ClaimContent]] = []
    if i < len(lines) and lines[i] == _ABSTENTION_LINE:
        i += 1
    else:
        while i < len(lines) and lines[i] != _HEADER_EVIDENCE:
            findings.append(_parse_finding_line(lines[i], i))
            i += 1

    i = _expect_header(lines, i, _HEADER_EVIDENCE)
    evidence: List[ParsedEvidence] = []
    if i < len(lines) and lines[i] == _NONE_LINE:
        i += 1
    else:
        while i < len(lines) and lines[i] != _HEADER_OBLIGATIONS:
            evidence.append(_parse_evidence_line(lines[i], i))
            i += 1

    i = _expect_header(lines, i, _HEADER_OBLIGATIONS)
    obligations: List[ParsedObligation] = []
    if i < len(lines) and lines[i] == _NONE_LINE:
        i += 1
    else:
        while i < len(lines) and lines[i] != _HEADER_RETRACTIONS:
            obligations.append(_parse_obligation_line(lines[i], i))
            i += 1

    i = _expect_header(lines, i, _HEADER_RETRACTIONS)
    retractions: List[ParsedRetraction] = []
    if i < len(lines) and lines[i] == _NONE_LINE:
        i += 1
    else:
        while i < len(lines):
            retractions.append(_parse_retraction_line(lines[i], i))
            i += 1

    if i != len(lines):
        raise RenderGrammarError("trailing content after the retractions zone (line %d)" % i)

    return ParsedDocument(
        findings=frozenset(c for _cid, c in findings),
        finding_ids=tuple(cid for cid, _c in findings),
        evidence=tuple(evidence),
        obligations=tuple(obligations),
        retractions=tuple(retractions),
    )


def denote(s: str) -> FrozenSet[ClaimContent]:
    """denote(s): the set of typed claim contents the AUTHORIZED FINDINGS
    zone of ``s`` asserts.  Raises :class:`RenderGrammarError` if ``s`` is
    not a member of the controlled grammar at all."""
    return parse(s).findings


# ---------------------------------------------------------------------------
# Renderer soundness
# ---------------------------------------------------------------------------


def is_authorized_rendering(candidate: str, obj: RenderObject) -> bool:
    """True iff ``candidate`` both parses under the controlled grammar AND
    denotes only content in ``AuthContents(obj)``.  This is the primitive
    the R2 mutation suite calls: a MUTATED string must make this False."""
    try:
        parsed = parse(candidate)
    except RenderGrammarError:
        return False
    return parsed.findings <= auth_contents(obj)


def renderer_sound(obj: RenderObject) -> bool:
    """Definition: Renderer soundness --

        denote(render(O)) subseteq AuthContents(O)

    checked exactly, by construction, for this object."""
    return is_authorized_rendering(render(obj), obj)


# ---------------------------------------------------------------------------
# Retraction (R4)
# ---------------------------------------------------------------------------


def retract(
    obj: RenderObject,
    claim_id: str,
    prior_ledger_version: int,
    new_ledger_version: int,
    conflicting_atom: str,
) -> RenderObject:
    """Remove ``claim_id`` from the authorized findings and append a
    retraction referencing the exact earlier claim id and ledger version.
    The returned object, once rendered, no longer denotes the retracted
    claim as active (R4)."""
    remaining = tuple(c for c in obj.authorized if c.claim_id != claim_id)
    if len(remaining) == len(obj.authorized):
        raise ValueError("claim_id %r is not among the authorized claims" % claim_id)
    entry = Retraction(
        claim_id=claim_id,
        ledger_version=new_ledger_version,
        prior_claim_id=claim_id,
        prior_ledger_version=prior_ledger_version,
        conflicting_atom=conflicting_atom,
    )
    return RenderObject(
        authorized=remaining,
        evidence=obj.evidence,
        unresolved=obj.unresolved,
        retractions=obj.retractions + (entry,),
    )
