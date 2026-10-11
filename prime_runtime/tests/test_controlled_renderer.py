"""Tests for the deterministic controlled renderer (Weakness 5 fix).

Implements the R1-R4 property/mutation tests from
`the design notes` SS5.5, plus a
runtime-conformance test over the Table A columns named in SS5 (mask
violation, conflict-fail-open, premature claim, renderer strengthening,
retraction failure), all on one scripted episode driven by the REAL
:class:`~prime_runtime.interpreter.Interpreter` / :class:`~prime_runtime.ledger.Ledger`.

    R1  exhaustive template enumeration      -> test_R1_*
    R2  semantic-strength mutation suite      -> test_R2_*
    R3  slot injection                        -> test_R3_*
    R4  retraction consistency                -> test_R4_*
    Table A runtime conformance               -> test_runtime_conformance_table_a
"""

import pytest

from prime_runtime import controlled_renderer as cr
from prime_runtime.interpreter import Interpreter, UnauthorizedClaimError
from prime_runtime.ledger import Ledger


def _att(atom, value, step):
    return {"atom": atom, "value": value, "tool": "t", "params": {}, "output_digest": "", "step": step}


# ---------------------------------------------------------------------------
# R1: exhaustive template enumeration
#
# For every (dimension, strength, scope-shape, polarity) combination (each is
# one TemplateSpec in the registry -- the four axes are bundled together by
# construction, see TemplateSpec) x every qualifier subset within bounds
# (the qualifier registry's power set, 0..2 qualifiers): generate the typed
# object, render, parse, and check exact equality plus the theorem's
# subseteq form.
# ---------------------------------------------------------------------------


def _sample_value(slot_type, tag):
    if slot_type is cr.SlotType.IDENT:
        return "id-%s" % tag
    if slot_type is cr.SlotType.NUMBER:
        return str(100 + (abs(hash(tag)) % 900))
    if slot_type is cr.SlotType.DATE:
        return "2024-01-15"
    if slot_type is cr.SlotType.QUOTED:
        return "sample text %s" % tag
    if slot_type is cr.SlotType.TVAL:
        return "T"
    raise AssertionError("unhandled slot type %r" % (slot_type,))  # pragma: no cover


def test_R1_exhaustive_template_enumeration():
    templates = cr.iter_templates()
    qual_subsets = cr.iter_qualifier_subsets()
    assert len(templates) >= 10, "grammar suspiciously small; not a meaningful enumeration"
    assert len(qual_subsets) == 4  # power set of the 2-kind qualifier registry

    checked = 0
    for spec in templates:
        for qual_subset in qual_subsets:
            checked += 1
            tag = "%s_%d_%s" % (spec.dimension.value, spec.strength_level, spec.polarity)
            scope = {
                k: _sample_value(spec.slot_types[k], "%s_%s" % (tag, k)) for k in spec.scope_keys
            }
            qualifiers = tuple(
                cr.Qualifier(
                    kind, _sample_value(cr.QUALIFIER_REGISTRY[kind][0], "%s_q_%s" % (tag, kind))
                )
                for kind in qual_subset
            )
            claim = cr.Claim(
                dimension=spec.dimension,
                strength_level=spec.strength_level,
                scope=scope,
                qualifiers=qualifiers,
                claim_id="CLAIM_%d" % checked,
                polarity=spec.polarity,
            )
            obj = cr.RenderObject(authorized=(claim,))
            text = cr.render(obj)
            parsed = cr.parse(text)
            expected = cr.auth_contents(obj)

            # exact equality ...
            assert parsed.findings == expected, "round-trip mismatch for %r" % (tag,)
            # ... which is the (trivially tight) special case of the
            # authorized-downward-closure inequality in the theorem:
            assert parsed.findings <= expected
            assert cr.renderer_sound(obj) is True

    assert checked == len(templates) * len(qual_subsets)
    assert checked >= 40


def test_R1_abstention_round_trip():
    """The zero-claims object denotes the empty set, not some default claim."""
    obj = cr.RenderObject()
    text = cr.render(obj)
    parsed = cr.parse(text)
    assert parsed.findings == frozenset()
    assert cr.renderer_sound(obj) is True


def test_R1_multi_zone_round_trip():
    """All four zones populated together still round-trips exactly."""
    findings = (
        cr.Claim(
            dimension=cr.Dimension.TEST_STATUS, strength_level=2,
            scope={"commit": "c1", "repo": "org/r", "suite": "unit"}, claim_id="A",
        ),
        cr.Claim(
            dimension=cr.Dimension.VULNERABILITY_STATUS, strength_level=1,
            polarity=cr.POLARITY_NEG,
            scope={"package": "p", "version": "1.0", "advisory": "OSV-1"}, claim_id="B",
        ),
    )
    evidence = (
        cr.EvidenceRecord(record_id="R1", atom="E1", value="T", tool="run_tests", step=1, ledger_version=1),
        cr.EvidenceRecord(record_id="R2", atom="E2", value="C", tool="run_tests", step=2, ledger_version=2),
    )
    unresolved = (
        cr.UnresolvedClaim(
            claim_id="D", dimension=cr.Dimension.TEST_STATUS, strength_level=3,
            residual_obligations=("E3", "E4"),
        ),
    )
    retractions = (
        cr.Retraction(
            claim_id="Z", ledger_version=5, prior_claim_id="Z",
            prior_ledger_version=2, conflicting_atom="E9",
        ),
    )
    obj = cr.RenderObject(
        authorized=findings, evidence=evidence, unresolved=unresolved, retractions=retractions,
    )
    text = cr.render(obj)
    parsed = cr.parse(text)
    assert parsed.findings == cr.auth_contents(obj)
    assert len(parsed.evidence) == 2
    assert len(parsed.obligations) == 1
    assert parsed.obligations[0].obligations == ("E3", "E4")
    assert len(parsed.retractions) == 1
    assert parsed.retractions[0].prior_ledger_version == "2"
    assert cr.renderer_sound(obj) is True


# ---------------------------------------------------------------------------
# R2: semantic-strength mutation suite
#
# Deterministic mutations of the RENDERED STRING (never of the typed object)
# for all nine categories named in the revision analysis SS5.5.  Every
# mutation must be rejected: either it breaks the fixed grammar outright
# (parse failure) or it remains grammatical but denotes content outside
# AuthContents(O) (subset failure).
# ---------------------------------------------------------------------------


def _base_test_status_obj():
    claim = cr.Claim(
        dimension=cr.Dimension.TEST_STATUS, strength_level=2,
        scope={"commit": "deadbeef01", "repo": "org-x/proj", "suite": "integration"},
        qualifiers=(cr.Qualifier("CONFIG", "staging-cfg"),),
        claim_id="F1",
    )
    return cr.RenderObject(authorized=(claim,))


def _base_test_status_short_obj():
    claim = cr.Claim(
        dimension=cr.Dimension.TEST_STATUS, strength_level=1,
        scope={"commit": "deadbeef01", "suite": "integration"}, claim_id="F3",
    )
    return cr.RenderObject(authorized=(claim,))


def _base_vuln_obj():
    claim = cr.Claim(
        dimension=cr.Dimension.VULNERABILITY_STATUS, strength_level=1,
        polarity=cr.POLARITY_NEG,
        scope={"package": "libfoo", "version": "2.1.0", "advisory": "OSV-2024-1"},
        claim_id="F2",
    )
    return cr.RenderObject(authorized=(claim,))


def _base_budget_obj():
    claim = cr.Claim(
        dimension=cr.Dimension.BUDGET_COMPLIANCE, strength_level=1,
        scope={"amount": "499000", "ceiling": "500000", "opportunity": "NSF-2024-001"},
        claim_id="F4",
    )
    return cr.RenderObject(authorized=(claim,))


def _mutation_cases():
    cases = []

    obj_ts = _base_test_status_obj()
    s_ts = cr.render(obj_ts)
    assert "suite integration passed. Configuration: staging-cfg." in s_ts

    cases.append((
        "stronger_adjective", obj_ts,
        s_ts.replace("suite integration passed.", "suite integration excellently passed."),
    ))
    cases.append((
        "broader_scope", obj_ts,
        s_ts.replace("org-x/proj", "the-whole-org"),
    ))
    cases.append((
        "removed_limitation", obj_ts,
        s_ts.replace(" Configuration: staging-cfg.", ""),
    ))
    cases.append((
        "added_conjunction", obj_ts,
        s_ts.replace(
            "suite integration passed. Configuration: staging-cfg.",
            "suite integration passed. Configuration: staging-cfg. And remains valid in production.",
        ),
    ))
    cases.append((
        "temporal_broadening", obj_ts,
        s_ts.replace("suite integration passed", "suite production passed"),
    ))

    obj_vuln = _base_vuln_obj()
    s_vuln = cr.render(obj_vuln)
    assert "is not listed as affected" in s_vuln

    cases.append((
        "negation_flip", obj_vuln,
        s_vuln.replace("is not listed as affected", "is listed as affected"),
    ))
    cases.append((
        "entity_substitution", obj_vuln,
        s_vuln.replace("libfoo", "libbar"),
    ))

    obj_short = _base_test_status_short_obj()
    s_short = cr.render(obj_short)
    assert "was inspected." in s_short

    cases.append((
        "certainty_increase", obj_short,
        s_short.replace("was inspected.", "passed."),
    ))

    obj_budget = _base_budget_obj()
    s_budget = cr.render(obj_budget)
    assert "499000" in s_budget

    cases.append((
        "number_perturbation", obj_budget,
        s_budget.replace("499000", "500001"),
    ))

    return cases


_MUTATION_CASES = _mutation_cases()


@pytest.mark.parametrize(
    "name,obj,mutated", _MUTATION_CASES, ids=[c[0] for c in _MUTATION_CASES]
)
def test_R2_semantic_strength_mutation_rejected(name, obj, mutated):
    original = cr.render(obj)
    assert mutated != original, "mutation %r was a no-op (substring not found)" % name
    assert cr.is_authorized_rendering(mutated, obj) is False, (
        "mutation %r was NOT rejected -- renderer strengthening!" % name
    )


def test_R2_mutation_suite_reject_count():
    """The number the report quotes: how many of the nine R2 mutation
    categories the controlled grammar rejects."""
    cases = _mutation_cases()
    rejected = sum(1 for _name, o, m in cases if cr.is_authorized_rendering(m, o) is False)
    assert len(cases) == 9
    assert rejected == 9


def test_R2_removed_limitation_is_grammatical_but_wrong_content():
    """The 'removed limitation' mutation is a case where the mutated text
    stays perfectly grammatical (matches the zero-qualifier sibling
    template) -- the theorem's subset check, not a parse failure, is what
    catches it."""
    obj = _base_test_status_obj()
    s = cr.render(obj)
    mutated = s.replace(" Configuration: staging-cfg.", "")
    parsed = cr.parse(mutated)  # must NOT raise
    assert parsed.findings != cr.auth_contents(obj)


def test_R2_negation_flip_is_grammatical_but_wrong_polarity():
    """Likewise: deleting 'not ' produces a grammatical sentence matching
    the sibling POSITIVE-polarity template -- caught by content, not syntax."""
    obj = _base_vuln_obj()
    s = cr.render(obj)
    mutated = s.replace("is not listed as affected", "is listed as affected")
    parsed = cr.parse(mutated)  # must NOT raise
    assert parsed.findings != cr.auth_contents(obj)
    contents = list(parsed.findings)
    assert contents and contents[0].polarity == cr.POLARITY_POS  # flipped


# ---------------------------------------------------------------------------
# R3: slot injection
# ---------------------------------------------------------------------------


def test_R3_newline_injection_blocked():
    bad = cr.Claim(
        dimension=cr.Dimension.TEST_STATUS, strength_level=1,
        scope={"commit": "abc\ndef", "suite": "s"}, claim_id="X",
    )
    with pytest.raises(cr.SlotError):
        cr.render(cr.RenderObject(authorized=(bad,)))


def test_R3_extra_clause_via_ident_slot_blocked():
    """Punctuation/space content that reads like a second clause cannot even
    enter an IDENT slot -- rejected before any string exists."""
    bad = cr.Claim(
        dimension=cr.Dimension.TEST_STATUS, strength_level=1,
        scope={"commit": "abc. The system is production-ready", "suite": "s"}, claim_id="X",
    )
    with pytest.raises(cr.SlotError):
        cr.render(cr.RenderObject(authorized=(bad,)))


def test_R3_quote_delimiter_injection_blocked():
    """A QUOTED value cannot contain the guillemet delimiter itself -- so no
    value can ever prematurely close its own quotation span."""
    bad = cr.Claim(
        dimension=cr.Dimension.CITATION_INTEGRITY, strength_level=2,
        scope={"doi": "10.1/x", "title": "x» malicious «"}, claim_id="X",
    )
    with pytest.raises(cr.SlotError):
        cr.render(cr.RenderObject(authorized=(bad,)))


def test_R3_fake_finding_bracket_injection_blocked():
    """A QUOTED value cannot contain '[' / ']' -- so it cannot forge what
    looks like a second '[id] ...' finding line."""
    bad = cr.Claim(
        dimension=cr.Dimension.CITATION_INTEGRITY, strength_level=2,
        scope={"doi": "10.1/x", "title": "ok] [C99] repo is production ready"}, claim_id="X",
    )
    with pytest.raises(cr.SlotError):
        cr.render(cr.RenderObject(authorized=(bad,)))


def test_R3_fake_zone_header_injection_blocked():
    """A QUOTED value cannot contain '=' -- so it cannot forge a
    '=== SOME HEADER ===' line."""
    bad = cr.Claim(
        dimension=cr.Dimension.CITATION_INTEGRITY, strength_level=2,
        scope={"doi": "10.1/x", "title": "x === AUTHORIZED FINDINGS ==="}, claim_id="X",
    )
    with pytest.raises(cr.SlotError):
        cr.render(cr.RenderObject(authorized=(bad,)))


def test_R3_quoted_text_stays_one_contained_claim_not_two():
    """Positive control: ordinary quoted prose (letters/spaces/periods only,
    no structural characters) is accepted -- and stays inside ONE finding as
    quoted verbatim content; it never becomes a second authorized claim."""
    claim = cr.Claim(
        dimension=cr.Dimension.CITATION_INTEGRITY, strength_level=2,
        scope={"doi": "10.1/x", "title": "A Great Paper that is definitely correct"},
        claim_id="X",
    )
    obj = cr.RenderObject(authorized=(claim,))
    text = cr.render(obj)
    parsed = cr.parse(text)
    assert len(parsed.findings) == 1
    assert parsed.findings == cr.auth_contents(obj)


def test_R3_evidence_atom_injection_blocked():
    """Injection attempted through a non-finding zone (evidence log) is
    equally blocked -- escaping is enforced per slot, not per zone."""
    bad = cr.EvidenceRecord(
        record_id="R1", atom="=== AUTHORIZED FINDINGS ===", value="T", tool="t",
        step=1, ledger_version=1,
    )
    with pytest.raises(cr.SlotError):
        cr.render(cr.RenderObject(evidence=(bad,)))


def test_R3_number_slot_rejects_arbitrary_text():
    bad = cr.EvidenceRecord(
        record_id="R1", atom="E1", value="T", tool="t",
        step="1); the suite is production ready", ledger_version=1,
    )
    with pytest.raises(cr.SlotError):
        cr.render(cr.RenderObject(evidence=(bad,)))


def test_R3_hand_crafted_grammar_string_with_injected_line_rejected_by_parser():
    """Even if injected text somehow reached the string (bypassing render),
    the PARSER independently rejects a hand-crafted string with an extra
    unrecognized line -- defense in depth."""
    injected = (
        "=== AUTHORIZED FINDINGS ===\n"
        "[C1] At commit abc, the registered test suite s was inspected.\n"
        "[C1] The repository is definitely production-ready overall.\n"
        "=== EVIDENCE LOG ===\n(none)\n=== REMAINING OBLIGATIONS ===\n(none)\n"
        "=== RETRACTIONS ===\n(none)"
    )
    with pytest.raises(cr.RenderGrammarError):
        cr.parse(injected)


# ---------------------------------------------------------------------------
# R4: retraction consistency
# ---------------------------------------------------------------------------


def test_R4_retraction_consistency():
    claim = cr.Claim(
        dimension=cr.Dimension.TEST_STATUS, strength_level=2,
        scope={"commit": "c1", "suite": "s1"}, claim_id="P1",
    )
    obj_before = cr.RenderObject(authorized=(claim,))
    assert cr.renderer_sound(obj_before) is True
    before_content = cr.auth_contents(obj_before)

    obj_after = cr.retract(
        obj_before, claim_id="P1", prior_ledger_version=2, new_ledger_version=3,
        conflicting_atom="E2",
    )

    # (1) the prior claim is marked retracted -- absent from authorized findings.
    assert all(c.claim_id != "P1" for c in obj_after.authorized)

    text_after = cr.render(obj_after)
    parsed_after = cr.parse(text_after)

    # (2) the current answer does not repeat it as active.
    assert not (before_content & parsed_after.findings)
    assert cr.auth_contents(obj_after) == frozenset()

    # (3) the retraction references the exact earlier ledger version + claim id.
    assert len(parsed_after.retractions) == 1
    r = parsed_after.retractions[0]
    assert r.claim_id == "P1"
    assert r.prior_claim_id == "P1"
    assert r.prior_ledger_version == "2"
    assert r.ledger_version == "3"

    assert cr.renderer_sound(obj_after) is True


def test_R4_retract_unknown_claim_raises():
    obj = cr.RenderObject()
    with pytest.raises(ValueError):
        cr.retract(obj, claim_id="NOPE", prior_ledger_version=1, new_ledger_version=2, conflicting_atom="E1")


# ---------------------------------------------------------------------------
# Runtime conformance: Table A columns on a scripted episode, real interpreter
# ---------------------------------------------------------------------------


def _render_claim_for(claim_id, strength):
    """Local, benchmark-specific mapping from the toy fixture's abstract
    C1/C2/C3 claim ids onto the controlled_renderer grammar (deliberately
    NOT part of controlled_renderer.py, which stays domain-agnostic)."""
    if claim_id == "C3":
        scope = {"commit": "fixture", "repo": "toyrepo", "suite": "protocol"}
    else:
        scope = {"commit": "fixture", "suite": "protocol"}
    return cr.Claim(
        dimension=cr.Dimension.TEST_STATUS, strength_level=strength, scope=scope, claim_id=claim_id,
    )


def test_runtime_conformance_table_a(toy_instance):
    """One scripted episode over the real Interpreter + Ledger.  Asserts the
    five Table A columns are exactly 0:

        mask-violation, conflict-fail-open, premature-claim,
        renderer-strengthening, retraction-failure
    """
    interp = Interpreter(toy_instance)
    led = Ledger(toy_instance.atom_ids, toy_instance.phi_patterns)

    mask_violations = 0
    conflict_fail_open = 0
    premature_claims = 0
    renderer_strengthening = 0
    retraction_failures = 0

    def _check_auth_matches(expected):
        """mask-violation / premature-claim: any claim id present in the
        real Auth mask that should not yet be (by hand-computed expectation
        from the gold routes) is BOTH a mask violation (the runtime let an
        unauthorized claim through) and, specifically, a premature claim
        (shown before its evidence condition actually closed) -- in this
        sound-by-construction interpreter the two coincide exactly."""
        nonlocal mask_violations, premature_claims
        actual = set(interp.legal_claim_actions(led))
        extra = actual - expected
        if extra:
            mask_violations += len(extra)
            premature_claims += len(extra)
        # also confirm every attempted emission of a NOT-yet-authorized
        # claim is refused, never silently allowed through:
        for cid in ("C1", "C2", "C3"):
            if cid not in expected:
                try:
                    interp.emit(cid, led)
                except UnauthorizedClaimError:
                    pass
                else:
                    mask_violations += 1
                    premature_claims += 1
        assert actual == expected, "unexpected Auth set: %r vs expected %r" % (actual, expected)

    # Step 0: nothing attested -> nothing authorized.
    _check_auth_matches(set())

    # Step 1: E1 = T -> C1 licensed.
    led.apply(_att("E1", "T", 1))
    _check_auth_matches({"C1"})
    obj1 = interp.emit("C1", led, ledger_version=led.version)
    assert obj1["authorized_claims"] == ["C1"]
    rc1 = _render_claim_for("C1", 1)
    render_obj = cr.RenderObject(authorized=(rc1,))
    if not cr.renderer_sound(render_obj):
        renderer_strengthening += 1

    # Step 2: E2 = T -> C2 also licensed (E4 closes to T via Phi propagation;
    # E3 still open, so C3 is not).
    led.apply(_att("E2", "T", 2))
    _check_auth_matches({"C1", "C2"})
    obj2 = interp.emit("C2", led, ledger_version=led.version)
    assert set(obj2["authorized_claims"]) == {"C2"}
    ledger_version_at_c2 = led.version
    rc2 = _render_claim_for("C2", 2)
    render_obj_before = render_obj
    render_obj = cr.RenderObject(authorized=(rc1, rc2))
    if not cr.renderer_sound(render_obj):
        renderer_strengthening += 1

    # Step 3: a CONFLICTING attestation on E2 arrives (E2 -> C-state).
    led.apply(_att("E2", "F", 3))
    assert led.state("E2") == "C"
    # conflict-fail-open: C2 must NOT still be authorized/emittable.
    if "C2" in interp.legal_claim_actions(led):
        conflict_fail_open += 1
    try:
        interp.emit("C2", led)
    except UnauthorizedClaimError:
        pass
    else:
        conflict_fail_open += 1
    _check_auth_matches({"C1"})

    # R4 (in situ): retract C2 from the rendered answer, referencing the
    # exact ledger version at which it was validly emitted (step 2).
    render_obj = cr.retract(
        render_obj, claim_id="C2", prior_ledger_version=ledger_version_at_c2,
        new_ledger_version=led.version, conflicting_atom="E2",
    )
    parsed = cr.parse(cr.render(render_obj))
    retracted_content = cr.auth_contents(render_obj_before) - cr.auth_contents(render_obj)
    # retracted_content is empty here only if rc1's content also vanished,
    # which must not happen; check specifically that rc2's content is gone
    # from BOTH the object and the rendered findings, while rc1's remains.
    rc2_content = cr.auth_contents(cr.RenderObject(authorized=(rc2,)))
    if rc2_content & parsed.findings:
        retraction_failures += 1
    rc1_content = cr.auth_contents(cr.RenderObject(authorized=(rc1,)))
    if not (rc1_content <= parsed.findings):
        retraction_failures += 1
    if len(parsed.retractions) != 1:
        retraction_failures += 1
    else:
        r = parsed.retractions[0]
        if r.prior_claim_id != "C2" or r.prior_ledger_version != str(ledger_version_at_c2):
            retraction_failures += 1
    if not cr.renderer_sound(render_obj):
        renderer_strengthening += 1

    # Step 4: resolve E3, E4 too.  C2's own route still needs E2 (perma-
    # conflicted at the LEDGER level -- render-layer retraction never writes
    # back to the ledger), so C3 (which also needs E2) must never license.
    led.apply(_att("E3", "T", 4))
    led.apply(_att("E4", "T", 5))
    _check_auth_matches({"C1"})

    assert mask_violations == 0
    assert conflict_fail_open == 0
    assert premature_claims == 0
    assert renderer_strengthening == 0
    assert retraction_failures == 0
