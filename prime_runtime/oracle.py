"""Executable benchmark oracle F* — independent of PRIME's route representation.

Benchmark truth must be an EXECUTABLE PREDICATE over evidence, not a check
of whether some route in PRIME's authored license family is complete.
Concretely:

    F*_x : Gamma_x x Omega_x -> {0,1}

is built directly from each benchmark instance's authored ``gold_licenses``
SEMANTICS (the disjunctive-normal-form support relation an annotator
declared: claim c is supported iff some route's literals are all confirmed),
but is implemented here as three INDEPENDENT representations that must agree
everywhere (the logical-representation invariance check, Experiment B1 in
:mod:`analysis.oracle_invariance`):

    support               -- direct DNF evaluation ("support" = call the
                              predicate).
    support_via_bdd        -- a small self-contained reduced-ordered binary
                              decision diagram (no external solver deps).
    support_via_truthtable -- exhaustive enumeration over the same domain.

``prime_routes`` (PRIME's own DNF route family) are then DERIVED from F* via
:func:`derive_prime_routes` (Quine-McCluskey-style prime-implicant
minimization) — the dependency runs oracle -> routes, never
routes -> oracle, which is the point: scoring calls ``support`` (or an
equivalent representation) directly and never asks "is some route in the
authored family complete".

Representation-invariance design note
--------------------------------------
A benchmark ``facts`` evidence state is three-valued per atom
(``'T'|'F'|'U'``, occasionally ``'C'`` for a conflicted ledger atom — see
:func:`facts_to_ledger`), so a literal ``l`` is *confirmed* iff its atom is
determined to ``l``'s required polarity; an unresolved ('U') or conflicted
('C') atom confirms neither polarity of any literal that mentions it
(matching ``closure.entails`` — an unresolved/conflicted atom entails
nothing). Call this per-literal boolean ``sat(l, facts)``.

``support`` evaluates ``OR_routes AND_{l in route} sat(l, facts)`` directly.
For the BDD and truth-table representations to be a genuinely INDEPENDENT
data structure that is *provably* equal to ``support`` on every possible
``facts`` (not merely typical ones), both are compiled over the derived
boolean indicator variables ``y_l = sat(l, facts)`` for the distinct
literals ``l`` appearing in the claim's gold routes — never over the raw
three-valued atom domain. This sidesteps the well-known ambiguity of
evaluating a reduced binary decision diagram at a partial (three-valued)
assignment (there is no single non-arbitrary rule for what a two-terminal
BDD should return when a variable it branches on is simply "unknown"): once
each literal's confirmation bit is computed by the same ``sat`` rule
``support`` uses, the *remaining* combinatorics (which is exactly the part
the invariance experiment cares about) are represented three separate ways
over an unambiguous, fully-determined boolean input. All three therefore
agree on every ``facts`` assignment by construction, and the agreement test
(:mod:`analysis.oracle_invariance`, Experiment B1) is a genuine check on the
representations' *combinatorial* logic (OR of ANDs vs. ROBDD vs. table
lookup), not an artifact of a shared partial-assignment convention.

Phi (the instance's forbidden-pattern constraints) plays no role in
``support``/``support_via_bdd``/``support_via_truthtable`` — F* is a
function of confirmed evidence only. Phi
DOES matter for :func:`derive_prime_routes`, which must minimize only over
*realizable* complete atom assignments: an unrealizable
combination can never actually occur, so a route need not defend against it,
and treating unrealizable atom combinations as don't-cares is what makes the
derived routes match hand-authored "minimal, non-dominated" gold routes.

Scope: this module reads ``instance.gold_licenses`` (the committed
ground truth) and ``instance.phi_patterns`` but performs its OWN Boolean
evaluation — it never calls into ``interpreter``/``auth``'s route-matching
machinery to decide what F* says. ``auth.py`` and ``closure.py`` ARE used
(imported, not modified) where the task is genuinely about the RUNTIME's
own authorization semantics under a program, e.g. :func:`auth_fp`/
:func:`auth_fn`, which compare the runtime's ``auth.auth_ids`` against this
module's independent oracle.
"""

import itertools
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple

from . import auth as _auth
from . import closure as _closure
from .ledger import Ledger
from .schema import Instance, Program, MAX_ATOMS, atom_of, parse_literal

__all__ = [
    "support",
    "support_via_bdd",
    "support_via_truthtable",
    "representations_agree",
    "derive_prime_routes",
    "gold_route_mismatch",
    "oracle_authorization",
    "ofar",
    "facts_to_ledger",
    "enumerate_complete_states",
    "auth_fp",
    "auth_fn",
]

FactState = str  # one of 'T', 'F', 'U', 'C'


# ---------------------------------------------------------------------------
# Literal satisfaction over a (possibly partial, three/four-valued) facts map
# ---------------------------------------------------------------------------


def _sat_literal(literal: str, facts: Mapping[str, FactState]) -> bool:
    """Is ``literal`` CONFIRMED by ``facts``?

    A positive literal ``a`` is confirmed iff ``facts[a] == 'T'``; a negated
    literal ``!a`` iff ``facts[a] == 'F'``. Unresolved ('U'), conflicted
    ('C'), or simply absent atoms confirm neither polarity (fail-closed,
    matching ``closure.entails``)."""
    atom, positive = parse_literal(literal)
    value = facts.get(atom, "U")
    if positive:
        return value == "T"
    return value == "F"


def _routes_of(instance: Instance, claim_id: str) -> Tuple[Tuple[str, ...], ...]:
    return instance.gold_licenses.get(claim_id, ())


# ---------------------------------------------------------------------------
# Representation 1: direct DNF evaluation
# ---------------------------------------------------------------------------


def support(instance: Instance, claim_id: str, facts: Mapping[str, FactState]) -> bool:
    """F*(claim_id, facts): direct evaluation of the executable support
    predicate.

    ``OR`` over the instance's authored gold routes of ``AND`` over each
    route's literals of literal confirmation. This is the benchmark's
    ground truth; it is evaluated directly on ``facts`` and never consults
    PRIME's runtime route/closure machinery (auth.py / closure.py)."""
    routes = _routes_of(instance, claim_id)
    return any(all(_sat_literal(lit, facts) for lit in route) for route in routes)


# ---------------------------------------------------------------------------
# Representation 2: a small self-contained reduced-ordered BDD (no deps)
# ---------------------------------------------------------------------------


class BDD:
    """Minimal self-contained reduced-ordered binary decision diagram.

    Node ids 0 and 1 are the FALSE/TRUE terminals. Internal nodes are
    triples ``(var_index, low, high)`` stored in a unique table so that
    structurally identical subgraphs are shared (the "reduced" of ROBDD);
    ``var_index`` indexes into the diagram's fixed variable order, which is
    what makes the diagram "ordered". No external solver / package is used.
    """

    def __init__(self, order: Sequence[str]) -> None:
        self.order: Tuple[str, ...] = tuple(order)
        self._index: Dict[str, int] = {v: i for i, v in enumerate(self.order)}
        self._unique: Dict[Tuple[int, int, int], int] = {}
        self._nodes: List[Any] = [None, None]  # slots 0/1 are terminals

    def const(self, value: bool) -> int:
        return 1 if value else 0

    def var(self, name: str) -> int:
        """The elementary BDD for a single positive literal indicator."""
        vi = self._index[name]
        return self._make(vi, 0, 1)

    def _make(self, var_index: int, low: int, high: int) -> int:
        if low == high:  # reduction rule: a node testing nothing is its child
            return low
        key = (var_index, low, high)
        node_id = self._unique.get(key)
        if node_id is not None:  # reduction rule: share identical nodes
            return node_id
        node_id = len(self._nodes)
        self._nodes.append(key)
        self._unique[key] = node_id
        return node_id

    def _level(self, node: int) -> int:
        return self._nodes[node][0] if node > 1 else len(self.order)

    def apply_and(self, a: int, b: int) -> int:
        return self._apply(a, b, "and", {})

    def apply_or(self, a: int, b: int) -> int:
        return self._apply(a, b, "or", {})

    def _apply(self, a: int, b: int, op: str, cache: Dict[Tuple[int, int], int]) -> int:
        if a in (0, 1) and b in (0, 1):
            return 1 if ((a and b) if op == "and" else (a or b)) else 0
        key = (a, b)
        if key in cache:
            return cache[key]
        va, vb = self._level(a), self._level(b)
        v = min(va, vb)
        a_lo, a_hi = (self._nodes[a][1], self._nodes[a][2]) if va == v else (a, a)
        b_lo, b_hi = (self._nodes[b][1], self._nodes[b][2]) if vb == v else (b, b)
        low = self._apply(a_lo, b_lo, op, cache)
        high = self._apply(a_hi, b_hi, op, cache)
        result = self._make(v, low, high)
        cache[key] = result
        return result

    def evaluate(self, node: int, assignment: Mapping[str, bool]) -> bool:
        """Walk a single root-to-terminal path under a COMPLETE assignment
        over ``self.order`` (every branch taken is fully determined — this
        diagram is only ever evaluated over the derived-literal domain,
        never a partial one; see the module docstring)."""
        while node > 1:
            var_index, low, high = self._nodes[node]
            name = self.order[var_index]
            node = high if assignment[name] else low
        return bool(node)


def _literal_order(routes: Sequence[Sequence[str]]) -> Tuple[str, ...]:
    lits = set()
    for route in routes:
        lits.update(route)
    return tuple(sorted(lits))


def _compile_bdd(routes: Sequence[Sequence[str]]) -> Tuple[BDD, int, Tuple[str, ...]]:
    order = _literal_order(routes)
    bdd = BDD(order)
    acc = bdd.const(False)
    for route in routes:
        term = bdd.const(True)
        for lit in route:
            term = bdd.apply_and(term, bdd.var(lit))
        acc = bdd.apply_or(acc, term)
    return bdd, acc, order


_BDD_CACHE: Dict[Tuple[str, str, Tuple[Tuple[str, ...], ...]], Tuple[BDD, int, Tuple[str, ...]]] = {}


def _get_bdd(instance: Instance, claim_id: str) -> Tuple[BDD, int, Tuple[str, ...]]:
    routes = _routes_of(instance, claim_id)
    key = (instance.id, claim_id, routes)
    cached = _BDD_CACHE.get(key)
    if cached is None:
        cached = _compile_bdd(routes)
        _BDD_CACHE[key] = cached
    return cached


def support_via_bdd(instance: Instance, claim_id: str, facts: Mapping[str, FactState]) -> bool:
    """F*(claim_id, facts) evaluated via a reduced-ordered BDD over the
    derived per-literal confirmation variables (the second evaluator
    representation)."""
    bdd, root, order = _get_bdd(instance, claim_id)
    assignment = {lit: _sat_literal(lit, facts) for lit in order}
    return bdd.evaluate(root, assignment)


# ---------------------------------------------------------------------------
# Representation 3: exhaustive truth table
# ---------------------------------------------------------------------------


_TABLE_CACHE: Dict[
    Tuple[str, str, Tuple[Tuple[str, ...], ...]], Tuple[Dict[Tuple[bool, ...], bool], Tuple[str, ...]]
] = {}


def _compile_table(
    routes: Sequence[Sequence[str]],
) -> Tuple[Dict[Tuple[bool, ...], bool], Tuple[str, ...]]:
    order = _literal_order(routes)
    table: Dict[Tuple[bool, ...], bool] = {}
    for bits in itertools.product((False, True), repeat=len(order)):
        y = dict(zip(order, bits))
        table[bits] = any(all(y[lit] for lit in route) for route in routes)
    return table, order


def _get_table(
    instance: Instance, claim_id: str
) -> Tuple[Dict[Tuple[bool, ...], bool], Tuple[str, ...]]:
    routes = _routes_of(instance, claim_id)
    key = (instance.id, claim_id, routes)
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        cached = _compile_table(routes)
        _TABLE_CACHE[key] = cached
    return cached


def support_via_truthtable(
    instance: Instance, claim_id: str, facts: Mapping[str, FactState]
) -> bool:
    """F*(claim_id, facts) via exhaustive lookup: the full ``2^|literals|``
    truth table of the claim's gold-route DNF, indexed by the same derived
    per-literal confirmation bits as :func:`support_via_bdd` (the third
    representation in Experiment B1).
    ``|literals| <= len(instance.atoms) <= MAX_ATOMS`` (20) for any
    well-formed benchmark instance."""
    table, order = _get_table(instance, claim_id)
    bits = tuple(_sat_literal(lit, facts) for lit in order)
    return table[bits]


def representations_agree(
    instance: Instance, claim_id: str, facts: Mapping[str, FactState]
) -> Tuple[bool, Dict[str, bool]]:
    """Convenience: evaluate all three representations and report whether
    they agree, plus each individual verdict (used by the invariance audit
    and by test_oracle.py)."""
    values = {
        "support": support(instance, claim_id, facts),
        "bdd": support_via_bdd(instance, claim_id, facts),
        "truthtable": support_via_truthtable(instance, claim_id, facts),
    }
    return (len(set(values.values())) == 1), values


# ---------------------------------------------------------------------------
# Prime-route derivation: routes are DERIVED FROM F*, never the reverse
# ---------------------------------------------------------------------------


def _combine(a: str, b: str) -> "str | None":
    """Quine-McCluskey merge: two equal-length cubes differing in exactly
    one non-dash position merge into a cube with a dash there; else None."""
    if len(a) != len(b):
        return None
    diff_pos = -1
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            if ca == "-" or cb == "-":
                return None  # dash positions must align to combine
            if diff_pos != -1:
                return None  # more than one differing position
            diff_pos = i
    if diff_pos == -1:
        return None  # identical cubes are not a merge instance
    return a[:diff_pos] + "-" + a[diff_pos + 1 :]


def _covers(cube: str, bits: str) -> bool:
    return all(c == "-" or c == b for c, b in zip(cube, bits))


def _quine_mccluskey(onset: Sequence[str], dontcare: Sequence[str]) -> set:
    """Classic prime-implicant extraction with don't-cares. ``onset`` and
    ``dontcare`` are equal-length '0'/'1' strings. Returns the set of prime
    implicant cubes ('0'/'1'/'-') that cover at least one ONSET point (pure
    don't-care-only primes are dropped, standard don't-care minimization)."""
    onset_set = set(onset)
    current = set(onset) | set(dontcare)
    if not current:
        return set()
    primes: set = set()
    while True:
        groups: Dict[int, set] = {}
        for term in current:
            groups.setdefault(term.count("1"), set()).add(term)
        used: set = set()
        next_terms: set = set()
        for k in sorted(groups):
            if k + 1 not in groups:
                continue
            for a in groups[k]:
                for b in groups[k + 1]:
                    merged = _combine(a, b)
                    if merged is not None:
                        next_terms.add(merged)
                        used.add(a)
                        used.add(b)
        primes |= current - used
        if not next_terms:
            break
        current = next_terms
    return {p for p in primes if any(_covers(p, o) for o in onset_set)}


def derive_prime_routes(instance: Instance, claim_id: str) -> List[FrozenSet[str]]:
    """Derive the prime implicants of F*(claim_id, .) directly from the
    oracle, for PRIME's gold-program condition, so that the condition
    does not rely on a manually authored and possibly incomplete route
    family.

    Enumerates all ``2^len(atoms)`` complete atom assignments; assignments
    forbidden by Phi (``instance.phi_patterns``) are treated as
    don't-cares in the Quine-McCluskey minimization (they can never occur,
    so a route need not defend against them — this is what lets derived
    routes match hand-authored minimal routes under a Phi constraint).
    Bounded by ``schema.MAX_ATOMS`` (20), same bound ``closure.py`` uses for
    exhaustive enumeration.

    Returns a list of ``frozenset`` literal cubes. Compare against
    ``instance.gold_licenses[claim_id]`` with :func:`gold_route_mismatch` —
    a mismatch is a benchmark-authoring finding to REPORT, never to silently
    reconcile by preferring one side.
    """
    atoms = instance.atom_ids
    m = len(atoms)
    if m == 0:
        return []
    if m > MAX_ATOMS:  # pragma: no cover - schema.validate_instance already
        raise ValueError(  # rejects this; defensive symmetry with closure.py
            "instance %s has %d atoms; exhaustive derivation requires <= %d"
            % (instance.id, m, MAX_ATOMS)
        )
    realizable = {
        tuple(omega[a] for a in atoms)
        for omega in _closure.realizable_assignments(atoms, instance.phi_patterns)
    }
    onset: List[str] = []
    dontcare: List[str] = []
    for bits in itertools.product((False, True), repeat=m):
        cube = "".join("1" if b else "0" for b in bits)
        if bits not in realizable:
            dontcare.append(cube)
            continue
        facts = {a: ("T" if v else "F") for a, v in zip(atoms, bits)}
        if support(instance, claim_id, facts):
            onset.append(cube)
    if not onset:
        return []
    primes = _quine_mccluskey(onset, dontcare)
    routes: List[FrozenSet[str]] = []
    for cube in primes:
        lits = set()
        for i, ch in enumerate(cube):
            if ch == "1":
                lits.add(atoms[i])
            elif ch == "0":
                lits.add("!" + atoms[i])
        routes.append(frozenset(lits))
    return routes


def gold_route_mismatch(instance: Instance, claim_id: str) -> Dict[str, Any]:
    """Compare :func:`derive_prime_routes` against the authored
    ``gold_licenses`` family for one claim. Never "fixes" a mismatch —
    reports it as a benchmark-authoring finding."""
    derived = {frozenset(r) for r in derive_prime_routes(instance, claim_id)}
    gold = {frozenset(r) for r in _routes_of(instance, claim_id)}
    return {
        "instance": instance.id,
        "claim_id": claim_id,
        "match": derived == gold,
        "derived": sorted(sorted(r) for r in derived),
        "gold": sorted(sorted(r) for r in gold),
        "derived_only": sorted(sorted(r) for r in (derived - gold)),
        "gold_only": sorted(sorted(r) for r in (gold - derived)),
    }


# ---------------------------------------------------------------------------
# Authorization-aware oracle support (downward closure over the claim order)
# ---------------------------------------------------------------------------


def _preceq(weaker: str, stronger: str, order: Sequence[Tuple[str, str]]) -> bool:
    """weaker <=* stronger under the reflexive-transitive closure of
    ``instance.order`` (mirrors ``auth.preceq``, computed independently
    here directly from the instance so this module never has to construct
    a ``Program`` to ask a question about the ORACLE's own claim lattice)."""
    if weaker == stronger:
        return True
    edges: Dict[str, set] = {}
    for w, s in order:
        edges.setdefault(w, set()).add(s)
    seen: set = set()
    stack = [weaker]
    while stack:
        node = stack.pop()
        for nxt in edges.get(node, ()):
            if nxt == stronger:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def oracle_authorization(
    instance: Instance, claim_id: str, facts: Mapping[str, FactState]
) -> bool:
    """Authorization-aware oracle support: claim ``claim_id`` is
    oracle-authorized at ``facts`` iff F*-support holds for it OR for some
    claim that dominates it under the instance's declared order
    (``F*(c, z) = 1`` used the same authorization-aware
    way ``goldledger.complete_auth`` uses the gold PROGRAM's Auth: a
    dominating supported claim entails every weaker comparable one)."""
    for cid in instance.claim_ids:
        if _preceq(claim_id, cid, instance.order) and support(instance, cid, facts):
            return True
    return False


def ofar(
    emitted: Sequence[Tuple[Instance, str]],
    facts_at_emission: Sequence[Mapping[str, FactState]],
) -> Dict[str, Any]:
    """Oracle False-Authorization Rate:

        OFAR = #{emitted claims c : F*(c, omega_t) = 0} / #{emitted claims}

    ``emitted``/``facts_at_emission`` are parallel sequences: one
    ``(instance, claim_id)`` pair and the evidence state at that claim's
    emission ledger version, per emitted claim (across one trace or an
    entire corpus — this helper is corpus-agnostic). Uses
    :func:`oracle_authorization` (authorization-aware, matching how
    PRIME's own runtime treats a weaker claim as justified once a
    dominating claim's evidence is complete) rather than raw ``support``.
    """
    n = len(emitted)
    if n != len(facts_at_emission):
        raise ValueError("emitted and facts_at_emission must be the same length")
    if n == 0:
        return {"n_emitted": 0, "n_unsupported": 0, "rate": 0.0, "unsupported": []}
    unsupported = []
    for (inst, cid), facts in zip(emitted, facts_at_emission):
        if not oracle_authorization(inst, cid, facts):
            unsupported.append({"instance": inst.id, "claim_id": cid})
    return {
        "n_emitted": n,
        "n_unsupported": len(unsupported),
        "rate": len(unsupported) / n,
        "unsupported": unsupported,
    }


# ---------------------------------------------------------------------------
# Program-oracle disagreement: AuthFP / AuthFN
# ---------------------------------------------------------------------------


def facts_to_ledger(instance: Instance, facts: Mapping[str, FactState]) -> Ledger:
    """Build a runtime :class:`~prime_runtime.ledger.Ledger` whose closure
    reproduces ``facts``: 'T'/'F' atoms get one synthetic attestation, 'C'
    atoms get two conflicting ones (so ``ledger.state(atom) == 'C'``), 'U'
    atoms are left untouched. Lets :func:`auth_fp`/:func:`auth_fn` query the
    runtime's OWN ``auth.auth_ids`` (through ``closure.cl``/Phi) at an
    enumerated or sampled evidence state, independently of this module."""
    led = Ledger(instance.atom_ids, instance.phi_patterns)
    step = 0
    for atom in instance.atom_ids:
        value = facts.get(atom, "U")
        if value == "U":
            continue
        if value in ("T", "F"):
            step += 1
            led.apply(
                {"atom": atom, "value": value, "tool": "oracle_sample", "params": {}, "step": step}
            )
        elif value == "C":
            step += 1
            led.apply({"atom": atom, "value": "T", "tool": "oracle_sample", "params": {}, "step": step})
            step += 1
            led.apply({"atom": atom, "value": "F", "tool": "oracle_sample", "params": {}, "step": step})
        else:
            raise ValueError("facts[%r] must be one of T/F/U/C, got %r" % (atom, value))
    return led


def enumerate_complete_states(instance: Instance) -> Iterable[Dict[str, FactState]]:
    """All Phi-realizable complete evidence states (Q; exhaustive
    enumeration is feasible because benchmark instances have at most 20
    Boolean evidence atoms). Yields ``dict[atom_id, 'T'|'F']``."""
    for omega in _closure.realizable_assignments(instance.atom_ids, instance.phi_patterns):
        yield {a: ("T" if v else "F") for a, v in omega.items()}


def auth_fp(
    induced_program: Program,
    instance: Instance,
    sample_states: Sequence[Mapping[str, FactState]],
) -> Dict[str, Any]:
    """AuthFP(P_hat, F*):

        Pr_z~Q[ exists c : c in Auth_Phat(z) and F*(c, z) = 0 ]

    i.e. the induced/committed program authorizes something the independent
    oracle does not support. ``sample_states`` is the enumerated/sampled
    state set Q (see :func:`enumerate_complete_states` for the exhaustive
    case). Returns the state-level rate plus the flagged states/claims for
    diagnosis (which claims and under what evidence the program overclaims).
    """
    n = len(sample_states)
    flagged: List[Dict[str, Any]] = []
    for facts in sample_states:
        led = facts_to_ledger(instance, facts)
        authed = _auth.auth_ids(induced_program, led)
        bad = [c for c in authed if not oracle_authorization(instance, c, facts)]
        if bad:
            flagged.append({"facts": dict(facts), "claims": bad})
    return {
        "n_states": n,
        "n_flagged": len(flagged),
        "rate": (len(flagged) / n) if n else 0.0,
        "flagged_states": flagged,
    }


def auth_fn(
    induced_program: Program,
    instance: Instance,
    sample_states: Sequence[Mapping[str, FactState]],
) -> Dict[str, Any]:
    """AuthFN(P_hat, F*):

        Pr_z~Q[ exists c : F*(c, z) = 1 and c not in Auth_Phat(z) ]

    i.e. the induced/committed program blocks a claim the independent
    oracle supports (overblocking). See :func:`auth_fp` for the dual and
    the shared ``sample_states``/return-shape convention."""
    n = len(sample_states)
    flagged: List[Dict[str, Any]] = []
    for facts in sample_states:
        led = facts_to_ledger(instance, facts)
        authed = set(_auth.auth_ids(induced_program, led))
        bad = [
            c
            for c in instance.claim_ids
            if oracle_authorization(instance, c, facts) and c not in authed
        ]
        if bad:
            flagged.append({"facts": dict(facts), "claims": bad})
    return {
        "n_states": n,
        "n_flagged": len(flagged),
        "rate": (len(flagged) / n) if n else 0.0,
        "flagged_states": flagged,
    }
