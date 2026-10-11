"""Instance schema, validation, and gold-program construction.

Implements the SPEC.md "Instance JSON" contract and the constraint checks the
runtime enforces on benchmark instances (SPEC.md SS"Instance JSON"): order
coherence, route realizability under Phi, min-cardinality route
representatives, and tool-resolvability of every atom.

Semantics source: PRIME-master.md SS5 (task model), SS6 (licenses), SS10.4
(benchmark instance shape).

Literals
--------
A literal is an atom id, optionally negated with a leading "!": "E1" means
z(E1) must close to T, "!E1" means it must close to F (signed literals,
master SS5.4).  Gold routes in the pilot are typically positive.

Order
-----
``order`` is a list of pairs ``[weaker, stronger]``: ``["C1", "C2"]`` states
C1 <= C2 (C2 dominates C1), matching master SS5.2 where higher levels are
stronger claims.
"""

import json
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple


class SchemaError(ValueError):
    """Raised on any instance/program constraint violation (hard error)."""


MAX_ATOMS = 20  # closure enumerates 2^m complete assignments; m <= 20 (SPEC).


def parse_literal(literal: str) -> Tuple[str, bool]:
    """Return (atom_id, positive) for a signed literal string."""
    if literal.startswith("!"):
        return literal[1:], False
    return literal, True


def negate_literal(literal: str) -> str:
    """Return the opposite-polarity literal."""
    atom, pos = parse_literal(literal)
    return atom if not pos else "!" + atom


def atom_of(literal: str) -> str:
    return parse_literal(literal)[0]


@dataclass(frozen=True)
class Resolver:
    """Registered resolver of a typed evidence atom (master SS5.3).

    ``tool``          evidence action able to establish the atom;
    ``params_schema`` declared parameter names/types (documentation);
    ``params``        optional binding: the attestation fires only when the
                      executed action's params include these key/values;
    ``attest``        attestation rule mapping a tool observation to T/F
                      (see goldledger.eval_attest).
    """

    tool: str
    params_schema: Mapping[str, str] = field(default_factory=dict)
    attest: str = "span_read"
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Atom:
    id: str
    type: str
    proposition: str
    resolver: Resolver
    # Locator-uncertain split (NAACL review response): the specification
    # view says WHAT the atom must read (``what``) and lists the candidate
    # field paths (``candidates``, the true one among them) but not which;
    # the resolver keeps the true locator, so attestation is unchanged and
    # only the agent's search is affected. None everywhere else.
    hint: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    level: int


@dataclass(frozen=True)
class Program:
    """A committed claim-license program P_0 (master SS7.2).

    Immutable by construction: tuples plus a read-only licenses mapping.
    Any attempted mutation raises (dataclasses.FrozenInstanceError for
    attribute assignment, TypeError for item assignment on ``licenses``) —
    the runtime half of the full in-episode freeze (master SS7.2).
    """

    claims: Tuple[str, ...]
    order: Tuple[Tuple[str, str], ...]  # (weaker, stronger) pairs
    licenses: Mapping[str, Tuple[Tuple[str, ...], ...]]
    scope: str
    atoms: Tuple[str, ...]

    def routes(self, claim_id: str) -> Tuple[Tuple[str, ...], ...]:
        return self.licenses.get(claim_id, ())


def make_program(
    claims: Sequence[str],
    order: Sequence[Sequence[str]],
    licenses: Mapping[str, Sequence[Sequence[str]]],
    scope: str,
    atoms: Sequence[str],
) -> Program:
    """Build an immutable Program from plain containers."""
    lic = {
        str(c): tuple(tuple(str(l) for l in route) for route in routes)
        for c, routes in licenses.items()
    }
    return Program(
        claims=tuple(str(c) for c in claims),
        order=tuple((str(w), str(s)) for w, s in order),
        licenses=MappingProxyType(lic),
        scope=str(scope),
        atoms=tuple(str(a) for a in atoms),
    )


@dataclass(frozen=True)
class Instance:
    id: str
    domain: str
    seed: str
    variant: str
    user_prompt: str
    artifact: Mapping[str, Any]
    scope: str
    actions: Tuple[str, ...]
    claims: Tuple[Claim, ...]
    order: Tuple[Tuple[str, str], ...]
    atoms: Tuple[Atom, ...]
    phi: Tuple[Mapping[str, Any], ...]  # [{"forbid": {atom: "T"|"F"}, "why": str}]
    gold_licenses: Mapping[str, Tuple[Tuple[str, ...], ...]]
    decoys: Tuple[str, ...]
    notes_for_annotators: str
    path: str = ""
    # Structured-record capability (Tier-M predicate-atom domains, e.g. the
    # synthetic bench/instances/_smoke_record/ instance and, on top of this
    # infra, security/OSV, financial/SEC-XBRL, drug-safety/openFDA and
    # clinical-trials/CTgov): named, embedded, real JSON records the
    # ``inspect_record`` tool and ``record_predicate:*`` attest rules read
    # (prime_runtime/predicates.py). Empty for every non-record instance.
    records: Mapping[str, Any] = field(default_factory=dict)

    @property
    def atom_ids(self) -> Tuple[str, ...]:
        return tuple(a.id for a in self.atoms)

    @property
    def claim_ids(self) -> Tuple[str, ...]:
        return tuple(c.id for c in self.claims)

    @property
    def phi_patterns(self) -> Tuple[Mapping[str, str], ...]:
        """Forbidden determined-value patterns (master SS10.6 stage 2)."""
        return tuple(MappingProxyType(dict(p.get("forbid", {}))) for p in self.phi)

    @property
    def artifact_dir(self) -> str:
        """Absolute artifact path.

        Per SPEC.md, ``artifact.path`` (e.g. ``artifacts/<name>@<sha>``) is
        relative to the BENCH ROOT — the directory containing ``instances/``
        — not to the instance file's own directory.  Walk up from the
        instance file until a parent named ``instances`` is found; its parent
        is the bench root.  Falls back to the instance dir, then cwd.
        """
        raw = str(self.artifact.get("path", ""))
        if os.path.isabs(raw):
            return raw
        base = os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()
        probe = base
        while probe and probe != os.path.dirname(probe):
            if os.path.basename(probe) == "instances":
                base = os.path.dirname(probe)
                break
            probe = os.path.dirname(probe)
        candidate = os.path.normpath(os.path.join(base, raw))
        if not os.path.isdir(candidate):
            # legacy fallback: relative to the instance file dir
            alt = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(self.path or "")), raw)
            )
            if os.path.isdir(alt):
                return alt
        return candidate

    def claim(self, claim_id: str) -> Claim:
        for c in self.claims:
            if c.id == claim_id:
                return c
        raise KeyError(claim_id)

    def atom(self, atom_id: str) -> Atom:
        for a in self.atoms:
            if a.id == atom_id:
                return a
        raise KeyError(atom_id)

    def gold_program(self) -> Program:
        """The expert-authored gold program f_bench (master SS10.6)."""
        return make_program(
            claims=self.claim_ids,
            order=self.order,
            licenses=self.gold_licenses,
            scope=self.scope,
            atoms=self.atom_ids,
        )


def load_instance(path: str, validate: bool = True) -> Instance:
    """Load an instance JSON file (SPEC.md SS"Instance JSON")."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    try:
        atoms = tuple(
            Atom(
                id=str(a["id"]),
                type=str(a.get("type", "inspect")),
                proposition=str(a.get("proposition", "")),
                # resolver: null is an authoring-time terminal-unavailability
                # declaration (evidence-unavailable split, master SS10.5) —
                # the atom is 'impossible' for this artifact/scope version.
                resolver=(
                    None
                    if a.get("resolver") is None
                    else Resolver(
                        tool=str(a["resolver"]["tool"]),
                        params_schema=MappingProxyType(dict(a["resolver"].get("params_schema", {}))),
                        attest=str(a["resolver"].get("attest", "span_read")),
                        params=MappingProxyType(dict(a["resolver"].get("params", {}))),
                    )
                ),
                hint=(MappingProxyType(dict(a["spec_hint"]))
                      if isinstance(a.get("spec_hint"), dict) else None),
            )
            for a in raw["atoms"]
        )
        claims = tuple(
            Claim(id=str(c["id"]), text=str(c.get("text", "")), level=int(c.get("level", 1)))
            for c in raw["claims"]
        )
        gold = MappingProxyType(
            {
                str(cid): tuple(tuple(str(l) for l in route) for route in routes)
                for cid, routes in raw.get("gold_licenses", {}).items()
            }
        )
        inst = Instance(
            id=str(raw["id"]),
            domain=str(raw.get("domain", "")),
            seed=str(raw.get("seed", raw["id"])),
            variant=str(raw.get("variant", "base")),
            user_prompt=str(raw["user_prompt"]),
            artifact=MappingProxyType(dict(raw.get("artifact", {}))),
            scope=str(raw.get("scope", "")),
            actions=tuple(str(a) for a in raw.get("actions", ())),
            claims=claims,
            order=tuple((str(w), str(s)) for w, s in raw.get("order", ())),
            atoms=atoms,
            phi=tuple(MappingProxyType(dict(p)) for p in raw.get("phi", ())),
            gold_licenses=gold,
            decoys=tuple(str(d) for d in raw.get("decoys", ())),
            notes_for_annotators=str(raw.get("notes_for_annotators", "")),
            path=os.path.abspath(path),
            records=MappingProxyType(
                {str(k): v for k, v in raw.get("records", {}).items()}
            ),
        )
    except (KeyError, TypeError) as exc:
        raise SchemaError("malformed instance %s: %r" % (path, exc)) from exc
    if validate:
        validate_instance(inst)
    return inst


# ---------------------------------------------------------------------------
# Validation (SPEC.md constraints; hard errors on violation)
# ---------------------------------------------------------------------------


def _preceq_pairs(order: Sequence[Tuple[str, str]], claims: Sequence[str]) -> FrozenSet[Tuple[str, str]]:
    """Reflexive-transitive closure of the (weaker, stronger) pairs."""
    reach = {c: {c} for c in claims}  # reach[w] = {s : w <=* s}
    edges: Dict[str, set] = {c: set() for c in claims}
    for w, s in order:
        edges.setdefault(w, set()).add(s)
    changed = True
    while changed:
        changed = False
        for w in list(reach):
            for mid in list(reach[w]):
                for s in edges.get(mid, ()):
                    if s not in reach[w]:
                        reach[w].add(s)
                        changed = True
    return frozenset((w, s) for w, ss in reach.items() for s in ss)


def validate_instance(inst: Instance) -> None:
    """Run the runtime-side constraint checks (SPEC.md).

    Raises SchemaError on: duplicate ids; unknown atoms/claims; > MAX_ATOMS
    atoms; atom resolvable only via an unlisted tool; Phi unsatisfiable
    (rejected pre-commitment, master SS6.4); unrealizable gold routes;
    dominated (non-min-cardinality) route representatives; order-incoherence.
    """
    # Local import to avoid a module cycle (closure imports schema helpers).
    from . import closure

    atom_ids = list(inst.atom_ids)
    claim_ids = list(inst.claim_ids)
    if len(set(atom_ids)) != len(atom_ids):
        raise SchemaError("duplicate atom ids in %s" % inst.id)
    if len(set(claim_ids)) != len(claim_ids):
        raise SchemaError("duplicate claim ids in %s" % inst.id)
    if len(atom_ids) > MAX_ATOMS:
        raise SchemaError(
            "instance %s has %d atoms; closure enumeration requires <= %d"
            % (inst.id, len(atom_ids), MAX_ATOMS)
        )
    for w, s in inst.order:
        if w not in claim_ids or s not in claim_ids:
            raise SchemaError("order pair (%s,%s) references unknown claim" % (w, s))
    for a in inst.atoms:
        if a.resolver is not None and a.resolver.tool not in inst.actions:
            raise SchemaError(
                "atom %s resolvable only via unlisted tool %r" % (a.id, a.resolver.tool)
            )
        # Structured-record atoms (Tier-M predicate grammar,
        # prime_runtime/predicates.py): record_id must name a record actually
        # embedded in this instance, else the atom can never fire.
        if a.resolver is not None and a.resolver.tool == "inspect_record":
            rid = a.resolver.params.get("record_id")
            if not rid or rid not in inst.records:
                raise SchemaError(
                    "atom %s: inspect_record record_id %r not declared in "
                    "instance records" % (a.id, rid)
                )
    for pat in inst.phi_patterns:
        for atom, val in pat.items():
            if atom not in atom_ids or val not in ("T", "F"):
                raise SchemaError("malformed phi pattern %r" % (dict(pat),))
    # Phi itself must be satisfiable (master SS6.4: Phi-unsat is rejected
    # before commitment; constraints are never repair targets).
    if not closure.realizable_assignments(tuple(atom_ids), inst.phi_patterns):
        raise SchemaError("Phi is unsatisfiable for instance %s" % inst.id)
    validate_program(inst.gold_program(), inst)


def validate_program(program: Program, inst: Instance) -> None:
    """Validate a (gold or induced) program against an instance.

    Checks (SPEC.md): claims/atoms known; every route realizable under Phi;
    min-cardinality representatives (no route in a family is a superset of a
    sibling route — a dominated representative creates phantom obligations,
    master SS9.1); order-coherence (every route of a stronger claim
    literal-contains some route of each weaker comparable claim — the
    documented sufficient condition).
    """
    from . import closure

    atom_ids = set(inst.atom_ids)
    claim_ids = set(inst.claim_ids)
    for cid in program.claims:
        if cid not in claim_ids:
            raise SchemaError("program claim %s not in instance %s" % (cid, inst.id))
    for cid, routes in program.licenses.items():
        if cid not in claim_ids:
            raise SchemaError("license for unknown claim %s" % cid)
        seen = set()
        for route in routes:
            for lit in route:
                if atom_of(lit) not in atom_ids:
                    raise SchemaError("route literal %r references unknown atom" % lit)
            if frozenset(route) in seen:
                raise SchemaError("duplicate route %r for claim %s" % (route, cid))
            seen.add(frozenset(route))
            if not closure.route_realizable(route, inst.atom_ids, inst.phi_patterns):
                raise SchemaError(
                    "route %r for claim %s unrealizable under Phi" % (route, cid)
                )
        # Min-cardinality representatives: no dominated superset siblings.
        rsets = [frozenset(r) for r in routes]
        for i, r1 in enumerate(rsets):
            for j, r2 in enumerate(rsets):
                if i != j and r1 > r2:
                    raise SchemaError(
                        "claim %s route %r strictly contains sibling %r "
                        "(dominated representative)" % (cid, sorted(r1), sorted(r2))
                    )
    # Order coherence (sufficient syntactic condition, SPEC.md).
    pre = _preceq_pairs(program.order, program.claims)
    for w, s in pre:
        if w == s:
            continue
        w_routes = [frozenset(r) for r in program.routes(w)]
        s_routes = [frozenset(r) for r in program.routes(s)]
        if not w_routes or not s_routes:
            continue
        for sr in s_routes:
            if not any(wr <= sr for wr in w_routes):
                raise SchemaError(
                    "order-incoherent: route %r of stronger claim %s contains "
                    "no route of weaker claim %s" % (sorted(sr), s, w)
                )
