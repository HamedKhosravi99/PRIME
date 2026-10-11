"""Frozen-program license interpreter and typed claim objects.

Implements PRIME-master.md SS7.2 (full in-episode freeze: at commitment the
complete program P_0 is frozen; during the episode the policy's only state
change is the ledger via attestations — no license/claim/order/scope edits of
any kind), SS7.4 (claim-action masking via Auth), and SS7.8 (typed claim
object; refusal of unauthorized emission).

Freeze enforcement is structural: the committed :class:`~prime_runtime.schema.Program`
is a frozen dataclass whose containers are immutable, and the interpreter
itself rejects attribute rebinding after construction.  Attempted program
mutation raises (FrozenProgramError / dataclasses.FrozenInstanceError /
TypeError) — the policy can NEVER modify the program; only the ledger
changes, and only via ``Ledger.apply``.
"""

from typing import Any, Dict, List, Mapping, Optional

from . import absorption as _absorption
from . import auth as _auth
from . import closure as _closure
from . import status as _status
from .closure import cl, is_inconsistent, satisfies
from .schema import Instance, Program, atom_of, make_program

# Revision-freedom ablation modes (master SS12.3 mechanism curve 4).
# "fixed" is the DEPLOYED system (full in-episode freeze, SS7.2); every other
# mode is an ABLATION-ONLY relaxation, explicitly flagged at construction.
REVISION_MODES = (
    "fixed",            # deployed default: no revision language at all
    "strengthen_only",  # literal addition / route deletion only
    "unfrozen_lattice", # + new claims with declared dominance
    "arbitrary",        # any well-formed program edit
    "posthoc",          # no upfront commitment; program supplied at emission
)

# Revision kinds accepted per mode (acceptance rule = kind gate + well-formedness).
_MODE_KINDS: Dict[str, frozenset] = {
    "fixed": frozenset(),
    "strengthen_only": frozenset({"add_literal", "delete_route"}),
    "unfrozen_lattice": frozenset({"add_literal", "delete_route", "add_claim"}),
    "arbitrary": frozenset(
        {
            "add_literal",
            "remove_literal",
            "add_route",
            "delete_route",
            "add_claim",
            "delete_claim",
            "replace_program",
        }
    ),
    "posthoc": frozenset({"replace_program"}),
}


def allowed_revision_kinds(revision_mode: str) -> List[str]:
    """Revision kinds accepted in ``revision_mode`` (empty for 'fixed')."""
    if revision_mode not in REVISION_MODES:
        raise ValueError("unknown revision_mode %r" % revision_mode)
    return sorted(_MODE_KINDS[revision_mode])


class FrozenProgramError(AttributeError):
    """Attempted in-episode mutation of the committed program (master SS7.2)."""


class IllFormedRevisionError(ValueError):
    """A proposed revision is not a well-formed program edit (rejected)."""


class UnauthorizedClaimError(RuntimeError):
    """Emission refused: claim not in Auth_P(z_t) (master SS7.4, Prop. 2)."""


class StaleLedgerError(RuntimeError):
    """Emission requested against a ledger version that is not current."""


class Interpreter:
    """Deterministic license interpreter over a committed frozen program.

    Parameters
    ----------
    instance : the benchmark instance (supplies scope, atoms, resolvers).
    program  : the committed program; defaults to the instance gold program
               (gold-license condition, SPEC.md arms).
    revision_mode : "fixed" (default, deployed full-freeze behavior) or one of
               the ABLATION-ONLY relaxations in :data:`REVISION_MODES`
               (master SS12.3 mechanism curve 4).  In non-fixed modes,
               :meth:`propose_revision` accepts revision requests per the
               mode's acceptance rule; EVERY accepted revision is logged with
               an ``auth_expanding`` flag computed by the exact robust route
               absorption check against the ORIGINAL committed program B_0
               (:func:`prime_runtime.absorption.check_patch`) — an accepted
               revision that fails it is authorization-expanding: a DRIFT
               EVENT (the empirical face of Props. 3-4).

               In ``posthoc`` mode there is NO upfront commitment: the working
               program starts with empty licenses (nothing authorized) and the
               policy supplies the program that licenses its claim at emission
               time via ``propose_revision("replace_program", ...)``,
               validated for well-formedness only (Prop. 4's selector).  The
               drift reference B_0 remains the program that the deployed
               full-freeze system would have committed (``program`` /
               gold), so drift is comparable across arms.
    """

    def __init__(
        self,
        instance: Instance,
        program: Optional[Program] = None,
        revision_mode: str = "fixed",
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        if revision_mode not in REVISION_MODES:
            raise ValueError("unknown revision_mode %r" % revision_mode)
        self.instance = instance
        self.revision_mode = revision_mode
        committed = program if program is not None else instance.gold_program()
        # B_0: the ORIGINAL committed program, the fixed reference for every
        # auth_expanding check.  NEVER rebound, in any mode.
        self.program0 = committed
        if revision_mode == "posthoc":
            # No upfront commitment: empty license map — nothing authorized
            # until the policy supplies a program at emission time.
            self.program = make_program(
                claims=instance.claim_ids,
                order=instance.order,
                licenses={},
                scope=instance.scope,
                atoms=instance.atom_ids,
            )
        else:
            self.program = committed
        # Append-only revision log (list mutation, not attribute rebinding):
        # one entry per proposal, accepted or not; accepted entries carry the
        # exact auth_expanding flag.
        self.revision_log: List[Dict[str, Any]] = []
        self.resolvers = _status.resolver_table(instance)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise FrozenProgramError(
                "in-episode freeze: interpreter attribute %r is immutable "
                "(master SS7.2 — only the ledger changes, via attestations)" % name
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise FrozenProgramError("in-episode freeze: cannot delete %r" % name)

    # -- masking -------------------------------------------------------------

    def legal_claim_actions(self, ledger) -> List[str]:
        """Claim actions available to the decoder: exactly Auth_P(z_t)
        (master SS7.4).  Empty on an inconsistent ledger."""
        return _auth.auth_ids(self.program, ledger)

    def frontier(self, ledger) -> Dict[str, List[List[str]]]:
        """Obligation frontier per claim (master SS6.4), for prompt
        injection by the prime arm."""
        return {
            cid: _status.frontier(cid, self.program, ledger, self.resolvers)
            for cid in self.program.claims
        }

    # -- revision-freedom ablation (master SS12.3 curve 4; ABLATION ONLY) -----

    @property
    def revisions_proposed(self) -> int:
        return len(self.revision_log)

    @property
    def revisions_accepted(self) -> int:
        return sum(1 for e in self.revision_log if e["accepted"])

    @property
    def drift_events(self) -> int:
        """Accepted authorization-expanding revisions (standard drift)."""
        return sum(
            1 for e in self.revision_log if e["accepted"] and e["auth_expanding"]
        )

    def propose_revision(
        self, kind: str, payload: Mapping[str, Any], step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Ablation-only revision request (NOT available in the deployed
        ``fixed`` mode, which raises FrozenProgramError — master SS7.2).

        Acceptance rule = (kind permitted by the mode) AND (the edit is a
        well-formed program edit: known claims/atoms per kind, routes over the
        instance vocabulary, realizable under Phi — unrealizable routes are
        ill-formed, master SS9.3).  No safety check gates acceptance: safety
        is MEASURED, not enforced — every accepted revision is logged with

            auth_expanding = not check_patch(B_0, delta(P_old, P_new), Phi)

        the exact robust route absorption decision (master SS9.3) applied to
        everything THIS revision introduces — added claims, added dominance
        pairs, added/modified routes — with per-route containment checked
        against the ORIGINAL committed program's absorption families B_0,
        never against the current (possibly already weakened) program, so
        strengthen-then-weaken chains cannot leak; deletions never expand.
        An accepted revision with auth_expanding=True is a DRIFT EVENT
        (mechanism curve 4).
        """
        if self.revision_mode == "fixed":
            raise FrozenProgramError(
                "in-episode freeze: program revision is not a policy action "
                "(master SS7.2); revision_mode='fixed'"
            )
        entry: Dict[str, Any] = {
            "kind": str(kind),
            "payload": _jsonable(payload),
            "step": step,
            "accepted": False,
            "reason": "",
            "auth_expanding": None,
        }
        if kind not in _MODE_KINDS[self.revision_mode]:
            entry["reason"] = "kind %r not permitted in mode %r (allowed: %s)" % (
                kind,
                self.revision_mode,
                sorted(_MODE_KINDS[self.revision_mode]),
            )
            self.revision_log.append(entry)
            return entry
        try:
            candidate = _apply_edit(self.program, kind, payload, self.instance)
        except IllFormedRevisionError as exc:
            entry["reason"] = "ill-formed: %s" % exc
            self.revision_log.append(entry)
            return entry
        # Exact drift flag: per-route containment of THIS revision's
        # additions against the ORIGINAL committed program B_0 over all
        # four-valued ledgers (absorption.check_patch, master SS9.3).
        delta = _revision_delta(self.program, candidate, self.program0)
        expanding = not _absorption.check_patch(
            self.program0, delta, self.instance.phi_patterns
        )
        entry["accepted"] = True
        entry["reason"] = "accepted (%s)" % self.revision_mode
        entry["auth_expanding"] = expanding
        # Ablation-only bypass of the structural freeze: rebind the WORKING
        # program.  self.program0 (B_0) is never rebound.
        object.__setattr__(self, "program", candidate)
        self.revision_log.append(entry)
        return entry

    # -- emission ------------------------------------------------------------

    def emit(
        self,
        claim_id: str,
        ledger,
        ledger_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Emit a typed claim object (master SS7.8); raises if unauthorized.

        ``ledger_version`` is a stale-emission guard: when supplied it must
        equal ``ledger.version`` (claims carry the ledger version under which
        they were licensed, master SS7.8).
        """
        if claim_id not in self.program.claims:
            raise UnauthorizedClaimError("unknown claim %r" % claim_id)
        if ledger_version is not None and ledger_version != ledger.version:
            raise StaleLedgerError(
                "emission requested at ledger version %s but current is %s"
                % (ledger_version, ledger.version)
            )
        authorized = self.legal_claim_actions(ledger)
        if claim_id not in authorized:
            raise UnauthorizedClaimError(
                "claim %s is not authorized at ledger version %d (Auth=%r)"
                % (claim_id, ledger.version, authorized)
            )
        return self._claim_object([claim_id], ledger, authorized)

    def emit_max_auth(self, ledger) -> Dict[str, Any]:
        """Emit the maximal authorized antichain MaxAuth_P(z) (master SS7.5).
        Raises when nothing is authorized (abstention is not an emission)."""
        maximal = _auth.max_auth(self.program, ledger)
        if not maximal:
            raise UnauthorizedClaimError("nothing is authorized; abstain")
        return self._claim_object(maximal, ledger, self.legal_claim_actions(ledger))

    # -- typed object construction --------------------------------------------

    def _claim_object(
        self, emitted: List[str], ledger, authorized: List[str]
    ) -> Dict[str, Any]:
        c = cl(ledger)
        assert not is_inconsistent(c)  # authorized non-empty implies consistent
        # Evidence: literals of complete routes of licensing claims that
        # dominate an emitted claim (downward closure, master SS9.2).
        evidence = set()
        for e in emitted:
            for d in self.program.claims:
                if _auth.preceq(e, d, self.program):
                    for route in self.program.routes(d):
                        if satisfies(c, route):
                            evidence.update(route)
        unresolved = []
        for cid in self.program.claims:
            if cid in authorized:
                continue
            fr = _status.frontier(cid, self.program, ledger, self.resolvers)
            fr_sorted = sorted(fr, key=lambda r: (len(r), r))
            residual = fr_sorted[0] if fr_sorted else ["all_routes_dead"]
            unresolved.append({"id": cid, "residual_obligations": residual})
        qualifications = [
            "Atom %s has conflicting attestations." % a
            for a in ledger.atoms
            if ledger.state(a) == "C"
        ]
        return {
            "authorized_claims": sorted(emitted),
            "evidence": sorted(evidence),
            "unresolved_claims": unresolved,
            "scope": self.program.scope,
            "qualifications": qualifications,
            "ledger_version": ledger.version,
        }


# ---------------------------------------------------------------------------
# Revision-edit helpers (ablation only; used by Interpreter.propose_revision)
# ---------------------------------------------------------------------------


def _revision_delta(old: Program, new: Program, p0: Program) -> Program:
    """The program fragment introduced by one revision step, for the exact
    per-revision drift check.

    Contains: claims of P_0 plus any claims new to this revision (a claim
    absent from P_0 makes check_patch fail — authorization over new content);
    order pairs not present in the pre-revision program (checked against
    P_0's transitive order by check_patch); and, per claim, exactly the
    routes not present pre-revision (each checked for containment of some
    ORIGINAL B_0 route).  Deletions contribute nothing — contractions never
    expand authorization (Prop. 3)."""
    old_claims = set(old.claims)
    p0_claims = set(p0.claims)
    added_claims = [c for c in new.claims if c not in old_claims and c not in p0_claims]
    old_pairs = {(w, s) for w, s in old.order}
    added_order = [[w, s] for w, s in new.order if (w, s) not in old_pairs]
    added_lic: Dict[str, List[List[str]]] = {}
    for cid, routes in new.licenses.items():
        old_routes = {frozenset(r) for r in old.licenses.get(cid, ())}
        novel = [list(r) for r in routes if frozenset(r) not in old_routes]
        if novel:
            added_lic[cid] = novel
    return make_program(
        claims=list(p0.claims) + added_claims,
        order=added_order,
        licenses=added_lic,
        scope=new.scope,
        atoms=new.atoms,
    )


def _jsonable(payload: Mapping[str, Any]) -> Dict[str, Any]:
    import json as _json

    try:
        return _json.loads(_json.dumps(dict(payload), default=str))
    except (TypeError, ValueError):
        return {"_repr": repr(payload)}


def _check_route(route: Any, instance: Instance) -> List[str]:
    """Well-formedness of one route: a list of literal strings over the
    instance atom vocabulary, realizable under Phi (master SS9.3:
    unrealizable revised routes are ill-formed)."""
    if not isinstance(route, (list, tuple)):
        raise IllFormedRevisionError("route must be a list of literals, got %r" % (route,))
    atom_ids = set(instance.atom_ids)
    out = []
    for lit in route:
        if not isinstance(lit, str) or atom_of(lit) not in atom_ids:
            raise IllFormedRevisionError("unknown literal %r" % (lit,))
        out.append(lit)
    if not _closure.route_realizable(out, instance.atom_ids, instance.phi_patterns):
        raise IllFormedRevisionError("route %r unrealizable under Phi" % (out,))
    return out


def _route_index(payload: Mapping[str, Any], routes: List[List[str]]) -> int:
    try:
        idx = int(payload.get("route_index", -1))
    except (TypeError, ValueError):
        raise IllFormedRevisionError("route_index must be an integer")
    if not 0 <= idx < len(routes):
        raise IllFormedRevisionError(
            "route_index %r out of range (claim has %d route(s))" % (idx, len(routes))
        )
    return idx


def _apply_edit(
    program: Program, kind: str, payload: Mapping[str, Any], instance: Instance
) -> Program:
    """Apply one well-formed edit to ``program``; raise IllFormedRevisionError
    otherwise.  Pure: returns a new immutable Program."""
    claims = list(program.claims)
    order = [list(p) for p in program.order]
    licenses: Dict[str, List[List[str]]] = {
        c: [list(r) for r in rs] for c, rs in program.licenses.items()
    }

    def _claim_routes() -> (str, List[List[str]]):
        cid = str(payload.get("claim", ""))
        if cid not in claims:
            raise IllFormedRevisionError("unknown claim %r" % cid)
        return cid, licenses.setdefault(cid, [])

    if kind == "add_literal":
        cid, routes = _claim_routes()
        idx = _route_index(payload, routes)
        lit = payload.get("literal")
        if not isinstance(lit, str) or atom_of(lit) not in set(instance.atom_ids):
            raise IllFormedRevisionError("unknown literal %r" % (lit,))
        if lit in routes[idx]:
            raise IllFormedRevisionError("literal %r already in route" % lit)
        routes[idx] = _check_route(routes[idx] + [lit], instance)
    elif kind == "remove_literal":
        cid, routes = _claim_routes()
        idx = _route_index(payload, routes)
        lit = payload.get("literal")
        if lit not in routes[idx]:
            raise IllFormedRevisionError("literal %r not in route %d" % (lit, idx))
        routes[idx] = [l for l in routes[idx] if l != lit]
    elif kind == "delete_route":
        cid, routes = _claim_routes()
        idx = _route_index(payload, routes)
        del routes[idx]
    elif kind == "add_route":
        cid, routes = _claim_routes()
        new_route = _check_route(payload.get("route"), instance)
        if any(frozenset(new_route) == frozenset(r) for r in routes):
            raise IllFormedRevisionError("route %r already present" % (new_route,))
        routes.append(new_route)
    elif kind == "add_claim":
        cid = str(payload.get("claim", ""))
        if not cid:
            raise IllFormedRevisionError("add_claim requires a claim id")
        if cid in claims:
            raise IllFormedRevisionError("claim %r already exists" % cid)
        new_routes = [
            _check_route(r, instance) for r in (payload.get("routes") or [])
        ]
        dominates = payload.get("dominates") or []
        known = set(claims) | {cid}
        pairs = []
        for pair in dominates:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or pair[0] not in known
                or pair[1] not in known
                or cid not in pair
            ):
                raise IllFormedRevisionError(
                    "declared dominance pair %r must be [weaker, stronger] over "
                    "known claims and mention the new claim" % (pair,)
                )
            pairs.append([str(pair[0]), str(pair[1])])
        claims.append(cid)
        order.extend(pairs)
        licenses[cid] = new_routes
    elif kind == "delete_claim":
        cid = str(payload.get("claim", ""))
        if cid not in claims:
            raise IllFormedRevisionError("unknown claim %r" % cid)
        claims.remove(cid)
        order = [p for p in order if cid not in p]
        licenses.pop(cid, None)
    elif kind == "replace_program":
        raw_lic = payload.get("licenses")
        if not isinstance(raw_lic, Mapping):
            raise IllFormedRevisionError(
                "replace_program requires 'licenses': {claim: [[literal,...],...]}"
            )
        claims = [str(c) for c in (payload.get("claims") or instance.claim_ids)]
        if len(set(claims)) != len(claims):
            raise IllFormedRevisionError("duplicate claim ids")
        raw_order = payload.get("order")
        if raw_order is None:
            order = [
                [w, s] for w, s in instance.order if w in claims and s in claims
            ]
        else:
            order = []
            for pair in raw_order:
                if (
                    not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                    or pair[0] not in claims
                    or pair[1] not in claims
                ):
                    raise IllFormedRevisionError("malformed order pair %r" % (pair,))
                order.append([str(pair[0]), str(pair[1])])
        licenses = {}
        for cid, routes in raw_lic.items():
            if str(cid) not in claims:
                raise IllFormedRevisionError("license for unknown claim %r" % cid)
            if not isinstance(routes, (list, tuple)):
                raise IllFormedRevisionError(
                    "licenses[%r] must be a list of routes" % cid
                )
            licenses[str(cid)] = [_check_route(r, instance) for r in routes]
    else:
        raise IllFormedRevisionError("unknown revision kind %r" % kind)

    return make_program(
        claims=claims,
        order=order,
        licenses=licenses,
        scope=program.scope,
        atoms=instance.atom_ids,
    )
