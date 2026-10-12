"""Method-independent gold-ledger reconstruction and CompleteAuth scoring.

The evaluator reconstructs
z^gold_0 = U^E, z^gold_{t+1} = Delta_gold(z^gold_t, a_t, o_t) from each
system's logged environment trace, identically for PRIME and every baseline.
A system's chain of thought, claimed evidence summary, internal ledger,
attestation log, or emissions are NEVER read — only (action, observation)
pairs, replayed through the instance's registered resolver definitions.

CompleteAuth(c, t) is evaluated at each claim's emission
ledger step against the GOLD program: 1 iff c is in AuthIDs_gold(z^gold_t)
— i.e., some dominating claim's gold license is closed at step t.

Attestation rules (``resolver.attest``) understood by Delta_gold:

    "span_read"      T when the observation has non-null ``content`` and no
                     ``error``; otherwise no attestation (fail-closed: a
                     failed read establishes nothing).
    "answer_given"   T when a substantive ``answer`` is present.
    "<key>==<val>"   generic comparison against an observation field, e.g.
                     "exit_code==0", "match_count==0"; T/F accordingly; no
                     attestation when the field is absent (never guess).
    also supported:  "!=", ">=", "<=", ">", "<".
    "record_predicate:<type>:<json>"
                     structured-record closed predicate grammar (Tier-M;
                     prime_runtime/predicates.py) -- evaluates the atom's
                     fixed predicate type + expected argument against the
                     observation's ``value`` field (the raw field the
                     ``inspect_record`` tool read). Fail-closed exactly like
                     every other rule here: no attestation on an observation
                     ``error``, a missing ``value``, or an inapplicable
                     predicate/value combination.
"""

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from . import auth as _auth
from . import predicates as _predicates
from .ledger import Ledger
from .schema import Instance
from .trace import read_trace

_CMP_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$")


def observation_digest(observation: Mapping[str, Any]) -> str:
    """Stable content digest of a tool observation (attestation provenance)."""
    blob = json.dumps(dict(observation), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _coerce(raw: str) -> Any:
    raw = raw.strip()
    if raw in ("true", "True"):
        return True
    if raw in ("false", "False"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw.strip("\"'")


def eval_attest(rule: str, observation: Mapping[str, Any]) -> Optional[str]:
    """Evaluate an attestation rule against a tool observation.

    Returns 'T', 'F', or None (no attestation — fail-closed on missing or
    inapplicable observations; the gold ledger never guesses)."""
    if rule == "span_read":
        ok = observation.get("error") is None and observation.get("content") is not None
        return "T" if ok else None
    if rule == "attested":
        # text-evidence atoms: the model is the resolver.
        # The executor accepts an attestation only when the atom's source text
        # was read in this episode and the quoted span occurs in it; the
        # value is then recorded as the model gave it, right or wrong, which
        # is exactly the resolver-error term of the decomposition theorem.
        if observation.get("accepted") is not True:
            return None
        v = observation.get("value")
        return v if v in ("T", "F") else None
    if rule == "file_exists":
        # T when the tool found the file; F on an explicit not-found error;
        # None (no attestation) on any other failure mode.
        err = observation.get("error")
        if err is None:
            return "T"
        return "F" if "not found" in str(err).lower() else None
    if rule == "answer_given":
        ans = observation.get("answer")
        return "T" if ans and ans != "no answer available" else None
    if rule.startswith("record_predicate:"):
        # Structured-record closed predicate grammar (Tier-M contract; see
        # prime_runtime/predicates.py for the full grammar documentation).
        if observation.get("error") is not None or "value" not in observation:
            return None  # fail-closed: a failed/absent field read attests nothing
        parsed = _predicates.parse_predicate_rule(rule)
        if parsed is None:
            return None  # malformed rule: never guess
        ptype, expected = parsed
        result = _predicates.evaluate_predicate(ptype, expected, observation["value"])
        return None if result is None else ("T" if result else "F")
    m = _CMP_RE.match(rule)
    if not m:
        return None
    key, op, raw = m.group(1), m.group(2), m.group(3)
    if key not in observation:
        return None
    left, right = observation[key], _coerce(raw)
    try:
        result = {
            "==": left == right,
            "!=": left != right,
            ">=": left >= right,
            "<=": left <= right,
            ">": left > right,
            "<": left < right,
        }[op]
    except TypeError:
        return None
    return "T" if result else "F"


def delta(
    instance: Instance,
    ledger: Ledger,
    action: str,
    params: Mapping[str, Any],
    observation: Mapping[str, Any],
    step: int,
) -> List[Dict[str, Any]]:
    """Delta_gold: convert one (action, observation) pair into ledger
    attestations via the instance's registered resolver definitions
    (closed typed atom schema) and apply them.

    An atom fires when its resolver tool matches the action, its optional
    ``params`` binding is included in the executed params, and its attest
    rule yields a determinate value.  Returns the applied attestations.
    """
    applied: List[Dict[str, Any]] = []
    for atom in instance.atoms:
        r = atom.resolver
        if r is None or r.tool != action:
            # None: terminal-unavailability atom — no attestation can ever fire.
            continue
        if r.params and not all(params.get(k) == v for k, v in r.params.items()):
            continue
        value = eval_attest(r.attest, observation)
        if value is None:
            continue
        att = {
            "atom": atom.id,
            "value": value,
            "tool": action,
            "params": dict(params),
            "output_digest": observation_digest(observation),
            "step": step,
        }
        ledger.apply(att)
        applied.append(att)
    return applied


class GoldReplay:
    """Per-step gold ledgers reconstructed from a logged trace."""

    def __init__(self, instance: Instance, attestations: List[Dict[str, Any]]) -> None:
        self.instance = instance
        self.attestations = attestations
        self.program = instance.gold_program()

    def ledger_at(self, step: int) -> Ledger:
        """Gold ledger after all observations with entry step <= ``step``."""
        led = Ledger(self.instance.atom_ids, self.instance.phi_patterns)
        for att in self.attestations:
            if att["step"] <= step:
                led.apply(att)
        return led

    def states_at(self, step: int) -> Dict[str, str]:
        return self.ledger_at(step).states()

    def complete_auth(self, claim_id: str, step: int) -> int:
        """CompleteAuth(c, t) against the gold program at the gold ledger of
        step t: authorization-aware — 1 iff c is authorized,
        directly or through a dominating claim's completed license."""
        led = self.ledger_at(step)
        return 1 if claim_id in _auth.auth_ids(self.program, led) else 0

    def completion_at_final(self) -> float:
        """Max obligation completion over all gold claims at the final gold
        ledger: per claim, max over non-dead routes of the
        fraction of route literals satisfied under Cl_Phi; empty route -> 1;
        0 when the ledger is inconsistent or all routes dead."""
        last = max((a["step"] for a in self.attestations), default=0)
        led = self.ledger_at(last)
        from .closure import cl, entails, is_inconsistent  # local import: no cycle
        closed = cl(led)
        if is_inconsistent(closed):
            return 0.0
        sat = lambda lit: entails(closed, lit)  # noqa: E731
        best = 0.0
        for claim in self.instance.claims:
            for route in self.program.licenses.get(claim.id, ()):
                if len(route) == 0:
                    best = max(best, 1.0)
                    continue
                n_sat = sum(1 for lit in route if sat(lit))
                best = max(best, n_sat / float(len(route)))
        return best


def reconstruct(
    trace: Union[str, Iterable[Mapping[str, Any]]], instance: Instance
) -> GoldReplay:
    """Replay a trace's (action, observation) pairs through the instance's
    resolver definitions into a step-indexed gold ledger.

    Reads ONLY entries of type "observation" (which carry the producing
    action and params) — never the agent's own attestation or emission
    entries (method-independent scoring).
    """
    entries = read_trace(trace)
    led = Ledger(instance.atom_ids, instance.phi_patterns)
    attestations: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("type") != "observation":
            continue
        applied = delta(
            instance,
            led,
            entry.get("action", ""),
            entry.get("params", {}),
            entry.get("data", {}),
            int(entry.get("step", 0)),
        )
        attestations.extend(applied)
    return GoldReplay(instance, attestations)


def complete_auth(
    trace: Union[str, Iterable[Mapping[str, Any]]],
    instance: Instance,
    claim_id: str,
    step: int,
) -> int:
    """Convenience wrapper: reconstruct then score CompleteAuth(c, t)."""
    return reconstruct(trace, instance).complete_auth(claim_id, step)
