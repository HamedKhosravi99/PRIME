"""Attested four-valued evidence ledger.

Implements the evidence ledger and its integrity rule (the ledger is part of
the trusted base): the policy cannot write the ledger; ``Ledger.apply`` is
the ONLY mutation path and consumes attestations carrying tool provenance.

State semantics
---------------
Each atom's state is derived from its *active* attestation records:

    no active records            -> 'U'
    all active records say T     -> 'T'
    all active records say F     -> 'F'
    disagreeing active records   -> 'C'   (permanently, until an explicit
                                           retraction removes one side)

A retraction is itself an attestation (``kind="retract"``, ``references`` =
record id): repair operates on provenance-bearing attestations, never on the
frozen constraints (unsat cores over labeled assumptions).
"""

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_STATES = ("U", "T", "F", "C")


class LedgerError(ValueError):
    """Malformed attestation or unknown atom/record."""


class Ledger:
    """Four-valued evidence ledger z_t over a fixed atom vocabulary.

    Parameters
    ----------
    atoms : atom ids (the declared evidence vocabulary E_x).
    phi   : forbidden determined-value patterns, each a mapping
            {atom_id: "T"|"F"}; carried here so closure/status functions can
            operate on a ledger alone.

    The only mutator is :meth:`apply`; there is no setter for atom states
    (ledger-integrity condition).
    """

    def __init__(
        self,
        atoms: Sequence[str],
        phi: Sequence[Mapping[str, str]] = (),
    ) -> None:
        self._atoms: Tuple[str, ...] = tuple(str(a) for a in atoms)
        if len(set(self._atoms)) != len(self._atoms):
            raise LedgerError("duplicate atom ids")
        self._phi: Tuple[Dict[str, str], ...] = tuple(
            dict(p.get("forbid", p)) if not _is_pattern(p) else dict(p) for p in phi
        )
        for pat in self._phi:
            for a, v in pat.items():
                if a not in self._atoms or v not in ("T", "F"):
                    raise LedgerError("malformed phi pattern %r" % (pat,))
        self._records: List[Dict[str, Any]] = []
        self._version: int = 0

    # -- read-only views ----------------------------------------------------

    @property
    def atoms(self) -> Tuple[str, ...]:
        return self._atoms

    @property
    def phi_patterns(self) -> Tuple[Dict[str, str], ...]:
        return tuple(dict(p) for p in self._phi)

    @property
    def version(self) -> int:
        """Monotone ledger version; increments on every applied attestation."""
        return self._version

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Deep copy of the full attestation history (audit view)."""
        return copy.deepcopy(self._records)

    def active_records(self, atom: Optional[str] = None) -> List[Dict[str, Any]]:
        out = [
            copy.deepcopy(r)
            for r in self._records
            if r.get("kind", "attest") == "attest"
            and not r.get("retracted", False)
            and (atom is None or r["atom"] == atom)
        ]
        return out

    def state(self, atom: str) -> str:
        """Current four-valued state of ``atom``."""
        if atom not in self._atoms:
            raise LedgerError("unknown atom %r" % atom)
        values = {
            r["value"]
            for r in self._records
            if r.get("kind", "attest") == "attest"
            and not r.get("retracted", False)
            and r["atom"] == atom
        }
        if not values:
            return "U"
        if values == {"T"}:
            return "T"
        if values == {"F"}:
            return "F"
        return "C"

    def states(self) -> Dict[str, str]:
        return {a: self.state(a) for a in self._atoms}

    # -- the single mutation path (ledger integrity) -------------------------

    def apply(self, attestation: Mapping[str, Any]) -> int:
        """Apply one attestation; returns the new record id.

        Regular attestation keys: ``atom``, ``value`` ("T"|"F"),
        ``tool``, ``params``, ``output_digest``, ``step``.

        Retraction: ``kind="retract"``, ``references`` (record id to
        retract), plus ``tool``/``step`` provenance.  Retracting one side of
        a conflict returns the atom to the surviving side's value.
        """
        att = dict(attestation)
        kind = att.get("kind", "attest")
        if kind == "retract":
            ref = att.get("references")
            target = None
            for r in self._records:
                if r["id"] == ref and r.get("kind", "attest") == "attest":
                    target = r
                    break
            if target is None:
                raise LedgerError("retraction references unknown record %r" % ref)
            if target.get("retracted", False):
                raise LedgerError("record %r already retracted" % ref)
            target["retracted"] = True
        elif kind == "attest":
            for key in ("atom", "value", "tool", "step"):
                if key not in att:
                    raise LedgerError("attestation missing %r" % key)
            if att["atom"] not in self._atoms:
                raise LedgerError("attestation for unknown atom %r" % att["atom"])
            if att["value"] not in ("T", "F"):
                raise LedgerError("attestation value must be 'T' or 'F'")
        else:
            raise LedgerError("unknown attestation kind %r" % kind)
        record = dict(att)
        record["kind"] = kind
        record["id"] = len(self._records)
        record.setdefault("params", {})
        record.setdefault("output_digest", "")
        record["retracted"] = record.get("retracted", False)
        self._records.append(record)
        self._version += 1
        return record["id"]


def _is_pattern(p: Mapping[str, Any]) -> bool:
    """True when ``p`` is already a bare {atom: "T"|"F"} pattern."""
    return bool(p) and all(v in ("T", "F") for v in p.values()) and "forbid" not in p
