"""JSONL trace logging (SPEC.md Runtime API; PRIME-master.md SS2.3).

Every action, tool observation, attestation, and emission is logged with its
step and, where applicable, the ledger version.  The gold-ledger evaluator
(:mod:`prime_runtime.goldledger`) replays ONLY the (action, observation)
entries — attestation and emission entries are the evaluated system's own
records and are never used for protocol-compliance scoring.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


class TraceLogger:
    """Append-only JSONL trace writer."""

    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, entry: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(dict(entry), sort_keys=True, default=str) + "\n")
        self._fh.flush()

    def log_action(self, step: int, name: str, params: Mapping[str, Any]) -> None:
        self.log({"type": "action", "step": step, "name": name, "params": dict(params)})

    def log_observation(
        self, step: int, action: str, params: Mapping[str, Any], data: Mapping[str, Any]
    ) -> None:
        """A tool observation, linked to the action that produced it — the
        (a_t, o_t) pair the gold evaluator replays."""
        self.log(
            {
                "type": "observation",
                "step": step,
                "action": action,
                "params": dict(params),
                "data": dict(data),
            }
        )

    def log_attestation(
        self, step: int, attestation: Mapping[str, Any], ledger_version: int
    ) -> None:
        self.log(
            {
                "type": "attestation",
                "step": step,
                "attestation": dict(attestation),
                "ledger_version": ledger_version,
            }
        )

    def log_emission(
        self,
        step: int,
        claim_object: Mapping[str, Any],
        ledger_version: int,
        claim_ids: Optional[List[str]] = None,
    ) -> None:
        emitted = claim_ids
        if emitted is None:
            emitted = list(claim_object.get("authorized_claims", ()))
        self.log(
            {
                "type": "emission",
                "step": step,
                "claims": emitted,
                "claim_object": dict(claim_object),
                "ledger_version": ledger_version,
            }
        )

    def log_meta(self, **fields: Any) -> None:
        entry = {"type": "meta"}
        entry.update(fields)
        self.log(entry)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_trace(source: Union[str, Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Read a trace from a JSONL path, or pass through an entry list."""
    if isinstance(source, str):
        entries: List[Dict[str, Any]] = []
        with open(source, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
    return [dict(e) for e in source]
