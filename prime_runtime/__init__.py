"""prime_runtime — deterministic PRIME runtime for the pilot.

Implements the PRIME semantics:

- four-valued attested evidence ledger (ledger integrity)        -> ledger.py
- fail-closed solver closure Cl_Phi                              -> closure.py
- typed availability / route status / obligation frontier        -> status.py
- licensing, downward-closed authorization, MaxAuth              -> auth.py
- frozen-program interpreter and typed claim objects             -> interpreter.py
- exact robust route absorption checker                          -> absorption.py
- constrained renderer                                           -> renderer.py
- JSONL trace logging (gold-trace requirements)                  -> trace.py
- method-independent gold-ledger reconstruction + CompleteAuth   -> goldledger.py
- instance schema, validation, gold program                      -> schema.py
- obligation-guided completion: reachable ceiling + minimal route
  from the committed program, for the ``prime_guided`` arm         -> guidance.py
- executable benchmark oracle F*, independent of PRIME's route
  representation; prime routes DERIVED from F* (also the
  AuthFP / AuthFN program-oracle disagreement metrics)           -> oracle.py
- specification/evidence split (S_x/E_x), hash-gated commitment,
  and the A_commit/A_episode capability boundary                 -> commitment.py

Stdlib only. Python >= 3.9.
"""

from . import absorption, auth, closure, commitment, goldledger, guidance, interpreter, ledger
from . import oracle, renderer, schema, status, trace

__all__ = [
    "absorption",
    "auth",
    "closure",
    "commitment",
    "goldledger",
    "guidance",
    "interpreter",
    "ledger",
    "oracle",
    "renderer",
    "schema",
    "status",
    "trace",
]
