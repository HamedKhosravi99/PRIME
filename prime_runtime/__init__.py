"""prime_runtime — deterministic PRIME runtime for the pilot.

Implements the semantics of PRIME-master.md (v15) sections 5, 6, 7 and 9:

- four-valued attested evidence ledger (SS5.4, T4 SS9.3)         -> ledger.py
- fail-closed solver closure Cl_Phi (SS5.4)                      -> closure.py
- typed availability / route status / obligation frontier (SS6.4)-> status.py
- licensing, downward-closed authorization, MaxAuth (SS7.4-7.5,
  SS9.2)                                                         -> auth.py
- frozen-program interpreter and typed claim objects (SS7.2,
  SS7.8)                                                         -> interpreter.py
- exact robust route absorption checker (SS9.3)                  -> absorption.py
- constrained renderer (SS7.8)                                   -> renderer.py
- JSONL trace logging (SS2.3 gold-trace requirements)            -> trace.py
- method-independent gold-ledger reconstruction + CompleteAuth
  (SS2.3, SS11.7)                                                -> goldledger.py
- instance schema, validation, gold program (SPEC.md)            -> schema.py
- obligation-guided completion: reachable ceiling + minimal route
  from the committed program, for the ``prime_guided`` arm         -> guidance.py
- executable benchmark oracle F*, independent of PRIME's route
  representation; prime routes DERIVED from F* (revision-analysis
  memo, Common infra A; Weakness 7; Weakness 2 §2.3 C1)           -> oracle.py
- specification/evidence split (S_x/E_x), hash-gated commitment,
  and the A_commit/A_episode capability boundary (revision-analysis
  memo Weakness 4 §4.2-§4.7)                                      -> commitment.py

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
