"""Specification/evidence split and hash-gated commitment (Weakness 4, paper
§4.2-§4.7: "Before evidence" is not formally defined strongly enough).

Reviewer objection addressed
-----------------------------
An agent that sees outcome-bearing artifact content (test results, CI logs,
paper tables, funding-eligibility values, ...) before it freezes its
license program could tailor the "standard" to the observed result -- the
architectural guarantee ("commit before evidence") would then be vacuous.
This module makes "before evidence" a checked runtime property rather than
a prose claim, by:

1. splitting every task instance into a **specification view** ``S_x``
   (:func:`spec_manifest`) and an **evidence view** ``E_x`` (everything
   else -- resolver outcomes, artifact contents, gold ground truth);
2. computing a **commitment hash** ``h0`` over the canonicalized program the
   moment it is fixed (:func:`commit_program`), so the freeze point is a
   verifiable artifact in the episode log, not an assertion;
3. enforcing an **enforceable capability boundary** (:class:`CommitmentGate`)
   under which evidence resolvers are technically uncallable until ``h0``
   has been recorded -- calling one earlier raises
   :class:`PreCommitEvidenceError` rather than merely being disallowed by
   policy;
4. providing an **audit** helper that checks a logged episode trace never
   invoked an evidence action before the commit record appears in it.

Formal statement (§4.5, Commitment-View Noninterference)
----------------------------------------------------------
Let two task instances ``x`` and ``x'`` share the same specification view
(``spec_manifest(x) == spec_manifest(x')``) but differ in hidden evidence.
If the commitment policy ``P0`` is invoked only on the specification view
and fixed randomness ``r`` (i.e. it is a function ``P0(S, r)``, with no
capability to call an evidence resolver before ``h0`` is recorded), then

    P0(x; r) = P0(x'; r).

The proof is by the capability/input boundary, not by any semantic argument
about what the policy "would" do: hidden evidence is simply not an argument
to ``P0``, and no evidence tool is invocable before commitment
(:class:`CommitmentGate` makes this executable). See
``pilot/prime_runtime/COMMITMENT_REPORT.md`` for the full statement/proof
and ``pilot/analysis/noninterference_audit.py`` (Experiment I1) for the
empirical audit that exercises exactly this boundary.

Strict vs artifact-aware modes (§4.7)
--------------------------------------
Two modes are distinguished explicitly (never conflated):

``strict``
    The commitment policy receives :func:`spec_manifest` ONLY. Evidence
    resolvers are unavailable until ``h0`` is recorded. Receives the full
    noninterference guarantee above.
``artifact_aware``
    The commitment policy may additionally inspect artifact content before
    freezing, but the program is still hashed and frozen before any LEDGER
    update or post-inspection revision. Receives only the weaker
    freeze-after-inspection guarantee: the program cannot be revised after
    evidence *resolution* begins, but it may already be a function of
    evidence *inspected* pre-freeze, so noninterference does not hold.

Timestamps
----------
The runtime forbids ``Date.now``-style wall-clock reads for reproducibility.
Every place a paper timestamp would go, this module instead takes an
injected integer **step counter** supplied by the caller (the episode's own
step index) -- see ``timestamp_placeholder`` in :func:`commit_program`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import Atom, Instance, Program

__all__ = [
    "SPEC_MANIFEST_VERSION",
    "PERMITTED_MANIFEST_FIELDS",
    "EXCLUDED_EVIDENCE_FIELDS",
    "DEFAULT_EVIDENCE_ACTIONS",
    "CommitmentMode",
    "PreCommitEvidenceError",
    "spec_manifest",
    "canonical_json",
    "sha256_hex",
    "spec_manifest_hash",
    "canonicalize_program",
    "canonical_program_json",
    "program_hash",
    "commit_program",
    "verify_commitment",
    "CommitmentGate",
    "GatedEvidenceExecutor",
    "verify_no_precommit_evidence",
    "audit_episode_trace",
]

SPEC_MANIFEST_VERSION = 1

# Documentation-grade manifests of what IS / IS NOT permitted pre-commit
# (paper §4.3). These constants are not consulted programmatically by
# spec_manifest() (its field selection below is authoritative and hand
# audited against this list) -- they exist so a reviewer or test can check
# the implementation against the prose contract line by line.
PERMITTED_MANIFEST_FIELDS = (
    "repository identifier (artifact.path)",
    "artifact kind / ecosystem label (artifact.kind)",
    "pinned commit hash (artifact.commit)",
    "domain label (instance.domain)",
    "user request text (instance.user_prompt) and declared scope (instance.scope)",
    "registered tool/action types (instance.actions)",
    "candidate claim SCHEMA: claim ids, level/order -- what a claim would "
    "assert, not whether it holds",
    "evidence atom SCHEMA: atom id, type label, resolver TOOL NAME, "
    "resolver parameter SCHEMA (declared names/types), resolver LOCATOR "
    "params (config/dependency file PATHS, declared suite/search-target "
    "NAMES -- i.e. where to look), resolver ATTESTATION RULE (how an "
    "outcome would be interpreted, e.g. 'exit_code==0' -- not the outcome "
    "itself)",
)
EXCLUDED_EVIDENCE_FIELDS = (
    "atom.gold_value (ground-truth resolver outcome -- evaluator-only)",
    "atom.proposition (in this benchmark's authoring convention this field "
    "is frequently written in outcome-revealing prose, e.g. '... exits 0 "
    "(668 passed, 3 skipped)' or '... the adjudication is that none does'; "
    "excluding it is load-bearing, not incidental -- see COMMITMENT_REPORT.md)",
    "instance.gold_licenses (the answer key)",
    "instance.decoys (knowing an action is a decoy already reveals which "
    "atoms matter)",
    "instance.notes_for_annotators (annotator/evidence commentary)",
    "any resolver output, tool observation, or ledger attestation",
    "any field not explicitly listed in PERMITTED_MANIFEST_FIELDS",
)

# Tool names that produce EVIDENCE (mirrors runner/arms.py's tool catalog,
# duplicated as a constant here rather than imported so this module has no
# dependency on the LLM-calling runner). emit_claim and revise_program are
# NOT evidence actions -- they are the licensed-output and program-mutation
# actions, gated separately by the interpreter / revision-mode ablations.
DEFAULT_EVIDENCE_ACTIONS: Tuple[str, ...] = (
    "inspect_file",
    "search_repo",
    "run_tests",
    "inspect_config",
    "ask_user",
)


class CommitmentMode:
    """Strict vs artifact-aware commitment modes (§4.7) -- flagged, never
    conflated. Every :class:`CommitmentGate` instance is constructed with
    exactly one of these; the mode is carried into every commit record so
    downstream analysis can never silently mix the two guarantees."""

    STRICT = "strict"
    ARTIFACT_AWARE = "artifact_aware"
    _ALL = (STRICT, ARTIFACT_AWARE)


class PreCommitEvidenceError(RuntimeError):
    """Raised when an evidence-producing resolver/tool is invoked before the
    commitment hash ``h0`` has been recorded (§4.4 capability boundary).

    This is a CAPABILITY failure, not a policy-compliance check: under
    strict mode, :class:`CommitmentGate` makes the call itself
    unavailable, so this error is the runtime's own enforcement, not a
    post-hoc audit finding.
    """


# ---------------------------------------------------------------------------
# Specification view S_x (§4.2, §4.3)
# ---------------------------------------------------------------------------


def _resolver_manifest(atom: Atom) -> Dict[str, Any]:
    r = atom.resolver
    if r is None:
        # Declared-unresolvable is itself part of the ATOM SCHEMA (an
        # authoring-time property of the instance's evidence vocabulary,
        # schema.py's "terminal-unavailability declaration"), not a
        # resolver OUTCOME -- both variants of a pair used to test
        # noninterference must keep this identical, or the manifests are
        # not byte-identical by construction (see noninterference_audit.py
        # for why "disable a resolver" is modeled as a hidden RUNTIME
        # override rather than as resolver=None).
        return {"resolvable": False, "tool": None, "params_schema": {}, "locator": {}, "attest_rule": None}
    return {
        "resolvable": True,
        "tool": r.tool,
        "params_schema": dict(r.params_schema),
        # Locator/config params (paths, suite names, search patterns): WHERE
        # to look / what tool call to make. Never a resolver OUTPUT.
        "locator": dict(r.params),
        "attest_rule": r.attest,
    }


def spec_manifest(instance: Instance) -> Dict[str, Any]:
    """Build the STRICT specification view ``S_x`` (§4.2/§4.3).

    Includes ONLY fields declared permissible before program commitment:
    repository identity, pinned commit hash, domain/ecosystem labels, the
    registered tool registry, the candidate claim SCHEMA (ids/text/level),
    the claim dominance SCHEMA (order edges), and the evidence-atom SCHEMA
    (id, type, resolver tool name, resolver parameter schema, resolver
    locator params such as file paths / suite names / search patterns, and
    the resolver's attestation RULE). See :data:`PERMITTED_MANIFEST_FIELDS`.

    Excludes everything else -- most importantly ``atom.gold_value``,
    ``atom.proposition``, ``instance.gold_licenses``, ``instance.decoys``
    and ``instance.notes_for_annotators`` (:data:`EXCLUDED_EVIDENCE_FIELDS`)
    -- which together constitute the evidence view ``E_x``. Everything not
    read out of ``instance`` by this function is, by construction, part of
    ``E_x`` and unavailable to a strict-mode commitment policy.

    The returned dict is plain JSON-serializable data (safe to hash via
    :func:`spec_manifest_hash` and log verbatim in an episode trace).
    """
    artifact = dict(instance.artifact)
    manifest: Dict[str, Any] = {
        "manifest_version": SPEC_MANIFEST_VERSION,
        "domain": instance.domain,
        "user_prompt": instance.user_prompt,
        "scope": instance.scope,
        "repository": {
            "path": artifact.get("path", ""),
            "kind": artifact.get("kind", ""),
            "commit": artifact.get("commit", ""),
        },
        "registered_tools": sorted(instance.actions),
        "claims": [
            {"id": c.id, "text": c.text, "level": c.level}
            for c in sorted(instance.claims, key=lambda c: c.id)
        ],
        "order": sorted([list(pair) for pair in instance.order]),
        "atoms": [
            dict({"id": a.id, "type": a.type}, **_resolver_manifest(a))
            for a in sorted(instance.atoms, key=lambda a: a.id)
        ],
    }
    return manifest


# ---------------------------------------------------------------------------
# Canonicalization + hashing
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Stable, whitespace-free JSON encoding used for every hash in this
    module (sorted keys, fixed separators) -- the same object always
    serializes to the same bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spec_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """SHA-256 of the canonicalized specification manifest."""
    return sha256_hex(canonical_json(manifest))


def _dedup_routes(routes: Sequence[Sequence[str]]) -> List[List[str]]:
    seen: List[frozenset] = []
    out: List[List[str]] = []
    for route in routes:
        key = frozenset(route)
        if key in seen:
            continue
        seen.append(key)
        out.append(sorted(route))
    out.sort()
    return out


def canonicalize_program(program: Program) -> Dict[str, Any]:
    """Canonical, JSON-serializable form of a committed :class:`Program`
    (Weakness 3 §3.2 canonicalization, lexical variant): claims sorted by
    id; order edges sorted; per claim, routes de-duplicated (as literal
    sets) and sorted lexically. This is enough for byte-stable hashing and
    exact-match testing; it is deliberately NOT a semantic (Phi-modulo)
    minimal-route reduction -- that reduction is schema.validate_program's
    and closure.py's job, run separately, before a program is ever passed
    to :func:`commit_program`.
    """
    return {
        "claims": sorted(program.claims),
        "order": sorted([list(pair) for pair in program.order]),
        "scope": program.scope,
        "atoms": sorted(program.atoms),
        "licenses": {
            cid: _dedup_routes(routes)
            for cid, routes in sorted(program.licenses.items())
        },
    }


def canonical_program_json(program: Program) -> str:
    return canonical_json(canonicalize_program(program))


def program_hash(program: Program) -> str:
    """``h0`` -- SHA-256 of the canonical program serialization."""
    return sha256_hex(canonical_program_json(program))


def commit_program(
    program: Program,
    manifest_hash: str = "",
    step: int = 0,
) -> Dict[str, Any]:
    """Freeze ``program`` and produce its commitment record (§4.4 steps 1-3).

    Parameters
    ----------
    program : the program to commit (already schema-validated by the
        caller, e.g. via ``schema.validate_program``).
    manifest_hash : ``spec_manifest_hash(spec_manifest(instance))`` for the
        instance this program was induced/authored against; recorded
        alongside ``h0`` so a downstream audit can confirm which
        specification view licensed this commitment. Optional so the
        function is usable standalone (e.g. in tests); callers that care
        about the full audit trail (i.e. everyone in strict mode) should
        always supply it -- :meth:`CommitmentGate.commit` does this for you.
    step : an injected step counter standing in for a wall-clock timestamp
        (``Date.now`` is forbidden in this runtime for reproducibility).
        Typically the episode's step index at the moment of commitment
        (0 for "before the first tool call").

    Returns
    -------
    A plain dict ``{h0, manifest_hash, timestamp_placeholder, program}``
    where ``program`` is the CANONICAL serializable form (not the live
    frozen dataclass) -- the record is self-contained and independently
    re-hashable (:func:`verify_commitment`).
    """
    canonical = canonicalize_program(program)
    h0 = sha256_hex(canonical_json(canonical))
    return {
        "h0": h0,
        "manifest_hash": manifest_hash,
        "timestamp_placeholder": step,
        "program": canonical,
    }


def verify_commitment(record: Mapping[str, Any], program: Program) -> bool:
    """Recompute ``h0`` from ``program`` and check it matches ``record["h0"]``
    -- tamper-evidence check for a logged commit record."""
    return program_hash(program) == record.get("h0")


# ---------------------------------------------------------------------------
# Enforceable capability boundary (§4.4)
# ---------------------------------------------------------------------------


class CommitmentGate:
    """Runtime enforcement of the ``A_commit`` / ``A_episode`` capability
    split (§4.4).

    Before :meth:`commit` (equivalently :meth:`record_commitment`) is
    called, :meth:`evidence_resolver` raises :class:`PreCommitEvidenceError`
    in strict mode -- evidence resolvers are technically uncallable, not
    merely disallowed by convention. After commitment, they are enabled.
    Every gate call (commit, evidence access, violation) is appended to
    :attr:`events`, an in-memory audit log consumable by
    :func:`verify_no_precommit_evidence`.

    In :data:`CommitmentMode.ARTIFACT_AWARE` mode, evidence access is
    permitted before commitment (the weaker freeze-after-inspection
    guarantee, §4.7) -- but re-commitment is still forbidden after the
    first :meth:`commit` call, and the mode is stamped into every event so
    strict- and artifact-aware-mode runs are never silently pooled.
    """

    def __init__(self, mode: str = CommitmentMode.STRICT) -> None:
        if mode not in CommitmentMode._ALL:
            raise ValueError("unknown commitment mode %r" % (mode,))
        self.mode = mode
        self._committed = False
        self._commit_record: Optional[Dict[str, Any]] = None
        self._program: Optional[Program] = None
        self._manifest: Optional[Dict[str, Any]] = None
        self._events: List[Dict[str, Any]] = []

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def commit_record(self) -> Optional[Dict[str, Any]]:
        return self._commit_record

    @property
    def program(self) -> Optional[Program]:
        return self._program

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def _log(self, entry: Mapping[str, Any]) -> None:
        e = dict(entry)
        e.setdefault("mode", self.mode)
        self._events.append(e)

    def record_commitment(
        self, program: Program, manifest_hash: str = "", step: int = 0
    ) -> Dict[str, Any]:
        """Low-level commit: compute + record ``h0`` for ``program``. Raises
        if a commitment has already been recorded on this gate (one gate ==
        one episode's commitment point)."""
        if self._committed:
            raise RuntimeError("commitment already recorded for this episode")
        record = commit_program(program, manifest_hash=manifest_hash, step=step)
        self._commit_record = record
        self._program = program
        self._committed = True
        self._log({"type": "commit", "step": step, "h0": record["h0"],
                    "manifest_hash": manifest_hash})
        return record

    def commit(self, instance: Instance, program: Program, step: int = 0) -> Dict[str, Any]:
        """Convenience commit: derive ``manifest_hash`` from
        :func:`spec_manifest` (``instance``) and record the commitment.
        Also stashes the manifest (:attr:`manifest`) for later audit."""
        manifest = spec_manifest(instance)
        self._manifest = manifest
        return self.record_commitment(
            program, manifest_hash=spec_manifest_hash(manifest), step=step
        )

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        return self._manifest

    def evidence_resolver(self, name: str, step: Optional[int] = None) -> bool:
        """Capability check gating one evidence-producing resolver/tool call.

        Strict mode, pre-commit: raises :class:`PreCommitEvidenceError`.
        Otherwise: logs the access and returns ``True`` (capability
        granted) -- the caller (e.g. a tool executor) proceeds to actually
        invoke the tool.
        """
        pre_commit = not self._committed
        if self.mode == CommitmentMode.STRICT and pre_commit:
            self._log({"type": "violation", "resolver": name, "step": step})
            raise PreCommitEvidenceError(
                "evidence resolver %r invoked before the commitment hash h0 "
                "was recorded (strict-mode capability boundary, paper §4.4)"
                % (name,)
            )
        self._log({"type": "evidence_access", "resolver": name, "step": step,
                    "pre_commit": pre_commit})
        return True


class GatedEvidenceExecutor:
    """Optional, non-invasive wrapper: gate any object exposing
    ``execute(name, params)`` (e.g. ``runner.arms.ToolExecutor``) behind a
    :class:`CommitmentGate`. Not wired into ``runner/arms.py`` by this
    module -- kept as an opt-in composition so the runner can adopt it
    without a breaking change:

        gate = CommitmentGate(mode=CommitmentMode.STRICT)
        gated = GatedEvidenceExecutor(ToolExecutor(instance), gate)
        ...
        gate.commit(instance, program, step=0)
        gated.execute("run_tests", {"suite": "tests"})  # now permitted
    """

    def __init__(self, inner: Any, gate: CommitmentGate) -> None:
        self._inner = inner
        self._gate = gate

    def execute(self, name: str, params: Mapping[str, Any], step: Optional[int] = None) -> Any:
        self._gate.evidence_resolver(name, step=step)
        return self._inner.execute(name, params)


# ---------------------------------------------------------------------------
# Episode-trace audit
# ---------------------------------------------------------------------------


def verify_no_precommit_evidence(gate: CommitmentGate) -> Dict[str, Any]:
    """Audit a :class:`CommitmentGate`'s own event log: confirm no evidence
    resolver access occurred before the commit event, and that a commit
    event exists at all.

    Returns ``{"ok": bool, "commit_recorded": bool, "violations": [...]}``.
    In strict mode ``ok`` can only be False if the gate's own
    :meth:`CommitmentGate.evidence_resolver` enforcement was bypassed
    (e.g. a caller invoked the underlying tool directly without going
    through the gate) -- this function is the independent check that the
    gate's raised exceptions are the whole story.
    """
    events = gate.events
    commit_index = next((i for i, e in enumerate(events) if e.get("type") == "commit"), None)
    violations = [
        {"index": i, **e}
        for i, e in enumerate(events)
        if e.get("type") in ("evidence_access", "violation")
        and (commit_index is None or i < commit_index)
    ]
    return {
        "ok": not violations and commit_index is not None,
        "commit_recorded": commit_index is not None,
        "commit_index": commit_index,
        "violations": violations,
        "n_events": len(events),
    }


def audit_episode_trace(
    entries: Sequence[Mapping[str, Any]],
    evidence_action_names: Sequence[str] = DEFAULT_EVIDENCE_ACTIONS,
    commit_marker_type: str = "commit",
) -> Dict[str, Any]:
    """Audit a generic, ordered episode trace (e.g. JSONL entries in the
    shape written by :class:`prime_runtime.trace.TraceLogger`, PLUS one
    entry of ``type == commit_marker_type`` marking where ``h0`` was
    recorded -- callers that log commitment should append such a marker,
    e.g. via ``trace.log(dict(gate.commit(...), type="commit"))``).

    Confirms that no entry naming an evidence action (``type`` in
    ``{"action", "observation"}`` and ``name`` -- or ``action`` -- in
    ``evidence_action_names``) occurs before the FIRST commit-marker entry.
    Position in ``entries`` (not the logged ``step`` number) is the
    ordering used, since step numbering is not guaranteed to reserve a
    slot for the commitment point itself.

    Returns ``{"ok", "commit_recorded", "commit_index", "violations",
    "n_entries"}`` -- ``ok`` requires both a recorded commit marker and
    zero violations (an episode with no commit marker at all cannot be
    certified "evidence came after commitment").
    """
    evidence_set = set(evidence_action_names)
    commit_index = next(
        (i for i, e in enumerate(entries) if e.get("type") == commit_marker_type), None
    )
    violations: List[Dict[str, Any]] = []
    for i, e in enumerate(entries):
        if e.get("type") not in ("action", "observation"):
            continue
        name = e.get("name", e.get("action"))
        if name in evidence_set and (commit_index is None or i < commit_index):
            violations.append({"index": i, "step": e.get("step"), "name": name,
                                "type": e.get("type")})
    return {
        "ok": not violations and commit_index is not None,
        "commit_recorded": commit_index is not None,
        "commit_index": commit_index,
        "violations": violations,
        "n_entries": len(entries),
    }
