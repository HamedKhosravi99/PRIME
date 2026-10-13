"""The four experimental arms and the shared tool-calling loop.

Arms (all sharing the same tool loop and 20-tool-call budget):

    direct  answer immediately, no evidence tools (emit_claim only);
    rubric  generate criteria first, then act with tools;
    react   standard interleaved reasoning + acting;
    prime   frozen gold (or validated induced) program; the interpreter
            masks emissions; the current obligation frontier and
            authorized-claim set are injected into the system prompt each
            step; emit_claim is allowed only for authorized claims.

Tools are executed HERE, by the runner, against the artifact directory; tool
outcomes are converted to attestations via the instance's registered
resolver definitions and ``Ledger.apply`` — the policy never writes the
ledger (ledger integrity).
"""

import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_PILOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PILOT not in sys.path:
    sys.path.insert(0, _PILOT)
    sys.path.insert(0, os.path.join(_PILOT, "src"))

from prime_runtime import auth as prime_auth  # noqa: E402
from prime_runtime import closure as _closure  # noqa: E402
from prime_runtime import goldledger  # noqa: E402
from prime_runtime import guidance as prime_guidance  # noqa: E402
from prime_runtime.interpreter import (  # noqa: E402
    REVISION_MODES,
    Interpreter,
    UnauthorizedClaimError,
    allowed_revision_kinds,
)
from prime_runtime.ledger import Ledger  # noqa: E402
from prime_runtime.schema import (  # noqa: E402
    Instance,
    Program,
    SchemaError,
    atom_of,
    make_program,
    validate_program,
)
from prime_runtime.trace import TraceLogger  # noqa: E402

# Four further baseline arms beyond the original four:
#   reflexion         answer -> self-critique -> revise -> emit (tools allowed)
#   cove              answer -> verification questions -> answer via tools ->
#                      revise -> emit (tools allowed)
#   static_contract   AUTHORED (gold) non-minimal checklist gate: no learned
#                      induction, no minimal-route search, no graded lattice
#   prime_nonminimal  PRIME gold arm, routes padded with 1-2 redundant
#                      still-satisfiable literals (tests the minimality factor)
# "PRIME without commitment" is NOT a new arm: it is already the existing
# prime arm's posthoc/arbitrary revision_mode ablations (see run_arm below).
#
# prime_guided (see src/prime_runtime/guidance.py): a CORE
# mechanism change, not a prompt tweak. PRIME's obligations are otherwise
# PERMISSIONS -- nothing drives the agent to keep gathering evidence once
# the weakest claim is authorized (measured under-claiming in structured-
# record domains, tableC_{security,drugsafety,financial}.json). Every step,
# the interpreter (src/prime_runtime/guidance.py, built only from
# closure.py/auth.py/status.py -- no re-derived semantics) computes the
# REACHABLE CEILING: the strongest claim whose minimal route from the
# COMMITTED program is already-true-or-still-open atoms within the
# remaining tool-call budget. It is surfaced as an ACTIVE OBLIGATION, and a
# TERMINATION RULE refuses a premature emit_claim / silent stop while a
# strictly stronger claim is still reachable and its obligation is open.
# SAFETY IS UNCHANGED: emission is still gated by the same auth mask
# (interpreter.emit) against the committed program -- guidance only changes
# which atoms get gathered and when the agent is allowed to stop, never
# what gets authorized. The unguided ``prime`` arm is kept byte-for-byte
# intact so guided-vs-unguided is a clean ablation.
ARMS = (
    "direct", "rubric", "react", "prime",
    "reflexion", "cove", "static_contract", "prime_nonminimal",
    "prime_guided", "prime_guided_gated", "masked_reactive", "conformal",
    "prime_complete", "prime_flat",
    # v5 information/enforcement ladder (docs/EXECUTION_PLAN.md, Part 1):
    "react_verify", "react_spec", "react_spec_verify",
    "react_program", "react_program_verify", "react_program_attested",
    "masked_reactive_v",
    # closest external systems, instantiated on the same tasks (extsys.py)
    "ext_agentspec", "ext_agentspec_examine", "ext_researchloop",
    "ext_prooforstop", "ext_hesp", "ext_glean",
)

# Frozen compliance instruction (docs/ANALYSIS_PLAN.md, F7): IDENTICAL on
# both sides of the primary enforcement contrast; the mask is the only
# difference.
COMPLIANCE_INSTR = (
    "\n\nBefore emitting any claim, check every route of that claim against "
    "the evidence you have actually gathered with tools in this session; "
    "emit a claim only if one of its routes is fully satisfied by that "
    "evidence, and otherwise emit the strongest claim that is, or abstain."
)
MAX_TOOL_CALLS = 20
RUN_TESTS_TIMEOUT = 120  # seconds
GUIDED_MAX_STALLS = 3  # bound on consecutive no-tool-call "hold" nudges
# (prime_guided): a safety valve against a runaway loop when the model
# repeatedly refuses to call a tool after being told a stronger claim is
# still reachable -- NOT part of the termination-rule semantics itself
# (which only ever blocks a stop while forward progress is still possible);
# it exists purely to bound worst-case wall-clock/API cost, exactly the way
# ``max_tool_calls`` already bounds tool-call cost.

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"}
# The only pytest options run_tests passes on (output and stop-first flags,
# and two options that narrow the run by a path inside the artifact).
_PYTEST_FLAGS = frozenset({"-q", "-qq", "-v", "-vv", "-x", "-s"})
_PYTEST_PATH_OPTIONS = ("--ignore=", "--deselect=")
_CONFIG_GLOBS = (".json", ".yaml", ".yml", ".toml", ".cfg", ".ini")


def _resolve_field_path(record: Any, field_path: str) -> Tuple[Any, bool]:
    """Walk a dotted field path (e.g. ``"severity.cvss_score"``,
    ``"affected.0.version"``) through nested JSON dict/list ``record``.

    Returns ``(value, found)``. A list segment is a non-negative or negative
    integer index; a dict segment is a key. The empty path returns the whole
    record. Mirrors ``src/suffice/validate.py``'s standalone duplicate (kept
    independent there so the offline validator has no runtime import)."""
    if not field_path:
        return record, True
    cur = record
    for part in field_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.lstrip("-").isdigit():
            idx = int(part)
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return None, False
        else:
            return None, False
    return cur, True


class ToolExecutor:
    """Executes the instance's evidence actions against the artifact dir."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.root = instance.artifact_dir
        # Call-class cache: one underlying
        # execution per (tool, canonical params); repeats are served from
        # cache and MUST NOT consume budget. `charges` tells an agent loop,
        # before a call executes, whether it consumes budget;
        # `last_was_cached` reports whether the call just executed was a
        # cache hit; `observed_classes` is the executor-side ground truth the
        # planner's cost model is tested against (planner-predicted ==
        # executor-observed).
        self._call_cache: Dict[Tuple[str, Tuple], Dict[str, Any]] = {}
        self.last_was_cached: bool = False
        self._texts_read: Dict[Tuple[str, str], str] = {}
        self.observed_classes: set = set()

    def _safe_path(self, rel: str) -> Optional[str]:
        """Resolve ``rel`` inside the artifact directory, or return None.

        The root and the candidate are both resolved (symlinks followed)
        before the containment test, so ``..`` segments, absolute paths,
        sibling directories whose names share the root's prefix, and
        symlinks that leave the root are all refused."""
        root = os.path.realpath(self.root)
        full = os.path.realpath(os.path.join(root, rel))
        if os.path.commonpath([root, full]) != root:
            return None
        return full

    def _class_key(self, name: str, params: Mapping[str, Any]) -> Tuple[str, Tuple]:
        """A call class: the tool and its arguments, and for attest_atom also
        whether the check's source text has been read, because the outcome
        depends on it (an attestation refused before the read is evaluated
        afresh after it)."""
        key = tuple(sorted((str(k), str(v)) for k, v in dict(params).items()))
        if name == "attest_atom":
            key += (("<source read>", str(self._source_read(params))),)
        return (name, key)

    def _source_read(self, params: Mapping[str, Any]) -> bool:
        aid = str(dict(params).get("atom_id", ""))
        atom = next((a for a in self.instance.atoms if a.id == aid), None)
        src = (getattr(atom, "hint", None) or {}).get("source") or {}
        return (str(src.get("record_id", "")), str(src.get("field_path", ""))) in self._texts_read

    def was_called(self, name: str, params: Mapping[str, Any]) -> bool:
        """Whether this (tool, params) call class has been executed in the episode."""
        return self._class_key(name, params) in self._call_cache

    def charges(self, name: str, params: Mapping[str, Any]) -> bool:
        """Whether a call about to be made consumes budget: every call does,
        except a repeat of an executed call class, which execute() serves
        from the cache. Decided before the call executes. (The logged runs
        charged each call by the cache status of the previously executed
        call; see docs/KNOWN_ISSUES.md.)"""
        return not self.was_called(name, params)

    def execute(self, name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        key = self._class_key(name, params)
        if key in self._call_cache:
            self.last_was_cached = True
            return self._call_cache[key]
        self.last_was_cached = False
        handler = getattr(self, "_tool_%s" % name, None)
        if handler is None:
            obs = {"error": "unknown tool %r" % name}
        else:
            try:
                obs = handler(dict(params))
            except Exception as exc:  # tool errors are observations, not crashes
                obs = {"error": "%s: %s" % (type(exc).__name__, exc)}
        self._call_cache[key] = obs
        self.observed_classes.add(key)
        return obs

    def _tool_read_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """The full text of a free-text field; remembered for attest_atom."""
        rid = str(params.get("record_id", ""))
        fp = str(params.get("field_path", ""))
        rec = (self.instance.records or {}).get(rid)
        if rec is None:
            return {"record_id": rid, "field_path": fp, "error": "unknown record"}
        value, ok = _resolve_field_path(rec, fp)
        if not ok or not isinstance(value, str):
            return {"record_id": rid, "field_path": fp, "error": "no text at this field"}
        self._texts_read[(rid, fp)] = value
        return {"record_id": rid, "field_path": fp, "text": value}

    def _tool_attest_atom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a model attestation of a text-evidence atom when its source
        text was read in this episode and the quote occurs in it verbatim
        (whitespace-normalized); the value is recorded as given."""
        aid = str(params.get("atom_id", ""))
        value = str(params.get("value", "")).strip().upper()
        quote = " ".join(str(params.get("quote", "")).split())
        atom = next((a for a in self.instance.atoms if a.id == aid), None)
        hint = getattr(atom, "hint", None) if atom is not None else None
        src = (hint or {}).get("source") or {}
        if atom is None or atom.resolver is None or atom.resolver.tool != "attest_atom":
            return {"atom_id": aid, "accepted": False, "reason": "not a text-evidence atom"}
        if value not in ("T", "F"):
            return {"atom_id": aid, "accepted": False, "reason": "value must be T or F"}
        key = (str(src.get("record_id", "")), str(src.get("field_path", "")))
        text = self._texts_read.get(key)
        if text is None:
            return {"atom_id": aid, "accepted": False, "reason": "read the source text first (read_text %s %s)" % key}
        if len(quote) < 8 or quote.lower() not in " ".join(text.split()).lower():
            return {"atom_id": aid, "accepted": False, "reason": "the quote must occur verbatim in the source text"}
        return {"atom_id": aid, "accepted": True, "value": value, "quote": quote}

    def _tool_inspect_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rel = str(params.get("path", ""))
        start = int(params.get("start", 1))
        end = int(params.get("end", start + 199))
        full = self._safe_path(rel)
        if full is None or not os.path.isfile(full):
            return {"path": rel, "error": "file not found"}
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        span = lines[start - 1 : end]
        return {"path": rel, "lines": [start, min(end, len(lines))], "content": "\n".join(span)}

    def _tool_search_repo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pattern = str(params.get("pattern", ""))
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return {"pattern": pattern, "error": "bad pattern: %s" % exc}
        # Optional path scoping: an atom whose proposition is scoped to a
        # subtree (e.g. "no TODO markers under schedule/") declares
        # ``path``; the search must honor it, else a whole-repo scan resolves
        # the atom against the wrong evidence.
        scope_rel = str(params.get("path", "") or "").strip()
        root = os.path.realpath(self.root)
        search_root = root
        if scope_rel:
            scoped = self._safe_path(scope_rel)
            if scoped is None or not os.path.exists(scoped):
                return {"pattern": pattern, "path": scope_rel, "error": "path not found",
                        "matches": [], "match_count": 0}
            search_root = scoped
        matches: List[Dict[str, Any]] = []
        for fpath in self._iter_files(search_root):
            if self._safe_path(os.path.relpath(fpath, root)) is None:
                continue  # a symlinked file that points outside the root
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            matches.append({
                                "path": os.path.relpath(fpath, root),
                                "line": i,
                                "text": line.rstrip()[:200],
                            })
            except OSError:
                continue
        return {"pattern": pattern, "matches": matches[:50], "match_count": len(matches)}

    @staticmethod
    def _iter_files(root: str):
        """Yield file paths under ``root`` (a file yields just itself),
        skipping vendored/VCS dirs."""
        if os.path.isfile(root):
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in sorted(filenames):
                yield os.path.join(dirpath, fname)

    def _tool_run_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        suite = str(params.get("suite", "default"))
        # Honor a scoped test selection: "default" runs the whole suite, any
        # other value is a space-separated list of pytest targets (files /
        # node ids). An atom scoped to specific modules must run exactly
        # those, else it resolves against the wrong tests (e.g. the whole
        # suite is red while the named core modules are green). One refused
        # token refuses the whole call: dropping it instead would widen the
        # run, up to the full suite when every target is refused.
        if suite and suite != "default":
            requested = shlex.split(suite)
            rejected = [t for t in requested if not self._test_target_ok(t)]
            if rejected:
                return {"suite": suite, "exit_code": -1,
                        "error": "refused: %s (test paths must stay inside the artifact "
                                 "directory; accepted options: %s, --ignore=PATH, "
                                 "--deselect=PATH)"
                                 % (" ".join(rejected), ", ".join(sorted(_PYTEST_FLAGS)))}
            cmd += requested
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=RUN_TESTS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"suite": suite, "exit_code": -1, "error": "timeout after %ss" % RUN_TESTS_TIMEOUT}
        return {
            "suite": suite,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }

    def _test_target_ok(self, token: str) -> bool:
        """Accept a pytest path or node id inside the artifact directory, a
        flag in ``_PYTEST_FLAGS``, or ``--ignore=``/``--deselect=`` with a
        path inside it. Every other option is refused: options such as
        ``-p``, ``-c``, ``--rootdir`` or ``--basetemp`` load code or touch
        files outside the artifact. A path that does not exist is passed
        on, and pytest reports it."""
        if token in _PYTEST_FLAGS:
            return True
        for opt in _PYTEST_PATH_OPTIONS:
            if token.startswith(opt):
                token = token[len(opt):]
                break
        else:
            if token.startswith("-"):
                return False
        return bool(token) and self._safe_path(token.split("::", 1)[0]) is not None

    def _tool_inspect_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rel = str(params.get("path", "") or "")
        if not rel:
            candidates = sorted(
                f
                for f in os.listdir(self.root)
                if os.path.isfile(os.path.join(self.root, f))
                and f.endswith(_CONFIG_GLOBS)
            )
            if not candidates:
                return {"error": "no config file found"}
            rel = candidates[0]
        full = self._safe_path(rel)
        if full is None or not os.path.isfile(full):
            return {"path": rel, "error": "config file not found"}
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return {"path": rel, "content": fh.read()[:4000]}

    def _tool_ask_user(self, params: Dict[str, Any]) -> Dict[str, Any]:
        answer = self.instance.notes_for_annotators or "no answer available"
        return {"question": str(params.get("question", "")), "answer": answer}

    def _tool_inspect_record(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Structured-record capability (Tier-M; see src/prime_runtime/
        predicates.py): read one field of a named JSON record embedded in
        the instance (``instance.records``, never anything fetched live --
        offline-replayable and deterministic by construction). The
        predicate/expected value that turns this raw field into an atom's
        T/F is NOT supplied here -- it lives in the atom's ``resolver.
        attest`` rule and is applied downstream by goldledger.eval_attest;
        this tool only returns the observed raw value, exactly like
        ``inspect_file`` only returns raw file content."""
        record_id = str(params.get("record_id", ""))
        field_path = str(params.get("field_path", ""))
        records = self.instance.records
        if record_id not in records:
            return {"record_id": record_id, "field_path": field_path, "error": "record not found"}
        value, found = _resolve_field_path(records[record_id], field_path)
        if not found:
            return {"record_id": record_id, "field_path": field_path, "error": "field not found"}
        return {"record_id": record_id, "field_path": field_path, "value": value}


# One-line, factual documentation of each revision kind (ablation modes).
_REVISION_KIND_DOC = {
    "add_literal": "add_literal — append one evidence literal to an existing "
    "route (payload: claim, route_index, literal)",
    "remove_literal": "remove_literal — remove one literal from an existing "
    "route (payload: claim, route_index, literal)",
    "add_route": "add_route — add a new route to an existing claim "
    "(payload: claim, route=[atom,...])",
    "delete_route": "delete_route — delete one route of a claim "
    "(payload: claim, route_index)",
    "add_claim": "add_claim — add a new claim with declared dominance "
    "(payload: claim, routes=[[atom,...],...], "
    "dominates=[[weaker_id, stronger_id],...])",
    "delete_claim": "delete_claim — remove a claim and its routes "
    "(payload: claim)",
    "replace_program": "replace_program — supply a complete license program "
    "(payload: licenses={claim_id: [[atom,...],...]}; optional claims, order)",
}


def revision_tool_schema(revision_mode: str) -> Dict[str, Any]:
    """The extra ``revise_program`` tool offered in non-fixed, non-posthoc
    ablation modes (OpenAI format)."""
    kinds = allowed_revision_kinds(revision_mode)
    return {
        "type": "function",
        "function": {
            "name": "revise_program",
            "description": "Propose a revision to the license program for "
            "this episode. Permitted kinds in the current mode: "
            + "; ".join(_REVISION_KIND_DOC[k] for k in kinds)
            + ". Proposals of other kinds or ill-formed edits are rejected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": kinds},
                    "payload": {
                        "type": "object",
                        "description": "Fields for the chosen kind: claim, "
                        "route_index, literal, route, routes, dominates, "
                        "licenses (as applicable).",
                        "properties": {
                            "claim": {"type": "string"},
                            "route_index": {"type": "integer"},
                            "literal": {"type": "string"},
                            "route": {"type": "array", "items": {"type": "string"}},
                            "routes": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}},
                            },
                            "dominates": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}},
                            },
                            "licenses": {"type": "object"},
                        },
                    },
                },
                "required": ["kind", "payload"],
            },
        },
    }


def tool_schemas(
    instance: Instance,
    actions: Optional[Sequence[str]] = None,
    revision_mode: str = "fixed",
    claim_choices: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """JSON tool schemas for the instance's actions (OpenAI format).

    ``claim_choices`` restricts the ``emit_claim`` enum (default: every
    instance claim). Used by the mechanism-ablation ``graded`` factor
    (Table E): ``graded=False`` restricts the enum to just the
    top-level claim -- answer/abstain only, no intermediate-strength claims."""
    claim_ids = list(claim_choices) if claim_choices is not None else list(instance.claim_ids)
    catalog: Dict[str, Dict[str, Any]] = {
        "inspect_file": {
            "description": "Read a span of a file in the artifact repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
        "search_repo": {
            "description": "Regex-search all files in the artifact repository.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
        "run_tests": {
            "description": "Run `python -m pytest` inside the artifact repository.",
            "parameters": {
                "type": "object",
                "properties": {"suite": {"type": "string"}},
            },
        },
        "read_text": {
            "description": "Read a free-text field of a record (the full text).",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}, "field_path": {"type": "string"}},
                "required": ["record_id", "field_path"],
            },
        },
        "attest_atom": {
            "description": "Attest a text-evidence atom after reading its source text: give the atom id, "
                           "its value (T if the text establishes the stated condition, F if it establishes "
                           "the opposite) and a verbatim quote from the text that supports the value. "
                           "The attestation is accepted only if the source text was read and the quote occurs in it.",
            "parameters": {
                "type": "object",
                "properties": {"atom_id": {"type": "string"}, "value": {"type": "string", "enum": ["T", "F"]},
                               "quote": {"type": "string"}},
                "required": ["atom_id", "value", "quote"],
            },
        },
        "inspect_config": {
            "description": "Read a configuration file (default: first found at repo root).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
        "ask_user": {
            "description": "Ask the user a clarifying question.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
            },
        },
        "inspect_record": {
            "description": "Read one field of a named structured record embedded in "
            "this instance, addressed by a dotted field path (e.g. "
            "'severity.cvss_score', 'affected.0.version'). Returns the raw field value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "field_path": {"type": "string"},
                },
                "required": ["record_id", "field_path"],
            },
        },
        "emit_claim": {
            "description": "Emit a final evaluative claim by id and end the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "enum": claim_ids}
                },
                "required": ["claim_id"],
            },
        },
    }
    if revision_mode == "posthoc":
        # Post-hoc program generation (the post-hoc selector ablation):
        # emit_claim carries the license program that licenses the
        # claim, supplied at emission time.
        catalog["emit_claim"] = {
            "description": "Emit a final evaluative claim by id and end the "
            "episode. No license program is committed for this episode: "
            "supply `licenses` — the license program under which the claim "
            "is licensed — as a mapping from claim id to a list of evidence "
            "routes (each route a list of evidence atom ids that must all be "
            "established as T).",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "enum": claim_ids},
                    "licenses": {
                        "type": "object",
                        "description": "{claim_id: [[atom_id, ...], ...]}",
                    },
                },
                "required": ["claim_id", "licenses"],
            },
        }
    wanted = list(actions if actions is not None else instance.actions)
    out = []
    for name in wanted:
        if name in catalog:
            out.append({"type": "function", "function": dict(catalog[name], name=name)})
    if revision_mode in ("strengthen_only", "unfrozen_lattice", "arbitrary"):
        out.append(revision_tool_schema(revision_mode))
    return out


def _base_system_prompt(instance: Instance) -> str:
    claims = "\n".join(
        "  %s (level %d): %s" % (c.id, c.level, c.text) for c in instance.claims
    )
    return (
        "You are reviewing an artifact to answer the user's evaluative request.\n"
        "Scope: %s\n"
        "Candidate claims (emit exactly one via emit_claim when ready):\n%s\n"
        "Gather evidence with the available tools before claiming."
        % (instance.scope, claims)
    )


def _atom_catalog(instance: Instance) -> str:
    return "\n".join("  %s: %s" % (a.id, a.proposition) for a in instance.atoms)


def _revision_mode_statement(instance: Instance, revision_mode: str) -> str:
    """Neutral statement of the episode's revision allowance (ablation modes).

    States what the mode permits — nothing more; it does not suggest that
    revising is useful, needed, or expected."""
    if revision_mode == "fixed":
        return ""
    if revision_mode == "posthoc":
        return (
            "\n\nProgram policy for this episode (mode: posthoc): no license "
            "program is committed upfront. When you call emit_claim, supply "
            "`licenses` — the license program under which your claim is "
            "licensed: {claim_id: [[atom_id, ...], ...]}. A claim is "
            "authorized when some route of it (or of a claim dominating it) "
            "has every atom established as T in the ledger. The supplied "
            "program is validated for well-formedness only.\n"
            "Evidence atoms:\n" + _atom_catalog(instance)
        )
    kinds = allowed_revision_kinds(revision_mode)
    return (
        "\n\nProgram revision policy for this episode (mode: %s): the "
        "committed license program may be revised with the revise_program "
        "tool. Permitted revision kinds:\n%s\n"
        "Proposals of any other kind are rejected. Accepted revisions take "
        "effect immediately; emit_claim is checked against the current "
        "program. Revising is optional.\n"
        "Evidence atoms:\n%s"
        % (
            revision_mode,
            "\n".join("  " + _REVISION_KIND_DOC[k] for k in kinds),
            _atom_catalog(instance),
        )
    )


def _prime_system_prompt(
    instance: Instance,
    interpreter: Interpreter,
    ledger: Ledger,
    revision_mode: str = "fixed",
) -> str:
    authorized = interpreter.legal_claim_actions(ledger)
    if revision_mode == "posthoc":
        # No committed program: no frontier to show (nothing is committed);
        # show ledger state and the authorization under the last supplied
        # program, if any.
        return (
            _base_system_prompt(instance)
            + _revision_mode_statement(instance, revision_mode)
            + "\n\nLedger state (updated every step): %s\n"
            % json.dumps(ledger.states(), sort_keys=True)
            + "AUTHORIZED CLAIMS under the last supplied program "
            + "(the only claims emit_claim will accept): %s" % (authorized or "none")
        )
    frontier = interpreter.frontier(ledger)
    return (
        _base_system_prompt(instance)
        + "\n\nPRIME license state (updated every step):\n"
        + "AUTHORIZED CLAIMS (the only claims emit_claim will accept): %s\n" % (authorized or "none")
        + "OBLIGATION FRONTIER (open work per claim, minimal routes):\n%s\n"
        % json.dumps({k: v for k, v in frontier.items()}, sort_keys=True)
        + "Discharge obligations with tools; emit only an authorized claim."
        + _revision_mode_statement(instance, revision_mode)
    )


# ---------------------------------------------------------------------------
# prime_guided arm (CORE mechanism: obligation-guided completion, see
# src/prime_runtime/guidance.py)
# ---------------------------------------------------------------------------


def _spec_vocab_block(instance: Instance) -> str:
    """The specification-view vocabulary (claims with text/level, order,
    atom schema incl. resolver tool + locator params) -- exactly the
    information Definition 2 grants, no routes, no outcomes."""
    return "\n\nSPECIFICATION (candidate claims, order, evidence schema):\n" + \
        _spec_view_vocabulary(instance)


def _routes_block(instance: Instance) -> str:
    """The exact committed routes (gold licenses), serialized. Giving these
    to an UNMASKED arm is the support-logic-information rung of the v5
    ladder."""
    lic = {cid: [list(r) for r in routes]
           for cid, routes in instance.gold_program().licenses.items()}
    return ("\n\nEVIDENCE LICENSES (which sets of atoms suffice for each "
            "claim; a claim is licensed iff some route's atoms are all "
            "established):\n" + json.dumps(lic, sort_keys=True))


_SPEC_PAD_FILLER = (
    " The evaluation proceeds by inspecting fields of the embedded record "
    "and recording what is observed before any conclusion is stated.")


def _ladder_system_prompt(arm: str, instance: Instance, ledger) -> str:
    """System prompts for the v5 ladder's unmasked arms. Padding: when
    PRIME_PAD_SPEC=1, react_spec* prompts are padded with a schema-neutral
    filler sentence to the length of the corresponding react_program*
    prompt (context-length sensitivity analysis; unpadded is primary)."""
    base = _base_system_prompt(instance)
    p = base
    if arm in ("react_spec", "react_spec_verify", "react_program",
               "react_program_verify", "react_program_attested"):
        p += _spec_vocab_block(instance)
    if arm in ("react_program", "react_program_verify", "react_program_attested"):
        p += _routes_block(instance)
    if arm == "react_program_attested":
        p += ("\n\nATTESTED LEDGER (updated every step; the recorded truth "
              "state of each evidence atom): %s"
              % json.dumps(ledger.states(), sort_keys=True))
    if arm in ("react_verify", "react_spec_verify", "react_program_verify",
               "react_program_attested"):
        p += COMPLIANCE_INSTR
    if arm in ("react_spec", "react_spec_verify") and os.environ.get("PRIME_PAD_SPEC") == "1":
        target = len(base + _spec_vocab_block(instance) + _routes_block(instance)
                     + (COMPLIANCE_INSTR if arm.endswith("verify") else ""))
        while len(p) < target:
            p += _SPEC_PAD_FILLER
    return p


def _masked_reactive_system_prompt(
    instance: Instance,
    interpreter: Interpreter,
    ledger: Ledger,
    revision_mode: str = "fixed",
) -> str:
    """``masked_reactive`` arm: PRIME's evidence GATE without PRIME's
    evidence-SEEKING control law.

    This is the strong formal-contract baseline. It is deliberately
    identical to ``_prime_system_prompt`` except that the
    ``OBLIGATION FRONTIER (open work per claim, minimal routes)`` block is
    withheld. The arm keeps, bit-for-bit, the same:

      * committed gold (or induced) DNF program -- the SAME disjunctive
        routes, NOT the ``static_contract`` conjunctive collapse that
        unions every literal across all routes and thereby destroys the
        alternatives;
      * graded claim lattice and the MaxAuth antichain fallback at budget
        exhaustion;
      * enforced interpreter mask (``emit_claim`` raises
        ``UnauthorizedClaimError`` on an unauthorized claim).

    What it does NOT get is obligation-directed action selection: nothing
    tells it which atoms are still open, which route is minimal, or what to
    do next. It must choose actions reactively and discover the route by
    exploration, exactly like the ``react`` arm, while being gated by an
    equally expressive evidence contract.

    The contrast ``prime`` vs ``masked_reactive`` therefore isolates the
    contribution of the control law alone, holding the gate fixed --
    the comparison ``static_contract`` cannot provide, because it differs
    from ``prime`` in gate expressiveness AND control law at once.
    """
    authorized = interpreter.legal_claim_actions(ledger)
    return (
        _base_system_prompt(instance)
        + "\n\nClaim gate state (updated every step):\n"
        + "AUTHORIZED CLAIMS (the only claims emit_claim will accept): %s\n" % (authorized or "none")
        + "Investigate with tools as you see fit; emit only an authorized claim."
        + _revision_mode_statement(instance, revision_mode)
    )


def _obligation_text(instance: Instance, g: Mapping[str, Any], executor=None) -> str:
    """The residual obligations as shown to the model: each open atom with
    its tool and, for an atom whose locator is withheld (spec_hint), what it
    must read, the candidate fields not yet read, and the candidates already
    read without establishing it. The runtime knows the last two from the
    executor's call log; this is the bookkeeping the deterministic candidate
    policy keeps as its tabu list, and withholding it would leave the model
    to infer from a silent ledger that a read went to the wrong field."""
    by_id = {a.id: a for a in instance.atoms}
    parts = []
    for lit in g["blocking_atoms"]:
        aid = atom_of(lit)
        tool = g["tools"].get(aid) or "?"
        a = by_id.get(aid)
        hint = getattr(a, "hint", None) if a is not None else None
        if not hint or a.resolver is None:
            parts.append("%s (via %s)" % (lit, tool))
            continue
        if "source" in hint:
            src = hint["source"]
            need = "F" if lit.startswith("!") else "T"
            parts.append("%s (read_text %s %s, then attest_atom %s with value %s and a quote; %s is T when %s)" % (
                lit, src.get("record_id"), src.get("field_path"), a.id, need, a.id, hint.get("what", "?")))
            continue
        cands = list(hint.get("candidates", []))
        read = set()
        if executor is not None:
            fixed = {k: v for k, v in dict(a.resolver.params).items() if k != "field_path"}
            read = {c for c in cands if executor.was_called(a.resolver.tool, dict(fixed, field_path=c))}
        untried = [c for c in cands if c not in read]
        tried = [c for c in cands if c in read]
        parts.append("%s (via %s; must read %s; candidate fields not yet read: %s%s)" % (
            lit, tool, hint.get("what", "?"), json.dumps(untried),
            "; read without establishing it: %s" % json.dumps(tried) if tried else ""))
    return ", ".join(parts)


def _prime_guided_system_prompt(
    instance: Instance,
    interpreter: Interpreter,
    ledger: Ledger,
    remaining_budget: int,
    levels: Mapping[str, int],
    executor=None,
) -> str:
    """Same masked AUTHORIZED CLAIMS line as ``_prime_system_prompt``, plus
    the REACHABLE CEILING / ACTIVE OBLIGATION / TERMINATION RULE directive
    computed by ``prime_runtime.guidance.reachable_ceiling`` from the
    COMMITTED program (never gold/oracle) and the current ledger. This is
    guidance only -- it never changes what emit_claim will accept; see
    ``run_arm``'s emit_claim handling for the actual (mask-preserving)
    enforcement."""
    authorized = interpreter.legal_claim_actions(ledger)
    g = prime_guidance.reachable_ceiling(
        interpreter.program, ledger, interpreter.resolvers, levels, remaining_budget
    )
    lines = [
        _base_system_prompt(instance),
        "",
        "PRIME GUIDED license state (updated every step; %d tool call(s) "
        "remaining):" % remaining_budget,
        "AUTHORIZED CLAIMS (the only claims emit_claim will accept): %s"
        % (authorized or "none"),
    ]
    ceiling = g["ceiling_claim"]
    if ceiling is None:
        lines.append(
            "REACHABLE CEILING: none of this instance's claims are licensable "
            "from here within the remaining budget."
        )
    elif g["must_continue"]:
        obligations = _obligation_text(instance, g, executor)
        lines.append(
            "REACHABLE CEILING: %s (level %d) -- the strongest claim licensable "
            "within your remaining budget." % (ceiling, g["ceiling_level"])
        )
        lines.append(
            "ACTIVE OBLIGATION: resolve %s to license %s." % (obligations, ceiling)
        )
        lines.append(
            "TERMINATION RULE: do not call emit_claim for a weaker claim, and "
            "do not stop, while this obligation is open and budget remains -- "
            "attempt the listed tool call(s) first. emit_claim for a weaker "
            "claim will be held (not accepted) until %s is authorized, the "
            "ceiling changes (e.g. an atom resolves against it), or your "
            "budget is exhausted." % ceiling
        )
    else:
        lines.append(
            "REACHABLE CEILING: %s (level %d) -- already authorized; no "
            "strictly stronger claim is reachable within the remaining "
            "budget. You may emit_claim now." % (ceiling, g["ceiling_level"])
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# static_contract arm (the "static formal contract" baseline)
# ---------------------------------------------------------------------------


def _static_contract_checklist(instance: Instance) -> Dict[str, Tuple[str, ...]]:
    """AUTHORED, non-adaptive checklist per claim: the UNION of literals
    across ALL of that claim's gold routes (not the minimal OR-of-routes
    PRIME uses). Fixed for the whole episode (no learned induction, no
    per-episode route search); a claim's checklist must be independently
    completed in full -- there is no OR of alternatives and no graded
    lattice auto-unlock of weaker claims from a stronger claim's evidence.
    This is the ResearchLoop-style "authored complete program used as a
    fixed checklist" baseline in the E1 methods list."""
    out: Dict[str, Tuple[str, ...]] = {}
    for cid in instance.claim_ids:
        seen: Dict[str, str] = {}
        for route in instance.gold_licenses.get(cid, ()):
            for lit in route:
                seen[atom_of(lit)] = lit  # real instances never disagree in polarity
        out[cid] = tuple(sorted(seen.values()))
    return out


def _checklist_satisfied(required: Sequence[str], ledger: Ledger) -> bool:
    c = _closure.cl(ledger)
    if _closure.is_inconsistent(c):
        return False  # fail-closed, matching the PRIME mask's own convention
    return _closure.satisfies(c, required)


# ---------------------------------------------------------------------------
# information-matched comparators
# ---------------------------------------------------------------------------
# The six external comparators are prompted the way their authors proposed
# and, like ReAct, are never told which fields the atoms read; part of their
# gap to the masked arms is therefore information, not mechanism. With
# PRIME_COMPARATOR_SPEC=1 each comparator's system prompt additionally
# carries the specification-view block (claim text and levels, atom types,
# resolver tools and locator params: exactly what +spec gives the ladder,
# no routes, no outcomes). Unset, every prompt is byte-identical to the
# published runs (pinned by test_comparator_spec_view.py).
COMPARATOR_ARMS = ("direct", "rubric", "cove", "reflexion", "conformal",
                   "static_contract")


def _comparator_spec_view() -> bool:
    return os.environ.get("PRIME_COMPARATOR_SPEC", "0") == "1"


def _comparator_system_prompt(instance: Instance) -> str:
    """Base prompt of a comparator arm, plus the specification view when the
    information-matched variant is requested."""
    p = _base_system_prompt(instance)
    if _comparator_spec_view():
        p += _spec_vocab_block(instance)
    return p


def _static_contract_system_prompt(
    instance: Instance, checklist: Mapping[str, Tuple[str, ...]], ledger: Ledger
) -> str:
    c = _closure.cl(ledger)
    inconsistent = _closure.is_inconsistent(c)
    lines = []
    for cid in instance.claim_ids:
        required = checklist.get(cid, ())
        if inconsistent:
            open_items = list(required)
        else:
            open_items = [lit for lit in required if not _closure.entails(c, lit)]
        status = "COMPLETE" if not open_items else "open: %s" % ", ".join(open_items)
        lines.append("  %s (fixed checklist %s): %s" % (cid, list(required), status))
    return (
        _comparator_system_prompt(instance)
        + "\n\nSTATIC CONTRACT (fixed authored checklist; updated every step): "
        "ALL listed items for a claim must be established True before that "
        "exact claim may be emitted. No partial credit, no alternate routes, "
        "no automatic unlocking of a weaker claim from a stronger claim's "
        "evidence.\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# prime_nonminimal arm ("PRIME, nonminimal routes")
# ---------------------------------------------------------------------------


def _nonminimal_gold_program(instance: Instance) -> Tuple[Program, Dict[str, Any]]:
    """The gold program with each route padded by 1-2 REDUNDANT,
    still-satisfiable literals -- isolates the minimality factor (does
    minimality save wasted work without changing what is supported?).

    The SAME 1-2 atoms are appended to every route of every claim (rather
    than a route-specific choice), which is what makes the result provably
    still schema-valid without a per-route realizability search FOR THE TWO
    STRUCTURAL CHECKS validate_program makes (non-domination within a claim's
    route family; cross-claim order coherence): the pad set is, by
    construction, disjoint from every atom already used anywhere in
    ``gold_licenses``, so adding it to both sides of every route / sibling
    route / order-dominance pair preserves both exactly.

    The THIRD check -- Phi-realizability of the padded route -- is NOT
    automatic: an instance's Phi may forbid the padding atom's polarity in
    combination with an atom the base route already requires (observed on
    swe-0006, where Phi forbids E3=T & E4=T and C3's gold route already
    requires E3=T). Several still-satisfiable candidates are tried in order
    -- 2 then 1 positive-polarity unused atoms, then 1 then 2 negated --
    before falling back to unpadded (reported, never silently fabricated).
    """
    used = set()
    for routes in instance.gold_licenses.values():
        for route in routes:
            for lit in route:
                used.add(atom_of(lit))
    unused = sorted(a for a in instance.atom_ids if a not in used)

    def _try_build(pad_atoms: List[str]) -> Program:
        licenses = {
            cid: [list(route) + pad_atoms for route in routes]
            for cid, routes in instance.gold_licenses.items()
        }
        program = make_program(
            claims=instance.claim_ids, order=instance.order, licenses=licenses,
            scope=instance.scope, atoms=instance.atom_ids,
        )
        validate_program(program, instance)  # raises SchemaError if unrealizable
        return program

    candidates: List[List[str]] = []
    if len(unused) >= 2:
        candidates.append(list(unused[:2]))
    if len(unused) >= 1:
        candidates.append([unused[0]])
        candidates.append(["!" + unused[0]])
    if len(unused) >= 2:
        candidates.append(["!" + unused[0], "!" + unused[1]])
        candidates.append([unused[0], "!" + unused[1]])
        candidates.append(["!" + unused[0], unused[1]])
    for pad_atoms in candidates:
        try:
            return _try_build(pad_atoms), {"pad_atoms": pad_atoms}
        except SchemaError:
            continue
    # No spare atom (or polarity of one) on this instance is Phi-consistent
    # with every route: cannot pad without reusing a load-bearing atom
    # (which could change what is supported, not just cost). Measured and
    # reported, never silently fabricated.
    return instance.gold_program(), {
        "pad_atoms": [], "note": "no still-satisfiable padding found"
    }


class InducedProgramInvalid(Exception):
    """Model failed to induce a valid license program (after retries).

    Not a crash: license-induction validity is a MEASURED quantity (the
    RubricBench risk). The episode records an
    invalid-induction outcome and abstains."""


def _spec_view_vocabulary(instance: Instance) -> str:
    """Render the claim and evidence-atom SCHEMA exactly as the specification
    view $S_x$ permits (``commitment.PERMITTED_MANIFEST_FIELDS``): each
    claim's id, level and TEXT ("what a claim would assert, not whether it
    holds"), and each atom's id, type label, resolver tool name and resolver
    LOCATOR params ("where to look").

    Deliberately excluded, because they are the evidence view $E_x$:
    ``atom.proposition`` (authored as outcome-revealing prose),
    ``atom.gold_value``, and ``instance.gold_licenses``. Passing any of those
    here would leak the answer into the induction prompt.
    """
    claims = "\n".join(
        "  %s (level %d): %s" % (c.id, c.level, c.text)
        for c in sorted(instance.claims, key=lambda c: (c.level, c.id))
    )
    atoms = []
    for a in sorted(instance.atoms, key=lambda a: a.id):
        r = a.resolver
        params = dict(getattr(r, "params", {}) or {})
        hint = getattr(a, "hint", None)
        if hint and "source" in hint:
            # text-evidence atom: the model reads the source text and attests
            src = hint["source"]
            atoms.append(
                "  %s (type text): read_text(record_id=%r, field_path=%r), then attest_atom(atom_id=%r, value, quote); "
                "it holds (T) when %s"
                % (a.id, src.get("record_id"), src.get("field_path"), a.id, hint.get("what", "?"))
            )
            continue
        if hint:
            # locator-uncertain split: the exact field is withheld; the agent
            # sees what the atom must read and the candidate fields
            fixed = {k: v for k, v in params.items() if k != "field_path"}
            atoms.append(
                "  %s (type %s): resolved by %s(%s, field_path=one of %s); it must read %s"
                % (a.id, a.type, getattr(r, "tool", "?"),
                   ", ".join("%s=%r" % (k, v) for k, v in sorted(fixed.items())),
                   json.dumps(list(hint.get("candidates", []))), hint.get("what", "?"))
            )
            continue
        atoms.append(
            "  %s (type %s): resolved by %s(%s)"
            % (a.id, a.type, getattr(r, "tool", "?"),
               ", ".join("%s=%r" % (k, v) for k, v in sorted(params.items())))
        )
    return ("Claims (weak to strong), with what each asserts:\n%s\n"
            "Evidence atoms, with what each one interrogates:\n%s\n"
            % (claims, "\n".join(atoms)))


def induce_program(
    instance: Instance, llm, max_retries: int = 2, view: str = "ids"
) -> Program:
    """Induced-license condition: the model proposes a program in schema;
    validated with up to ``max_retries`` feedback rounds (the validator error
    is fed back like a compiler message — fair, method-independent). A still-
    invalid program raises InducedProgramInvalid (measured, not an error).

    ``view`` selects HOW MUCH of the specification view the inducing policy
    is conditioned on. This is an experimental factor, not a mechanism
    change (see experiments/analysis/vocabulary_conditions.py and the paper's
    vocabulary-condition section):

        "ids"   the originally-run condition, preserved BIT-FOR-BIT as the
                default so every previously published induced-license number
                remains reproducible: the model sees the ordered claim IDs
                and atom IDs and the user prompt, but NOT what any claim
                asserts or what any atom interrogates.
        "spec"  the full specification view the paper's Definition 2 grants
                a commitment policy: claim text and level, atom type,
                resolver tool and locator params. Strictly more information,
                still strictly inside the information boundary (no atom
                proposition, no gold value, no gold licenses).

    Under "ids" the policy is connecting semantically empty identifiers,
    which understates what license induction
    from a specification would look like; "spec" is the matched condition
    that tests that.
    """
    if view not in ("ids", "spec"):
        raise ValueError("unknown induction view %r" % view)
    vocab = (
        _spec_view_vocabulary(instance) if view == "spec"
        else "Claims (weak to strong): %s. Available atoms: %s. Use ONLY these exact "
             "ids. Answer with JSON only." % (list(instance.claim_ids), list(instance.atom_ids))
    )
    prompt = (
        "Propose a claim-license program as JSON with one key 'licenses' "
        "mapping each claim id to a LIST OF ROUTES. Each route is ITSELF A "
        "LIST of evidence-atom ids that together suffice for the claim; a "
        "single-atom route is still a list, e.g. [\"E1\"].\n"
        "FORMAT (note the double brackets -- each claim's value is a list "
        "whose elements are routes, and each route is a list of atom ids):\n"
        "  {\"licenses\": {\n"
        "     \"CX\": [[\"Ea\"]],                 # one route needing atom Ea\n"
        "     \"CY\": [[\"Ea\", \"Eb\"]],           # one route needing Ea AND Eb\n"
        "     \"CZ\": [[\"Ea\", \"Eb\"], [\"Ea\", \"Ec\"]]  # two alternative routes\n"
        "  }}\n"
        "A stronger claim's route should generally CONTAIN a weaker claim's "
        "route (add the extra evidence the stronger claim needs). Include only "
        "atoms genuinely necessary for each claim -- do not pad routes.\n"
        + vocab
        + ("Use ONLY these exact claim and atom ids. Answer with JSON only."
           if view == "spec" else "")
    )
    messages = [
        {"role": "system", "content": "You design evidence license programs."},
        {"role": "user", "content": instance.user_prompt + "\n" + prompt},
    ]
    last_err = "no JSON found in model output"
    for _attempt in range(1 + max_retries):
        resp = llm.chat(messages, tools=None)
        text = resp.get("content", "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                licenses = json.loads(match.group(0)).get("licenses", {})
                return _build_validated_program(instance, licenses)
            except (SchemaError, ValueError, TypeError) as exc:
                last_err = str(exc)
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": "Invalid program: %s. Emit corrected JSON only, using "
                "ONLY the exact claim and atom ids listed above." % last_err[:300],
            }
        )
    raise InducedProgramInvalid(last_err)


def _build_validated_program(instance: Instance, licenses) -> Program:
    program = make_program(
        claims=instance.claim_ids,
        order=instance.order,
        licenses=licenses,
        scope=instance.scope,
        atoms=instance.atom_ids,
    )
    validate_program(program, instance)
    return program


# family -> induced licence family (docs/EXECUTION_PLAN.md, Part 6b, FB-05).
# Backed by disk because each ARM
# is a separate run_batch process: an in-memory cache alone would still
# re-induce every family once per arm, which is 3x the model calls (19 GPU-
# hours instead of 6) for programs that are identical by construction. The
# cache is keyed by model + policy + draws so a different induction
# configuration never silently reuses another's programs.
_INDUCTION_CACHE: Dict[Any, Any] = {}
# Counterfactual-gate crashes, kept visible rather than swallowed.
_GATE_FAILURES: List[str] = []
_INDUCTION_CACHE_DIR = os.environ.get(
    "PRIME_INDUCTION_CACHE",
    os.path.join(_PILOT, "results", "induction_cache"))


def _cache_path(fam: str, model: str, policy: str, draws: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", "%s__%s__%s__d%d"
                  % (fam, model, policy, draws))
    return os.path.join(_INDUCTION_CACHE_DIR, safe + ".json")


def _cache_load(fam, model, policy, draws):
    p = _cache_path(fam, model, policy, draws)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)["licenses"]
    except (OSError, ValueError, KeyError):
        return None


def _cache_store(fam, model, policy, draws, licenses) -> None:
    # A real run always names its model. "unknown" means a stub or a test
    # LLM, and caching that would silently hand fabricated programs to a
    # later confirmatory run -- which happened: four families were poisoned
    # with spy output before this guard existed.
    if not model or model == "unknown":
        return
    try:
        os.makedirs(_INDUCTION_CACHE_DIR, exist_ok=True)
        p = _cache_path(fam, model, policy, draws)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"family": fam, "model": model, "policy": policy,
                       "draws": draws, "licenses": licenses}, fh,
                      sort_keys=True)
        os.replace(tmp, p)          # atomic: a killed job leaves no partial
    except OSError:
        pass


def _family_of(instance: Instance) -> Optional[str]:
    try:
        return json.loads(instance.notes_for_annotators or "{}").get("family")
    except (ValueError, TypeError):
        return None


def induce_program_v5(instance: Instance, llm, policy: str = "unanimous",
                      draws: int = 3):
    """W3 conservative induction (experiments/analysis/induction_v5.py) as a runtime
    license source, returning (Program, eligibility).

    The W3 stack existed but was never wired into the runner:
    ``--license-source induced`` always called the one-shot induce_program
    above, so a W2b run would have measured the LEGACY induction while the
    paper described the conservative one. Both paths end at
    _build_validated_program, so the difference is purely how the licence
    family is obtained:

      one-shot     ask the model for routes directly
      state_first  enumerate realizable evidence states, ask the model
                   whether each claim holds in each state from a SPEC-ONLY
                   description, aggregate conservatively over draws, then
                   compile minimal routes mechanically

    The commit gates run here too, so an ineligible claim is known BEFORE
    the episode: counterfactual validation and the typed linter feed
    eligibility(), and prime_guided_gated guides only toward eligible
    claims. Nothing in this path reads gold_value or gold_licenses.
    """
    sys.path.insert(0, os.path.join(_PILOT, "experiments", "analysis"))
    import induction_v5 as iv

    # Induce ONCE PER TEMPLATE FAMILY, not per instance. Within a family the
    # user prompt and the claim texts are identical and only the record
    # IDENTIFIER differs -- a locator, not support logic -- so per-instance
    # induction repeats the same elicitation 4-15 times. At 1,152 calls per
    # induction that is the difference between ~6 GPU-hours and ~154 for
    # W2b, and it also makes the clustering honest: the family is already
    # the inferential unit, so one committed program per family is what the
    # design means rather than 4-15 noisy draws of the same one.
    # Logged as freeze-break FB-05 (docs/EXECUTION_PLAN.md, Part 6b).
    fam = _family_of(instance)
    model = getattr(llm, "model", "unknown")
    key = (fam, model, policy, draws)
    licenses = _INDUCTION_CACHE.get(key) if fam else None
    if licenses is None and fam:
        licenses = _cache_load(fam, model, policy, draws)
    if licenses is None:
        licenses = iv.induce_state_first(instance, llm, policy=policy, draws=draws)
        if fam:
            _cache_store(fam, model, policy, draws, licenses)
    if fam:
        _INDUCTION_CACHE[key] = licenses
    lint = iv.lint_program(instance, licenses)
    # The argument order here was (instance, llm, licenses) against a
    # signature of (instance, licenses, llm). licenses.items() was therefore
    # called on the LLM client, every call raised AttributeError, and the
    # bare except swallowed it -- so cf was ALWAYS {} and eligibility() has
    # only ever seen lint findings. Every number this paper has reported for
    # claim eligibility and for the gated arm was produced with the
    # counterfactual gate silently absent. Found by an adversarial audit of
    # our own code, not by a test, because the except hid it from every test.
    cf = {}
    try:
        cf = iv.counterfactual_validate(instance, licenses, llm)
    except Exception as exc:               # a gate must not crash an episode
        # ...but it must not vanish either. Count it so a dead gate cannot
        # masquerade as a gate that found nothing.
        _GATE_FAILURES.append(repr(exc))
    elig = iv.eligibility(instance, licenses, lint, cf)
    return _build_validated_program(instance, licenses), elig



class EpisodeResult:
    def __init__(self) -> None:
        self.emitted: List[Dict[str, Any]] = []  # {"claim","step","ledger_version"}
        self.refused: int = 0  # mask refusals (unauthorized emit attempts)
        self.induction_eligibility = None   # W3 per-claim guidance gate
        self.induced_program_valid: Optional[bool] = None  # induced condition only
        self.tool_calls: int = 0
        self.tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self.final_text: str = ""
        self.error: str = ""
        self.ledger_states: Dict[str, str] = {}
        # Revision-freedom ablation: zero in the deployed fixed mode.
        self.revisions_proposed: int = 0
        self.revisions_accepted: int = 0
        self.drift_events: int = 0  # accepted auth-expanding revisions
        self.revision_log: List[Dict[str, Any]] = []
        # prime_guided only (obligation-guided completion -- see
        # src/prime_runtime/guidance.py): counts of the termination rule
        # actually firing. Zero for every other arm.
        self.guided_holds: int = 0  # premature emit_claim held (mask-safe)
        self.guided_stalls: int = 0  # silent-stop nudges (bounded, see GUIDED_MAX_STALLS)
        # prime_complete only: resolver calls the runtime made itself after
        # the model stopped short of the reachable ceiling (scripted
        # completion of the open obligations); zero for every other arm.
        self.discharged_calls: int = 0
        # conformal arm only: model-elicited per-claim confidences (0-100),
        # thresholded at ANALYSIS time by cross-conformal calibration
        # (experiments/analysis/build_conformal.py); {} for every other arm.
        self.claim_confidences: Dict[str, float] = {}
        # external-system arms only (extsys.py): per-episode mechanism
        # counters (model calls, rule firings, repairs, ...); {} elsewhere.
        self.ext: Dict[str, Any] = {}


def run_arm(
    arm: str,
    instance: Instance,
    llm,
    trace: TraceLogger,
    license_source: str = "gold",
    revision_mode: str = "fixed",
    max_tool_calls: int = MAX_TOOL_CALLS,
    induction_view: str = "ids",
    induction_method: str = "oneshot",   # oneshot | state_first (W3)
) -> EpisodeResult:
    """Run one episode of ``arm`` on ``instance``; logs everything to trace.

    ``max_tool_calls``: the matched action-budget cap (default 20, Table C);
    E3's cost-support frontier reruns the same arms at 4/8/16 by
    passing this explicitly -- every arm still shares the identical loop.

    ``revision_mode``: "fixed" (deployed full-freeze default) or one of the
    revision-freedom ablation modes (prime / prime_nonminimal arms only)."""
    if arm not in ARMS:
        raise ValueError("unknown arm %r" % arm)
    if revision_mode not in REVISION_MODES:
        raise ValueError("unknown revision_mode %r" % revision_mode)
    if arm.startswith("ext_"):
        # closest external systems: own control loops over the same executor,
        # resolvers, budget accounting and trace (extsys.py)
        if license_source != "gold" or revision_mode != "fixed":
            raise ValueError("external-system arms run on the committed (gold) program only")
        import extsys  # noqa: E402 -- lazy: extsys imports this module
        return extsys.run_ext_arm(arm, instance, llm, trace, max_tool_calls)
    is_prime_like = arm in ("prime", "prime_nonminimal", "prime_guided",
                            "prime_guided_gated", "masked_reactive",
                            "masked_reactive_v", "prime_complete", "prime_flat")
    # prime_flat: an evidence-gated claim system without PRIME's additions.
    # The agent may state only the strongest conclusion, gated on its route
    # being established (no graded conclusions, no residual obligations, no
    # completion); otherwise it abstains. Same prompt family as the mask-only
    # arm. This isolates what ordered authorization and guidance add over
    # "attest, then gate the finished claim".
    flat = arm == "prime_flat"
    if revision_mode != "fixed" and not is_prime_like:
        raise ValueError(
            "revision_mode %r applies to the prime / prime_nonminimal / "
            "prime_guided arms only" % revision_mode
        )
    if arm == "prime_nonminimal" and license_source == "induced":
        raise ValueError(
            "prime_nonminimal pads GOLD routes; it does not support "
            "license_source='induced'"
        )
    # prime_guided's obligation-guided completion (reachable ceiling +
    # termination rule) is only meaningful under the deployed full-freeze
    # program (revision_mode="fixed"); under an ablation revision_mode it
    # falls back to plain (unguided) prime-arm behavior below.
    # prime_guided_gated is prime_guided restricted to claims that survive the
    # W3 commit gates (docs/EXECUTION_PLAN.md, Part R, P8: gold ->
    # prime_guided, induced -> gated primary, ungated -> amplification
    # diagnostic). The ungated arm on an INDUCED program is the configuration
    # that turned 0% OFAR into 100% in an earlier run, so guiding toward a
    # claim whose routes did not survive counterfactual validation is exactly
    # what the gate exists to prevent.
    guided = arm in ("prime_guided", "prime_guided_gated", "prime_complete") and revision_mode == "fixed"
    # prime_complete = prime_guided (same prompt, same mask, same holds and
    # nudges) plus scripted completion: when the model still stops short of
    # the reachable ceiling, the runtime discharges the open obligations
    # itself before stating MaxAuth (see _discharge_obligations).
    discharge = arm == "prime_complete" and revision_mode == "fixed"
    claim_levels = prime_guidance.claim_levels(instance) if guided else {}
    result = EpisodeResult()
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    executor = ToolExecutor(instance)

    interpreter: Optional[Interpreter] = None
    checklist: Optional[Dict[str, Tuple[str, ...]]] = None
    pad_info: Optional[Dict[str, Any]] = None
    if is_prime_like:
        if arm == "prime_nonminimal":
            program, pad_info = _nonminimal_gold_program(instance)
        elif license_source == "induced":
            try:
                if induction_method == "state_first":
                    program, _elig = induce_program_v5(instance, llm)
                    result.induction_eligibility = _elig
                    if arm == "prime_guided_gated":
                        claim_levels = {c: lv for c, lv in claim_levels.items()
                                        if _elig.get(c, False)}
                else:
                    program = induce_program(instance, llm, view=induction_view)
                result.induced_program_valid = True
            except InducedProgramInvalid as exc:
                # Measured outcome, not a crash: induction
                # below the validity floor -> the episode abstains, and the
                # induction-validity rate is reported alongside the arms.
                result.induced_program_valid = False
                result.final_text = "induced program invalid: %s" % exc
                result.ledger_states = ledger.states()
                trace.log_meta(
                    instance=instance.id, arm=arm, license_source=license_source,
                    induced_program_valid=False,
                )
                return result
        else:
            program = instance.gold_program()
        # revision_mode="fixed" is the deployed full-freeze path; any other
        # value is an explicitly flagged ablation (the interpreter keeps the
        # committed program as the drift reference B_0 in every mode).
        interpreter = Interpreter(instance, program, revision_mode=revision_mode)
    elif arm == "static_contract":
        checklist = _static_contract_checklist(instance)

    meta_extra: Dict[str, Any] = {}
    if pad_info is not None:
        meta_extra["nonminimal_pad"] = pad_info
    spec_matched = arm in COMPARATOR_ARMS and _comparator_spec_view()
    if spec_matched:
        meta_extra["comparator_spec_view"] = True
    trace.log_meta(
        instance=instance.id,
        arm=arm,
        license_source=license_source,
        revision_mode=revision_mode,
        **meta_extra,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": (_comparator_system_prompt(instance)
                                       if arm in COMPARATOR_ARMS
                                       else _base_system_prompt(instance))},
        {"role": "user", "content": instance.user_prompt},
    ]
    if arm == "rubric":
        rub = llm.chat(
            messages + [{"role": "user", "content": _RUBRIC_VARIANTS[_scaffold_variant() % len(_RUBRIC_VARIANTS)]}],
            tools=None,
        )
        _add_usage(result, rub)
        messages.append({"role": "assistant", "content": rub.get("content", "")})

    actions = ["emit_claim"] if arm == "direct" else list(instance.actions)
    top_claim = max(instance.claims, key=lambda c: (c.level, c.id)).id
    schemas = tool_schemas(
        instance, actions, revision_mode=revision_mode if is_prime_like else "fixed",
        claim_choices=[top_claim] if flat else None,
    )

    step = 0
    reflected = False  # reflexion/cove: has the one critique/verify round run?
    stall_count = 0  # prime_guided: consecutive silent-stop nudges, bounded
    while result.tool_calls < max_tool_calls:
        if is_prime_like:
            if guided:
                # v5: guided also carries the spec vocabulary + routes
                # (docs/ANALYSIS_PLAN.md, F7 ladder row) so its obligations are
                # dischargeable on deep-locator domains; the guidance
                # directives come on top.
                messages[0] = {
                    "role": "system",
                    "content": _spec_vocab_block(instance)
                    + _routes_block(instance)
                    + "\n\n" + _prime_guided_system_prompt(
                        instance, interpreter, ledger,
                        max_tool_calls - result.tool_calls, claim_levels,
                        executor=executor,
                    ),
                }
            elif arm == "masked_reactive":
                # legacy arm, unchanged (v1 comparisons)
                messages[0] = {
                    "role": "system",
                    "content": _masked_reactive_system_prompt(
                        instance, interpreter, ledger, revision_mode
                    ),
                }
            elif flat:
                authorized = [c for c in interpreter.legal_claim_actions(ledger) if c == top_claim]
                messages[0] = {
                    "role": "system",
                    "content": (_ladder_system_prompt("react_program_attested", instance, ledger)
                                + "\n\nAUTHORIZED CLAIMS (the only claim emit_claim will accept; weaker "
                                  "conclusions cannot be stated in this configuration): %s"
                                % (authorized or "none")),
                }
            elif arm == "masked_reactive_v":
                # FROZEN v5 ladder (docs/ANALYSIS_PLAN.md, F7):
                # masked_reactive_v carries vocab + routes + attested ledger +
                # compliance instruction -- i.e., EXACTLY
                # react_program_attested's prompt -- plus the authorized-claims
                # gate line; the MASK is the only mechanism difference in the
                # E1 contrast. Machine-checked by
                # test_e1_contrast_is_mask_only.
                authorized = interpreter.legal_claim_actions(ledger)
                mrp = (_ladder_system_prompt("react_program_attested",
                                             instance, ledger)
                       + "\n\nAUTHORIZED CLAIMS (the only claims emit_claim "
                         "will accept): %s" % (authorized or "none"))
                messages[0] = {"role": "system", "content": mrp}
            else:
                messages[0] = {
                    "role": "system",
                    "content": _prime_system_prompt(
                        instance, interpreter, ledger, revision_mode
                    ),
                }
        elif arm == "static_contract":
            messages[0] = {
                "role": "system",
                "content": _static_contract_system_prompt(instance, checklist, ledger),
            }
        elif arm in ("react_verify", "react_spec", "react_spec_verify",
                     "react_program", "react_program_verify",
                     "react_program_attested"):
            messages[0] = {
                "role": "system",
                "content": _ladder_system_prompt(arm, instance, ledger),
            }
        resp = llm.chat(messages, tools=schemas)
        _add_usage(result, resp)
        calls = resp.get("tool_calls") or []
        if not calls:
            if guided and stall_count < GUIDED_MAX_STALLS:
                g = prime_guidance.reachable_ceiling(
                    interpreter.program, ledger, interpreter.resolvers,
                    claim_levels, max_tool_calls - result.tool_calls,
                )
                if g["must_continue"]:
                    # Termination rule: a strictly stronger claim is still
                    # reachable within budget and its obligation is open --
                    # do not accept this as the end of the episode. Nudge
                    # once (bounded by GUIDED_MAX_STALLS, a wall-clock/cost
                    # safety valve, not a relaxation of the rule itself) and
                    # let the SAME loop continue.
                    stall_count += 1
                    result.guided_stalls += 1
                    obligations = _obligation_text(instance, g, executor)
                    messages.append({"role": "assistant", "content": resp.get("content", "")})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You stopped without calling a tool or emitting a "
                                "claim. %s (level %d) is still reachable within "
                                "your remaining budget: resolve %s, or call "
                                "emit_claim with your currently authorized claim "
                                "if you intend to stop now."
                                % (g["ceiling_claim"], g["ceiling_level"], obligations)
                            ),
                        }
                    )
                    continue
            result.final_text = resp.get("content", "")
            break
        call = calls[0]
        name, args = call["name"], dict(call.get("arguments", {}))
        step += 1
        if executor.charges(name, args):
            result.tool_calls += 1
        trace.log_action(step, name, args)

        if name == "revise_program" and is_prime_like and revision_mode not in ("fixed", "posthoc"):
            kind = str(args.get("kind", ""))
            payload = args.get("payload")
            if not isinstance(payload, Mapping):
                # Some models flatten payload fields to the top level.
                payload = {k: v for k, v in args.items() if k != "kind"}
            entry = interpreter.propose_revision(kind, payload, step=step)
            trace.log(dict(entry, type="revision", ledger_version=ledger.version))
            feedback = json.dumps(
                {"accepted": entry["accepted"], "reason": entry["reason"]}
            )
            messages.append({"role": "assistant", "content": json.dumps(call)})
            messages.append(
                {"role": "user", "content": "TOOL_RESULT revise_program: %s" % feedback}
            )
            continue

        if name == "emit_claim":
            claim_id = str(args.get("claim_id", ""))
            if flat and claim_id != top_claim:
                # the flat gate exposes one conclusion; a weaker one is not
                # a legal output in this configuration (mask-equivalent
                # refusal, counted as such)
                result.refused += 1
                messages.append({"role": "assistant", "content": json.dumps(call)})
                messages.append({"role": "user", "content": (
                    "TOOL_RESULT emit_claim refused: only %s can be stated in this "
                    "configuration; weaker conclusions cannot be stated. Establish a "
                    "route to %s or stop." % (top_claim, top_claim))})
                continue
            if is_prime_like:
                if revision_mode == "posthoc":
                    accepted, feedback = _posthoc_supply_program(
                        interpreter, args, step, ledger, trace
                    )
                    if not accepted and feedback:
                        messages.append({"role": "assistant", "content": json.dumps(call)})
                        messages.append(
                            {
                                "role": "user",
                                "content": "TOOL_RESULT emit_claim: %s" % feedback,
                            }
                        )
                        continue
                try:
                    obj = interpreter.emit(claim_id, ledger)
                except UnauthorizedClaimError as exc:
                    result.refused += 1
                    feedback = "emit_claim refused: %s" % exc
                    if revision_mode == "posthoc":
                        feedback += (
                            " (in this episode emit_claim must include "
                            "`licenses`: {claim_id: [[atom_id, ...], ...]})"
                        )
                    messages.append({"role": "assistant", "content": json.dumps(call)})
                    messages.append({"role": "user", "content": "TOOL_RESULT emit_claim: %s" % feedback})
                    continue
                if guided:
                    # TERMINATION RULE (core enforcement, distinct from the
                    # auth mask above): claim_id IS authorized -- masking is
                    # unchanged and already passed -- but a strictly stronger
                    # claim may still be reachable within budget. Hold this
                    # emission (never accept it) until the ceiling is
                    # reached, the ceiling changes, or budget is exhausted.
                    g = prime_guidance.reachable_ceiling(
                        interpreter.program, ledger, interpreter.resolvers,
                        claim_levels, max_tool_calls - result.tool_calls,
                    )
                    if g["must_continue"] and claim_id != g["ceiling_claim"]:
                        result.guided_holds += 1
                        obligations = _obligation_text(instance, g, executor)
                        feedback = (
                            "emit_claim held (not accepted): %s is authorized, "
                            "but strictly stronger claim %s (level %d) is still "
                            "reachable within your remaining budget. Resolve %s "
                            "before terminating." % (
                                claim_id, g["ceiling_claim"], g["ceiling_level"], obligations
                            )
                        )
                        messages.append({"role": "assistant", "content": json.dumps(call)})
                        messages.append(
                            {"role": "user", "content": "TOOL_RESULT emit_claim: %s" % feedback}
                        )
                        continue
                trace.log_emission(step, obj, ledger.version)
                result.emitted.append(
                    {"claim": claim_id, "step": step, "ledger_version": ledger.version}
                )
            elif arm == "static_contract":
                required = checklist.get(claim_id, ())
                if not _checklist_satisfied(required, ledger):
                    result.refused += 1
                    c = _closure.cl(ledger)
                    missing = required if _closure.is_inconsistent(c) else [
                        lit for lit in required if not _closure.entails(c, lit)
                    ]
                    feedback = (
                        "emit_claim refused: static checklist incomplete for %s; "
                        "outstanding items: %s" % (claim_id, list(missing))
                    )
                    messages.append({"role": "assistant", "content": json.dumps(call)})
                    messages.append({"role": "user", "content": "TOOL_RESULT emit_claim: %s" % feedback})
                    continue
                obj = {
                    "authorized_claims": [claim_id],
                    "unmasked": False,
                    "checklist_gated": True,
                }
                trace.log_emission(step, obj, ledger.version, claim_ids=[claim_id])
                result.emitted.append(
                    {"claim": claim_id, "step": step, "ledger_version": ledger.version}
                )
            elif arm in ("reflexion", "cove") and not reflected:
                # Draft claim: run the one self-critique / verification-then-
                # revise round, then let the SAME tool loop continue
                # (further tool calls allowed) until the model re-emits, which
                # is then accepted as the final answer below.
                reflected = True
                messages.append({"role": "assistant", "content": json.dumps(call)})
                _reflect_or_verify(arm, claim_id, messages, llm, result)
                continue
            else:
                obj = {"authorized_claims": [claim_id], "unmasked": True}
                trace.log_emission(step, obj, ledger.version, claim_ids=[claim_id])
                result.emitted.append(
                    {"claim": claim_id, "step": step, "ledger_version": ledger.version}
                )
            break  # terminal claim ends the episode

        observation = executor.execute(name, args)
        trace.log_observation(step, name, args, observation)
        attestations = goldledger.delta(instance, ledger, name, args, observation, step)
        for att in attestations:
            trace.log_attestation(step, att, ledger.version)
        messages.append({"role": "assistant", "content": json.dumps(call)})
        messages.append(
            {
                "role": "user",
                "content": "TOOL_RESULT %s: %s" % (name, json.dumps(observation, default=str)[:2000]),
            }
        )

    if discharge and not result.emitted and result.tool_calls < max_tool_calls:
        step = _discharge_obligations(instance, interpreter, ledger, executor, trace,
                                      result, claim_levels, max_tool_calls, step)

    if is_prime_like and not result.emitted:
        # Budget exhausted / model stopped: emit MaxAuth antichain,
        # or abstain when nothing is authorized.
        maximal = prime_auth.max_auth(interpreter.program, ledger)
        if flat:
            maximal = [c for c in maximal if c == top_claim]
        if maximal:
            step += 1
            obj = interpreter.emit_max_auth(ledger)
            trace.log_emission(step, obj, ledger.version)
            for cid in maximal:
                result.emitted.append(
                    {"claim": cid, "step": step, "ledger_version": ledger.version}
                )

    if not result.emitted:
        # Forced claim selection — method-independent emission capture.
        # Models (especially small ones) often answer in prose without calling
        # emit_claim; every arm's final assertion is mapped to claim IDs by
        # the SAME procedure so baseline gap rates are not silently deflated.
        # For the prime/prime_nonminimal arms the selections still pass
        # through the mask; for static_contract they still pass the checklist
        # gate (else the gate would be vacuous on a common fallback path).
        step += 1
        _force_claim_selection(
            instance, llm, messages, result, trace, step, ledger, interpreter, arm,
            revision_mode, checklist=checklist,
            claim_choices=[top_claim] if flat else None,
        )

    if is_prime_like and interpreter is not None:
        result.revisions_proposed = interpreter.revisions_proposed
        result.revisions_accepted = interpreter.revisions_accepted
        result.drift_events = interpreter.drift_events
        result.revision_log = [dict(e) for e in interpreter.revision_log]

    if arm == "conformal":
        _elicit_claim_confidences(instance, messages, llm, result)
    result.ledger_states = ledger.states()
    return result


def _discharge_obligations(instance, interpreter, ledger, executor, trace, result,
                           levels, max_tool_calls, step):
    """Scripted completion (the prime_complete arm).

    Every residual obligation of the committed program names a registered
    resolver call, so once the model has stopped short of the reachable
    ceiling the runtime can make those calls itself: at each step it takes
    the guidance's own target (the strongest conclusion still reachable
    within the remaining budget) and resolves the first open atom of that
    conclusion's cheapest live route, until the target is authorized, the
    budget is spent, or no executable obligation remains. An atom whose
    locator is withheld (``hint`` set) is discharged by enumeration: the
    candidates the model has not read are tried in the listed order, one
    call each, the same rule as the deterministic candidate policy; the
    model's reads come first, so its field choices are kept. The mask is
    untouched; the ledger is written by the same Delta as for the model's
    own calls, and every call is logged as a runtime action."""
    by_id = {a.id: a for a in instance.atoms}
    tried = set()          # fully specified atoms already called once
    exhausted = set()      # locator-uncertain atoms whose candidates are all read
    while result.tool_calls < max_tool_calls:
        g = prime_guidance.reachable_ceiling(
            interpreter.program, ledger, interpreter.resolvers, levels,
            max_tool_calls - result.tool_calls,
        )
        if not g["must_continue"]:
            break
        nxt, params = None, None
        for lit in g["blocking_atoms"]:
            a = by_id.get(atom_of(lit))
            if a is None or a.resolver is None or a.id in tried or a.id in exhausted:
                continue
            hint = getattr(a, "hint", None)
            if a.resolver.tool == "attest_atom":
                # a text-evidence atom is resolved by the model's reading;
                # the runtime cannot attest it
                exhausted.add(a.id)
                continue
            if hint:
                # locator withheld: the runtime cannot read the description,
                # so it enumerates the candidates the model has not read, in
                # the listed order, one call each (the blind policy's rule)
                fixed = {k: v for k, v in dict(a.resolver.params).items() if k != "field_path"}
                untried = [c for c in hint.get("candidates", [])
                           if not executor.was_called(a.resolver.tool, dict(fixed, field_path=c))]
                if not untried:
                    exhausted.add(a.id)
                    continue
                nxt, params = a, dict(fixed, field_path=untried[0])
            else:
                nxt, params = a, dict(a.resolver.params)
                tried.add(a.id)
            break
        if nxt is None:
            break
        step += 1
        trace.log_action(step, nxt.resolver.tool, params)
        trace.log({"type": "discharge", "step": step, "atom": nxt.id,
                   "target": g["ceiling_claim"], "tool": nxt.resolver.tool,
                   "params": params})
        observation = executor.execute(nxt.resolver.tool, params)
        if not executor.last_was_cached:
            result.tool_calls += 1
            result.discharged_calls += 1
        trace.log_observation(step, nxt.resolver.tool, params, observation)
        for att in goldledger.delta(instance, ledger, nxt.resolver.tool, params, observation, step):
            trace.log_attestation(step, att, ledger.version)
    return step


def _posthoc_supply_program(interpreter, args, step, ledger, trace):
    """Post-hoc mode: install the emission-time program supplied with
    emit_claim (validated for well-formedness only — the post-hoc selector).

    Returns (proceed, feedback): proceed=False with feedback when a supplied
    program was rejected as ill-formed; proceed=True otherwise (with or
    without a supplied program — absent licenses simply leave the current
    program, and the mask, in place)."""
    lic = args.get("licenses")
    if isinstance(lic, str):
        try:
            lic = json.loads(lic)
        except ValueError:
            return False, "ill-formed `licenses` (not valid JSON)"
    if lic is None:
        return True, ""
    if not isinstance(lic, Mapping):
        return False, "ill-formed `licenses` (must be {claim_id: [[atom_id,...],...]})"
    entry = interpreter.propose_revision(
        "replace_program", {"licenses": lic}, step=step
    )
    trace.log(dict(entry, type="revision", ledger_version=ledger.version))
    if not entry["accepted"]:
        return False, "license program rejected: %s" % entry["reason"]
    return True, ""


def _force_claim_selection(
    instance, llm, messages, result, trace, step, ledger, interpreter, arm,
    revision_mode="fixed", checklist=None, claim_choices=None,
):
    """One strict-JSON follow-up mapping the arm's final answer to claim IDs.

    ``claim_choices`` restricts which claims may be chosen (the ``graded``
    ablation factor, Table E: answer/abstain only offers the
    top-level claim)."""
    is_prime_like = arm in ("prime", "prime_nonminimal", "prime_guided", "prime_guided_gated",
                            "masked_reactive", "masked_reactive_v", "prime_complete", "prime_flat")
    eligible = [c for c in instance.claims if claim_choices is None or c.id in claim_choices]
    ids = [c.id for c in eligible]
    prompt = (
        "Based on your assessment above, which of these claims do you assert? "
        "Claims: %s. Respond with ONLY JSON: {\"claims\": [<ids you assert>]} "
        "(empty list to abstain)." % json.dumps({c.id: c.text for c in eligible})
    )
    if is_prime_like and revision_mode == "posthoc":
        # Post-hoc selector: the program licensing the asserted
        # claims is supplied together with the assertion.
        prompt = (
            "Based on your assessment above, which of these claims do you "
            "assert? Claims: %s. Respond with ONLY JSON: "
            "{\"claims\": [<ids you assert>], \"licenses\": {<claim id>: "
            "[[<evidence atom id>, ...], ...]}} — `licenses` is the license "
            "program under which your claims are licensed (empty claims list "
            "to abstain)." % json.dumps({c.id: c.text for c in eligible})
        )
    resp = llm.chat(messages + [{"role": "user", "content": prompt}], tools=None)
    _add_usage(result, resp)
    text = resp.get("content", "") or ""
    # Accept {"claims": [...]} as well as a bare JSON list ["C1", ...] —
    # small models frequently answer with the bare-list variant, and dropping
    # it would silently lose real assertions (measurement-validity bug).
    m = re.search(r"\{.*\}", text, re.S) or re.search(r"\[.*\]", text, re.S)
    if not m:
        return
    try:
        parsed = json.loads(m.group(0))
        raw = parsed.get("claims", []) if isinstance(parsed, dict) else parsed
        chosen = [c for c in (raw or []) if c in ids]
    except (ValueError, AttributeError, TypeError):
        return
    if (
        is_prime_like
        and revision_mode == "posthoc"
        and interpreter is not None
        and isinstance(parsed, dict)
        and chosen
    ):
        lic = parsed.get("licenses")
        if isinstance(lic, Mapping) and lic:
            entry = interpreter.propose_revision(
                "replace_program", {"licenses": lic}, step=step
            )
            trace.log(dict(entry, type="revision", ledger_version=ledger.version))
    for cid in chosen:
        if is_prime_like and interpreter is not None:
            try:
                obj = interpreter.emit(cid, ledger)
            except UnauthorizedClaimError:
                result.refused += 1
                continue
            trace.log_emission(step, obj, ledger.version)
        elif arm == "static_contract" and checklist is not None:
            # The gate applies on the forced-selection fallback too — else it
            # would be vacuous whenever the model never calls emit_claim.
            required = checklist.get(cid, ())
            if not _checklist_satisfied(required, ledger):
                result.refused += 1
                continue
            obj = {
                "authorized_claims": [cid],
                "unmasked": False,
                "checklist_gated": True,
                "forced_selection": True,
            }
            trace.log_emission(step, obj, ledger.version, claim_ids=[cid])
        else:
            obj = {"authorized_claims": [cid], "unmasked": True, "forced_selection": True}
            trace.log_emission(step, obj, ledger.version, claim_ids=[cid])
        result.emitted.append(
            {"claim": cid, "step": step, "ledger_version": ledger.version}
        )


def _elicit_claim_confidences(instance, messages, llm, result: "EpisodeResult") -> None:
    """conformal arm: one final non-tool turn eliciting a 0-100 confidence
    per candidate claim ("probability the claim is fully supported by the
    evidence you gathered"). The arm itself never thresholds -- calibration
    and the emission decision happen in experiments/analysis/build_conformal.py
    (leave-one-out cross-conformal against the executable oracle's labels),
    matching how the conformal-factuality family separates scoring from
    calibrated selection. Parse failure stores {} (measured, not a crash)."""
    ids = list(instance.claim_ids)
    prompt = (
        "For each candidate claim, give the probability (0-100) that it is "
        "FULLY supported by the evidence you actually gathered with tools "
        "in this session. Claims: %s. Respond with ONLY JSON: "
        '{"confidences": {"<claim id>": <0-100 number>, ...}}.'
        % json.dumps({c.id: c.text for c in instance.claims})
    )
    try:
        resp = llm.chat(messages + [{"role": "user", "content": prompt}], tools=None)
        _add_usage(result, resp)
        m = re.search(r"\{.*\}", resp.get("content", "") or "", re.DOTALL)
        raw = json.loads(m.group(0)).get("confidences", {}) if m else {}
        result.claim_confidences = {
            str(k): float(v) for k, v in raw.items()
            if str(k) in ids and isinstance(v, (int, float))
        }
    except Exception:  # noqa: BLE001 -- elicitation failure is a measured {}
        result.claim_confidences = {}


# Scaffold-sensitivity variants (an adapter of our design could test our
# choices rather than the original systems). PRIME_SCAFFOLD_VARIANT selects
# among meaning-preserving paraphrases of the self-standard arms' fixed
# instruction strings; variant 0 is BYTE-IDENTICAL to the originally-run
# scaffolds (pinned by test_scaffold_variants.py), so every published number is
# unchanged. Running the same arms under variants 1-2 turns "our prompt
# choices" from a confound into a measured sensitivity band.
def _scaffold_variant() -> int:
    try:
        return int(os.environ.get("PRIME_SCAFFOLD_VARIANT", "0"))
    except ValueError:
        return 0


_RUBRIC_VARIANTS = (
    "First, list the evaluation criteria that apply.",
    "Before doing anything else, enumerate the criteria you will judge this by.",
    "Start by writing out the standards relevant to this assessment.",
)

_REFLECT_PROMPT_VARIANTS = {
    "reflexion": (
        (  # variant 0: BYTE-IDENTICAL to the originally-run scaffold
            "Self-critique: for the claim you just proposed (%s), list any part "
            "of your reasoning that is not yet directly supported by evidence "
            "you have gathered with tools. Be specific about what is missing "
            "or merely assumed.",
            "Given that self-critique, revise your assessment. You may use "
            "tools again before finalizing. When ready, call emit_claim with "
            "your final claim id.",
        ),
        (
            "Critically examine the claim you just proposed (%s): which steps "
            "of your reasoning rest on evidence you actually collected with "
            "tools, and which are assumptions? Name the gaps explicitly.",
            "Now revise your assessment in light of that critique; further "
            "tool use is allowed. Finalize by calling emit_claim with your "
            "final claim id.",
        ),
        (
            "Review your proposed claim (%s) for unsupported reasoning: "
            "identify every statement not yet backed by a tool observation "
            "from this session, as concretely as you can.",
            "Update your assessment accordingly (you may gather more "
            "evidence first), then call emit_claim with your final claim id.",
        ),
        # variant 3: the VERBATIM released Reflexion prompt
        # (github.com/noahshinn/reflexion, hotpotqa_runs/prompts.py, fetched
        # 2026-07-30), with exactly two environment-noun substitutions,
        # each marked [orig: ...]: "tool-call environment" [orig: "an
        # Docstore API environment"] and "an evaluative request" [orig: "a
        # question to answer"], plus one appended sentence binding their
        # "previous reasoning trial" framing to this harness's draft claim.
        # Everything else, including the original's typo "self refection",
        # is theirs, untouched.
        (
            "You are an advanced reasoning agent that can improve based on "
            "self refection. You will be given a previous reasoning trial in "
            "which you were given access to a tool-call environment and an "
            "evaluative request. You were unsuccessful in answering the "
            "question either because you guessed the wrong answer with "
            "Finish[<answer>], or you used up your set number of reasoning "
            "steps. In a few sentences, Diagnose a possible reason for "
            "failure and devise a new, concise, high level plan that aims to "
            "mitigate the same failure. Use complete sentences. The trial is "
            "your reasoning so far toward the draft claim %s, which may be "
            "unsupported.",
            "Following your plan, revise your assessment. You may use tools "
            "again before finalizing. When ready, call emit_claim with your "
            "final claim id.",
        ),
    ),
    "cove": (
        (  # variant 0: BYTE-IDENTICAL to the originally-run scaffold
            "Chain-of-verification: for the claim you just proposed (%s), write "
            "2-4 specific verification questions whose answers would confirm or "
            "refute it, each answerable by one of the available tools.",
            "Answer each verification question using the available tools, then "
            "revise your assessment if any answer contradicts the draft claim. "
            "When ready, call emit_claim with your final claim id.",
        ),
        (
            "For your draft claim (%s), pose 2-4 concrete verification "
            "questions -- each checkable with one of the available tools -- "
            "whose answers would confirm or refute the claim.",
            "Work through those questions with the tools, revise your "
            "assessment if anything contradicts the draft, and finish by "
            "calling emit_claim with your final claim id.",
        ),
        (
            "Write 2-4 targeted questions that would verify or falsify the "
            "claim you just proposed (%s), each answerable via an available "
            "tool.",
            "Resolve each question using the tools; if any answer conflicts "
            "with the draft, revise. Then call emit_claim with your final "
            "claim id.",
        ),
    ),
}


def _reflect_or_verify(arm: str, claim_id: str, messages, llm, result: "EpisodeResult") -> None:
    """The one self-critique (reflexion) / verification-question (cove)
    round: two plain (non-tool) chat turns appended to ``messages`` in
    place. Neither turn counts against the shared tool-call budget — only
    the draft and the eventual final emit_claim do, matching how every other
    arm's terminal emit_claim is the thing that is budget-matched."""
    variants = _REFLECT_PROMPT_VARIANTS[arm]
    critique_prompt_tmpl, revise_prompt = variants[_scaffold_variant() % len(variants)]
    critique_prompt = critique_prompt_tmpl % claim_id
    resp = llm.chat(messages + [{"role": "user", "content": critique_prompt}], tools=None)
    _add_usage(result, resp)
    messages.append({"role": "user", "content": critique_prompt})
    messages.append({"role": "assistant", "content": resp.get("content", "")})
    messages.append({"role": "user", "content": revise_prompt})


def _add_usage(result: EpisodeResult, resp: Mapping[str, Any]) -> None:
    usage = resp.get("usage", {}) or {}
    result.tokens["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
    result.tokens["completion_tokens"] += int(usage.get("completion_tokens", 0))


# ---------------------------------------------------------------------------
# 2^4 mechanism-interaction ablation (Table E)
# ---------------------------------------------------------------------------

# Fractional-factorial half-design generator I = A*B*C*D (resolution IV: no
# main effect is aliased with any 2-factor interaction; 2-factor interactions
# are aliased only in pairs). The 8 cells below are the 3 free factors
# (minimal, frozen, masked) crossed fully, with graded set by the generator
# graded = minimal XOR frozen XOR masked. Kept as an explicit table (not
# computed ad hoc at call sites) so the design is auditable in one place.
ABLATION_CELLS: Tuple[Dict[str, bool], ...] = tuple(
    {
        "minimal": bool(minimal),
        "frozen": bool(frozen),
        "masked": bool(masked),
        "graded": bool(minimal ^ frozen ^ masked),
    }
    for minimal in (0, 1)
    for frozen in (0, 1)
    for masked in (0, 1)
)


def run_prime_ablation(
    instance: Instance,
    llm,
    trace: TraceLogger,
    *,
    minimal: bool,
    frozen: bool,
    masked: bool,
    graded: bool,
    license_source: str = "gold",
    max_tool_calls: int = MAX_TOOL_CALLS,
    induction_view: str = "ids",
) -> EpisodeResult:
    """One cell of the 2^4 mechanism-interaction ablation (Table E: minimal
    routes vs. redundant routes; frozen vs. revisable program; emission mask
    vs. prompt-only compliance; graded lattice vs. answer/abstain). Each factor
    is a concrete config toggle:

        minimal   True  -> PRIME's own minimal gold routes (arm ``prime``)
                  False -> routes padded with 1-2 redundant literals
                           (``prime_nonminimal``'s ``_nonminimal_gold_program``)
        frozen    True  -> deployed full in-episode freeze
                           (``revision_mode="fixed"``)
                  False -> allow posthoc revision (``revision_mode="posthoc"``)
        masked    True  -> interpreter mask enforced: emit_claim raises
                           UnauthorizedClaimError when the claim is not
                           authorized (refused, tool loop continues)
                  False -> prompt-only compliance: the SAME obligation-
                           frontier information is shown in the system
                           prompt, but emit_claim is NOT gated (accepted
                           pass-through, exactly like a baseline arm)
        graded    True  -> the full claim lattice is offered, with the
                           MaxAuth antichain fallback at budget exhaustion
                  False -> answer/abstain only: only the top-level claim is
                           offered; no partial-credit fallback (abstain if
                           it is not authorized/would not be emitted)

    A separate function from ``run_arm`` (duplicating its loop rather than
    threading these 4 booleans through it) so Table C's ``prime`` /
    ``prime_nonminimal`` results are provably unaffected by this harness.
    Reuses the same ToolExecutor / tool_schemas / Interpreter / goldledger
    building blocks and the identical scoring path downstream (run_episode.py
    replays the trace through goldledger.reconstruct regardless of which
    function produced it).
    """
    result = EpisodeResult()
    ledger = Ledger(instance.atom_ids, instance.phi_patterns)
    executor = ToolExecutor(instance)
    revision_mode = "fixed" if frozen else "posthoc"

    pad_info: Optional[Dict[str, Any]] = None
    if minimal:
        if license_source == "induced":
            try:
                program = induce_program(instance, llm, view=induction_view)
                result.induced_program_valid = True
            except InducedProgramInvalid as exc:
                result.induced_program_valid = False
                result.final_text = "induced program invalid: %s" % exc
                result.ledger_states = ledger.states()
                trace.log_meta(
                    instance=instance.id, arm="prime_ablation", license_source=license_source,
                    induced_program_valid=False, minimal=minimal, frozen=frozen,
                    masked=masked, graded=graded,
                )
                return result
        else:
            program = instance.gold_program()
    else:
        program, pad_info = _nonminimal_gold_program(instance)

    interpreter = Interpreter(instance, program, revision_mode=revision_mode)
    top_claim = max(instance.claims, key=lambda c: (c.level, c.id))
    claim_choices = None if graded else [top_claim.id]

    meta: Dict[str, Any] = dict(
        instance=instance.id, arm="prime_ablation", license_source=license_source,
        revision_mode=revision_mode, minimal=minimal, frozen=frozen, masked=masked,
        graded=graded,
    )
    if pad_info is not None:
        meta["nonminimal_pad"] = pad_info
    trace.log_meta(**meta)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _base_system_prompt(instance)},
        {"role": "user", "content": instance.user_prompt},
    ]
    schemas = tool_schemas(
        instance, list(instance.actions), revision_mode=revision_mode, claim_choices=claim_choices
    )
    unmasked_note = (
        "" if masked else
        "\n\n(NOTE: this episode's emit_claim is NOT enforced against the "
        "above license state this cell -- compliance is voluntary, per the "
        "'masked' ablation factor.)"
    )

    step = 0
    while result.tool_calls < max_tool_calls:
        messages[0] = {
            "role": "system",
            "content": _prime_system_prompt(instance, interpreter, ledger, revision_mode) + unmasked_note,
        }
        resp = llm.chat(messages, tools=schemas)
        _add_usage(result, resp)
        calls = resp.get("tool_calls") or []
        if not calls:
            result.final_text = resp.get("content", "")
            break
        call = calls[0]
        name, args = call["name"], dict(call.get("arguments", {}))
        step += 1
        if executor.charges(name, args):
            result.tool_calls += 1
        trace.log_action(step, name, args)

        if name == "emit_claim":
            claim_id = str(args.get("claim_id", ""))
            if masked:
                if revision_mode == "posthoc":
                    accepted, feedback = _posthoc_supply_program(
                        interpreter, args, step, ledger, trace
                    )
                    if not accepted and feedback:
                        messages.append({"role": "assistant", "content": json.dumps(call)})
                        messages.append(
                            {"role": "user", "content": "TOOL_RESULT emit_claim: %s" % feedback}
                        )
                        continue
                try:
                    obj = interpreter.emit(claim_id, ledger)
                except UnauthorizedClaimError as exc:
                    result.refused += 1
                    feedback = "emit_claim refused: %s" % exc
                    messages.append({"role": "assistant", "content": json.dumps(call)})
                    messages.append({"role": "user", "content": "TOOL_RESULT emit_claim: %s" % feedback})
                    continue
                trace.log_emission(step, obj, ledger.version)
            else:
                would_be = interpreter.legal_claim_actions(ledger)
                obj = {
                    "authorized_claims": [claim_id],
                    "unmasked": True,
                    "would_be_authorized": claim_id in would_be,
                }
                trace.log_emission(step, obj, ledger.version, claim_ids=[claim_id])
            result.emitted.append(
                {"claim": claim_id, "step": step, "ledger_version": ledger.version}
            )
            break

        observation = executor.execute(name, args)
        trace.log_observation(step, name, args, observation)
        attestations = goldledger.delta(instance, ledger, name, args, observation, step)
        for att in attestations:
            trace.log_attestation(step, att, ledger.version)
        messages.append({"role": "assistant", "content": json.dumps(call)})
        messages.append(
            {
                "role": "user",
                "content": "TOOL_RESULT %s: %s" % (name, json.dumps(observation, default=str)[:2000]),
            }
        )

    if not result.emitted:
        if masked and graded:
            maximal = prime_auth.max_auth(interpreter.program, ledger)
            if maximal:
                step += 1
                obj = interpreter.emit_max_auth(ledger)
                trace.log_emission(step, obj, ledger.version)
                for cid in maximal:
                    result.emitted.append(
                        {"claim": cid, "step": step, "ledger_version": ledger.version}
                    )
        elif masked and not graded:
            # answer/abstain only: no partial-credit antichain fallback.
            if top_claim.id in interpreter.legal_claim_actions(ledger):
                step += 1
                obj = interpreter.emit(top_claim.id, ledger)
                trace.log_emission(step, obj, ledger.version)
                result.emitted.append(
                    {"claim": top_claim.id, "step": step, "ledger_version": ledger.version}
                )
            # else: abstain -- the defining behavior of graded=False.

    if not result.emitted:
        step += 1
        _force_claim_selection(
            instance, llm, messages, result, trace, step, ledger,
            interpreter if masked else None,
            "prime" if masked else "react",  # arm label controls gating only
            revision_mode, claim_choices=claim_choices,
        )

    result.revisions_proposed = interpreter.revisions_proposed
    result.revisions_accepted = interpreter.revisions_accepted
    result.drift_events = interpreter.drift_events
    result.revision_log = [dict(e) for e in interpreter.revision_log]
    result.ledger_states = ledger.states()
    return result
