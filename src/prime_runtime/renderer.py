"""Constrained renderer for typed claim objects.

Every evaluative sentence must reference an authorized claim ID; the
renderer cannot introduce evaluative propositions without one.  The pilot
renderer is fully deterministic (template-based), so rendering fidelity is
exact by construction.
"""

from typing import Any, Dict, Mapping

from .schema import Instance


class RendererError(ValueError):
    """Claim object references a claim id with no registered content."""


def render(claim_object: Mapping[str, Any], claim_texts: Mapping[str, str]) -> str:
    """Render a typed claim object to natural language.

    ``claim_texts`` maps claim id -> benchmark-authored claim content
    (T1 trusted base: contents are fixed, not model-generated).
    """
    lines = []
    for cid in claim_object.get("authorized_claims", ()):
        if cid not in claim_texts:
            raise RendererError("no registered content for claim id %r" % cid)
        lines.append("[%s] %s" % (cid, claim_texts[cid]))
    if not lines:
        lines.append("No claim is authorized under the committed program; abstaining.")
    evidence = claim_object.get("evidence", ())
    if evidence:
        lines.append("Evidence: %s." % ", ".join(evidence))
    for item in claim_object.get("unresolved_claims", ()):
        cid = item["id"]
        text = claim_texts.get(cid, cid)
        lines.append(
            "Not authorized: [%s] %s — residual obligations: %s."
            % (cid, text, ", ".join(item.get("residual_obligations", ())) or "none")
        )
    scope = claim_object.get("scope", "")
    if scope:
        lines.append("Scope: %s." % scope)
    for q in claim_object.get("qualifications", ()):
        lines.append("Qualification: %s" % q)
    return "\n".join(lines)


def render_for_instance(claim_object: Mapping[str, Any], instance: Instance) -> str:
    return render(claim_object, {c.id: c.text for c in instance.claims})
