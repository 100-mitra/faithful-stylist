"""GroundingVerifier — the headline feature (deterministic faithfulness veto).

The LLM has no channel to emit a factual *value*: a ``RationaleDraft`` only names which
facts to surface (``FactRef``) and writes short opinion clauses. This module:

  * renders factual text from the catalog record in code (``render_factual``), and
  * audits every claim with a pure, network-free, LLM-free ``GroundingVerifier``:
      A. each referenced factual field must be present/true on the record,
      B. the draft's product id must match the candidate, and
      C. no number/currency/metal/stone/certification fact (expressed in the catalog's
         controlled vocabulary) may be smuggled into an opinion clause.

Any blocked factual claim is removed from the user-facing rationale. Within the templated
factual path the LLM has no value channel at all, so a fabricated price/metal/stone/carat
cannot appear there. We bias toward over-blocking: a false-positive block is safe; a
false-negative is the project-killer.

SCOPE / KNOWN LIMITS (do not overstate this): check C recognises only the controlled
vocabulary (a fixed metal/stone/cert set + a numeric/currency/units regex). A factual term
OUTSIDE that vocabulary (e.g. moissanite, palladium, titanium, lab-grown, gold-plated, an
out-of-list stone) is NOT detected in opinion text. Opinion substance beyond smuggled facts
(provenance, superlatives, value claims) is also unchecked — the opinion label bounds
framing, not factual accuracy. See README > Limitations.
"""

from __future__ import annotations

import re

from core.models import (
    FactRef,
    PreferenceProfile,
    Product,
    RationaleClaim,
    RationaleDraft,
    VerifierResult,
)
from core.vocab import CERT_TERMS, METAL_BASE_TOKENS, METALS, OCCASIONS, STONES, STYLES

# Any digit, currency token, or measurement unit appearing in an opinion clause is a
# smuggled fact — opinions never need them; all numbers flow through FactRef rendering.
_FACT_LEAK = re.compile(r"\d|₹|\$|\b(?:rs|inr|usd|ct|carat|carats|karat|kt|mm|gram|grams)\b", re.I)


# ---------------------------------------------------------------------------
# Factual rendering (code-generated from the record — never the LLM)
# ---------------------------------------------------------------------------
def _format_price(product: Product) -> str:
    if product.currency == "INR":
        return f"₹{product.price:,}"
    return f"{product.price:,} {product.currency}"


def render_factual(field: str, product: Product, profile: PreferenceProfile) -> str | None:
    """Render one factual phrase from the record, or None if unsupported/false.

    Returning None means the LLM referenced a fact the item does not have (e.g. a
    certification it lacks) — the verifier blocks that claim.
    """
    if field == "price":
        return _format_price(product)
    if field == "metal":
        return product.metal
    if field == "stone_primary":
        return product.stone_primary or None
    if field == "stone_accent":
        return f"{product.stone_accent} accents" if product.stone_accent else None
    if field == "carat":
        return f"{product.carat:g} ct" if product.carat is not None else None
    if field == "certification":
        return f"{product.certification}-certified" if product.certification else None
    if field == "category":
        return product.category
    if field == "under_budget":
        if profile.budget_max is not None and product.price <= profile.budget_max:
            return "within your budget"
        return None
    return None


def _scan_smuggled_fact(clause: str, product: Product) -> str | None:
    """Return a block reason if an opinion clause contains an unsupported factual token."""
    low = clause.lower()
    if _FACT_LEAK.search(low):
        return "numeric/measurement/currency value in opinion text"

    record_metal = product.metal.lower()
    # Full metal phrases first, then base tokens.
    for phrase in sorted(METALS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            if phrase not in record_metal and record_metal not in phrase:
                return f"metal '{phrase}' contradicts record ({product.metal})"
    for token in METAL_BASE_TOKENS:
        if re.search(rf"\b{token}\b", low) and token not in record_metal:
            return f"metal '{token}' contradicts record ({product.metal})"

    record_stones = {s for s in (product.stone_primary, product.stone_accent) if s}
    for stone in STONES:
        if re.search(rf"\b{stone}\b", low) and stone not in record_stones:
            return f"stone '{stone}' not on record"

    for term in CERT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", low) and not product.certification:
            return f"certification claim '{term}' but record is uncertified"
    return None


def _has_recognized_style(clause: str) -> bool:
    low = clause.lower()
    vocab = STYLES + OCCASIONS
    return any(re.search(rf"\b{re.escape(w)}\b", low) for w in vocab)


def _assemble(factual: list[str], subjective: list[str], recommended: bool) -> str:
    lead = "Recommended" if recommended else "Considered"
    parts = [f"{lead}: " + ", ".join(factual) + "." if factual else f"{lead}."]
    if subjective:
        parts.append("Style notes (my opinion): " + "; ".join(subjective) + ".")
    return " ".join(parts)


class GroundingVerifier:
    """Deterministic faithfulness veto. Pure: no network, no randomness, no LLM."""

    def verify(
        self,
        draft: RationaleDraft,
        product: Product,
        profile: PreferenceProfile,
    ) -> VerifierResult:
        claims: list[RationaleClaim] = []
        decision = "passed"

        # Check B — product identity.
        if draft.product_id != product.id:
            decision = "blocked"
            claims.append(
                RationaleClaim(
                    claim_text=f"recommends product {draft.product_id}",
                    claim_type="factual",
                    grounded=False,
                    blocked=True,
                    source_field="product_id",
                    note=f"product id mismatch (record is {product.id})",
                )
            )

        # Check A — factual slot validity.
        allowed_factual: list[str] = []
        for ref in draft.factual_slots:
            field = ref.field if isinstance(ref, FactRef) else ref["field"]
            text = render_factual(field, product, profile)
            if text is None:
                decision = "blocked"
                claims.append(
                    RationaleClaim(
                        claim_text=f"claims {field}",
                        claim_type="factual",
                        grounded=False,
                        blocked=True,
                        source_field=field,
                        note=f"{field} is not present/true on the record",
                    )
                )
            else:
                claims.append(
                    RationaleClaim(
                        claim_text=text,
                        claim_type="factual",
                        grounded=True,
                        source_field=field,
                    )
                )
                allowed_factual.append(text)

        # Check C — smuggled-fact scan over opinion clauses.
        allowed_subjective: list[str] = []
        for clause in draft.subjective_clauses:
            reason = _scan_smuggled_fact(clause, product)
            if reason is not None:
                decision = "blocked"
                claims.append(
                    RationaleClaim(
                        claim_text=clause,
                        claim_type="subjective",
                        grounded=False,
                        blocked=True,
                        note=reason,
                    )
                )
            else:
                flagged = not _has_recognized_style(clause)
                claims.append(
                    RationaleClaim(
                        claim_text=clause,
                        claim_type="subjective",
                        grounded=True,
                        blocked=False,
                        note="unrecognised style vocabulary" if flagged else None,
                    )
                )
                allowed_subjective.append(clause)

        rationale_text = _assemble(allowed_factual, allowed_subjective, draft.recommended)
        return VerifierResult(decision=decision, claims=claims, rationale_text=rationale_text)


def render_rationale(
    draft: RationaleDraft,
    product: Product,
    profile: PreferenceProfile,
) -> str:
    """Convenience: the safe, user-facing rationale text (blocked claims removed)."""
    return GroundingVerifier().verify(draft, product, profile).rationale_text
