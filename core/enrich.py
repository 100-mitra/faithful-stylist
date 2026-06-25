"""Attribute enrichment: the LLM reads title/description and emits subjective StyleTags
via a strict Pydantic schema. Inferred labels are constrained to the controlled vocab and
kept separate from the factual Product record.

Tagging accuracy is measured honestly: on synthetic data we report agreement with the
generator's ``style_hint`` (a pseudo-label, NOT a human label) — captioned as such.
"""

from __future__ import annotations

from core.llm import LLMProvider, get_provider
from core.models import Product, StyleTagDraft, StyleTags
from core.vocab import OCCASIONS, STYLES

_PROVIDER_VERSION = "v1"


def _prompt(product: Product) -> str:
    desc = (product.raw_attributes or {}).get("description", "")
    return (
        "Infer subjective style/occasion tags for this jewellery item. "
        f"Choose 1-3 styles from {STYLES} and 1-3 occasions from {OCCASIONS}. "
        "Do not restate factual attributes; give a short aesthetic note as opinion.\n"
        f"Title: {product.title}\nDescription: {desc}"
    )


def enrich_product(product: Product, provider: LLMProvider | None = None) -> StyleTags:
    provider = provider or get_provider()
    draft: StyleTagDraft = provider.complete_structured(
        task="enrich",
        schema=StyleTagDraft,
        prompt=_prompt(product),
        context={
            "title": product.title,
            "description": (product.raw_attributes or {}).get("description", ""),
            "style_hint": (product.raw_attributes or {}).get("style_hint"),
            "occasion_hint": (product.raw_attributes or {}).get("occasion_hint"),
        },
    )
    # Constrain to controlled vocab so factual/inferred data never blurs.
    styles = [s for s in draft.styles if s in STYLES] or ["classic"]
    occasions = [o for o in draft.occasions if o in OCCASIONS] or ["daily wear"]
    return StyleTags(
        product_id=product.id,
        styles=styles,
        occasions=occasions,
        aesthetic_notes=draft.aesthetic_notes,
        confidence=max(0.0, min(1.0, draft.confidence)),
        tagged_by=f"enrich-{_PROVIDER_VERSION}",
        qa_checked=False,
    )


def enrich_catalog(
    products: list[Product], provider: LLMProvider | None = None
) -> dict[str, StyleTags]:
    provider = provider or get_provider()
    return {p.id: enrich_product(p, provider) for p in products}


def measure_tagging_accuracy(
    products: list[Product], tags: dict[str, StyleTags]
) -> dict[str, float | int | str]:
    """Agreement between the leading inferred style and the synthetic ``style_hint``.

    Honest caveat: on synthetic data the ``style_hint`` is a pseudo-label, not a human
    annotation. For scraped data with no hint, this returns n=0 (hand-label a sample
    instead). Reported in the README WITH this limitation.
    """
    hits = 0
    n = 0
    for p in products:
        hint = (p.raw_attributes or {}).get("style_hint")
        if not hint or p.id not in tags:
            continue
        n += 1
        if hint in tags[p.id].styles:
            hits += 1
    accuracy = (hits / n) if n else 0.0
    return {
        "tagging_accuracy": round(accuracy, 4),
        "n": n,
        "label_source": "synthetic style_hint (pseudo-label, not human-annotated)",
    }
