"""P1.5: enrichment emits schema-valid StyleTags constrained to the controlled vocab."""

from core.enrich import enrich_catalog, enrich_product, measure_tagging_accuracy
from core.vocab import OCCASIONS, STYLES


def test_enrich_returns_valid_styletags(catalog, provider):
    tags = enrich_product(catalog[0], provider)
    assert tags.product_id == catalog[0].id
    assert tags.styles and all(s in STYLES for s in tags.styles)
    assert tags.occasions and all(o in OCCASIONS for o in tags.occasions)
    assert 0.0 <= tags.confidence <= 1.0
    assert tags.tagged_by.startswith("enrich-")
    assert tags.qa_checked is False


def test_enrich_catalog_covers_every_product(catalog, provider):
    tags = enrich_catalog(catalog, provider)
    assert set(tags.keys()) == {p.id for p in catalog}


def test_tagging_accuracy_is_reported_with_n_and_caveat(catalog, provider):
    tags = enrich_catalog(catalog, provider)
    metrics = measure_tagging_accuracy(catalog, tags)
    assert metrics["n"] > 0
    assert 0.0 <= metrics["tagging_accuracy"] <= 1.0
    # Honesty: the label source must be stated (pseudo-label, not human-annotated).
    assert "pseudo-label" in metrics["label_source"]
