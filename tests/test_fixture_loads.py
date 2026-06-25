"""Phase 0: the committed sample catalog + brief fixtures load and are well-formed.

This is the brief's Phase 0 "a test loads the fixture" gate, and also guards the
factual Product schema (Section 5) the rest of the pipeline depends on.
"""

from core.config import load_briefs, load_catalog_dicts
from core.ingest.synthetic import generate_catalog

REQUIRED_PRODUCT_FIELDS = {
    "id",
    "source",
    "source_url",
    "title",
    "price",
    "currency",
    "metal",
    "stone_primary",
    "stone_accent",
    "carat",
    "category",
    "image_path",
    "raw_attributes",
    "ingested_at",
}


def test_catalog_fixture_loads_and_is_wellformed():
    catalog = load_catalog_dicts()
    assert len(catalog) >= 60, "fixture should hold a meaningful sample (~60-100 items)"
    ids = set()
    for item in catalog:
        missing = REQUIRED_PRODUCT_FIELDS - item.keys()
        assert not missing, f"{item.get('id')} missing fields: {missing}"
        assert item["source"] == "synthetic"
        assert item["currency"] == "INR"
        assert isinstance(item["price"], int) and item["price"] > 0
        ids.add(item["id"])
    assert len(ids) == len(catalog), "product ids must be unique"


def test_briefs_fixture_loads():
    briefs = load_briefs()
    assert len(briefs) >= 3
    for b in briefs:
        assert b["id"] and b["text"]


def test_synthetic_generator_is_deterministic():
    a = generate_catalog(n=20, seed=7)
    b = generate_catalog(n=20, seed=7)
    assert a == b, "same seed must yield an identical catalog (reproducibility)"
    c = generate_catalog(n=20, seed=8)
    assert a != c, "different seed should yield a different catalog"
