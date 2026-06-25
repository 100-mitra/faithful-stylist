"""P1.7: hybrid retrieval — the retrieval-filter test and the adversarial
constraint-satisfaction test (brief §9). Hard constraints must yield ZERO violations.
"""

from core.profile import parse_preference
from core.retrieve import retrieve
from core.store import query_products


def test_retrieval_applies_budget_and_metal_filters(engine, provider, embedder):
    prof = parse_preference("platinum jewellery only, under 1,00,000", provider, embedder)
    survivors = query_products(engine, prof)
    assert survivors, "expected platinum items under 1L in the fixture"
    assert all(p.metal == "platinum" and p.price <= 100000 for p in survivors)


def test_adversarial_platinum_only_under_budget_zero_violations(engine, provider, embedder):
    prof = parse_preference(
        "Platinum only, nothing above 80,000. No gold at all.", provider, embedder
    )
    results = retrieve(prof, engine, embedder, top_k=50)
    assert results, "expected some valid platinum items"
    for product, _scores in results:
        assert product.metal == "platinum", f"metal violation: {product.metal}"
        assert product.price <= 80000, f"budget violation: {product.price}"


def test_adversarial_no_gemstones_zero_violations(engine, provider, embedder):
    prof = parse_preference(
        "a modern plain metal bangle, no gemstones, under 60,000", provider, embedder
    )
    results = retrieve(prof, engine, embedder, top_k=50)
    assert results
    for product, _ in results:
        assert product.stone_primary is None, "stone-exclusion violated"
        assert product.category == "bangle"
        assert product.price <= 60000


def test_excluded_stone_never_appears(engine, provider, embedder):
    prof = parse_preference("a ring, no diamonds, under 2,00,000", provider, embedder)
    results = retrieve(prof, engine, embedder, top_k=50)
    assert results
    for product, _ in results:
        assert product.stone_primary != "diamond"
        assert product.stone_accent != "diamond"


def test_results_are_ranked_by_total_score(engine, provider, embedder):
    prof = parse_preference("vintage white gold diamond ring under 2,00,000", provider, embedder)
    results = retrieve(prof, engine, embedder, top_k=10)
    totals = [s["total"] for _, s in results]
    assert totals == sorted(totals, reverse=True)
