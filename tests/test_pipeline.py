"""P1.8 + P1.9: rerank + the end-to-end grounded-rationale pipeline.

Asserts the full path is grounded and that an injected hallucination is still blocked
end-to-end (the headline guarantee, exercised through recommend()).
"""

from core.llm import CachingProvider, FakeProvider, ResponseCache
from core.models import FactRef, RationaleDraft
from core.pipeline import build_catalog, recommend


def _provider(overrides=None):
    return CachingProvider(FakeProvider(overrides=overrides), ResponseCache())


def test_recommend_returns_grounded_recommendations(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    profile, recs = recommend(
        "vintage engagement ring, white gold or platinum, diamond, up to 2,00,000",
        ctx,
        provider,
        embedder,
        top_k=5,
    )
    assert recs, "expected at least one recommendation"
    for rec in recs:
        assert rec.grounding_decision == "passed"
        # Every factual claim is grounded and traceable to a source field.
        for claim in rec.claims:
            if claim.claim_type == "factual":
                assert claim.grounded and claim.source_field and not claim.blocked
        # Rationale surfaces the item's real metal verbatim and labels opinion.
        assert rec.product.metal in rec.rationale
        assert "Style notes (my opinion):" in rec.rationale


def test_recommendations_respect_hard_constraints(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    _profile, recs = recommend("platinum only, under 90,000", ctx, provider, embedder, top_k=5)
    assert recs
    for rec in recs:
        assert rec.product.metal == "platinum"
        assert rec.product.price <= 90000


def test_hallucination_blocked_end_to_end(embedder):
    # Force the (faked) rationale LLM to hallucinate a carat into the opinion text.
    def poison(ctx, schema):
        pid = ctx.get("product", {}).get("id", "")
        return RationaleDraft(
            product_id=pid,
            factual_slots=[FactRef(field="metal")],
            subjective_clauses=["a dazzling 5-carat centre stone, very romantic"],
            recommended=True,
        )

    provider = _provider(overrides={"rationale": poison})
    ctx = build_catalog(provider)
    _profile, recs = recommend("a nice ring under 2,00,000", ctx, provider, embedder, top_k=3)
    assert recs
    for rec in recs:
        # The fabricated carat must never reach the user-facing rationale.
        assert "5-carat" not in rec.rationale
        assert "5 ct" not in rec.rationale
        assert rec.grounding_decision == "blocked"
        assert any(c.blocked for c in rec.claims)


def test_rerank_preserves_survivors_and_orders(embedder):
    from core.profile import parse_preference
    from core.rerank import rerank
    from core.retrieve import retrieve

    provider = _provider()
    ctx = build_catalog(provider)
    profile = parse_preference("white gold diamond ring under 2,00,000", provider, embedder)
    candidates = retrieve(profile, ctx.engine, embedder, ctx.tags_by_id, top_k=10)
    ranked = rerank(profile, candidates, provider, top_k=5)
    assert ranked
    ranked_ids = {p.id for p, _ in ranked}
    cand_ids = {p.id for p, _ in candidates}
    assert ranked_ids <= cand_ids  # rerank never invents items
