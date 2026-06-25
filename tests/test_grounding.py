"""P1.3 — the HEADLINE tests.

If a fabricated factual claim can reach the user, the build has failed. These tests are
fully deterministic: the verifier is a pure function, so we construct malicious drafts
directly (no live LLM) and assert they are blocked.
"""

from core.grounding import GroundingVerifier, render_rationale
from core.llm import FakeProvider
from core.models import FactRef, PreferenceProfile, Product, RationaleDraft


def make_product(**overrides) -> Product:
    base = dict(
        id="P1",
        source="synthetic",
        source_url="synthetic://catalog/P1",
        title="White Gold Vintage Diamond Ring",
        price=50000,
        currency="INR",
        metal="white gold",
        stone_primary="diamond",
        stone_accent=None,
        carat=1.0,
        certification=None,
        category="ring",
        image_path=None,
        raw_attributes={"style_hint": "vintage"},
        ingested_at="2026-01-01T00:00:00+00:00",
    )
    base.update(overrides)
    return Product(**base)


def make_profile(**overrides) -> PreferenceProfile:
    base = dict(raw_text="a vintage ring", budget_max=150000)
    base.update(overrides)
    return PreferenceProfile(**base)


# ---------------------------------------------------------------------------
# THE headline test
# ---------------------------------------------------------------------------
def test_blocks_hallucinated_carat_smuggled_into_opinion():
    product = make_product(carat=1.0)  # the real stone is 1.0 ct
    draft = RationaleDraft(
        product_id="P1",
        factual_slots=[FactRef(field="metal"), FactRef(field="stone_primary")],
        subjective_clauses=["this 2-carat stone reads as vintage"],  # 2-carat is a lie
        recommended=True,
    )
    result = GroundingVerifier().verify(draft, product, make_profile())

    assert result.decision == "blocked"
    smuggled = [c for c in result.claims if c.claim_type == "subjective" and c.blocked]
    assert smuggled and "2-carat" in smuggled[0].claim_text
    # The fabricated claim must NOT survive into the user-facing rationale.
    assert "2-carat" not in result.rationale_text
    assert "2 ct" not in result.rationale_text


def test_headline_block_holds_through_the_provider_seam():
    # Even when the (faked) LLM is forced to hallucinate, the verifier blocks it.
    def poison(ctx, schema):
        return RationaleDraft(
            product_id="P1",
            factual_slots=[FactRef(field="metal")],
            subjective_clauses=["a stunning 3 carat diamond, truly romantic"],
            recommended=True,
        )

    provider = FakeProvider(overrides={"rationale": poison})
    draft = provider.complete_structured(
        task="rationale", schema=RationaleDraft, prompt="explain", context={}
    )
    result = GroundingVerifier().verify(draft, make_product(), make_profile())
    assert result.decision == "blocked"
    assert "3 carat" not in result.rationale_text


# ---------------------------------------------------------------------------
# Other faithfulness blocks
# ---------------------------------------------------------------------------
def test_blocks_factref_to_absent_certification():
    product = make_product(certification=None)
    draft = RationaleDraft(product_id="P1", factual_slots=[FactRef(field="certification")])
    result = GroundingVerifier().verify(draft, product, make_profile())
    assert result.decision == "blocked"
    assert any(c.source_field == "certification" and c.blocked for c in result.claims)
    assert "certified" not in result.rationale_text


def test_blocks_wrong_metal_in_opinion():
    product = make_product(metal="white gold")
    draft = RationaleDraft(
        product_id="P1",
        factual_slots=[FactRef(field="metal")],
        subjective_clauses=["the platinum setting feels luxurious"],  # not platinum
    )
    result = GroundingVerifier().verify(draft, product, make_profile())
    assert result.decision == "blocked"
    assert "platinum" not in result.rationale_text


def test_allows_true_metal_mention_in_opinion():
    product = make_product(metal="white gold")
    draft = RationaleDraft(
        product_id="P1",
        factual_slots=[FactRef(field="metal")],
        subjective_clauses=["the gold tones feel warm and romantic"],  # gold ⊂ white gold
    )
    result = GroundingVerifier().verify(draft, product, make_profile())
    assert result.decision == "passed"


def test_blocks_wrong_stone_in_opinion():
    product = make_product(stone_primary="diamond", stone_accent=None)
    draft = RationaleDraft(
        product_id="P1",
        subjective_clauses=["the ruby gives it a romantic glow"],  # it's a diamond
    )
    result = GroundingVerifier().verify(draft, product, make_profile())
    assert result.decision == "blocked"


def test_blocks_product_id_mismatch():
    draft = RationaleDraft(product_id="WRONG", factual_slots=[FactRef(field="metal")])
    result = GroundingVerifier().verify(draft, make_product(id="P1"), make_profile())
    assert result.decision == "blocked"
    assert any(c.source_field == "product_id" and c.blocked for c in result.claims)


def test_blocks_under_budget_when_over_budget():
    product = make_product(price=200000)
    draft = RationaleDraft(product_id="P1", factual_slots=[FactRef(field="under_budget")])
    result = GroundingVerifier().verify(draft, product, make_profile(budget_max=150000))
    assert result.decision == "blocked"
    assert "within your budget" not in result.rationale_text


# ---------------------------------------------------------------------------
# Clean path + factual-fields-verbatim
# ---------------------------------------------------------------------------
def test_clean_draft_passes_and_factual_fields_are_verbatim():
    product = make_product(metal="white gold", price=50000, carat=1.0, category="ring")
    draft = RationaleDraft(
        product_id="P1",
        factual_slots=[
            FactRef(field="metal"),
            FactRef(field="price"),
            FactRef(field="carat"),
            FactRef(field="category"),
            FactRef(field="under_budget"),
        ],
        subjective_clauses=["reads as vintage to me"],
        recommended=True,
    )
    result = GroundingVerifier().verify(draft, product, make_profile(budget_max=150000))

    assert result.decision == "passed"
    text = result.rationale_text
    # Factual fields equal the record's values verbatim.
    assert product.metal in text  # "white gold"
    assert f"₹{product.price:,}" in text  # "₹50,000"
    assert "1 ct" in text  # carat 1.0
    assert product.category in text  # "ring"
    assert "Style notes (my opinion):" in text  # opinion is labelled, never stated as fact
    # Every factual claim is grounded and traceable to a source field.
    factual = [c for c in result.claims if c.claim_type == "factual"]
    assert factual and all(c.grounded and c.source_field for c in factual)


def test_unrecognized_opinion_is_flagged_but_not_blocked():
    draft = RationaleDraft(
        product_id="P1",
        factual_slots=[FactRef(field="metal")],
        subjective_clauses=["you will absolutely adore it"],  # no recognised style vocab
    )
    result = GroundingVerifier().verify(draft, make_product(), make_profile())
    assert result.decision == "passed"
    flagged = [c for c in result.claims if c.claim_type == "subjective"]
    assert flagged and flagged[0].grounded and flagged[0].note == "unrecognised style vocabulary"


def test_render_rationale_helper_matches_verifier_text():
    product = make_product()
    draft = RationaleDraft(product_id="P1", factual_slots=[FactRef(field="metal")])
    assert render_rationale(draft, product, make_profile()) == (
        GroundingVerifier().verify(draft, product, make_profile()).rationale_text
    )
