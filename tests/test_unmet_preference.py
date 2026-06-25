"""STEP 1(b): faithfulness extended to UNMET preferences.

When an explicit stone/metal preference cannot be met within the shopper's budget and
constraints, the rationale must SAY SO (a grounded, deterministic disclosure) instead of
silently substituting a non-matching item. Also covers STEP 1(a): a stated preference that
CAN be met is actually surfaced at the top.
"""

from core.llm import CachingProvider, FakeProvider, ResponseCache
from core.models import HardConstraints, PreferenceProfile, Product
from core.pipeline import build_catalog, recommend
from core.retrieve import preference_disclosure, preference_gaps

_T = "2026-01-01T00:00:00+00:00"


def _provider():
    return CachingProvider(FakeProvider(), ResponseCache())


def _prod(pid, metal, stone, price, category="ring", carat=None, style="classic"):
    return Product(
        id=pid,
        source="synthetic",
        source_url=f"synthetic://{pid}",
        title=f"{metal} {style} {stone or ''} {category}".strip(),
        price=price,
        currency="INR",
        metal=metal,
        stone_primary=stone,
        stone_accent=None,
        carat=carat,
        category=category,
        image_path=None,
        raw_attributes={"description": f"a {style} {category}", "style_hint": style},
        ingested_at=_T,
    )


def _profile(**kw):
    return PreferenceProfile(
        raw_text=kw.get("raw_text", "a ring"),
        budget_max=kw.get("budget_max"),
        styles=kw.get("styles", []),
        metal_prefs=kw.get("metal_prefs", []),
        stone_prefs=kw.get("stone_prefs", []),
        hard_constraints=HardConstraints(),
    )


# --- Unit tests of the deterministic disclosure logic -----------------------------------
def test_gap_flagged_when_no_in_budget_item_matches_stone():
    prof = _profile(budget_max=150000, stone_prefs=["diamond"], metal_prefs=["platinum"])
    rec = _prod("a", "platinum", "amethyst", 49000)
    survivors = [rec, _prod("b", "platinum", "pearl", 30000)]  # no diamond anywhere
    gaps = preference_gaps(prof, rec, survivors)
    stone_gap = next(g for g in gaps if g.attribute == "stone")
    assert stone_gap.none_in_catalog is True
    assert stone_gap.requested == ["diamond"] and stone_gap.actual == "amethyst"


def test_unmet_preference_is_disclosed_not_hidden():
    prof = _profile(budget_max=150000, stone_prefs=["diamond"], metal_prefs=["platinum"])
    rec = _prod("a", "platinum", "amethyst", 49000)
    survivors = [rec, _prod("b", "platinum", "pearl", 30000)]
    text, gaps = preference_disclosure(prof, rec, survivors)
    assert text is not None
    assert "No" in text and "diamond" in text and "₹150,000" in text
    assert "platinum amethyst ring" in text  # honestly describes what it actually is
    assert any(g["none_in_catalog"] for g in gaps)


def test_met_preference_yields_no_disclosure():
    prof = _profile(budget_max=150000, stone_prefs=["diamond"])
    rec = _prod("a", "platinum", "diamond", 98000, carat=0.5)
    text, _ = preference_disclosure(prof, rec, [rec])
    assert text is None  # nothing to disclose — the preference is met


def test_preference_available_elsewhere_gets_soft_note():
    prof = _profile(budget_max=150000, stone_prefs=["diamond"])
    rec = _prod("a", "platinum", "amethyst", 49000)  # this item lacks diamond ...
    survivors = [rec, _prod("b", "white gold", "diamond", 64000, carat=0.3)]  # ... but one exists
    gap = next(g for g in preference_gaps(prof, rec, survivors) if g.attribute == "stone")
    assert gap.none_in_catalog is False
    text, _ = preference_disclosure(prof, rec, survivors)
    assert text is not None and "does not match your stated preference" in text


# --- End-to-end: the disclosure surfaces through recommend() (and is grounded) ----------
def test_recommend_discloses_unmet_preference_end_to_end(embedder):
    provider = _provider()
    # A controlled catalog with NO diamond, so a diamond request genuinely cannot be met.
    products = [
        _prod("r1", "platinum", "amethyst", 40000, style="vintage"),
        _prod("r2", "rose gold", "pearl", 30000, style="romantic"),
    ]
    ctx = build_catalog(provider, products=products)
    _profile_out, recs = recommend(
        "a diamond ring, platinum preferred, under 1,50,000", ctx, provider, embedder, top_k=2
    )
    assert recs
    top = recs[0]
    assert top.product.stone_primary != "diamond"  # genuinely substituted, not faked
    assert "No" in top.rationale and "diamond" in top.rationale  # ... and it says so
    gap_claims = [c for c in top.claims if c.source_field == "preference_gap"]
    assert gap_claims, "the unmet preference must appear in the claims audit"
    assert gap_claims[0].claim_type == "factual" and gap_claims[0].grounded
    assert not gap_claims[0].blocked
    # Disclosing an honest substitution does NOT block — the rationale is faithful, not vetoed.
    assert top.grounding_decision == "passed"


# --- STEP 1(a): a preference that CAN be met is surfaced at the top ----------------------
def test_diamond_engagement_brief_now_returns_diamond_at_top(embedder):
    provider = _provider()
    ctx = build_catalog(provider)  # the committed fixture (now includes diamond anchors)
    _profile_out, recs = recommend(
        "engagement ring, vintage romantic, platinum or white gold, diamond, up to 1,50,000",
        ctx,
        provider,
        embedder,
        top_k=3,
    )
    assert recs
    assert recs[0].product.stone_primary == "diamond", "stated diamond preference must win"
    assert recs[0].product.metal in ("platinum", "white gold")
    assert recs[0].product.price <= 150000
