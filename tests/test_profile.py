"""P1.6: freeform -> PreferenceProfile, with cold-start handling."""

from core.profile import parse_preference


def test_rich_brief_parses_structured(provider, embedder):
    prof = parse_preference(
        "engagement ring, vintage feel, prefer platinum or white gold, ideally a diamond, "
        "up to 1,50,000",
        provider,
        embedder,
    )
    assert prof.budget_max == 150000
    assert prof.occasion == "engagement"
    assert "vintage" in prof.styles
    assert set(prof.metal_prefs) == {"platinum", "white gold"}
    assert prof.embedding is not None and len(prof.embedding) > 0
    assert prof.needs_clarification is False


def test_cold_start_sparse_input_asks_for_clarification(provider, embedder):
    prof = parse_preference("something nice for my mom", provider, embedder)
    assert prof.needs_clarification is True
    assert prof.clarifying_question
    assert prof.recipient in {"mom", "mother"}


def test_out_of_vocab_values_are_dropped(provider, embedder):
    # The deterministic parser only emits controlled-vocab values, so junk never appears.
    prof = parse_preference("a zorblax-style ring", provider, embedder)
    assert all(
        s
        in {
            "vintage",
            "minimalist",
            "boho",
            "statement",
            "classic",
            "modern",
            "romantic",
            "delicate",
        }
        for s in prof.styles
    )
