"""Deterministic NL parsing — the backbone of offline mode and the cold-start fallback.

The hard-constraint cases here are what the adversarial constraint test (P1.7) relies on.
"""

import pytest

from core.textparse import parse_budget, parse_profile_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("under 1,50,000", 150000),
        ("up to 1.5 lakh", 150000),
        ("nothing above 80,000", 80000),
        ("budget around 90,000", 90000),
        ("under 8,000", 8000),
        ("around 1,00,000 is fine", 100000),
        ("a tight budget under 30k", 30000),
        ("something nice for my mom", None),
    ],
)
def test_parse_budget(text, expected):
    assert parse_budget(text) == expected


def test_platinum_only_no_gold():
    d = parse_profile_text("Platinum only, nothing above 80,000. No gold at all.")
    assert d.budget_max == 80000
    assert d.allowed_metals == ["platinum"]
    # gold must be excluded and never allowed
    assert set(d.excluded_metals) == {"rose gold", "white gold", "yellow gold"}
    assert "platinum" not in d.excluded_metals


def test_no_gemstones_plain():
    d = parse_profile_text("A modern plain metal bangle, no gemstones, under 60,000.")
    assert d.require_no_stone is True
    assert "modern" in d.styles
    assert d.budget_max == 60000
    assert d.categories == ["bangle"]


def test_metal_and_stone_preferences_are_soft():
    d = parse_profile_text(
        "engagement ring, vintage feel, prefer platinum or white gold, ideally a diamond, "
        "up to 1,50,000"
    )
    assert d.budget_max == 150000
    assert d.occasion == "engagement"
    assert "vintage" in d.styles
    assert set(d.metal_prefs) == {"platinum", "white gold"}
    assert d.stone_prefs == ["diamond"]
    # "prefer" / "ideally" are soft — no hard allow-list, nothing excluded.
    assert d.allowed_metals == []
    assert d.categories == ["ring"]


def test_category_word_boundary_does_not_false_match():
    # "earrings" contains "ring" but must not be parsed as a ring.
    d = parse_profile_text("minimalist silver earrings for office")
    assert d.categories == ["earrings"]
