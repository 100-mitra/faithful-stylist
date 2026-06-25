"""Deterministic synthetic jewellery-catalog generator.

Produces realistic Indian-market (INR) catalog records with the same schema as the
real scraper, so the pipeline and tests never depend on a live scrape. Output is a
pure function of ``seed`` and ``n`` — identical inputs give identical catalogs.

Only *factual* attributes are generated here. Subjective style tags are produced
later by the LLM enrichment step (``core/enrich.py``); the descriptions carry a soft
``style_hint``/``occasion_hint`` so enrichment has something to read.

Run ``python -m core.ingest.synthetic`` to (re)write data/fixtures/catalog.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Fixed so generated rows are byte-stable across runs/machines.
INGESTED_AT = "2026-01-01T00:00:00+00:00"

METALS = ["yellow gold", "white gold", "rose gold", "platinum", "silver"]
GOLD_METALS = {"yellow gold", "white gold", "rose gold"}
# Stones that are sold by carat weight; others (pearl) or None get carat=None.
CARAT_STONES = ["diamond", "ruby", "emerald", "sapphire"]
OTHER_STONES = ["pearl", "amethyst", "topaz"]
CARAT_CHOICES = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]

CATEGORIES = ["ring", "pendant", "earrings", "necklace", "bracelet", "bangle", "nose pin"]
# Categories where a centre stone is common.
STONE_HEAVY = {"ring", "pendant", "earrings", "necklace"}

STYLES = ["vintage", "minimalist", "boho", "statement", "classic", "modern", "romantic", "delicate"]
OCCASIONS = ["engagement", "daily wear", "festive", "gift", "wedding", "office"]
COLLECTIONS = ["Aurelia", "Meher", "Solene", "Vasanti", "Indra", "Noor", "Rivaah", "Kalindi"]


def _price(metal: str, stone: str | None, carat: float | None, rng: random.Random) -> int:
    """A plausible INR price built from the factual attributes."""
    base = {
        "silver": 2500,
        "yellow gold": 22000,
        "white gold": 24000,
        "rose gold": 24000,
        "platinum": 38000,
    }[metal]
    price = float(base)
    if stone in CARAT_STONES and carat is not None:
        per_carat = {"diamond": 130000, "ruby": 55000, "emerald": 60000, "sapphire": 50000}[stone]
        price += per_carat * carat
    elif stone == "pearl":
        price += 8000
    elif stone in OTHER_STONES:
        price += 6000
    price *= rng.uniform(0.85, 1.25)  # spread
    return int(round(price, -2))  # round to nearest 100


def _title(metal: str, style: str, stone: str | None, category: str) -> str:
    parts = [metal.title(), style.title()]
    if stone:
        parts.append(stone.title())
    parts.append(category.title())
    return " ".join(parts)


def _description(metal: str, style: str, stone: str | None, category: str, occasion: str) -> str:
    stone_clause = f" set with a {stone}" if stone else ""
    return (
        f"A {style} {category} crafted in {metal}{stone_clause}. Designed with {occasion} in mind."
    )


def generate_catalog(n: int = 80, seed: int = 7) -> list[dict]:
    """Generate ``n`` synthetic Product records as plain dicts (deterministic)."""
    rng = random.Random(seed)
    items: list[dict] = []
    for i in range(n):
        category = rng.choice(CATEGORIES)
        metal = rng.choice(METALS)
        style = rng.choice(STYLES)
        occasion = rng.choice(OCCASIONS)

        # Decide the centre stone.
        if category in STONE_HEAVY and rng.random() < 0.85:
            stone_primary = rng.choice(CARAT_STONES + OTHER_STONES)
        elif rng.random() < 0.45:
            stone_primary = rng.choice(CARAT_STONES + OTHER_STONES)
        else:
            stone_primary = None

        carat = rng.choice(CARAT_CHOICES) if stone_primary in CARAT_STONES else None

        # Optional small accent stone (usually diamond), distinct from the primary.
        stone_accent = None
        if stone_primary is not None and rng.random() < 0.25:
            accent_pool = [s for s in (["diamond"] + OTHER_STONES) if s != stone_primary]
            stone_accent = rng.choice(accent_pool)

        price = _price(metal, stone_primary, carat, rng)

        raw: dict = {
            "description": _description(metal, style, stone_primary, category, occasion),
            "style_hint": style,
            "occasion_hint": occasion,
            "collection": rng.choice(COLLECTIONS),
        }
        if metal in GOLD_METALS:
            raw["gold_purity"] = rng.choice(["18k", "22k"])

        pid = f"syn-{i:04d}"
        items.append(
            {
                "id": pid,
                "source": "synthetic",
                "source_url": f"synthetic://catalog/{pid}",
                "title": _title(metal, style, stone_primary, category),
                "price": price,
                "currency": "INR",
                "metal": metal,
                "stone_primary": stone_primary,
                "stone_accent": stone_accent,
                "carat": carat,
                "category": category,
                "image_path": None,
                "raw_attributes": raw,
                "ingested_at": INGESTED_AT,
            }
        )
    return items


def write_fixture(path: Path, n: int = 80, seed: int = 7) -> int:
    catalog = generate_catalog(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return len(catalog)


if __name__ == "__main__":
    from core.config import CATALOG_FIXTURE

    count = write_fixture(CATALOG_FIXTURE)
    print(f"Wrote {count} synthetic products to {CATALOG_FIXTURE}")
