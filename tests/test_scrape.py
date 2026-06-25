"""P1.14: the responsible scraper respects robots.txt and maps to §5 facts (offline)."""

import json

import pytest

from core.ingest.scrape import Scraper, parse_factuals
from core.models import Product

_ALLOW_ROBOTS = "User-agent: *\nAllow: /\n"
_BLOCK_ROBOTS = "User-agent: *\nDisallow: /products.json\nDisallow: /products\n"

_PAGE1 = {
    "products": [
        {
            "id": 111,
            "title": "Platinum 0.5 ct Solitaire Diamond Ring",
            "handle": "plat-solitaire",
            "tags": "engagement, solitaire",
            "product_type": "Rings",
            "vendor": "GIVA",
            "variants": [{"price": "84999.00"}],
            "images": [{"src": "https://cdn.example/img1.jpg"}],
        },
        {
            "id": 222,
            "title": "Sterling Silver Minimalist Pendant",
            "handle": "silver-pendant",
            "tags": "daily",
            "product_type": "Pendants",
            "vendor": "GIVA",
            "variants": [{"price": "1999.00"}],
            "images": [],
        },
    ]
}


def _fake_fetcher(url: str) -> bytes:
    if "page=2" in url:
        return json.dumps({"products": []}).encode()
    return json.dumps(_PAGE1).encode()


def test_parse_factuals_extracts_stated_facts():
    assert parse_factuals("Platinum 0.5 ct Solitaire Diamond Ring") == ("platinum", "diamond", 0.5)
    assert parse_factuals("Sterling Silver Minimalist Pendant") == ("silver", None, None)
    metal, _stone, _carat = parse_factuals("22k Gold Bangle")
    assert metal == "yellow gold"


def test_robots_disallow_blocks_fetch(tmp_path):
    s = Scraper(
        "https://shop.example",
        "ex",
        cache_dir=tmp_path,
        robots_txt=_BLOCK_ROBOTS,
        fetcher=_fake_fetcher,
    )
    assert s.allowed("https://shop.example/products.json") is False
    with pytest.raises(PermissionError):
        s.get("https://shop.example/products.json?limit=50&page=1")


def test_shopify_mapping_yields_valid_factual_products(tmp_path):
    s = Scraper(
        "https://shop.example",
        "ex",
        cache_dir=tmp_path,
        robots_txt=_ALLOW_ROBOTS,
        fetcher=_fake_fetcher,
    )
    rows = s.scrape_shopify(max_items=10, page_size=50)
    assert len(rows) == 2
    products = [Product(**r) for r in rows]  # must validate as §5 factual records
    p0 = products[0]
    assert p0.source == "ex"
    assert p0.metal == "platinum" and p0.stone_primary == "diamond" and p0.carat == 0.5
    assert p0.price == 84999 and p0.category == "ring"
    assert p0.source_url.endswith("/products/plat-solitaire")
    # Thumbnail is a URL reference, never downloaded/redistributed.
    assert p0.image_path == "https://cdn.example/img1.jpg"


def test_cache_prevents_refetch(tmp_path):
    calls = {"n": 0}

    def counting_fetcher(url):
        calls["n"] += 1
        return _fake_fetcher(url)

    s = Scraper(
        "https://shop.example",
        "ex",
        cache_dir=tmp_path,
        robots_txt=_ALLOW_ROBOTS,
        fetcher=counting_fetcher,
    )
    url = "https://shop.example/products.json?limit=50&page=1"
    s.get(url)
    s.get(url)  # second call served from disk cache
    assert calls["n"] == 1
