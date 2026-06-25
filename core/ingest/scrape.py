"""Responsible real-brand scraper (brief §7).

Defaults to a Shopify storefront's public, structured ``/products.json`` endpoint —
structured factual data, far gentler than scraping rendered HTML. The scraper fetches and
honours each store's own robots.txt at runtime (with our descriptive User-Agent) before
any request; we do not assert any brand's permission, we check it. Guardrails, all
enforced here:

  * respects robots.txt (urllib.robotparser, our descriptive User-Agent) and FAILS CLOSED
    if robots.txt cannot be fetched/parsed (a 404 / no-robots is the only "allow" fallback),
  * rate-limits (>= 3s between requests, single thread),
  * caches every response to disk so a page is never re-fetched,
  * extracts ONLY the Section 5 factual fields (+ a thumbnail URL, not the bytes),
  * writes to a gitignored path — the dataset is NEVER committed (only the small
    synthetic fixture is). Style tags are still inferred later by enrich.py.

If a site disallows crawling or is unreachable, callers fall back to the synthetic
generator — the pipeline never depends on a live scrape.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from core.config import CACHE_DIR
from core.vocab import METALS, STONES

USER_AGENT = "FaithfulStylistBot/0.1 (+educational portfolio; contact: soumitrachavan@gmail.com)"
_CARAT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ct|carat)", re.I)


def parse_factuals(text: str) -> tuple[str | None, str | None, float | None]:
    """Extract (metal, stone, carat) FACTS stated in the brand's own listing text."""
    low = text.lower()
    metal = next((m for m in sorted(METALS, key=len, reverse=True) if m in low), None)
    if metal is None and "gold" in low:
        metal = "yellow gold"  # brand stated "gold" without a colour qualifier
    stone = next((s for s in STONES if re.search(rf"\b{s}\b", low)), None)
    m = _CARAT_RE.search(low)
    carat = float(m.group(1)) if m else None
    return metal, stone, carat


class Scraper:
    """Robots-aware, rate-limited, disk-cached fetcher for one storefront."""

    def __init__(
        self,
        base_url: str,
        brand: str,
        delay: float = 3.0,
        cache_dir: Path | None = None,
        robots_txt: str | None = None,
        fetcher=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.brand = brand
        self.delay = delay
        self.cache_dir = cache_dir or (CACHE_DIR / "scrape" / brand)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetcher = fetcher  # tests inject a fake fetcher; live uses urlopen
        self._last = 0.0
        self._rp = RobotFileParser()
        if robots_txt is not None:
            self._rp.parse(robots_txt.splitlines())
        else:
            self._load_robots_with_ua()

    def _load_robots_with_ua(self) -> None:
        """Fetch robots.txt with OUR descriptive User-Agent and parse it.

        RobotFileParser.read() uses urllib's default UA, which some sites block (403),
        causing it to (incorrectly) treat everything as disallowed. We fetch with our own
        UA and FAIL CLOSED on uncertainty: only a 404 (no robots.txt published, which the
        standard treats as unrestricted) falls back to allow; any other failure
        (401/403/5xx/network/parse) disallows, so a flaky or malformed robots.txt can never
        silently disable the guard.
        """
        robots_url = urljoin(self.base_url + "/", "robots.txt")
        try:
            req = Request(robots_url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=15) as resp:  # pragma: no cover - live network
                raw = resp.read()
            self._rp.parse(raw.decode("utf-8", "replace").splitlines())
        except HTTPError as exc:  # pragma: no cover - live network
            if exc.code == 404:
                self._rp.allow_all = True  # no robots.txt published -> unrestricted (standard)
            else:
                self._rp.disallow_all = True  # 401/403/5xx -> cannot confirm policy: fail closed
        except Exception:  # pragma: no cover - network/parse failure -> fail closed
            self._rp.disallow_all = True

    def allowed(self, url: str) -> bool:
        return self._rp.can_fetch(USER_AGENT, url)

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")

    def get(self, url: str) -> bytes:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        cache = self._cache_path(url)
        if cache.exists():
            return cache.read_bytes()
        self._throttle()
        if self._fetcher is not None:
            data = self._fetcher(url)
        else:  # pragma: no cover - live network
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=20) as resp:
                data = resp.read()
        cache.write_bytes(data)
        return data

    def _map_product(self, item: dict) -> dict:
        from core.textparse import parse_categories

        variants = item.get("variants") or [{}]
        price_raw = variants[0].get("price", "0")
        try:
            price = int(round(float(price_raw)))
        except (TypeError, ValueError):
            price = 0
        tags = item.get("tags") or ""
        if isinstance(tags, list):  # Shopify products.json returns tags as an array
            tags = " ".join(str(t) for t in tags)
        text = " ".join([item.get("title", ""), tags, item.get("product_type", "") or ""])
        metal, stone, carat = parse_factuals(text)
        cats = parse_categories(item.get("title", "").lower())
        category = cats[0] if cats else (item.get("product_type") or "jewellery").lower()
        handle = item.get("handle", "")
        images = item.get("images") or []
        return {
            "id": f"{self.brand}-{item.get('id')}",
            "source": self.brand,
            "source_url": urljoin(self.base_url + "/", f"products/{handle}"),
            "title": item.get("title", ""),
            "price": price,
            "currency": "INR",
            "metal": metal or "unspecified",
            "stone_primary": stone,
            "stone_accent": None,
            "carat": carat,
            "certification": None,
            "category": category,
            "image_path": images[0].get("src") if images else None,  # URL only; bytes not stored
            "raw_attributes": {
                "tags": item.get("tags"),
                "product_type": item.get("product_type"),
                "vendor": item.get("vendor"),
                "handle": handle,
            },
            "ingested_at": datetime.now(UTC).isoformat(),
        }

    def scrape_shopify(self, max_items: int = 20, page_size: int = 50) -> list[dict]:
        """Page through the public products.json endpoint, mapping to §5 Product dicts."""
        out: list[dict] = []
        page = 1
        while len(out) < max_items:
            url = urljoin(self.base_url + "/", f"products.json?limit={page_size}&page={page}")
            items = json.loads(self.get(url)).get("products", [])
            if not items:
                break
            for it in items:
                out.append(self._map_product(it))
                if len(out) >= max_items:
                    break
            page += 1
        return out


def scrape_to_file(base_url: str, brand: str, out_path: Path, max_items: int = 20) -> int:
    scraper = Scraper(base_url, brand)
    rows = scraper.scrape_shopify(max_items=max_items)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from core.config import DATA_DIR

    ap = argparse.ArgumentParser(description="Responsible Shopify catalog scraper.")
    ap.add_argument("--base", default="https://www.giva.co")
    ap.add_argument("--brand", default="giva")
    ap.add_argument("--max", type=int, default=20)
    args = ap.parse_args()
    # data/scraped/ is gitignored — the dataset is never committed.
    out = DATA_DIR / "scraped" / f"{args.brand}.json"
    n = scrape_to_file(args.base, args.brand, out, max_items=args.max)
    print(f"Scraped {n} products from {args.base} -> {out} (gitignored)")
