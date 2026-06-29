"""Deterministic product-image renderer (Phase 2).

The synthetic catalog has no photographs, so we render a simple but visually-meaningful
image per product from its *factual* attributes: metal -> colour, stone -> gem colour,
category -> silhouette/layout. Rendering is a pure function of the product id + attributes
(seeded via hashlib, not Python's salted hash()), so images are byte-stable across runs and
machines — which keeps the CLIP/colour embeddings, and the visual sanity eval, reproducible.

These are renders, NOT photographs: CLIP/colour similarity here reflects the rendered
attributes (colour + coarse shape), so the visual eval is a pipeline sanity check, not a
claim about real-photo aesthetic similarity. See README > Phase 2.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 224  # CLIP's native input size
BG = (244, 244, 246)

METAL_RGB: dict[str, tuple[int, int, int]] = {
    "yellow gold": (212, 175, 55),
    "white gold": (229, 228, 226),
    "rose gold": (200, 140, 130),
    "platinum": (181, 184, 188),
    "silver": (192, 192, 196),
}
STONE_RGB: dict[str, tuple[int, int, int]] = {
    "diamond": (222, 234, 246),
    "ruby": (198, 40, 62),
    "emerald": (38, 158, 92),
    "sapphire": (40, 82, 200),
    "pearl": (243, 238, 228),
    "amethyst": (150, 92, 190),
    "topaz": (222, 178, 60),
}


def _seed(pid: str) -> int:
    return int(hashlib.md5(pid.encode("utf-8")).hexdigest()[:8], 16)


def _jit(seed: int, lo: int, hi: int, salt: int) -> int:
    """Tiny deterministic jitter so identical-attribute items aren't pixel-identical."""
    return lo + ((seed >> salt) % (hi - lo + 1))


def _gem(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(60, 60, 70))


def render_product_image(product: dict, size: int = SIZE) -> Image.Image:
    """Render one product to a deterministic PIL image from its factual attributes."""
    pid = product["id"]
    seed = _seed(pid)
    metal = METAL_RGB.get(product.get("metal", ""), (200, 200, 200))
    stone = product.get("stone_primary")
    gem = STONE_RGB.get(stone) if stone else None
    category = product.get("category", "ring")

    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    c = size // 2
    dx, dy = _jit(seed, -8, 8, 0), _jit(seed, -8, 8, 8)

    if category == "ring":
        r = 64 + _jit(seed, -6, 6, 4)
        d.ellipse([c - r + dx, c - r + dy, c + r + dx, c + r + dy], outline=metal, width=18)
        if gem:
            _gem(d, c + dx, c - r + dy, 16, gem)
    elif category == "bangle":
        r = 84
        d.ellipse([c - r + dx, c - r + dy, c + r + dx, c + r + dy], outline=metal, width=30)
        if gem:
            _gem(d, c + dx, c - r + dy, 14, gem)
    elif category == "bracelet":
        d.rounded_rectangle([28, c - 26 + dy, size - 28, c + 26 + dy], radius=26, fill=metal)
        if gem:
            _gem(d, c + dx, c + dy, 18, gem)
    elif category == "necklace":
        d.arc([34, 40, size - 34, size - 20], start=20, end=160, fill=metal, width=10)
        if gem:
            _gem(d, c + dx, size - 70 + dy, 20, gem)
    elif category == "earrings":
        for off in (-40, 40):
            d.ellipse(
                [c + off - 22, c - 40 - 22, c + off + 22, c - 40 + 22], outline=metal, width=12
            )
            if gem:
                _gem(d, c + off, c - 40 + 24, 12, gem)
    elif category == "nose pin":
        d.ellipse([c - 16 + dx, c - 16 + dy, c + 16 + dx, c + 16 + dy], outline=metal, width=8)
        if gem:
            _gem(d, c + dx, c + dy, 9, gem)
    else:  # pendant (default): chain + hanging gem
        d.line([c + dx, 36, c + dx, c + dy], fill=metal, width=6)
        d.ellipse([c - 30 + dx, c - 30 + dy, c + 30 + dx, c + 30 + dy], outline=metal, width=12)
        if gem:
            _gem(d, c + dx, c + dy, 18, gem)
    return img


def generate_images(products: list[dict], out_dir: Path) -> dict[str, str]:
    """Render every product to ``out_dir/<id>.png`` (idempotent). Returns {id: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for product in products:
        path = out_dir / f"{product['id']}.png"
        if not path.exists():
            render_product_image(product).save(path)
        paths[product["id"]] = str(path)
    return paths
