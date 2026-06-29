"""Honest visual-similarity sanity eval (Phase 2).

Leave-one-out nearest-neighbour over the product-image embeddings: for each item, how often
do its top-k visual neighbours share its category / metal / primary stone? It is a *sanity
check* that the visual pipeline recovers same-attribute neighbours — NOT a claim about
real-photo aesthetic similarity (the images are synthetic renders; see core/ingest/images.py).

Run with the default colour embedder (reproducible in CI) or the real CLIP embedder:
    python -m eval.visual_eval                       # ColorGridEmbedder
    STYLIST_IMAGE_EMBEDDER=clip python -m eval.visual_eval   # real CLIP (needs the [clip] extra)
"""

from __future__ import annotations

import json

from core.config import ROOT_DIR, get_settings, load_catalog_dicts
from core.models import Product
from core.visual import VisualIndex, build_visual_index, nearest

CAVEAT = (
    "Synthetic renders (metal->colour, stone->gem colour, category->silhouette), NOT "
    "photographs. A pipeline sanity check that visual nearest-neighbour recovers same-attribute "
    "items; NOT a claim about real-photo aesthetic similarity. Baselines: random same-category "
    "with 7 categories ~14%, same-metal with 5 metals ~20%."
)


def run_visual_eval(products: list[Product], index: VisualIndex, k: int = 5) -> dict:
    by_id = {p.id: p for p in products}
    cat = met = sto = 0.0
    n = 0
    for p in products:
        qvec = index.matrix[index.ids.index(p.id)]
        nbrs = nearest(qvec, index, top_k=k, exclude=p.id)
        if not nbrs:
            continue
        n += 1
        cat += sum(by_id[i].category == p.category for i, _ in nbrs) / len(nbrs)
        met += sum(by_id[i].metal == p.metal for i, _ in nbrs) / len(nbrs)
        sto += sum(by_id[i].stone_primary == p.stone_primary for i, _ in nbrs) / len(nbrs)
    return {
        "k": k,
        "n": n,
        "embedder": index.embedder_name,
        "same_category_rate": round(cat / n, 4) if n else None,
        "same_metal_rate": round(met / n, 4) if n else None,
        "same_stone_rate": round(sto / n, 4) if n else None,
        "n_categories": len({p.category for p in products}),
        "n_metals": len({p.metal for p in products}),
        "caveat": CAVEAT,
    }


def record_visual_run(k: int = 5) -> dict:
    products = [Product(**d) for d in load_catalog_dicts()]
    index = build_visual_index(products)
    metrics = run_visual_eval(products, index, k=k)
    out = ROOT_DIR / "eval" / f"visual_run_{get_settings().image_embedder}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Visual eval [{metrics['embedder']}] @k={k}: "
        f"same_category={metrics['same_category_rate']}, "
        f"same_metal={metrics['same_metal_rate']}, "
        f"same_stone={metrics['same_stone_rate']} (N={metrics['n']})\nWrote {out}"
    )
    return metrics


if __name__ == "__main__":  # pragma: no cover
    record_visual_run()
