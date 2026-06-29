"""Visual search (Phase 2): index product-image embeddings, fuse with the text pipeline.

The flow mirrors the text recommender — candidate retrieval -> LLM rerank -> grounded
rationale -> GroundingVerifier — but the candidate signal is image similarity (optionally
fused with the text brief). Grounding is unchanged: visual similarity only orders candidates;
every factual claim in the rationale is still templated from the record and verified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import DATA_DIR
from core.embed import Embedder, get_embedder
from core.image_embed import ImageEmbedder, get_image_embedder
from core.ingest.images import generate_images
from core.llm import LLMProvider, get_provider
from core.models import PreferenceProfile, Product, RationaleClaim, Recommendation
from core.pipeline import CatalogContext
from core.profile import parse_preference
from core.rationale import build_rationale
from core.rerank import rerank
from core.retrieve import preference_disclosure, score_candidates
from core.store import query_products

IMAGE_DIR = DATA_DIR / "images"

# Fusion weights when a text brief accompanies the inspiration image (scores are min-max
# normalized across the candidate set first, so the weights are comparable).
W_VISUAL = 0.5
W_TEXT = 0.5


@dataclass
class VisualIndex:
    ids: list[str]
    matrix: np.ndarray  # (n, dim) L2-normalized image embeddings
    embedder_name: str
    image_dir: str


def build_visual_index(
    products: list[Product],
    image_embedder: ImageEmbedder | None = None,
    image_dir=IMAGE_DIR,
) -> VisualIndex:
    """Render (if needed) and embed every product image into a cosine-searchable matrix."""
    image_embedder = image_embedder or get_image_embedder()
    paths_by_id = generate_images([p.model_dump() for p in products], image_dir)
    ids = [p.id for p in products]
    matrix = image_embedder.embed_images([paths_by_id[i] for i in ids])
    return VisualIndex(
        ids=ids, matrix=matrix, embedder_name=image_embedder.name, image_dir=str(image_dir)
    )


def visual_scores(query_vec: np.ndarray, index: VisualIndex) -> dict[str, float]:
    """Cosine similarity (vectors are L2-normalized) of the query against every product."""
    sims = index.matrix @ np.asarray(query_vec, dtype=np.float32)
    return {pid: float(s) for pid, s in zip(index.ids, sims, strict=True)}


def nearest(
    query_vec: np.ndarray, index: VisualIndex, top_k: int = 5, exclude: str | None = None
) -> list[tuple[str, float]]:
    scores = visual_scores(query_vec, index)
    if exclude is not None:
        scores.pop(exclude, None)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def _minmax(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def _visual_only_profile() -> PreferenceProfile:
    """A minimal profile for image-only search (no stated text preferences)."""
    return PreferenceProfile(raw_text="(visual similarity search)")


def recommend_visual(
    image_path: str,
    raw_text: str,
    ctx: CatalogContext,
    index: VisualIndex,
    image_embedder: ImageEmbedder | None = None,
    provider: LLMProvider | None = None,
    embedder: Embedder | None = None,
    top_k: int = 5,
) -> tuple[PreferenceProfile, list[Recommendation]]:
    provider = provider or get_provider()
    embedder = embedder or get_embedder()
    image_embedder = image_embedder or get_image_embedder()

    qvec = image_embedder.embed_images([image_path])[0]
    vscores = visual_scores(qvec, index)
    survivors_for_gap: list[Product] = ctx.products

    if raw_text and raw_text.strip():
        # Fuse image similarity with the text brief, over the hard-filtered survivors.
        profile = parse_preference(raw_text, provider, embedder)
        survivors = query_products(ctx.engine, profile)
        survivors_for_gap = survivors
        text_scored = score_candidates(profile, survivors, embedder, ctx.tags_by_id)
        if text_scored:
            nv = _minmax([vscores.get(p.id, 0.0) for p, _ in text_scored])
            nt = _minmax([s["total"] for _, s in text_scored])
            fused = []
            for (product, s), v, t in zip(text_scored, nv, nt, strict=True):
                scores = {
                    **s,
                    "visual": round(vscores.get(product.id, 0.0), 4),
                    "fused": round(W_VISUAL * v + W_TEXT * t, 4),
                }
                fused.append((product, scores))
            fused.sort(key=lambda x: x[1]["fused"], reverse=True)
        else:
            fused = []
        candidates = fused[: max(top_k * 3, 15)]
        ranked = rerank(profile, candidates, provider, top_k=top_k)
    else:
        # Pure visual search: rank everything by image similarity (no LLM rerank).
        profile = _visual_only_profile()
        by_id = ctx.products_by_id
        ordered = sorted(vscores.items(), key=lambda kv: kv[1], reverse=True)
        ranked = [
            (by_id[pid], {"visual": round(v, 4), "fused": round(v, 4)})
            for pid, v in ordered
            if pid in by_id
        ][:top_k]

    recs: list[Recommendation] = []
    for rank, (product, scores) in enumerate(ranked, start=1):
        verdict = build_rationale(profile, product, provider)
        rationale = verdict.rationale_text
        claims = verdict.claims
        disclosure, gaps = preference_disclosure(profile, product, survivors_for_gap)
        if disclosure:
            none_avail = any(g["none_in_catalog"] for g in gaps)
            note = (
                "honest substitution: no in-budget catalog item matches this stated preference"
                if none_avail
                else "faithful disclosure: a match exists at a higher rank; this is complementary"
            )
            rationale = f"{disclosure} {rationale}"
            claims = [
                RationaleClaim(
                    claim_text=disclosure,
                    claim_type="factual",
                    grounded=True,
                    source_field="preference_gap",
                    note=note,
                ),
                *claims,
            ]
        recs.append(
            Recommendation(
                product=product,
                rank=rank,
                retrieval_scores=scores,
                rationale=rationale,
                claims=claims,
                grounding_decision=verdict.decision,
            )
        )
    return profile, recs
