"""Phase 2: deterministic renders, the colour image embedder, visual retrieval + fusion
(grounding intact), and the honest visual sanity eval. All offline, no torch."""

import numpy as np
import pytest

from core.image_embed import ColorGridEmbedder
from core.ingest.images import render_product_image
from core.llm import CachingProvider, FakeProvider, ResponseCache
from core.pipeline import build_catalog
from core.visual import build_visual_index, nearest, recommend_visual
from eval.visual_eval import run_visual_eval


def _provider():
    return CachingProvider(FakeProvider(), ResponseCache())


@pytest.fixture(scope="module")
def visual_ctx(tmp_path_factory):
    provider = _provider()
    ctx = build_catalog(provider)
    img_dir = tmp_path_factory.mktemp("images")
    index = build_visual_index(ctx.products, ColorGridEmbedder(), image_dir=img_dir)
    return provider, ctx, index, img_dir


def test_render_is_deterministic():
    p = {"id": "syn-0001", "metal": "platinum", "stone_primary": "diamond", "category": "ring"}
    assert list(render_product_image(p).getdata()) == list(render_product_image(p).getdata())


def test_color_embedder_is_normalized_and_shaped(visual_ctx):
    _, _, index, _ = visual_ctx
    assert index.matrix.shape == (len(index.ids), ColorGridEmbedder().dim)
    assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1.0, atol=1e-5)


def test_nearest_excludes_self_and_returns_k(visual_ctx):
    _, _, index, _ = visual_ctx
    pid = index.ids[0]
    nbrs = nearest(index.matrix[0], index, top_k=5, exclude=pid)
    assert len(nbrs) == 5 and all(i != pid for i, _ in nbrs)


def test_pure_visual_recommendations_are_grounded(visual_ctx):
    provider, ctx, index, img_dir = visual_ctx
    qpath = f"{img_dir}/{index.ids[0]}.png"
    _profile, recs = recommend_visual(qpath, "", ctx, index, ColorGridEmbedder(), provider, top_k=3)
    assert recs
    for r in recs:
        assert r.grounding_decision == "passed"
        assert "visual" in r.retrieval_scores
        for claim in r.claims:
            if claim.claim_type == "factual":
                assert claim.grounded and not claim.blocked


def test_fused_visual_plus_text_respects_hard_constraints(visual_ctx):
    provider, ctx, index, img_dir = visual_ctx
    qpath = f"{img_dir}/{index.ids[0]}.png"
    _profile, recs = recommend_visual(
        qpath, "platinum only, under 90,000", ctx, index, ColorGridEmbedder(), provider, top_k=5
    )
    assert recs
    for r in recs:
        assert r.product.metal == "platinum" and r.product.price <= 90000
        assert r.grounding_decision == "passed"
        assert "fused" in r.retrieval_scores


def test_visual_sanity_eval_beats_random_baselines(visual_ctx):
    _, ctx, index, _ = visual_ctx
    m = run_visual_eval(ctx.products, index, k=5)
    # Random baselines: same-metal ~1/5=0.20, same-category ~1/7=0.14. The colour embedder
    # should clearly beat the metal baseline (colour-dominated renders).
    assert m["same_metal_rate"] > 0.40
    assert m["same_category_rate"] > 0.14
    assert m["n"] == len(ctx.products)
