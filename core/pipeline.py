"""End-to-end recommendation pipeline (the full §6 flow, wired together).

freeform brief -> parse -> hybrid retrieve (hard filter + semantic) -> LLM rerank ->
grounded rationale + GroundingVerifier -> Recommendation with a claims audit.

``build_catalog`` ingests + enriches + indexes once; the API caches the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import get_settings, load_catalog_dicts
from core.embed import Embedder, get_embedder
from core.enrich import enrich_catalog, measure_tagging_accuracy
from core.llm import LLMProvider, get_provider
from core.models import PreferenceProfile, Product, Recommendation, StyleTags
from core.profile import parse_preference
from core.rationale import build_rationale
from core.rerank import rerank
from core.retrieve import retrieve
from core.store import init_db, make_engine, seed_catalog


@dataclass
class CatalogContext:
    engine: object
    products: list[Product]
    tags_by_id: dict[str, StyleTags]
    snapshot_hash: str

    @property
    def products_by_id(self) -> dict[str, Product]:
        return {p.id: p for p in self.products}


def _snapshot_hash(products: list[Product]) -> str:
    import hashlib
    import json

    rows = sorted(json.dumps(p.model_dump(), sort_keys=True, default=str) for p in products)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def build_catalog(
    provider: LLMProvider | None = None,
    db_url: str = "sqlite://",
    products: list[Product] | None = None,
) -> CatalogContext:
    """Ingest a catalog, enrich it, and index it into SQL (once).

    Defaults to the committed synthetic fixture; pass ``products`` (e.g. rows from the
    responsible scraper) to run the same pipeline on real-brand data — the system works
    end-to-end on either source.
    """
    provider = provider or get_provider()
    if products is None:
        products = [Product(**d) for d in load_catalog_dicts()]
    tags_by_id = enrich_catalog(products, provider)
    engine = make_engine(db_url)
    init_db(engine)
    seed_catalog(engine, products, list(tags_by_id.values()))
    return CatalogContext(
        engine=engine,
        products=products,
        tags_by_id=tags_by_id,
        snapshot_hash=_snapshot_hash(products),
    )


def recommend(
    raw_text: str,
    ctx: CatalogContext,
    provider: LLMProvider | None = None,
    embedder: Embedder | None = None,
    top_k: int = 5,
) -> tuple[PreferenceProfile, list[Recommendation]]:
    provider = provider or get_provider()
    embedder = embedder or get_embedder()

    profile = parse_preference(raw_text, provider, embedder)
    candidates = retrieve(profile, ctx.engine, embedder, ctx.tags_by_id, top_k=max(top_k * 3, 15))
    ranked = rerank(profile, candidates, provider, top_k=top_k)

    recs: list[Recommendation] = []
    for rank, (product, scores) in enumerate(ranked, start=1):
        verdict = build_rationale(profile, product, provider)
        recs.append(
            Recommendation(
                product=product,
                rank=rank,
                retrieval_scores=scores,
                rationale=verdict.rationale_text,
                claims=verdict.claims,
                grounding_decision=verdict.decision,
            )
        )
    return profile, recs


def tagging_accuracy(ctx: CatalogContext) -> dict:
    """Tagging agreement — suppressed offline (the fake tagger echoes the hint it's scored
    against, so any number would be a tautology). Only emits a value on a real/cassette run.
    """
    if get_settings().llm_provider not in ("anthropic", "cassette"):
        return {
            "tagging_accuracy": None,
            "n": 0,
            "offline_placeholder": True,
            "caveat": (
                "OFFLINE PLACEHOLDER: the fake tagger echoes the style_hint it is then scored "
                "against, so any number is a tautology. Run STYLIST_LLM=anthropic for a real "
                "agreement number."
            ),
            "label_source": "synthetic style_hint (pseudo-label, not human-annotated)",
        }
    return {**measure_tagging_accuracy(ctx.products, ctx.tags_by_id), "offline_placeholder": False}
