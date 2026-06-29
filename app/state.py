"""Application state: the catalog is ingested, enriched, and indexed ONCE at startup
and reused across requests. Enrichment is cached on disk, so repeated startups (and real
LLM runs) don't re-pay for tagging.
"""

from __future__ import annotations

from core.config import get_settings
from core.embed import get_embedder
from core.image_embed import get_image_embedder
from core.llm import get_provider
from core.pipeline import CatalogContext, build_catalog
from core.visual import VisualIndex, build_visual_index


class AppState:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = get_provider(self.settings)
        self.embedder = get_embedder(self.settings)
        self.ctx: CatalogContext = build_catalog(self.provider, db_url=self.settings.db_url)
        # Phase 2: render + embed product images once for visual search (default colour
        # embedder needs no torch, so this is cheap and works on the free-tier deploy).
        self.image_embedder = get_image_embedder(self.settings)
        self.visual: VisualIndex = build_visual_index(self.ctx.products, self.image_embedder)


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state


def reset_state() -> None:  # tests
    global _state
    _state = None
