"""Project paths and environment-driven settings.

Safe-by-default: with no environment configured, the LLM provider is the offline
``fake`` and the embedder is the dependency-light ``hashing`` one, so the whole
system runs with no API key and no network (this is what CI uses).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CORE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"
CACHE_DIR = DATA_DIR / "cache"

CATALOG_FIXTURE = FIXTURES_DIR / "catalog.json"
BRIEFS_FIXTURE = FIXTURES_DIR / "briefs.json"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    llm_provider: str  # "fake" | "anthropic" | "cassette"
    embedder: str  # "hashing" | "minilm" | "voyage"
    image_embedder: str  # "color" | "clip"  (Phase 2 visual search)
    anthropic_api_key: str | None
    voyage_api_key: str | None
    llm_budget_usd: float
    db_url: str


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val is not None and val != "" else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        llm_provider=_env("STYLIST_LLM", "fake").lower(),
        embedder=_env("STYLIST_EMBEDDER", "hashing").lower(),
        image_embedder=_env("STYLIST_IMAGE_EMBEDDER", "color").lower(),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        voyage_api_key=os.environ.get("VOYAGE_API_KEY") or None,
        llm_budget_usd=float(_env("STYLIST_LLM_BUDGET_USD", "1.00")),
        db_url=_env("STYLIST_DB_URL", f"sqlite:///{DATA_DIR / 'stylist.db'}"),
    )


# ---------------------------------------------------------------------------
# Fixture loaders (plain dicts; validated into models by callers)
# ---------------------------------------------------------------------------
def load_catalog_dicts(path: Path | None = None) -> list[dict]:
    """Load the committed sample catalog fixture as a list of plain dicts."""
    path = path or CATALOG_FIXTURE
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_briefs(path: Path | None = None) -> list[dict]:
    """Load the seeded demo brief set."""
    path = path or BRIEFS_FIXTURE
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
