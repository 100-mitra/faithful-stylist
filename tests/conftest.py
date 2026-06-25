"""Shared fixtures: a seeded in-memory catalog + offline providers.

Everything here uses the deterministic FakeProvider and the hashing embedder, so the
whole suite runs with no API key and no network.
"""

import pytest

from core.config import load_catalog_dicts
from core.embed import get_embedder
from core.enrich import enrich_catalog
from core.llm import FakeProvider, ResponseCache
from core.models import Product
from core.store import init_db, make_engine, seed_catalog


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def cached_provider():
    from core.llm import CachingProvider

    return CachingProvider(FakeProvider(), ResponseCache())


@pytest.fixture
def embedder():
    return get_embedder()


@pytest.fixture
def catalog() -> list[Product]:
    return [Product(**d) for d in load_catalog_dicts()]


@pytest.fixture
def tags_by_id(catalog, provider):
    return enrich_catalog(catalog, provider)


@pytest.fixture
def engine(catalog, tags_by_id):
    eng = make_engine("sqlite://")
    init_db(eng)
    seed_catalog(eng, catalog, list(tags_by_id.values()))
    return eng
