"""P1.2: LLM cache key stability + reproducibility (brief §4 reproducibility/hash test)."""

from core.llm import (
    CachingProvider,
    FakeProvider,
    ResponseCache,
    get_provider,
    make_cache_key,
)
from core.models import ProfileDraft, StyleTagDraft


def test_cache_key_is_stable_and_input_sensitive():
    a = make_cache_key("parse_profile", "claude-haiku-4-5", ProfileDraft, "sys", "p", {"x": 1})
    b = make_cache_key("parse_profile", "claude-haiku-4-5", ProfileDraft, "sys", "p", {"x": 1})
    c = make_cache_key("parse_profile", "claude-haiku-4-5", ProfileDraft, "sys", "p", {"x": 2})
    assert a == b
    assert a != c


def test_identical_requests_hit_cache_once():
    inner = FakeProvider()
    provider = CachingProvider(inner, ResponseCache())
    kwargs = dict(
        task="parse_profile",
        schema=ProfileDraft,
        prompt="vintage ring under 1,50,000",
        context={"raw_text": "vintage ring under 1,50,000"},
    )
    first = provider.complete_structured(**kwargs)
    second = provider.complete_structured(**kwargs)
    assert first == second
    assert inner.calls() == 1, "second identical request must be served from cache"
    assert provider.cache_hits() == 1


def test_disk_cache_round_trip(tmp_path):
    cache = ResponseCache(disk_dir=tmp_path)
    provider = CachingProvider(FakeProvider(), cache)
    out = provider.complete_structured(
        task="enrich",
        schema=StyleTagDraft,
        prompt="tag this",
        context={"title": "Rose Gold Vintage Diamond Ring", "style_hint": "vintage"},
    )
    # A fresh cache pointed at the same dir replays the stored result with no inner call.
    fresh_inner = FakeProvider()
    replay = CachingProvider(fresh_inner, ResponseCache(disk_dir=tmp_path))
    out2 = replay.complete_structured(
        task="enrich",
        schema=StyleTagDraft,
        prompt="tag this",
        context={"title": "Rose Gold Vintage Diamond Ring", "style_hint": "vintage"},
    )
    assert out == out2
    assert fresh_inner.calls() == 0


def test_default_provider_is_offline_fake():
    # With no env configured, the factory must yield a keyless, offline provider.
    provider = get_provider(cache=ResponseCache())
    draft = provider.complete_structured(
        task="parse_profile",
        schema=ProfileDraft,
        prompt="platinum only under 80,000",
        context={"raw_text": "platinum only under 80,000"},
    )
    assert isinstance(draft, ProfileDraft)
    assert provider.cost_usd() == 0.0
