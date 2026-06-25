"""LLM provider seam — a thin, swappable interface over the model provider.

Three implementations behind one Protocol:
  * ``AnthropicProvider``  — real Claude (Haiku for tagging/parse, Sonnet for rerank/
    rationale/judge), structured outputs, with a hard cost cap.
  * ``FakeProvider``       — deterministic, no network, no key (CI/default). Produces a
    genuinely working pipeline via the textparse backbone.
  * ``CassetteProvider``   — replays recorded real responses (reproducible eval, no key).

Every call is cached by a content hash so identical requests return identical results
(reproducibility, brief §4) and cost is bounded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from core import textparse
from core.config import CACHE_DIR, Settings, get_settings
from core.models import RationaleDraft, RerankResult

# Task -> model tier (brief §3). Haiku for high-volume tagging/parse; Sonnet for reasoning.
TASK_MODEL: dict[str, str] = {
    "enrich": "claude-haiku-4-5",
    "parse_profile": "claude-haiku-4-5",
    "rerank": "claude-sonnet-4-6",
    "rationale": "claude-sonnet-4-6",
    "judge": "claude-sonnet-4-6",
}

# USD per 1M tokens (input, output). Source: claude-api reference, verified at build time.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

_DEFAULT_SYSTEM = (
    "You are a precise jewellery-domain assistant. Return only the requested structured "
    "data. Never invent factual product attributes (price, metal, stone, carat); those are "
    "supplied separately. Subjective style language must be framed as opinion."
)


class LLMError(RuntimeError):
    pass


class LLMBudgetExceeded(LLMError):
    pass


# ---------------------------------------------------------------------------
# Cache key + store
# ---------------------------------------------------------------------------
def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def make_cache_key(
    task: str,
    model: str,
    schema: type[BaseModel],
    system: str,
    prompt: str,
    context: dict | None,
) -> str:
    payload = {
        "task": task,
        "model": model,
        "schema": schema.__name__,
        "system": system,
        "prompt": prompt,
        "context": _canonical(context or {}),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    """In-memory cache with optional JSON-on-disk persistence (reproducibility/cost)."""

    def __init__(self, disk_dir: Path | None = None):
        self._mem: dict[str, dict] = {}
        self._disk = disk_dir
        if self._disk is not None:
            self._disk.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        if key in self._mem:
            return self._mem[key]
        if self._disk is not None:
            path = self._disk / f"{key}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self._mem[key] = data
                return data
        return None

    def set(self, key: str, data: dict) -> None:
        self._mem[key] = data
        if self._disk is not None:
            (self._disk / f"{key}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class LLMProvider(Protocol):
    def complete_structured(
        self,
        *,
        task: str,
        schema: type[BaseModel],
        prompt: str,
        system: str = "",
        context: dict | None = None,
    ) -> BaseModel: ...

    def cost_usd(self) -> float: ...

    def calls(self) -> int: ...


class _BaseProvider:
    def __init__(self) -> None:
        self._cost = 0.0
        self._calls = 0

    def cost_usd(self) -> float:
        return self._cost

    def calls(self) -> int:
        return self._calls


class AnthropicProvider(_BaseProvider):
    """Real Claude via the Anthropic SDK structured-output (`messages.parse`) path."""

    def __init__(self, api_key: str, budget_usd: float):
        super().__init__()
        import anthropic  # imported lazily so the package isn't required for fake/CI runs.

        self._client = anthropic.Anthropic(api_key=api_key)
        self._budget = budget_usd

    def _add_cost(self, model: str, in_tok: int, out_tok: int) -> None:
        pin, pout = PRICING_PER_MTOK[model]
        self._cost += (in_tok * pin + out_tok * pout) / 1_000_000

    def complete_structured(self, *, task, schema, prompt, system="", context=None):
        if self._cost >= self._budget:
            raise LLMBudgetExceeded(f"LLM cost cap ${self._budget:.2f} reached")
        model = TASK_MODEL[task]
        resp = self._client.messages.parse(
            model=model,
            max_tokens=2048,
            system=system or _DEFAULT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        usage = resp.usage
        self._add_cost(model, usage.input_tokens, usage.output_tokens)
        self._calls += 1
        return resp.parsed_output


def _fake_enrich(ctx: dict, schema: type[BaseModel]) -> BaseModel:
    return textparse.style_tags_from_text(
        title=ctx.get("title", ""),
        description=ctx.get("description", ""),
        style_hint=ctx.get("style_hint"),
        occasion_hint=ctx.get("occasion_hint"),
    )


def _fake_parse_profile(ctx: dict, schema: type[BaseModel]) -> BaseModel:
    return textparse.parse_profile_text(ctx.get("raw_text", ""))


def _fake_rerank(ctx: dict, schema: type[BaseModel]) -> BaseModel:
    # Identity ordering: retrieval already ranked; a real LLM would reorder.
    return RerankResult(ranked_ids=list(ctx.get("candidate_ids", [])))


def _fake_rationale(ctx: dict, schema: type[BaseModel]) -> BaseModel:
    p = ctx.get("product", {})
    prof = ctx.get("profile", {})
    slots: list[dict] = [{"field": "metal"}, {"field": "price"}, {"field": "category"}]
    if p.get("stone_primary"):
        slots.append({"field": "stone_primary"})
    if p.get("carat") is not None:
        slots.append({"field": "carat"})
    budget = prof.get("budget_max")
    if budget and p.get("price") is not None and p["price"] <= budget:
        slots.append({"field": "under_budget"})
    style = (p.get("raw_attributes") or {}).get("style_hint")
    styles = prof.get("styles") or []
    chosen = style or (styles[0] if styles else "classic")
    return RationaleDraft(
        product_id=p.get("id", ""),
        factual_slots=slots,  # type: ignore[arg-type]
        subjective_clauses=[f"reads as {chosen} to me"],
        recommended=True,
    )


# Registry of deterministic fake handlers. The "judge" handler is registered in P1.10.
_FAKE_HANDLERS: dict[str, Any] = {
    "enrich": _fake_enrich,
    "parse_profile": _fake_parse_profile,
    "rerank": _fake_rerank,
    "rationale": _fake_rationale,
}


class FakeProvider(_BaseProvider):
    """Deterministic, offline. ``overrides`` lets a test inject behaviour (e.g. poison)."""

    def __init__(self, overrides: dict[str, Any] | None = None):
        super().__init__()
        self._overrides = overrides or {}

    def complete_structured(self, *, task, schema, prompt, system="", context=None):
        self._calls += 1
        ctx = context or {}
        handler = self._overrides.get(task) or _FAKE_HANDLERS.get(task)
        if handler is None:
            return schema()  # minimal schema-valid default
        return handler(ctx, schema)


class CassetteProvider(_BaseProvider):
    """Replays recorded real responses keyed by cache key (no network/key needed)."""

    def __init__(self, cassette_dir: Path):
        super().__init__()
        self._dir = cassette_dir

    def complete_structured(self, *, task, schema, prompt, system="", context=None):
        model = TASK_MODEL[task]
        key = make_cache_key(task, model, schema, system or _DEFAULT_SYSTEM, prompt, context)
        path = self._dir / f"{key}.json"
        if not path.exists():
            raise LLMError(
                f"No cassette for {task} (key {key[:12]}…). Record one with a real run first."
            )
        self._calls += 1
        return schema(**json.loads(path.read_text(encoding="utf-8")))


class CachingProvider(_BaseProvider):
    """Wraps a provider with a content-hash cache (reproducibility + cost control)."""

    def __init__(self, inner: LLMProvider, cache: ResponseCache):
        super().__init__()
        self._inner = inner
        self._cache = cache
        self._hits = 0

    def cost_usd(self) -> float:
        return self._inner.cost_usd()

    def calls(self) -> int:
        return self._inner.calls()

    def cache_hits(self) -> int:
        return self._hits

    def complete_structured(self, *, task, schema, prompt, system="", context=None):
        model = TASK_MODEL[task]
        sys_text = system or _DEFAULT_SYSTEM
        key = make_cache_key(task, model, schema, sys_text, prompt, context)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return schema(**cached)
        result = self._inner.complete_structured(
            task=task, schema=schema, prompt=prompt, system=sys_text, context=context
        )
        self._cache.set(key, result.model_dump())
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_inner_provider(settings: Settings) -> LLMProvider:
    kind = settings.llm_provider
    if kind == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("STYLIST_LLM=anthropic but ANTHROPIC_API_KEY is not set.")
        return AnthropicProvider(settings.anthropic_api_key, settings.llm_budget_usd)
    if kind == "cassette":
        from core.config import ROOT_DIR

        return CassetteProvider(ROOT_DIR / "eval" / "cassettes")
    return FakeProvider()


def get_provider(
    settings: Settings | None = None,
    cache: ResponseCache | None = None,
) -> CachingProvider:
    settings = settings or get_settings()
    inner = build_inner_provider(settings)
    cache = cache if cache is not None else ResponseCache(CACHE_DIR / "llm")
    return CachingProvider(inner, cache)
