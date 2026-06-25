"""Record ONE real, cost-capped Anthropic eval run and write reproducible cassettes.

Usage (with ANTHROPIC_API_KEY set in the environment):

    python -m eval.record --n 10 --top-k 3 --budget 1.50

Writes:
  * eval/cassettes/<key>.json  — every real LLM response, keyed by the same content hash the
    CassetteProvider replays (so CI reproduces the numbers with no key and no spend).
  * eval/real_run.json         — the EvalRun metrics + report + real tagging accuracy
    (committed; the README's subjective numbers come from here).

Prints the actual USD cost. The AnthropicProvider enforces the budget cap, so a too-tight cap
aborts mid-run (raise it and re-run; cassettes already written are reused).
"""

from __future__ import annotations

import argparse
import json
import os

from core.config import ROOT_DIR, get_settings
from core.embed import get_embedder
from core.llm import CachingProvider, ResponseCache, build_inner_provider
from core.pipeline import build_catalog, tagging_accuracy
from eval.briefs import make_eval_briefs
from eval.harness import format_report, run_eval

CASSETTE_DIR = ROOT_DIR / "eval" / "cassettes"
REAL_RUN_PATH = ROOT_DIR / "eval" / "real_run.json"


def record(n: int, top_k: int, budget: float) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set — cannot do a real run.")

    # Force a real anthropic run; run_eval reads get_settings() to label the judge as real.
    os.environ["STYLIST_LLM"] = "anthropic"
    os.environ["STYLIST_LLM_BUDGET_USD"] = str(budget)
    get_settings.cache_clear()
    settings = get_settings()

    # The CachingProvider's disk cache IS the cassette store (same key + payload format),
    # so a real run records cassettes automatically.
    provider = CachingProvider(build_inner_provider(settings), ResponseCache(CASSETTE_DIR))
    embedder = get_embedder(settings)

    ctx = build_catalog(provider, db_url="sqlite://")  # real Haiku enrichment (cached -> cassettes)
    run = run_eval(ctx, provider, embedder, briefs=make_eval_briefs(n), top_k=top_k)
    tagging = tagging_accuracy(ctx)  # real inference vs the synthetic style_hint pseudo-label

    payload = {
        "recorded_n": n,
        "top_k": top_k,
        "budget_usd": budget,
        "eval": run.model_dump(),
        "report": format_report(run),
        "tagging": tagging,
    }
    REAL_RUN_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    n_cassettes = len(list(CASSETTE_DIR.glob("*.json")))
    print(
        f"Real run complete: n={n}, top_k={top_k}, cost=${run.cost_usd:.4f}, "
        f"cassettes={n_cassettes}\nWrote {REAL_RUN_PATH}"
    )
    return payload


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description="Record a real, cost-capped eval run + cassettes.")
    ap.add_argument("--n", type=int, default=10, help="number of eval briefs (keep small)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=float, default=1.50, help="hard USD cost cap")
    args = ap.parse_args()
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    record(args.n, args.top_k, args.budget)
