"""Reproducibility: replay recorded real LLM responses (cassettes) with NO key, NO network,
and assert the published real numbers reproduce. Skips until a real run has been recorded
(`python -m eval.record`), so CI stays green before the cassettes exist.
"""

import json

import pytest

from core.config import ROOT_DIR, get_settings
from core.embed import get_embedder
from core.llm import CachingProvider, ResponseCache, build_inner_provider
from core.pipeline import build_catalog, tagging_accuracy
from eval.briefs import make_eval_briefs
from eval.harness import run_eval

_CASSETTE_DIR = ROOT_DIR / "eval" / "cassettes"
_REAL_RUN = ROOT_DIR / "eval" / "real_run.json"

pytestmark = pytest.mark.skipif(
    not _REAL_RUN.exists() or not any(_CASSETTE_DIR.glob("*.json")),
    reason="no recorded real run yet (run `python -m eval.record` with a key)",
)


@pytest.fixture
def cassette_env(monkeypatch):
    monkeypatch.setenv("STYLIST_LLM", "cassette")
    monkeypatch.setenv("STYLIST_EMBEDDER", "hashing")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _cassette_provider():
    settings = get_settings()
    return CachingProvider(build_inner_provider(settings), ResponseCache())


def test_cassette_reproduces_recorded_real_numbers(cassette_env):
    expected = json.loads(_REAL_RUN.read_text(encoding="utf-8"))
    n, top_k = expected["recorded_n"], expected["top_k"]

    provider = _cassette_provider()
    embedder = get_embedder(get_settings())
    ctx = build_catalog(provider, db_url="sqlite://")
    run = run_eval(ctx, provider, embedder, briefs=make_eval_briefs(n), top_k=top_k)

    em, rm = expected["eval"]["metrics"], run.metrics
    # Subjective numbers are real (cassette replays real judge) and reproduce exactly.
    assert (
        rm["subjective_relevance"]["mean_score_1to5"]
        == em["subjective_relevance"]["mean_score_1to5"]
    )
    assert (
        rm["pairwise_prompt_iteration"]["variant_b_win_rate"]
        == em["pairwise_prompt_iteration"]["variant_b_win_rate"]
    )
    # Structural metrics reproduce the recorded values exactly under replay. (Groundedness
    # on a real run is < 1.0 — the verifier blocks some of the model's attempted claims — so
    # assert reproduction of the recorded number, not a hardcoded 1.0.)
    assert rm["groundedness_rate"] == em["groundedness_rate"]
    assert rm["constraint_satisfaction_rate"] == em["constraint_satisfaction_rate"]
    assert rm["adversarial_grounding_block_rate"] == em["adversarial_grounding_block_rate"]
    # Real tagging accuracy reproduces.
    assert tagging_accuracy(ctx)["tagging_accuracy"] == expected["tagging"]["tagging_accuracy"]
