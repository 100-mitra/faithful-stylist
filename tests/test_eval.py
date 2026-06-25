"""P1.10: the eval harness emits real, honestly-captioned numbers (no fabrication)."""

from core.llm import CachingProvider, FakeProvider, ResponseCache
from core.pipeline import build_catalog
from eval.briefs import make_eval_briefs
from eval.harness import format_report, run_eval


def _provider():
    return CachingProvider(FakeProvider(), ResponseCache())


def test_eval_objective_metrics_are_real(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    briefs = make_eval_briefs(12)
    run = run_eval(ctx, provider, embedder, briefs=briefs, top_k=3)
    m = run.metrics

    assert m["n_briefs"] == 12
    # Honest pipeline => fully grounded, fully constraint-satisfying.
    assert m["groundedness_rate"] == 1.0 and m["n_factual_claims"] > 0
    assert m["constraint_satisfaction_rate"] == 1.0 and m["n_returned_items"] > 0
    assert m["retrieval_validity_rate"] == 1.0
    # The verifier actively blocks every injected hallucination.
    assert m["adversarial_grounding_block_rate"] == 1.0 and m["n_adversarial"] > 0
    # Offline fake run costs nothing.
    assert run.cost_usd == 0.0


def test_subjective_relevance_is_estimate_with_n_and_caveat(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    run = run_eval(ctx, provider, embedder, briefs=make_eval_briefs(12), top_k=3)
    sr = run.metrics["subjective_relevance"]
    assert sr["n"] > 0
    assert sr["mean_score_1to5"] is not None
    assert "NOT validated" in sr["caveat"]
    assert "judge_prompt" in sr


def test_pairwise_prompt_iteration_reported(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    run = run_eval(ctx, provider, embedder, briefs=make_eval_briefs(12), top_k=3)
    pw = run.metrics["pairwise_prompt_iteration"]
    assert pw["n"] > 0
    assert pw["variant_b_win_rate"] is not None
    assert "shuffled" in pw["method"]


def test_report_renders_without_overclaiming(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    run = run_eval(ctx, provider, embedder, briefs=make_eval_briefs(8), top_k=3)
    report = format_report(run)
    assert "Groundedness" in report
    assert "not validated accuracy" in report.lower()
    assert "Adversarial grounding block rate" in report
