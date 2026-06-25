"""P1.10: the eval harness emits real, honestly-captioned numbers (no fabrication)."""

from core.llm import CachingProvider, FakeProvider, ResponseCache
from core.pipeline import build_catalog
from eval.briefs import make_eval_briefs
from eval.harness import format_report, run_eval


def _provider():
    return CachingProvider(FakeProvider(), ResponseCache())


def test_structural_guarantees_are_real_and_provider_independent(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    briefs = make_eval_briefs(12)
    run = run_eval(ctx, provider, embedder, briefs=briefs, top_k=3)
    m = run.metrics

    assert m["n_briefs"] == 12
    # Structural (deterministic code): grounded by construction, constraints enforced by SQL.
    assert m["groundedness_rate"] == 1.0 and m["n_factual_claims"] > 0
    assert m["constraint_satisfaction_rate"] == 1.0 and m["n_returned_items"] > 0
    assert m["retrieval_validity_rate"] == 1.0
    # The verifier actively blocks every injected hallucination (self-authored set).
    assert m["adversarial_grounding_block_rate"] == 1.0 and m["n_adversarial"] > 0
    # The honesty caveats must accompany the structural numbers.
    assert "PARSED profile" in m["constraint_note"]
    assert "self-authored" in m["adversarial_note"]
    assert run.cost_usd == 0.0


def test_offline_subjective_numbers_are_suppressed_not_faked(embedder):
    # On the fake provider, NO subjective number may be presented as a result.
    provider = _provider()
    ctx = build_catalog(provider)
    run = run_eval(ctx, provider, embedder, briefs=make_eval_briefs(12), top_k=3)
    sr = run.metrics["subjective_relevance"]
    pw = run.metrics["pairwise_prompt_iteration"]
    assert sr["offline_placeholder"] is True
    assert sr["mean_score_1to5"] is None  # never a fake number
    assert "OFFLINE PLACEHOLDER" in sr["caveat"]
    assert "judge_prompt" in sr
    assert pw["offline_placeholder"] is True and pw["variant_b_win_rate"] is None


def test_report_frames_structurally_and_hides_offline_subjective(embedder):
    provider = _provider()
    ctx = build_catalog(provider)
    run = run_eval(ctx, provider, embedder, briefs=make_eval_briefs(8), top_k=3)
    report = format_report(run)
    assert "Structural guarantees" in report
    assert "NOT accuracy" in report
    assert "self-authored" in report  # adversarial caveat surfaced
    assert "PARSED profile" in report  # constraint caveat surfaced
    assert "offline placeholder" in report.lower()
    # No fabricated subjective score in the offline report.
    assert "/5 (N=" not in report
