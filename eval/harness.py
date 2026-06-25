"""Honest evaluation harness (brief §4).

Measures the things that ARE objective (groundedness, constraint-satisfaction, retrieval
validity, adversarial block rate) directly. Reports subjective relevance ONLY as an
LLM-as-judge estimate WITH N, the judge prompt, and an explicit caveat. Includes >=2
measured rationale prompt iterations via order-shuffled pairwise comparison, and the
total $ cost of the run. Never fabricates, reuses, or oversells a metric.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from statistics import mean

from core.config import get_settings
from core.embed import Embedder, get_embedder
from core.grounding import GroundingVerifier
from core.llm import LLMProvider, get_provider
from core.models import EvalRun, FactRef, RationaleDraft
from core.pipeline import CatalogContext, recommend
from core.rationale import build_rationale
from core.retrieve import constraint_violations
from eval.briefs import make_eval_briefs
from eval.judge import JUDGE_PROMPT, judge_pairwise_variants, judge_relevance

_VERIFIER = GroundingVerifier()


def run_eval(
    ctx: CatalogContext,
    provider: LLMProvider | None = None,
    embedder: Embedder | None = None,
    briefs: list[dict] | None = None,
    top_k: int = 3,
) -> EvalRun:
    provider = provider or get_provider()
    embedder = embedder or get_embedder()
    briefs = briefs if briefs is not None else make_eval_briefs()

    total_factual = grounded_factual = 0
    constraint_checks = constraint_ok = 0
    adversarial_total = adversarial_blocked = 0
    relevance_scores: list[int] = []
    pairwise_total = v2_wins = 0

    for brief in briefs:
        profile, recs = recommend(brief["text"], ctx, provider, embedder, top_k=top_k)
        for rec in recs:
            for claim in rec.claims:
                if claim.claim_type == "factual":
                    total_factual += 1
                    if claim.grounded and not claim.blocked:
                        grounded_factual += 1
            constraint_checks += 1
            if not constraint_violations(rec.product, profile):
                constraint_ok += 1

        if not recs:
            continue
        top = recs[0]

        # Subjective relevance (LLM-as-judge estimate).
        relevance_scores.append(judge_relevance(profile, top, provider).score)

        # Adversarial groundedness: a forced hallucination MUST be blocked.
        poisoned = RationaleDraft(
            product_id=top.product.id,
            factual_slots=[FactRef(field="metal")],
            subjective_clauses=["a breathtaking 9-carat centre stone"],
            recommended=True,
        )
        adversarial_total += 1
        if _VERIFIER.verify(poisoned, top.product, profile).decision == "blocked":
            adversarial_blocked += 1

        # Prompt-iteration evidence: v1 vs v2, order-shuffled pairwise.
        v1 = build_rationale(profile, top.product, provider, variant="v1")
        v2 = build_rationale(profile, top.product, provider, variant="v2")
        pairwise_total += 1
        if judge_pairwise_variants(profile, v1, v2, provider, brief["id"]) == "v2":
            v2_wins += 1

    settings = get_settings()
    judge_is_real = settings.llm_provider == "anthropic"
    metrics = {
        "n_briefs": len(briefs),
        # --- Objective ---
        "groundedness_rate": round(grounded_factual / total_factual, 4) if total_factual else None,
        "n_factual_claims": total_factual,
        "constraint_satisfaction_rate": round(constraint_ok / constraint_checks, 4)
        if constraint_checks
        else None,
        "retrieval_validity_rate": round(constraint_ok / constraint_checks, 4)
        if constraint_checks
        else None,
        "n_returned_items": constraint_checks,
        "adversarial_grounding_block_rate": round(adversarial_blocked / adversarial_total, 4)
        if adversarial_total
        else None,
        "n_adversarial": adversarial_total,
        # --- Subjective (estimate only) ---
        "subjective_relevance": {
            "mean_score_1to5": round(mean(relevance_scores), 3) if relevance_scores else None,
            "n": len(relevance_scores),
            "scale": "1-5 (LLM-as-judge)",
            "judge_prompt": JUDGE_PROMPT,
            "caveat": (
                "LLM-as-judge estimate over a small self-built brief set. This is NOT "
                "validated relevance accuracy: there is no human ground truth and the judge "
                "is itself an LLM. Treat as directional only."
            ),
        },
        # --- Prompt iteration ---
        "pairwise_prompt_iteration": {
            "variant_b_win_rate": round(v2_wins / pairwise_total, 4) if pairwise_total else None,
            "n": pairwise_total,
            "method": "randomized order-shuffled pairwise (position bias neutralized)",
            "variants": "v1 (concise) vs v2 (occasion-enriched)",
        },
        # --- Provenance ---
        "provider": settings.llm_provider,
        "judge_model": "claude-sonnet-4-6" if judge_is_real else "deterministic-offline-fake",
        "embedder": (embedder or get_embedder()).name,
        "catalog_snapshot": ctx.snapshot_hash,
    }

    created_at = datetime.now(UTC).isoformat()
    run_id = (
        "eval-"
        + hashlib.sha256(
            f"{ctx.snapshot_hash}-{len(briefs)}-{settings.llm_provider}-{created_at}".encode()
        ).hexdigest()[:12]
    )
    return EvalRun(
        id=run_id,
        dataset_ref=f"generated:eval_briefs(n={len(briefs)})",
        metrics=metrics,
        cost_usd=round(provider.cost_usd(), 6),
        created_at=created_at,
    )


def format_report(run: EvalRun) -> str:
    m = run.metrics
    sr = m["subjective_relevance"]
    pw = m["pairwise_prompt_iteration"]

    def pct(x):
        return "n/a" if x is None else f"{x * 100:.1f}%"

    return "\n".join(
        [
            f"# Eval report — {run.id}",
            f"Dataset: {run.dataset_ref}  |  Provider: {m['provider']}  |  "
            f"Embedder: {m['embedder']}  |  Cost: ${run.cost_usd:.4f}",
            "",
            "## Objective metrics",
            f"- Groundedness (factual claims grounded): {pct(m['groundedness_rate'])} "
            f"(N={m['n_factual_claims']} claims)",
            f"- Constraint satisfaction: {pct(m['constraint_satisfaction_rate'])} "
            f"(N={m['n_returned_items']} returned items)",
            f"- Retrieval validity: {pct(m['retrieval_validity_rate'])} "
            f"(N={m['n_returned_items']})",
            f"- Adversarial grounding block rate: {pct(m['adversarial_grounding_block_rate'])} "
            f"(N={m['n_adversarial']} injected hallucinations)",
            "",
            "## Subjective relevance (ESTIMATE — not validated accuracy)",
            f"- Mean LLM-judge score: {sr['mean_score_1to5']}/5 (N={sr['n']}, "
            f"judge={m['judge_model']})",
            f"- Caveat: {sr['caveat']}",
            "",
            "## Prompt iteration",
            f"- Variant B win rate: {pct(pw['variant_b_win_rate'])} (N={pw['n']}, "
            f"{pw['method']}; {pw['variants']})",
        ]
    )
