# Faithful Stylist

**A grounded, evaluated conversational jewellery recommender (an "AI Stylist").**

The recommender is the vehicle; the payload is a **deterministic faithfulness verifier**: every
*factual* claim in a recommendation's rationale (price, metal, stone, carat, certification, "under
budget") is templated from the catalog record and **verified** — any factual claim not supported by
the record is **blocked** before it can reach the user. Subjective style language ("reads as vintage
to me") is allowed, but always labelled as opinion, never as a product fact.

> Status: **Phase 0 + Phase 1** (the complete, shippable deliverable). Phases 2–3 (CLIP visual
> search, learned ranker) are deliberately out of scope until Phase 1 ships. This is a
> portfolio/learning artifact, not a production system — see **Limitations**.

## Architecture (Phase 1)

```
freeform brief
   -> preference parse (LLM, Pydantic, fail-loud)
   -> hybrid retrieval (hard constraints as SQL filters FIRST, then semantic similarity)
   -> LLM rerank (position-bias-aware)
   -> grounded rationale: LLM picks WHICH facts + writes opinion clauses;
        code renders factual text from the record; GroundingVerifier audits every claim
   -> claims audit (each claim tagged factual/subjective + grounded flag) exposed in the UI/API
   -> honest eval harness (groundedness, constraint-satisfaction, retrieval validity;
        subjective relevance only as an LLM-judge estimate with N + caveat)
```

`core/` is import-free of FastAPI so it is unit-testable and reusable on its own.

## Quickstart

```bash
docker-compose up --build        # boots the API + minimal UI on http://localhost:8000
# health check:
curl http://localhost:8000/healthz/
```

By default everything runs on the **deterministic fake LLM provider** (no API key, no network). For
real recommendations, copy `.env.example` to `.env`, set `STYLIST_LLM=anthropic` and your
`ANTHROPIC_API_KEY`.

## Development

```bash
python -m venv .venv && . .venv/Scripts/activate   # (Windows) or .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

CI (GitHub Actions) runs ruff + pytest with the fake provider — green without any secret.

## Measured numbers

_Filled in at the end of Phase 1 from a real eval run — every number freshly measured on this
project, reported with N and explicit caveats. No borrowed or fabricated metrics._

## Limitations

_Offline evaluation without real user-interaction data is inherently weak; subjective recommendation
relevance has no ground truth here; synthetic/sparse style labels. Detailed in the final README._
