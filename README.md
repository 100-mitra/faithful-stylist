# The Faithful Stylist

**A grounded, evaluated conversational jewellery recommender (an "AI Stylist").**

The recommender is the vehicle; the payload is a **deterministic faithfulness verifier**.
Every *factual* claim in a recommendation's rationale (price, metal, stone, carat,
certification, "under budget") is **templated from the catalog record and verified** — any
factual claim not supported by the record is **blocked before it can reach the user**.
Subjective style language ("reads as vintage to me") is allowed, but always **labelled as
opinion**, never stated as a product fact.

> Scope: **Phase 0 + Phase 1** — a complete, shippable artifact. This is a portfolio/learning
> project, not a production system. The evaluation is deliberately honest about its limits
> (see **Limitations**).

## Why this is different

A conversational stylist that asks occasion/style/budget is table stakes, and off-the-shelf
bots routinely **hallucinate product claims**. The wedge here is **measured faithfulness +
honest evaluation**: a deterministic verifier that makes the stylist *structurally unable to
misstate a product*, plus an eval harness that measures what is objective and refuses to
dress up what isn't.

## Architecture

```
freeform brief
  └─▶ preference parse            (LLM → Pydantic profile; vocab-constrained, fail-loud)
       └─▶ hybrid retrieval       (HARD constraints as SQL filters FIRST, then semantic)
            └─▶ LLM rerank         (reorders survivors only — cannot break a constraint)
                 └─▶ grounded rationale
                      • LLM picks WHICH facts + writes opinion clauses (no values)
                      • code renders factual text from the record
                      • GroundingVerifier audits every claim (pure, no LLM, no network)
                      └─▶ recommendation + per-claim CLAIMS AUDIT  ◀── the differentiator
  └─▶ honest eval harness         (groundedness · constraint-sat · retrieval validity ·
                                   adversarial block rate · LLM-judge estimate · pairwise · $)
```

- **The LLM has no value channel for facts.** It returns a `RationaleDraft` of *field
  references* (`FactRef(field="carat")` — never a value) plus opinion clauses. Code renders
  `"1 ct"` from the record. So a fabricated price/metal/stone/carat is structurally impossible
  in the templated path, and a number/metal/stone smuggled into the opinion text is caught by
  a deterministic scan and blocked.
- `core/` is import-free of FastAPI — unit-testable and reusable on its own.
- Provider, embedder, and vector store all sit behind thin interfaces (swappable).

## Quickstart

```bash
docker compose up --build          # API + UI on http://localhost:8000
curl http://localhost:8000/healthz/
```

Everything runs by default on a **deterministic offline `fake` provider** — no API key, no
network, no cost — and the headline grounding demo works fully. For real LLM recommendations,
copy `.env.example` → `.env`, set `STYLIST_LLM=anthropic` and your `ANTHROPIC_API_KEY`.

### Try the headline in 10 seconds
Open `http://localhost:8000`, tick **"demo: inject a hallucinated claim"**, and hit
**Recommend** — watch the verifier block a fabricated carat before it reaches the rationale.

## API

```
POST /api/profile/parse     freeform text -> structured PreferenceProfile
POST /api/recommend         text -> ranked recs + grounded rationales + claims audit
GET  /api/products/{id}     catalog record + tags (factual vs inferred kept separate)
POST /api/eval/run          run the eval harness -> EvalRun
GET  /api/eval/{id}         metrics + human-readable report
GET  /api/tagging-accuracy  enrichment agreement metric (with N + caveat)
GET  /healthz/
```

### curl examples

```bash
# Recommend (grounded)
curl -s localhost:8000/api/recommend -H 'content-type: application/json' \
  -d '{"text":"vintage engagement ring, platinum or white gold, diamond, up to 1,50,000","top_k":3}'

# See a hallucination get blocked (claims_audit shows the blocked claim)
curl -s localhost:8000/api/recommend -H 'content-type: application/json' \
  -d '{"text":"a ring under 2,00,000","top_k":2,"demo_hallucination":true}'

# Adversarial hard constraints -> zero violations
curl -s localhost:8000/api/recommend -H 'content-type: application/json' \
  -d '{"text":"platinum only, nothing above 80,000, no gold","top_k":5}'

# Run the eval and read the report
curl -s localhost:8000/api/eval/run -H 'content-type: application/json' -d '{"n":40,"top_k":3}'
```

## Measured numbers

There are two kinds of number here and they are **not** the same kind of claim. Read the
distinction — it is the whole point of the project.

### A. Structural guarantees (provider-independent)

These come from **deterministic code** — the grounding verifier and the SQL filters — **not
from model accuracy**. They hold under *any* provider (measured offline at N=40; reconfirmed
in the real run). They are guarantees about the *architecture*, not claims that the LLM is
accurate or that the recommendations are good.

| Guarantee | Result | N | What it means — and what it does **not** |
|---|---|---|---|
| Factual claims grounded in output | 100% | 559 claims | **By construction**: factual text is rendered from the record, so a fabricated value cannot appear in the templated path. This is structural, not a measure of model accuracy. |
| Hard-constraint satisfaction | 100% | 114 items | Enforced by SQL, re-checked per returned item — **against the _parsed_ profile** (see caveats). |
| Retrieval validity | 100% | 114 items | Every returned item satisfies all parsed hard constraints. |
| Adversarial grounding block rate | 100% | 40 injected | The verifier blocks every hallucination in a **self-authored** set of 40 attacks (numbers / wrong metal / wrong stone / absent certification / over-budget smuggled into opinion text). |

Caveats — these bound what the 100%s actually claim:
- **"Grounded" ≠ "accurate" ≠ "good."** Groundedness means no *unsupported* factual claim
  survives to the user. It does not validate that the catalog data is itself correct, nor that
  the recommendation is a *good* pick.
- **Constraint satisfaction is measured against the _parsed_ profile.** Natural-language →
  structured parsing is a separate, upstream error source **not** captured by this number: if
  the parser misreads "no gold," retrieval cannot filter what it never parsed. Parsing quality
  is a distinct (and unmeasured-here) concern.
- **The adversarial set is self-authored (N=40) and covers _known_ attack patterns.** A 100%
  block rate shows the verifier catches the attacks I designed; it is **not** a proof of
  robustness against all phrasings — e.g. a true-but-misleading claim, or an attack built only
  from allowed vocabulary. It is a floor of demonstrated coverage, not a robustness guarantee.

### B. Subjective / model-dependent numbers — _pending a real run_

These require a real LLM and are **deliberately not reported from the offline provider** (the
fake judge and tagger are deterministic placeholders, so any number from them would be
meaningless dressed up as a result). They will be filled in from **one small, cost-capped
`anthropic` run**, each as an estimate with N and an explicit caveat:

| Metric | Status | How it will be reported |
|---|---|---|
| Subjective relevance | _pending real run_ | LLM-as-judge **estimate** with N, the judge prompt, and a "not validated accuracy" caveat. |
| Prompt-iteration pairwise win rate | _pending real run_ | Randomized order-shuffled pairwise (position-bias neutralized), with N. |
| Enrichment tagging agreement | _pending real run_ | Real inference vs the synthetic `style_hint` **pseudo-label** (not human-annotated), N=80. |
| Eval cost (USD) | _pending real run_ | Actual per-token cost of the capped run. |

> Reproducibility: the real run's responses are recorded as cassettes (`eval/cassettes/`) and a
> CI test replays them, so the published subjective numbers are reproducible without a key or
> further spend.

## Faithfulness & evaluation guarantees (the §4 contract)

- **Grounding invariant** — every factual claim is checkable against the record; unsupported
  ones are blocked. Headline test: `tests/test_grounding.py` injects a hallucinated claim and
  asserts it is blocked. If a fabricated factual claim can reach the user, the test fails.
- **Subjective vs factual separation** — factual fields are templated from the record (the LLM
  never generates them free-form) and are verbatim in output; opinion is labelled as opinion.
- **Constraint satisfaction** — hard constraints are retrieval-time SQL filters, not LLM
  promises; adversarial test asserts zero violations.
- **No fabricated relevance numbers** — subjective relevance is only ever an LLM-judge estimate
  with N + judge prompt + caveat. No metric is reused from another project.
- **Prompt-iteration evidence** — ≥2 rationale prompts compared via randomized, order-shuffled
  pairwise (neutralizes position bias); win rate reported.
- **Reproducibility & cost** — every LLM call is cached by `hash(task+prompt+context+params)`;
  identical requests return cached results; total $ is reported per eval run.

## Data

- **Synthetic generator** (`core/ingest/synthetic.py`) — deterministic, ships an 80-item
  committed fixture (`data/fixtures/catalog.json`). The whole pipeline runs end-to-end on it
  with no external dependency.
- **Responsible real-brand scraper** (`core/ingest/scrape.py`) — targets a Shopify storefront's
  public `products.json`; **respects robots.txt** (fetched with our descriptive User-Agent),
  rate-limits (≥3s, single thread), caches to disk, extracts **only** the factual fields (+ a
  thumbnail URL, never the bytes), and writes to **gitignored** `data/scraped/` — the dataset
  is **never committed**. Style tags are still inferred by the LLM ("scrape facts, infer
  style"). The pipeline runs end-to-end on either source.

## Development

```bash
python -m venv .venv && . .venv/Scripts/activate     # (Windows) or .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest                                                # offline, no key, fully deterministic
```

CI (GitHub Actions) runs ruff + pytest on the `fake` provider — green without any secret.
Required tests (all green): grounding headline (hallucination blocked), factual-fields-verbatim,
constraint-satisfaction adversarial, retrieval-filter, enrichment-schema validation, API
integration, reproducibility/hash, scraper robots/parse.

## Deploy (Render)

`render.yaml` deploys the API + UI as a single Docker web service. The public demo runs on the
offline `fake` provider by default (no key, no cost — the grounding demo still works). To serve
real LLM recommendations, set `STYLIST_LLM=anthropic` and add `ANTHROPIC_API_KEY` as a secret
in the Render dashboard.

```
1. Push to GitHub.
2. Render → New → Blueprint → pick this repo (reads render.yaml).
3. (Optional) Dashboard → Environment → set STYLIST_LLM=anthropic + ANTHROPIC_API_KEY.
```

> Live demo URL: _added once deployed._

## Limitations (read these)

- **Offline evaluation has no real users.** Without interaction/click data, recommender
  evaluation is inherently weak; nothing here measures real-world conversion or satisfaction.
- **Faithfulness is bounded by parsing.** The grounding guarantee covers what the system
  *outputs*; it does not cover the natural-language → structured-profile parse. If the parser
  misreads a constraint, retrieval filters the wrong thing — and the verifier won't catch that,
  because the returned facts are still grounded in the (wrongly-retrieved) record. Parsing
  quality is a separate, upstream concern not measured by the grounding numbers.
- **Adversarial coverage is self-authored.** The 100% block rate is over a set of attacks I
  wrote (N=40). It demonstrates the verifier catches known patterns; it is not a robustness
  proof against adversarial phrasings I didn't think of (e.g. true-but-misleading claims, or
  attacks confined to the allowed vocabulary).
- **Subjective relevance has no ground truth.** It is an LLM-as-judge *estimate*, never
  validated accuracy — and it is reported only from a real run, never from the fake provider.
- **Synthetic / sparse labels.** Style tags are LLM-inferred; on synthetic data the tagging
  metric compares against a generator hint (a pseudo-label), not human annotation.
- **The hashing embedder is lexical**, not a learned semantic model — adequate for a small
  catalog after SQL filtering, but not state-of-the-art retrieval. (MiniLM/Voyage are swappable.)
- **Not production-ready.** No auth, no real persistence guarantees on free-tier, no scale.

## Repo layout

```
core/      framework-free pipeline: ingest (synthetic + scraper), enrich, embed, profile,
           retrieve, rerank, rationale, grounding (the headline), llm (provider seam + cache)
app/       FastAPI routes + the single-file demo UI (templates/index.html)
eval/      brief set, LLM-judge, order-shuffled pairwise, metrics + report
tests/     incl. test_grounding.py (headline), test_retrieve.py (adversarial constraints)
data/      fixtures/ (committed synthetic sample) · cache/ + scraped/ (gitignored)
```
