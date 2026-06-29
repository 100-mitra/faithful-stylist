# The Faithful Stylist

**A grounded, evaluated conversational jewellery recommender (an "AI Stylist").**

The recommender is the vehicle; the payload is a **deterministic faithfulness verifier**.
Every *factual* claim in a recommendation's rationale (price, metal, stone, carat,
certification, "under budget") is **templated from the catalog record and verified** — any
factual claim not supported by the record is **blocked before it can reach the user**.
Subjective style language ("reads as vintage to me") is allowed, but always **labelled as
opinion**, never stated as a product fact.

**▶ Live demo: https://faithful-stylist.onrender.com** — submit a brief, get ranked recs with a
visible claims audit, tick **"inject a hallucinated claim"** to watch the verifier block it, and
run the eval. Keyless and free (runs on the offline provider). Free tier spins down when idle, so
the first request may cold-start for ~30–60s.

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
  └─▶ preference parse            (vocab-constrained → Pydantic profile; deterministic
                                   textparse on the offline provider, LLM on anthropic)
       └─▶ hybrid retrieval       (HARD constraints as SQL filters FIRST, then semantic)
            └─▶ LLM rerank         (reorders survivors only — cannot break a constraint)
                 └─▶ grounded rationale
                      • LLM picks WHICH facts + writes opinion clauses (no values)
                      • code renders factual text from the record
                      • GroundingVerifier audits every claim (pure, no LLM, no network)
                      └─▶ recommendation + per-claim CLAIMS AUDIT  ◀── the differentiator
  └─▶ honest eval harness         (structural: groundedness · constraint-sat · retrieval
                                   validity · adversarial block rate │ real-run-only:
                                   LLM-judge estimate · pairwise · $ — null/placeholder offline)
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

The headline result and the structural guarantees are different kinds of claim — read the
distinction, it's the whole point of the project. Every number is from this project (no borrowed
metrics). The real-LLM figures come from **one small, cost-capped `anthropic` run** (N=10
self-built briefs, `claude-haiku-4-5` + `claude-sonnet-4-6`, **total cost $0.29**), recorded as
cassettes (`eval/cassettes/`, committed) and replayed by `tests/test_cassette.py` so they
reproduce in CI with no key and no spend (`STYLIST_LLM=cassette pytest tests/test_cassette.py`).

### Headline — the verifier earns its keep on a real model

> **12 / 172 attempted factual claims were blocked on the real run.** The model *tried* 172
> factual references across the run; ~7% were unsupported (e.g. a certification the item lacks),
> and the deterministic verifier caught **every one** and removed it — so **none reached the user**.

This is the thesis made measurable: on the offline fake model groundedness is a trivial 100%,
but a *real* model genuinely attempts unsupported claims, and the grounding layer is what stops
them. (Output groundedness is still 100% by construction — see below; this number is the
verifier's catch rate on the model's *attempts*.)

### Structural guarantees (provider-independent)

Deterministic code — the grounding verifier and SQL filters — **not model accuracy**.
Provider-independent (measured offline at N=40, reconfirmed on the real run). Guarantees about
the *architecture*, not claims that the LLM is accurate or that the recommendations are good.

| Guarantee | Result | N | What it means — and what it does **not** |
|---|---|---|---|
| Output groundedness (no fabricated factual claim reaches the user) | 100% **by construction** | — | Factual text is code-rendered from the record; unsupported claims are removed before output (the headline catch rate above is what gets removed). |
| Hard-constraint satisfaction | 100% | 114 offline / 29 real | SQL-enforced, re-checked per returned item — **against the _parsed_ profile** (see caveats). |
| Retrieval validity | 100% | 114 offline / 29 real | Every returned item satisfies all parsed hard constraints. |
| Adversarial grounding block rate | 100% | 40 offline / 10 real | Blocks every injected hallucination — one per brief, **rotated** across 5 self-authored attack categories (carat/number, wrong metal, wrong stone, absent certification, currency-in-opinion). See caveat: known patterns, not robustness. |

Caveats — these bound what the 100%s actually claim:
- **"Grounded" ≠ "accurate" ≠ "good."** Groundedness means no *unsupported* factual claim
  survives to the user. It does not validate that the catalog data is itself correct, nor that
  the recommendation is a *good* pick.
- **Constraint satisfaction is measured against the _parsed_ profile.** Natural-language →
  structured parsing is a separate, upstream error source **not** captured by this number: if
  the parser misreads "no gold," retrieval cannot filter what it never parsed. Parsing quality
  is a distinct (and unmeasured-here) concern.
- **The adversarial set is self-authored and covers _known_ attack patterns.** A 100% block
  rate shows the verifier catches the attacks I designed; it is **not** a proof of robustness.
  The scan only recognises the controlled vocabulary (a fixed metal/stone/cert set + a
  numeric/currency regex), so a factual term *outside* it (moissanite, palladium, titanium,
  lab-grown, gold-plated, an out-of-list stone) is **not** detected, and a true-but-misleading
  claim is not caught. It is a floor of demonstrated coverage, not a robustness guarantee.

### Subjective estimates (real run — directional, never headlined)

Reported only from the real run, always as estimates with N and a caveat:
- **Subjective relevance: 3.4 / 5 (N=10)** — an LLM-as-judge estimate, **NOT validated
  accuracy**: an LLM scoring an LLM over a tiny self-built set with no human ground truth.
  Directional only; the judge prompt is in `eval/real_run.json`.
- **Prompt-iteration pairwise: 90% to v2 (N=10)** — v2 (occasion-enriched) vs v1 (concise),
  randomized order-shuffled to neutralize position bias. Tiny N; directional evidence the prompt
  change helped, not a precise effect size.

> **Footnote — enrichment tagging is a sanity check, not an accuracy number.** The real run's
> tagging agreement is 100% (N=86), but it is **near-tautological**: the synthetic product
> description contains the style word verbatim, so the model trivially recovers it. It confirms
> the enrichment path runs end-to-end on a real model; it is **not** real-world tagging accuracy
> (that needs human-labelled data, out of scope here).

## Faithfulness & evaluation guarantees (the §4 contract)

- **Grounding invariant** — every factual claim *expressed in the catalog's controlled
  vocabulary* is checkable against the record; unsupported ones are blocked (terms outside the
  vocabulary are not scanned — see Limitations). Headline test: `tests/test_grounding.py`
  injects a hallucinated claim and asserts it is blocked.
- **Subjective vs factual separation** — factual fields are templated from the record (the LLM
  never generates them free-form) and are verbatim in output; opinion is labelled as opinion.
- **Constraint satisfaction** — hard constraints are retrieval-time SQL filters, not LLM
  promises; adversarial test asserts zero violations (against the parsed profile).
- **No fabricated relevance numbers** — subjective relevance is only ever an LLM-judge estimate
  with N + judge prompt + caveat, reported only from a real run. No metric is reused from another project.
- **Prompt-iteration evidence** — ≥2 rationale prompts compared via randomized, order-shuffled
  pairwise (neutralizes position bias); win rate reported from a real run (suppressed offline).
- **Reproducibility & cost** — every LLM call is cached by
  `hash(task+model+schema+system+prompt+context)`; identical requests return cached results;
  total $ is reported per eval run.

## Data

- **Synthetic generator** (`core/ingest/synthetic.py`) — deterministic, ships an 80-item
  committed fixture (`data/fixtures/catalog.json`). The whole pipeline runs end-to-end on it
  with no external dependency.
- **Responsible real-brand scraper** (`core/ingest/scrape.py`) — targets a Shopify storefront's
  public `products.json`; **honours that store's robots.txt** (fetched with our descriptive
  User-Agent) and **fails closed** if robots.txt can't be fetched/parsed, rate-limits (≥3s,
  single thread), caches to disk, and writes to **gitignored** `data/scraped/` — the dataset is
  **never committed**. It extracts the factual fields it can parse from the listing (metal,
  primary stone, carat, price, category, title); certification and accent stones aren't exposed
  by `products.json`, so they're recorded as absent (any cert claim on a scraped item is thus
  always blocked as uncertified). Style tags are still inferred by the LLM ("scrape facts, infer
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
integration, cache-key/offline-replay reproducibility, scraper robots/parse. (The real-run
**cassette** replay test is present but **skipped** until a real run is recorded — it is not
counted among the green tests above.)

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

> **Live demo: https://faithful-stylist.onrender.com** (deployed from this repo via `render.yaml`).

## Limitations (read these)

- **Offline evaluation has no real users.** Without interaction/click data, recommender
  evaluation is inherently weak; nothing here measures real-world conversion or satisfaction.
- **Faithfulness is bounded by parsing.** The grounding guarantee covers what the system
  *outputs*; it does not cover the natural-language → structured-profile parse. If the parser
  misreads a constraint, retrieval filters the wrong thing — and the verifier won't catch that,
  because the returned facts are still grounded in the (wrongly-retrieved) record. Concretely:
  "within your budget" is rendered grounded against the *parsed* `budget_max`, so a budget
  mis-parse (lakh notation, or a max read as a min) can make the verifier *stamp* a budget claim
  that's false to the user's actual intent. Parsing quality is a separate, upstream concern not
  measured by the grounding numbers.
- **The smuggled-fact scan is vocabulary-bounded.** It recognises only a fixed metal/stone/cert
  set plus a numeric/currency/units regex. A factual material or gemstone claim phrased with a
  term *outside* that vocabulary (moissanite, cubic zirconia, palladium, titanium, lab-grown,
  gold-plated, conflict-free, an out-of-list stone like opal/garnet) is **not** detected and can
  reach the user. So the 100% adversarial block rate covers *known, in-vocabulary* patterns —
  not robustness to attacks I didn't think of.
- **Opinion substance is unchecked.** Opinion clauses are scanned only for smuggled
  numeric/metal/stone/cert facts; their substance is otherwise unverified. An unsupported
  provenance, popularity, or value claim phrased as opinion ("ethically sourced", "best value in
  India", "our bestseller") passes through with only a non-blocking note. The opinion label
  bounds *framing*, not factual accuracy.
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
