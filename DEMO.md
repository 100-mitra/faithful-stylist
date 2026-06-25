# 60–90s demo storyboard

Record after (a) the real eval run is done (so the eval numbers shown are genuine) and
(b) the live demo is deployed. The public demo runs on the `fake` provider, so the
grounding/hallucination shots work keyless; for the recommendation prose you can run
locally with `STYLIST_LLM=anthropic` if you want real rationales on camera.

Tooling: any screen recorder (Loom / OBS / QuickTime). Keep it ≤90s, no narration
required — on-screen captions are enough. Target the live URL.

---

**0:00–0:10 — The hook (1 sentence).**
Caption: *"Off-the-shelf shopping bots hallucinate product claims. This one structurally
can't."* Show the landing page (the tagline already says it).

**0:10–0:30 — A real recommendation.**
- Leave the seeded engagement-ring brief in the box; click **Recommend**.
- Show the parsed-preferences chips (budget / occasion / styles / metals) — proves it
  understood the freeform brief.
- Scroll to a recommendation card. Caption: *"Every factual line — metal, price, carat —
  is templated from the catalog and verified."* Hover the green **factual / grounded**
  tags in the claims audit.

**0:30–0:55 — The headline: a blocked hallucination.**
- Tick **"demo: inject a hallucinated claim"**, click **Recommend** again.
- The red banner appears: *"the grounding verifier blocked a fabricated factual claim."*
- Zoom the claims audit: the injected `"…4-carat diamond…"` row is tagged **BLOCKED** (red)
  with the reason, and point out it is **absent from the rationale text above**.
- Caption: *"The LLM was forced to fabricate a carat. The deterministic verifier caught it
  before it reached the user."*  ← this is the whole pitch.

**0:55–1:15 — Honest evaluation.**
- Click **Run eval**. Show the report panel.
- Caption: *"Structural guarantees are provider-independent: groundedness, constraint
  satisfaction, adversarial block rate — 100%, with N."*
- Then point to the subjective section. Caption: *"Subjective relevance is reported only as
  an LLM-judge estimate with N and a caveat — never dressed up as accuracy."*
  (If recording the offline demo, it honestly says the subjective number is a placeholder;
  show the real `eval/real_run.json` numbers in a quick cut instead.)

**1:15–1:30 — Close.**
Caption: *"A hybrid recommender with an LLM rerank + rationale layer, where a deterministic
grounding verifier makes the stylist unable to misstate a product. FastAPI, Dockerized,
live demo, honest eval."* End on the live URL.
