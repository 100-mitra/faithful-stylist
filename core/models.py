"""Domain model (brief Section 5).

Hard separation between two classes of data:

* **Factual** attributes live on ``Product`` (and are persisted to SQL so hard
  constraints can be enforced as queries). They are scraped/generated, never invented
  by the LLM.
* **Subjective / inferred** data (``StyleTags``) is kept in a separate table so factual
  and inferred data never blur.

The rationale-generation contract (``FactRef`` / ``RationaleDraft``) deliberately gives
the LLM no channel to emit a factual *value*: it only names *which* fact to surface; the
value is rendered from the record in code and audited by the GroundingVerifier.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

# Which factual fields a rationale is allowed to *reference* (never to value-fill).
FactField = Literal[
    "price",
    "metal",
    "stone_primary",
    "stone_accent",
    "carat",
    "certification",
    "category",
    "under_budget",
]
ClaimType = Literal["factual", "subjective"]


# ---------------------------------------------------------------------------
# Persisted factual record
# ---------------------------------------------------------------------------
class Product(SQLModel, table=True):
    """Factual catalog record. Factual fields only — never LLM-invented."""

    id: str = SQLField(primary_key=True)
    source: str
    source_url: str
    title: str
    price: int = SQLField(index=True)
    currency: str = "INR"
    metal: str = SQLField(index=True)
    stone_primary: str | None = SQLField(default=None, index=True)
    stone_accent: str | None = None
    carat: float | None = None
    certification: str | None = None  # e.g. "IGI" / "BIS"; None for synthetic rows.
    category: str = SQLField(index=True)
    image_path: str | None = None
    raw_attributes: dict = SQLField(default_factory=dict, sa_column=Column(JSON))
    ingested_at: str


class StyleTags(SQLModel, table=True):
    """LLM-inferred style/occasion labels. Kept separate from Product on purpose."""

    product_id: str = SQLField(primary_key=True, foreign_key="product.id")
    styles: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))
    occasions: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))
    aesthetic_notes: str = ""
    confidence: float = 0.0
    tagged_by: str = ""  # model + version
    qa_checked: bool = False


class StyleTagDraft(BaseModel):
    """Structured output the LLM enrichment step must return (subjective only)."""

    styles: list[str] = Field(default_factory=list)
    occasions: list[str] = Field(default_factory=list)
    aesthetic_notes: str = ""
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# Preference profile (parsed from freeform text)
# ---------------------------------------------------------------------------
class HardConstraints(BaseModel):
    """Strict, retrieval-time filters. Enforced by SQL, never by the LLM.

    ``allowed_*`` (when non-empty) means the attribute MUST be one of the listed
    values; ``excluded_*`` removes values; ``require_no_stone`` keeps only plain pieces.
    """

    allowed_metals: list[str] = Field(default_factory=list)
    excluded_metals: list[str] = Field(default_factory=list)
    allowed_stones: list[str] = Field(default_factory=list)
    excluded_stones: list[str] = Field(default_factory=list)
    require_no_stone: bool = False
    categories: list[str] = Field(default_factory=list)
    size: str | None = None


class PreferenceProfile(BaseModel):
    raw_text: str
    styles: list[str] = Field(default_factory=list)
    occasion: str | None = None
    budget_max: int | None = None  # hard ceiling (INR)
    metal_prefs: list[str] = Field(default_factory=list)  # soft preference (boost)
    stone_prefs: list[str] = Field(default_factory=list)  # soft preference (boost)
    recipient: str | None = None
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    embedding: list[float] | None = None
    # Cold-start handling: sparse input sets a clarifying follow-up.
    needs_clarification: bool = False
    clarifying_question: str | None = None


class ProfileDraft(BaseModel):
    """Structured output the LLM preference-parse step returns (pre-embedding)."""

    styles: list[str] = Field(default_factory=list)
    occasion: str | None = None
    budget_max: int | None = None
    metal_prefs: list[str] = Field(default_factory=list)
    stone_prefs: list[str] = Field(default_factory=list)
    recipient: str | None = None
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)


# ---------------------------------------------------------------------------
# Rerank + rationale contract
# ---------------------------------------------------------------------------
class RerankResult(BaseModel):
    """LLM rerank output: an ordering of candidate product ids (best first)."""

    ranked_ids: list[str] = Field(default_factory=list)


class FactRef(BaseModel):
    """A reference to a factual field to surface — carries NO value."""

    field: FactField


class RationaleDraft(BaseModel):
    """What the LLM is allowed to produce for a rationale.

    ``factual_slots`` name which facts to mention (values come from the record);
    ``subjective_clauses`` are short opinion phrases that get labelled as opinion.
    """

    product_id: str
    factual_slots: list[FactRef] = Field(default_factory=list)
    subjective_clauses: list[str] = Field(default_factory=list)
    recommended: bool = True


class RationaleClaim(BaseModel):
    """One audited claim (Section 5). Powers the grounding audit + headline test."""

    claim_text: str
    claim_type: ClaimType
    grounded: bool
    source_field: str | None = None
    blocked: bool = False  # True => excluded from the user-facing rationale
    note: str | None = None  # why it was blocked / flagged


class VerifierResult(BaseModel):
    """Output of the deterministic GroundingVerifier."""

    decision: Literal["passed", "blocked"]
    claims: list[RationaleClaim] = Field(default_factory=list)
    rationale_text: str = ""  # safe, user-facing text (blocked claims removed)


# ---------------------------------------------------------------------------
# Recommendation (returned by the API)
# ---------------------------------------------------------------------------
class Recommendation(BaseModel):
    product: Product
    rank: int
    retrieval_scores: dict = Field(default_factory=dict)
    rationale: str = ""
    claims: list[RationaleClaim] = Field(default_factory=list)
    grounding_decision: Literal["passed", "blocked"] = "passed"

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Eval run (persisted)
# ---------------------------------------------------------------------------
class EvalRun(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    dataset_ref: str
    metrics: dict = SQLField(default_factory=dict, sa_column=Column(JSON))
    cost_usd: float = 0.0
    created_at: str = ""
