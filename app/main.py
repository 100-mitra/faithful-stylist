"""FastAPI entrypoint: the Faithful Stylist API + a minimal demo UI.

Exposing the per-claim grounding audit is the differentiator, so /api/recommend returns
it for every recommendation, and the UI surfaces it (including a one-click demo that
shows a fabricated claim being blocked).
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import UnidentifiedImageError
from pydantic import BaseModel
from sqlmodel import Session, select

from app.state import get_state
from core import __version__
from core.llm import CachingProvider, FakeProvider, ResponseCache
from core.models import (
    FactRef,
    PreferenceProfile,
    RationaleDraft,
    Recommendation,
    StyleTags,
)
from core.pipeline import recommend, tagging_accuracy
from core.store import get_eval_run, save_eval_run
from eval.harness import format_report, run_eval

_TEMPLATES = Path(__file__).resolve().parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_state()  # warm the catalog (ingest -> enrich -> index) once at startup
    yield


app = FastAPI(
    title="Faithful Stylist",
    version=__version__,
    summary="A grounded, evaluated conversational jewellery recommender.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ParseRequest(BaseModel):
    text: str


class RecommendRequest(BaseModel):
    text: str
    top_k: int = 5
    demo_hallucination: bool = False  # inject a fabricated claim to show it blocked


class EvalRequest(BaseModel):
    n: int | None = None
    top_k: int = 3


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------
def _profile_dict(profile: PreferenceProfile) -> dict:
    data = profile.model_dump(exclude={"embedding"})
    return data


def _rec_dict(rec: Recommendation) -> dict:
    return {
        "rank": rec.rank,
        "product": rec.product.model_dump(),
        "retrieval_scores": rec.retrieval_scores,
        "rationale": rec.rationale,
        "grounding_decision": rec.grounding_decision,
        "claims_audit": [c.model_dump() for c in rec.claims],
    }


def _demo_poison(ctx: dict, schema):
    pid = ctx.get("product", {}).get("id", "")
    return RationaleDraft(
        product_id=pid,
        factual_slots=[FactRef(field="metal"), FactRef(field="price")],
        subjective_clauses=["a dazzling 4-carat diamond — pure romance"],
        recommended=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz/")
def healthz() -> dict:
    return {"status": "ok", "service": "faithful-stylist", "version": __version__}


@app.post("/api/profile/parse")
def parse_profile(req: ParseRequest) -> dict:
    state = get_state()
    from core.profile import parse_preference

    profile = parse_preference(req.text, state.provider, state.embedder)
    return {"profile": _profile_dict(profile)}


@app.post("/api/recommend")
def post_recommend(req: RecommendRequest) -> dict:
    state = get_state()
    provider = state.provider
    if req.demo_hallucination:
        # Force a fabricated claim into the rationale to demonstrate the veto.
        provider = CachingProvider(
            FakeProvider(overrides={"rationale": _demo_poison}), ResponseCache()
        )
    profile, recs = recommend(req.text, state.ctx, provider, state.embedder, top_k=req.top_k)
    return {
        "profile": _profile_dict(profile),
        "recommendations": [_rec_dict(r) for r in recs],
        "blocked_any": any(r.grounding_decision == "blocked" for r in recs),
    }


@app.post("/api/recommend/visual")
async def post_recommend_visual(
    file: UploadFile = File(...),
    text: str = Form(""),
    top_k: int = Form(5),
) -> dict:
    """Phase 2: an inspiration image (+ optional text brief) -> visually similar recs.

    Visual similarity only orders candidates; every factual claim is still grounded + audited.
    """
    from core.visual import recommend_visual

    state = get_state()
    data = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        profile, recs = recommend_visual(
            tmp_path,
            text,
            state.ctx,
            state.visual,
            state.image_embedder,
            state.provider,
            state.embedder,
            top_k=top_k,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"could not read image: {exc}") from exc
    finally:
        os.unlink(tmp_path)
    return {
        "profile": _profile_dict(profile),
        "recommendations": [_rec_dict(r) for r in recs],
        "blocked_any": any(r.grounding_decision == "blocked" for r in recs),
        "image_embedder": state.visual.embedder_name,
    }


@app.get("/api/products/{product_id}")
def get_product(product_id: str) -> dict:
    state = get_state()
    product = state.ctx.products_by_id.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="product not found")
    with Session(state.ctx.engine) as session:
        tag = session.exec(select(StyleTags).where(StyleTags.product_id == product_id)).first()
    return {
        "factual": product.model_dump(),  # never LLM-invented
        "inferred": tag.model_dump() if tag else None,  # LLM-inferred, kept separate
    }


@app.post("/api/eval/run")
def post_eval(req: EvalRequest) -> dict:
    state = get_state()
    from eval.briefs import make_eval_briefs

    briefs = make_eval_briefs(req.n) if req.n else None
    run = run_eval(state.ctx, state.provider, state.embedder, briefs=briefs, top_k=req.top_k)
    save_eval_run(state.ctx.engine, run)
    return {"eval": run.model_dump(), "report": format_report(run)}


@app.get("/api/eval/{eval_id}")
def get_eval(eval_id: str) -> dict:
    state = get_state()
    run = get_eval_run(state.ctx.engine, eval_id)
    if run is None:
        raise HTTPException(status_code=404, detail="eval run not found")
    return {"eval": run.model_dump(), "report": format_report(run)}


@app.get("/api/tagging-accuracy")
def get_tagging_accuracy() -> dict:
    return tagging_accuracy(get_state().ctx)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_TEMPLATES / "index.html").read_text(encoding="utf-8")
