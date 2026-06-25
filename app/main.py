"""FastAPI entrypoint.

Phase 0 exposes only the health endpoint. Recommendation/eval routes and the minimal
UI are added in Phase 1.
"""

from __future__ import annotations

from fastapi import FastAPI

from core import __version__

app = FastAPI(
    title="Faithful Stylist",
    version=__version__,
    summary="A grounded, evaluated conversational jewellery recommender.",
)


@app.get("/healthz/")
def healthz() -> dict:
    """Liveness probe. Returns 200 with basic build info."""
    return {"status": "ok", "service": "faithful-stylist", "version": __version__}
