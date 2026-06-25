"""SQLite/SQLAlchemy persistence for the factual catalog.

Hard-filter fields (price, metal, stone, category) live in SQL so retrieval can enforce
hard constraints as queries (Section 6.5) rather than trusting the LLM. The filtered
query itself is added in the retrieval step.
"""

from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from core.config import get_settings
from core.models import Product, StyleTags


def make_engine(url: str | None = None, echo: bool = False):
    """Create an engine. In-memory SQLite uses a shared StaticPool so multiple
    sessions see the same database (needed for tests and a single-process app)."""
    url = url or get_settings().db_url
    if url in ("sqlite://", "sqlite:///:memory:"):
        return create_engine(
            "sqlite://",
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, echo=echo, connect_args=connect_args)


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)


def seed_catalog(
    engine,
    products: list[Product],
    tags: list[StyleTags] | None = None,
) -> None:
    """Upsert products (and optional style tags) into the database."""
    with Session(engine) as session:
        for product in products:
            session.merge(product)
        for tag in tags or []:
            session.merge(tag)
        session.commit()
