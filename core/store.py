"""SQLite/SQLAlchemy persistence for the factual catalog.

Hard-filter fields (price, metal, stone, category) live in SQL so retrieval can enforce
hard constraints as queries (Section 6.5) rather than trusting the LLM. The filtered
query itself is added in the retrieval step.
"""

from __future__ import annotations

from sqlalchemy import or_
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from core.config import get_settings
from core.models import PreferenceProfile, Product, StyleTags


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


def query_products(engine, profile: PreferenceProfile) -> list[Product]:
    """Apply the profile's HARD constraints as SQL filters (brief §6.5).

    Hard constraints are enforced here, in the database, never left to the LLM. This is
    the guarantee the adversarial constraint-satisfaction test checks.
    """
    hc = profile.hard_constraints
    with Session(engine) as session:
        stmt = select(Product)
        if profile.budget_max is not None:
            stmt = stmt.where(Product.price <= profile.budget_max)
        if hc.allowed_metals:
            stmt = stmt.where(Product.metal.in_(hc.allowed_metals))
        if hc.excluded_metals:
            stmt = stmt.where(Product.metal.not_in(hc.excluded_metals))
        if hc.require_no_stone:
            stmt = stmt.where(Product.stone_primary.is_(None))
        if hc.allowed_stones:
            stmt = stmt.where(Product.stone_primary.in_(hc.allowed_stones))
        if hc.excluded_stones:
            stmt = stmt.where(
                or_(
                    Product.stone_primary.is_(None),
                    Product.stone_primary.not_in(hc.excluded_stones),
                )
            )
            stmt = stmt.where(
                or_(
                    Product.stone_accent.is_(None),
                    Product.stone_accent.not_in(hc.excluded_stones),
                )
            )
        if hc.categories:
            stmt = stmt.where(Product.category.in_(hc.categories))
        return list(session.exec(stmt).all())
