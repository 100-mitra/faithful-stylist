"""P1.1: domain models validate from the fixture and round-trip through SQLite."""

from sqlmodel import Session, select

from core.config import load_catalog_dicts
from core.models import (
    EvalRun,
    FactRef,
    HardConstraints,
    PreferenceProfile,
    Product,
    RationaleDraft,
    StyleTags,
)
from core.store import init_db, make_engine, seed_catalog


def test_products_validate_from_fixture():
    catalog = load_catalog_dicts()
    products = [Product(**d) for d in catalog]
    assert len(products) == len(catalog)
    # raw_attributes survives as a dict; factual fields keep their types.
    p = products[0]
    assert isinstance(p.raw_attributes, dict)
    assert isinstance(p.price, int)
    assert p.certification is None  # synthetic rows carry no certification


def test_catalog_round_trips_through_sqlite():
    catalog = load_catalog_dicts()
    products = [Product(**d) for d in catalog]
    tags = [
        StyleTags(
            product_id=products[0].id,
            styles=["vintage"],
            occasions=["engagement"],
            aesthetic_notes="reads as romantic",
            confidence=0.8,
            tagged_by="test",
            qa_checked=True,
        )
    ]
    engine = make_engine("sqlite://")
    init_db(engine)
    seed_catalog(engine, products, tags)

    with Session(engine) as s:
        # Hard-constraint style query: platinum under 1L.
        rows = s.exec(
            select(Product).where(Product.metal == "platinum", Product.price < 100000)
        ).all()
        assert rows, "expected some platinum items under 1L in the fixture"
        assert all(r.metal == "platinum" and r.price < 100000 for r in rows)

        loaded_tags = s.exec(select(StyleTags)).all()
        assert loaded_tags[0].styles == ["vintage"]


def test_evalrun_json_metrics_round_trip():
    engine = make_engine("sqlite://")
    init_db(engine)
    run = EvalRun(
        id="run-1",
        dataset_ref="fixtures/briefs.json",
        metrics={"groundedness_rate": 1.0, "n": 8},
        cost_usd=0.0,
        created_at="2026-01-01T00:00:00+00:00",
    )
    with Session(engine) as s:
        s.add(run)
        s.commit()
    with Session(engine) as s:
        got = s.get(EvalRun, "run-1")
        assert got.metrics["groundedness_rate"] == 1.0


def test_rationale_contract_models():
    draft = RationaleDraft(
        product_id="syn-0001",
        factual_slots=[FactRef(field="metal"), FactRef(field="price")],
        subjective_clauses=["reads as vintage to me"],
    )
    assert draft.factual_slots[0].field == "metal"
    # FactRef carries no value channel.
    assert not hasattr(draft.factual_slots[0], "value")


def test_preference_profile_defaults():
    prof = PreferenceProfile(raw_text="something nice")
    assert prof.hard_constraints == HardConstraints()
    assert prof.budget_max is None
    assert prof.needs_clarification is False
