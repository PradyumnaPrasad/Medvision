import pytest
from sqlmodel import SQLModel, create_engine

from backend.app import db


@pytest.fixture(scope="session", autouse=True)
def isolated_database(tmp_path_factory):
    """Tests must never write to the developer's real data/medvision.db: they delete their
    image files afterwards, which used to leave broken entries in the app's History."""
    engine = create_engine(f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    original = db._engine
    db._engine = engine
    yield
    db._engine = original
