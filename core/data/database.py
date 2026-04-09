"""Database connection and session management."""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: str) -> None:
    """Create all tables if they don't exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


@contextmanager
def get_session(db_path: str):
    engine = get_engine(db_path)
    Session_ = sessionmaker(bind=engine)
    session = Session_()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
