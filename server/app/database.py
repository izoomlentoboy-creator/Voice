"""Database engine and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from . import config

engine = create_engine(
    config.DATABASE_URL,
    # SQLite-specific: allow multi-thread access
    connect_args={"check_same_thread": False}
    if config.DATABASE_URL.startswith("sqlite")
    else {},
    echo=config.DEBUG,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Session:
    """Dependency: yield a database session, close on teardown."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=engine)
