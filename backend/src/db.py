"""Database setup using SQLAlchemy.

Reads DATABASE_URL from env, defaults to a local Postgres-style URL placeholder.
Provides:
  - engine
  - SessionLocal
  - Base (declarative)
  - get_db() dependency for FastAPI
  - init_db() to create tables

Example DATABASE_URL values:
  postgresql+psycopg2://user:password@localhost:5432/blackline
  postgresql://user:password@/dbname?host=/var/run/postgresql  (domain sockets)

Note: For dev without Postgres, you can use SQLite by setting
  DATABASE_URL=sqlite:///./backend/data/dev.db
"""
from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
try:
  # Optional: load .env if present
  from dotenv import load_dotenv  # type: ignore
  load_dotenv()
except Exception:
  pass


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./backend/data/dev.db")

# For SQLite, need check_same_thread 
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(models_module=None):
    """Create all tables. Optionally accepts a module to ensure models are imported."""
    # Ensure models are imported so metadata is populated
    if models_module is not None:
        _ = models_module
    Base.metadata.create_all(bind=engine)
