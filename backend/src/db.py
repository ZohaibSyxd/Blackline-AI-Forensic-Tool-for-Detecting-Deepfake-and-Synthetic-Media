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


_default_sqlite = "sqlite:////tmp/data/dev.db" if os.getenv("PORT") else "sqlite:///./backend/data/dev.db"
DATABASE_URL = os.environ.get("DATABASE_URL", _default_sqlite)

# For SQLite, need check_same_thread 
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# If using SQLite on filesystem, ensure parent directory exists
if DATABASE_URL.startswith("sqlite"):
  try:
    # Extract path after 'sqlite:///' or 'sqlite:////'
    # Formats: sqlite:///relative/path.db  or sqlite:////absolute/path.db
    url = DATABASE_URL[len("sqlite://"):]
    # strip leading slashes to detect absolute vs relative
    path_part = url.lstrip("/")
    # Reconstruct filesystem path respecting absolute indicator
    if url.startswith("/"):
      fs_path = "/" + path_part
    else:
      fs_path = path_part
    # Ignore special cases like :memory:
    if fs_path and not fs_path.startswith(":"):
      import os as _os
      parent = _os.path.dirname(fs_path)
      if parent and not _os.path.exists(parent):
        _os.makedirs(parent, exist_ok=True)
  except Exception:
    pass

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
