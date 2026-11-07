"""
db_config.py
------------
MySQL connection configuration with SQLAlchemy engine.

Provides a get_engine() function that returns a SQLAlchemy engine.
Falls back to an in-memory SQLite database when MySQL is unavailable,
so the CrickIQ application always starts successfully.
"""

import os
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MySQL connection parameters (override via environment variables)
# ---------------------------------------------------------------------------
MYSQL_HOST = os.getenv("CRICKIQ_DB_HOST", "localhost")
MYSQL_PORT = int(os.getenv("CRICKIQ_DB_PORT", "3306"))
MYSQL_USER = os.getenv("CRICKIQ_DB_USER", "root")
MYSQL_PASSWORD = os.getenv("CRICKIQ_DB_PASSWORD", "password")
MYSQL_DATABASE = os.getenv("CRICKIQ_DB_NAME", "crickiq")

MYSQL_URL = (
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

SQLITE_URL = "sqlite:///crickiq_local.db"

_engine = None  # module-level singleton


def get_engine(force_sqlite: bool = False):
    """
    Return a SQLAlchemy engine.

    Tries MySQL first; falls back to SQLite on connection failure.

    Parameters
    ----------
    force_sqlite : bool
        If True, skip MySQL attempt and use SQLite directly.

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    global _engine
    if _engine is not None:
        return _engine

    if not force_sqlite:
        try:
            engine = create_engine(MYSQL_URL, pool_pre_ping=True, connect_args={"connect_timeout": 5})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to MySQL at %s:%s/%s", MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE)
            _engine = engine
            return _engine
        except Exception as exc:
            logger.warning("MySQL unavailable (%s). Falling back to SQLite.", exc)

    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    logger.info("Using SQLite database: %s", SQLITE_URL)
    _engine = engine
    return _engine


def reset_engine():
    """Force a new engine on next call (useful for testing)."""
    global _engine
    _engine = None
