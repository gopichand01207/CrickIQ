"""
setup_db.py
-----------
CrickIQ database setup script.

Creates the MySQL (or SQLite) tables and loads the generated CSV data.

Usage
-----
    python setup_db.py              # auto-detects MySQL / SQLite fallback
    python setup_db.py --sqlite     # force SQLite
    python setup_db.py --regenerate # regenerate CSVs before loading
"""

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from database.db_config import get_engine, reset_engine
from database.queries import bootstrap_tables, load_csv_to_db
from utils.preprocessor import load_matches, load_deliveries

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CrickIQ Database Setup")
    parser.add_argument("--sqlite", action="store_true", help="Force SQLite instead of MySQL")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate CSV data before loading")
    args = parser.parse_args()

    logger.info("=== CrickIQ Database Setup ===")

    # Get engine
    reset_engine()
    engine = get_engine(force_sqlite=args.sqlite)
    logger.info("Engine: %s", engine.url)

    # Load data
    logger.info("Loading match data …")
    matches = load_matches(regenerate=args.regenerate)
    logger.info("  → %d matches loaded", len(matches))

    logger.info("Loading delivery data …")
    deliveries = load_deliveries(regenerate=args.regenerate)
    logger.info("  → %d deliveries loaded", len(deliveries))

    # Write to DB
    logger.info("Writing to database …")
    load_csv_to_db(engine, matches, deliveries)
    logger.info("  ✓ matches table loaded")
    logger.info("  ✓ deliveries table loaded")

    logger.info("Database setup complete!")


if __name__ == "__main__":
    main()
