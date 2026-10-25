#!/usr/bin/env python3
"""Download the tar1090-db aircraft registry and build it for target labelling.

    python scripts/fetch_aircraft_db.py [--out PATH]

Writes services.aircraft_db's file (AIRCRAFT_DB_PATH, or backend/data/
aircraft_db.sqlite).  The server picks it up on its next lookup, so there is no
need to restart.  Where AIRCRAFT_DB_ENABLED=1 the server keeps it current
itself; this is for everywhere else.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import aircraft_db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="registry file to write")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = args.out or aircraft_db.db_path()
    count = aircraft_db.refresh(out)
    print(f"{count} aircraft written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
