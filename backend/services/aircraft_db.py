"""The aircraft registry: what an ICAO hex address is, looked up offline.

ADS-B tells us a transponder's 24-bit address and nothing about the airframe
behind it.  The registry answers that from the tar1090-db export of Mictronics'
readsb database, the same source tar1090 itself labels aircraft with: one row
per address (registration, ICAO type designator, flags, description, operator)
and a type table mapping each designator to its ICAO aircraft class (``L2J``:
landplane, two jet engines) and wake-turbulence category.

The export is ~620k rows.  Held as a Python dict it costs well over 100 MB of
resident memory for a lookup that touches a few hundred addresses at a time, so
it is converted once into a SQLite file under ``backend/data`` and read through
a small LRU cache instead.  The file is built by ``refresh()`` (the
aircraft_db_refresh task where AIRCRAFT_DB_ENABLED=1, or
``scripts/fetch_aircraft_db.py`` by hand); with no file every lookup answers
None and nothing else changes.
"""

import csv
import gzip
import io
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from services.id_utils import normalize_hex_key

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "aircraft_db.sqlite"
AIRCRAFT_CSV_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/csv/aircraft.csv.gz"
AIRCRAFT_TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types2.js"
_USER_AGENT = "retina-server/1.0 (+https://github.com/offworldlabs/retina-server)"
_TIMEOUT_S = 60
_LOOKUP_CACHE = 8192


def db_path() -> Path:
    return Path(os.getenv("AIRCRAFT_DB_PATH") or _DEFAULT_PATH)


_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None
_conn_mtime: float | None = None


def _connection() -> sqlite3.Connection | None:
    """The open registry, reopened when the file on disk has been replaced.

    Called under ``_lock``.  A missing file is checked again on every call, so a
    registry built after boot is picked up without a restart.
    """
    global _conn, _conn_path, _conn_mtime
    path = db_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if _conn is not None and (_conn_path != path or _conn_mtime != mtime):
        _conn.close()
        _conn = None
        _lookup_cached.cache_clear()
    if _conn is None and mtime is not None:
        _conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn_path, _conn_mtime = path, mtime
    return _conn


@lru_cache(maxsize=_LOOKUP_CACHE)
def _lookup_cached(hex_code: str, _generation: float) -> dict | None:
    conn = _conn
    if conn is None:
        return None
    row = conn.execute(
        "SELECT a.registration, a.type_code, a.flags, a.description, a.operator,"
        " t.name AS type_name, t.icao_class, t.wtc"
        " FROM aircraft a LEFT JOIN types t ON t.type_code = a.type_code"
        " WHERE a.hex = ?",
        (hex_code,),
    ).fetchone()
    if row is None:
        return None
    return {k: v for k, v in dict(row).items() if v not in (None, "")}


def lookup(hex_code) -> dict | None:
    """The registry row for one ICAO address, or None.

    Keys present only where the registry has a value: ``registration``,
    ``type_code``, ``flags``, ``description``, ``operator``, and from the type
    table ``type_name``, ``icao_class`` and ``wtc``.  A non-ICAO address
    (tar1090's ``~`` prefix, a solver-minted ``mn``/``pr`` hex) is never looked up.
    """
    h = normalize_hex_key(hex_code)
    if len(h) != 6 or any(c not in "0123456789abcdef" for c in h):
        return None
    with _lock:
        if _connection() is None:
            return None
        try:
            return _lookup_cached(h, _conn_mtime)
        except sqlite3.Error:
            log.exception("aircraft registry lookup failed for %s", h)
            return None


def available() -> bool:
    with _lock:
        return _connection() is not None


def age_s() -> float | None:
    """Seconds since the registry file was last built, or None without one."""
    try:
        return time.time() - db_path().stat().st_mtime
    except OSError:
        return None


def parse_aircraft_csv(text: str):
    """Rows of the tar1090-db export: ``hex;reg;type;flags;desc;year;ownop;``."""
    for row in csv.reader(io.StringIO(text), delimiter=";"):
        if not row or len(row[0]) != 6:
            continue
        row = row + [""] * (7 - len(row))
        yield (row[0].lower(), row[1].strip(), row[2].strip().upper(), row[3].strip(), row[4].strip(), row[6].strip())


def parse_types(data: dict):
    """Rows of the type table: ``{"B738": ["BOEING 737-800", "L2J", "M"]}``."""
    for code, value in data.items():
        if not isinstance(value, list):
            continue
        value = value + [""] * (3 - len(value))
        yield (code.upper(), value[0] or "", value[1] or "", value[2] or "")


def build(aircraft_csv: str, types: dict, out_path: Path) -> int:
    """Write a registry file atomically; returns the number of aircraft rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".aircraft_db.", suffix=".sqlite", dir=out_path.parent)
    os.close(fd)
    try:
        conn = sqlite3.connect(tmp)
        try:
            conn.executescript(
                "CREATE TABLE aircraft (hex TEXT PRIMARY KEY, registration TEXT, type_code TEXT,"
                " flags TEXT, description TEXT, operator TEXT) WITHOUT ROWID;"
                "CREATE TABLE types (type_code TEXT PRIMARY KEY, name TEXT, icao_class TEXT, wtc TEXT)"
                " WITHOUT ROWID;"
            )
            conn.executemany("INSERT OR REPLACE INTO aircraft VALUES (?,?,?,?,?,?)", parse_aircraft_csv(aircraft_csv))
            conn.executemany("INSERT OR REPLACE INTO types VALUES (?,?,?,?)", parse_types(types))
            conn.commit()
            (count,) = conn.execute("SELECT COUNT(*) FROM aircraft").fetchone()
        finally:
            conn.close()
        os.replace(tmp, out_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return count


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        raw = resp.read()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def refresh(out_path: Path | None = None) -> int:
    """Download the tar1090-db export and rebuild the registry file.

    Both files are gzip whatever their names say (``icao_aircraft_types2.js``
    is gzipped JSON), so each is sniffed rather than trusted by extension.
    """
    aircraft_csv = _download(os.getenv("AIRCRAFT_DB_URL") or AIRCRAFT_CSV_URL).decode("utf-8", "replace")
    types = json.loads(_download(os.getenv("AIRCRAFT_TYPES_URL") or AIRCRAFT_TYPES_URL))
    count = build(aircraft_csv, types, out_path or db_path())
    log.info("aircraft registry rebuilt: %d aircraft, %d types", count, len(types))
    return count


def _reset_for_tests() -> None:
    global _conn, _conn_path, _conn_mtime
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = _conn_path = _conn_mtime = None
        _lookup_cached.cache_clear()
