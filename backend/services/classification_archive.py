"""A persistent record of what the classifier saw, one row per target per minute.

The live scorecard (services.target_classification) keeps each aircraft's latest
verdict for six hours and forgets it.  This keeps the evidence: for every target
in the aircraft feed, a row per 60 s window holding the motion baseline's
verdict and the features behind it, the rotor-line (comb) summary, the label
where a hex or the simulator names one, and where the target was seen from.
Unlabelled targets are kept too, since a label can arrive later and the rows are
the only record of what the motion looked like.  scripts/classification_report.py
reads the files and can export the labelled rows as a training set.

Flow, all driven by classify_feed (the end of every feed build):

  observe(entry, label, now)   fold one feed entry into its target's open window
  close_expired(now)           emit a row for every window whose minute is over
  flush()                      write the buffered rows to disk (the hourly task,
                               services.tasks.classification_archive)

Files are ``<archive dir>/YYYY/MM/DD/HH.parquet`` in UTC, one per hour of window
start.  Parquet cannot be appended to, so a flush that finds the hour's file
already there reads it and writes the union back (via a temp file and a rename,
so a reader never sees half of one).  The column list is
services.classification_columns.ARCHIVE_COLUMNS, shared with the offline bench.

Everything here is enrichment: classify_feed counts and swallows an exception
from it (``classification_archive_errors``), and a write that fails leaves its
rows buffered for the next flush.  CLASSIFICATION_ARCHIVE_ENABLED=0 turns the
whole thing off.
"""

import logging
import math
import os
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from core import state
from services import classification_columns as cols

log = logging.getLogger(__name__)

ENABLED_ENV = "CLASSIFICATION_ARCHIVE_ENABLED"
ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "classification_archive"

WINDOW_S = cols.WINDOW_S
# A window in which the target was in fewer feed builds than this says too little
# to be a row: a target seen once as it left the beam.  The bench's --min-frames.
MIN_OBSERVATIONS = 3
# Closed rows waiting for the hourly flush.  A busy fleet makes a few hundred a
# minute, so this is hours of backlog; past it the oldest go, counted, rather
# than growing without bound while the disk is failing.
MAX_BUFFERED_ROWS = 200_000
FT_TO_M = 0.3048

_SCHEMA = pa.schema(list(cols.ARCHIVE_COLUMNS.items()))

_lock = threading.Lock()
_open: "dict[str, _Window]" = {}
_rows: list[dict] = []
_stats = {"rows_written": 0, "rows_dropped": 0, "last_flush": None, "last_flush_files": [], "last_error": None}


def enabled() -> bool:
    """On unless exactly ``0``, read at call time so an operator's change needs no restart."""
    return os.environ.get(ENABLED_ENV) != "0"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _alt_m(alt_ft) -> float | None:
    if alt_ft == "ground":
        return 0.0
    if isinstance(alt_ft, (int, float)) and not isinstance(alt_ft, bool) and math.isfinite(alt_ft):
        return float(alt_ft) * FT_TO_M
    return None


class _Window:
    """One target's observations inside one 60 s window.

    The verdict, comb and position source are the newest observation's: the
    verdict reads a trail that only grows, so the last one has seen the most of
    it.  Altitude and body SNR are medians over the window.
    """

    __slots__ = (
        "start", "n", "nodes", "sim_nodes", "position_source", "label", "type_code",
        "motion", "comb", "comb_full", "alts", "snrs",
    )  # fmt: skip

    def __init__(self, start: float):
        self.start = start
        self.n = 0
        self.nodes: set[str] = set()
        self.sim_nodes: set[str] = set()
        self.position_source = None
        self.label = None
        self.type_code = None
        self.motion = None
        self.comb = None
        self.comb_full = None
        self.alts: list[float] = []
        self.snrs: list[float] = []

    def add(self, key: str, entry: dict, label) -> None:
        self.n += 1
        for nid in [entry.get("node_id"), *(entry.get("contributing_node_ids") or [])]:
            if nid:
                self.nodes.add(nid)
                if state.node_world(nid) == "sim":
                    self.sim_nodes.add(nid)
        self.position_source = entry.get("position_source") or self.position_source
        if label is not None:
            self.label = label
        self.type_code = (entry.get("identity") or {}).get("type_code") or self.type_code
        self.motion = entry.get("motion_class") or self.motion
        if entry.get("comb"):
            self.comb = entry["comb"]
            self.comb_full = entry.get("_comb_full") or self.comb_full
        alt = _alt_m(entry.get("alt_baro"))
        if alt is not None:
            self.alts.append(alt)
        snr = (entry.get("_comb_full") or {}).get("median_body_snr_db")
        if snr is not None:
            self.snrs.append(snr)

    def world(self) -> str:
        """``sim`` when every node that saw the target is synthetic, else ``real``.

        A target no node is named on (an ADS-B-only entry) is the simulator's
        exactly when its label is the simulator's truth.
        """
        if self.nodes:
            return "sim" if self.nodes == self.sim_nodes else "real"
        return "sim" if self.label and self.label[2] == "sim_truth" else "real"

    def row(self, key: str) -> dict:
        label = self.label
        comb = self.comb_full
        if comb is None and self.comb:
            comb = {
                "frames": self.comb.get("frames"),
                "frac_frames_with_comb": self.comb.get("frac_frames_with_comb"),
                "median_n_lines": self.comb.get("n_lines"),
                "median_spacing_hz": self.comb.get("spacing_hz"),
            }
        return {
            cols.GROUP_COLUMN: key,
            "window_start_s": self.start,
            "world": self.world(),
            "node_ids": ",".join(sorted(self.nodes)),
            "n_nodes": len(self.nodes),
            "position_source": self.position_source,
            cols.LABEL_COLUMN: label[1] if label else None,
            "label_source": label[2] if label else None,
            "type_code": self.type_code,
            **cols.motion_row_from_dict(self.motion),
            **cols.comb_row(comb),
            "median_alt_m": _median(self.alts),
            "median_body_snr_db": _median(self.snrs),
            "n_frames": self.n,
        }


def _emit(key: str, window: _Window) -> None:
    """Move a finished window into the buffer.  The caller holds the lock."""
    if window.n < MIN_OBSERVATIONS or window.motion is None:
        return
    _rows.append(window.row(key))
    overflow = len(_rows) - MAX_BUFFERED_ROWS
    if overflow > 0:
        del _rows[:overflow]
        _stats["rows_dropped"] += overflow


def observe(entry: dict, label, now: float) -> None:
    """Fold one classified feed entry into its target's window.

    ``label`` is what target_classification scored the entry under, as
    ``(key, class, label_source)``, or None.  The window is keyed by the label's
    hex where there is one, so a target's rows stay together across the feed
    entries that carry it; otherwise by the entry's own hex.
    """
    if not enabled():
        return
    key = label[0] if label else entry.get("hex")
    if not key:
        return
    start = math.floor(now / WINDOW_S) * WINDOW_S
    with _lock:
        window = _open.get(key)
        if window is not None and window.start != start:
            _emit(key, _open.pop(key))
            window = None
        if window is None:
            window = _open[key] = _Window(start)
        window.add(key, entry, label)


def close_expired(now: float) -> None:
    """Emit a row for every window whose minute is over, so a target that left the feed still closes."""
    if not enabled():
        return
    with _lock:
        for key in [k for k, w in _open.items() if w.start + WINDOW_S <= now]:
            _emit(key, _open.pop(key))


def close_all() -> None:
    """Close every open window, short ones included: the process is going down."""
    with _lock:
        for key in list(_open):
            _emit(key, _open.pop(key))


def _hour_path(window_start_s: float) -> Path:
    t = datetime.fromtimestamp(window_start_s, tz=timezone.utc)
    return ARCHIVE_DIR / f"{t:%Y}" / f"{t:%m}" / f"{t:%d}" / f"{t:%H}.parquet"


def _write_hour(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    if path.exists():
        try:
            table = pa.concat_tables([pq.read_table(path).cast(_SCHEMA), table])
        except Exception:
            # An unreadable file would otherwise wedge this hour's rows
            # forever; keep it for a human and write the new rows beside it.
            aside = path.with_suffix(f".unreadable-{int(time.time())}")
            log.exception("classification archive: %s is unreadable, moving it to %s", path, aside.name)
            os.replace(path, aside)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def flush(now: float | None = None) -> list[str]:
    """Write every buffered row to its hour's file; returns the files written.

    Open windows stay open: only closed rows are written.  Raises on the first
    hour that fails to write, leaving that hour's rows (and every later one's)
    buffered; the hours before it are already on disk and already forgotten.
    """
    if not enabled():
        return []
    with _lock:
        pending = list(_rows)
    by_hour: dict[Path, list[dict]] = {}
    for row in pending:
        by_hour.setdefault(_hour_path(row["window_start_s"]), []).append(row)
    written = []
    for path, rows in sorted(by_hour.items()):
        try:
            _write_hour(path, rows)
        except Exception as exc:
            _stats["last_error"] = f"{type(exc).__name__}: {exc}"
            raise
        done = {id(r) for r in rows}
        with _lock:
            _rows[:] = [r for r in _rows if id(r) not in done]
            _stats["rows_written"] += len(rows)
        written.append(str(path.relative_to(ARCHIVE_DIR)))
    if written:
        _stats["last_flush"] = time.time() if now is None else now
        _stats["last_flush_files"] = written
        _stats["last_error"] = None
    return written


def stats() -> dict:
    """What the archive holds and has written, for the scorecard endpoint."""
    with _lock:
        return {
            "enabled": enabled(),
            "rows_buffered": len(_rows),
            "open_windows": len(_open),
            **{k: v for k, v in _stats.items()},
        }


def _reset_for_tests() -> None:
    with _lock:
        _open.clear()
        _rows.clear()
    _stats.update(rows_written=0, rows_dropped=0, last_flush=None, last_flush_files=[], last_error=None)
