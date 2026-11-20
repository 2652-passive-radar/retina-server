"""The columns a classification dataset row carries, shared by every writer and reader.

Two producers write one row per target per 60 s window: the offline bench
(scripts/classification_bench.py, from the simulator's world) and the live
archive (services.classification_archive, from the feed).  One consumer reads
both (scripts/classification_report.py).  A column whose meaning is the same in
both has the same name and type here, defined once, so a rename or a retyped
column cannot leave the report reading one file and not the other.

What differs is only identity and provenance, which each producer declares for
itself: the bench knows the simulator's ``object_id`` and its true ``airframe``;
the archive knows the ``hex``, which ``world`` the nodes belong to and where the
label came from.
"""

import pyarrow as pa

from services.target_identity import TARGET_CLASSES

WINDOW_S = 60.0

# Where a row's label came from; the strings services.target_classification
# records labels under.  A bench row carries no such column and reads as
# sim_truth.
LABEL_SOURCES = ("hex", "sim_truth")

MOTION_COLUMNS: dict[str, pa.DataType] = {
    "span_s": pa.float64(),
    "n_chords": pa.int64(),
    "speed_ms": pa.float64(),
    "max_speed_ms": pa.float64(),
    "speed_floor_ms": pa.float64(),
    "alt_m": pa.float64(),
    "turn_dps": pa.float64(),
    "climb_ms": pa.float64(),
    "hover_fraction": pa.float64(),
}

BASELINE_COLUMNS: dict[str, pa.DataType] = {
    "baseline_class": pa.string(),
    "baseline_confidence": pa.float64(),
    **{f"baseline_score_{c}": pa.float64() for c in TARGET_CLASSES},
}

# CombAccumulator.summary() key -> column.
COMB_FIELDS = {
    "frames": "comb_frames",
    "frac_frames_with_comb": "comb_frac_frames",
    "median_n_lines": "comb_median_n_lines",
    "max_n_lines": "comb_max_n_lines",
    "median_spacing_hz": "comb_median_spacing_hz",
    "spacing_spread_hz": "comb_spacing_spread_hz",
    "median_max_offset_hz": "comb_median_max_offset_hz",
    "median_symmetry": "comb_median_symmetry",
    "median_rel_snr_db": "comb_median_rel_snr_db",
}
COMB_COLUMNS = tuple(COMB_FIELDS.values())
COMB_TYPES: dict[str, pa.DataType] = {c: pa.float64() for c in COMB_COLUMNS}

GEOMETRY_COLUMNS: dict[str, pa.DataType] = {
    "n_nodes": pa.int64(),
    "node_ids": pa.string(),
    "median_alt_m": pa.float64(),
    "median_body_snr_db": pa.float64(),
    "n_frames": pa.int64(),
}

# What a report needs from either file; the rest is producer-specific.
LABEL_COLUMN = "airframe"
GROUP_COLUMN = "hex"

# The archive's own columns, after the shared ones.
ARCHIVE_COLUMNS: dict[str, pa.DataType] = {
    GROUP_COLUMN: pa.string(),
    "window_start_s": pa.float64(),
    "world": pa.string(),
    "node_ids": pa.string(),
    "n_nodes": pa.int64(),
    "position_source": pa.string(),
    LABEL_COLUMN: pa.string(),
    "label_source": pa.string(),
    "type_code": pa.string(),
    **MOTION_COLUMNS,
    **BASELINE_COLUMNS,
    **COMB_TYPES,
    "median_alt_m": pa.float64(),
    "median_body_snr_db": pa.float64(),
    "n_frames": pa.int64(),
}


def motion_row(motion) -> dict:
    """The motion and baseline columns for a services.motion_classifier.MotionClass."""
    f = motion.features
    return {
        "span_s": f.span_s,
        "n_chords": f.n_chords,
        "speed_ms": f.speed_ms,
        "max_speed_ms": f.max_speed_ms,
        "speed_floor_ms": f.speed_floor_ms,
        "alt_m": f.alt_m,
        "turn_dps": f.turn_dps,
        "climb_ms": f.climb_ms,
        "hover_fraction": f.hover_fraction,
        "baseline_class": motion.cls,
        "baseline_confidence": motion.confidence,
        **{f"baseline_score_{c}": motion.scores.get(c) for c in TARGET_CLASSES},
    }


def motion_row_from_dict(motion: dict | None) -> dict:
    """motion_row for a feed entry's ``motion_class`` (MotionClass.as_dict), which omits unmeasured features."""
    motion = motion or {}
    feats = motion.get("features") or {}
    scores = motion.get("scores") or {}
    return {
        **{k: feats.get(k) for k in MOTION_COLUMNS},
        "baseline_class": motion.get("class"),
        "baseline_confidence": motion.get("confidence"),
        **{f"baseline_score_{c}": scores.get(c) for c in TARGET_CLASSES},
    }


def comb_row(summary: dict | None) -> dict:
    """The comb columns for a CombAccumulator.summary(), all null where there is none."""
    summary = summary or {}
    return {col: summary.get(key) for key, col in COMB_FIELDS.items()}


def write_parquet(rows: list[dict], path, columns: dict[str, pa.DataType]) -> None:
    import os

    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(list(columns.items()))), path)
