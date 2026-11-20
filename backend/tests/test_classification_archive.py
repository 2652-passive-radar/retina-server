"""The classification archive: one row per target per minute, flushed to hourly Parquet."""

from datetime import datetime, timezone

import pyarrow.parquet as pq
import pytest

from core import state
from services import aircraft_db, classification_archive, target_classification
from services import classification_columns as cols

# 2026-11-17 05:00:00 UTC, on a minute and an hour boundary.
T0 = datetime(2026, 11, 17, 5, 0, 0, tzinfo=timezone.utc).timestamp()

REGISTRY_CSV = "4CA1F2;EI-DCL;B738;00;BOEING 737-800;;Ryanair;"
REGISTRY_TYPES = {"B738": ["BOEING 737-800", "L2J", "M"]}


@pytest.fixture(autouse=True)
def archive_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(classification_archive, "ARCHIVE_DIR", tmp_path / "classification_archive")
    monkeypatch.setenv(classification_archive.ENABLED_ENV, "1")
    return tmp_path / "classification_archive"


def entry(hex_code="mn0123456789", node="real-1", cls="jet", **kw):
    e = {
        "hex": hex_code,
        "node_id": node,
        "position_source": "multinode_solve",
        "alt_baro": 35_000,
        "motion_class": {
            "class": cls,
            "confidence": 0.9,
            "scores": {"jet": 0.9, "turboprop": 0.1},
            "features": {"span_s": 40.0, "n_chords": 4, "speed_ms": 230.0, "alt_m": 10_668.0},
        },
    }
    e.update(kw)
    return e


def feed(e, times, label=None):
    for t in times:
        classification_archive.observe(e, label, t)


def rows_buffered():
    return classification_archive._rows


class TestWindows:
    def test_a_window_closes_at_its_boundary_not_before(self):
        feed(entry(), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 59.9)
        assert rows_buffered() == []
        classification_archive.close_expired(T0 + 60.0)
        (row,) = rows_buffered()
        assert row["window_start_s"] == T0
        assert row["n_frames"] == 3

    def test_an_observation_in_the_next_window_closes_the_last(self):
        feed(entry(), [T0 + 10, T0 + 20, T0 + 30, T0 + 61])
        (row,) = rows_buffered()
        assert row["window_start_s"] == T0
        assert classification_archive.stats()["open_windows"] == 1

    def test_a_target_seen_too_few_times_makes_no_row(self):
        feed(entry(), [T0 + 1, T0 + 2])
        classification_archive.close_expired(T0 + 120)
        assert rows_buffered() == []

    def test_each_hex_has_its_own_window(self):
        feed(entry("mnaaaaaaaaaa"), [T0 + 1, T0 + 2, T0 + 3])
        feed(entry("mnbbbbbbbbbb"), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        assert {r["hex"] for r in rows_buffered()} == {"mnaaaaaaaaaa", "mnbbbbbbbbbb"}

    def test_the_row_carries_the_verdict_features_and_medians(self):
        e = entry(contributing_node_ids=["real-2"])
        for t, alt in zip((1, 2, 3), (30_000, 34_000, 38_000)):
            e["alt_baro"] = alt
            classification_archive.observe(e, None, T0 + t)
        classification_archive.close_expired(T0 + 60)
        (row,) = rows_buffered()
        assert row["baseline_class"] == "jet"
        assert row["baseline_confidence"] == 0.9
        assert row["baseline_score_jet"] == 0.9 and row["baseline_score_drone"] is None
        assert row["speed_ms"] == 230.0 and row["turn_dps"] is None
        assert row["median_alt_m"] == pytest.approx(34_000 * 0.3048)
        assert row["node_ids"] == "real-1,real-2" and row["n_nodes"] == 2
        assert row["position_source"] == "multinode_solve"
        assert set(row) == set(cols.ARCHIVE_COLUMNS)

    def test_a_ground_altitude_and_a_missing_one(self):
        feed(entry(alt_baro="ground"), [T0 + 1, T0 + 2, T0 + 3])
        feed(entry("mnbbbbbbbbbb", alt_baro=None), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        by_hex = {r["hex"]: r for r in rows_buffered()}
        assert by_hex["mn0123456789"]["median_alt_m"] == 0.0
        assert by_hex["mnbbbbbbbbbb"]["median_alt_m"] is None


class TestLabels:
    def test_a_hex_label_is_recorded_with_its_source_and_type(self):
        aircraft_db.build(REGISTRY_CSV, REGISTRY_TYPES, aircraft_db.db_path())
        e = {
            "hex": "4ca1f2",
            "node_id": "real-1",
            "position_source": "adsb_single_node",
            "gs": 450.0,
            "alt_baro": 35_000,
        }
        for t in (1, 2, 3):
            target_classification.classify_feed([e], now=T0 + t)
        classification_archive.close_expired(T0 + 60)
        (row,) = rows_buffered()
        assert (row["hex"], row["airframe"], row["label_source"], row["type_code"]) == ("4ca1f2", "jet", "hex", "B738")
        assert row["world"] == "real"
        assert row["baseline_class"] == "jet"

    def test_an_unlabelled_target_is_kept_without_a_label(self):
        e = {"hex": "mn0123456789", "node_id": "real-1", "position_source": "adsb_single_node", "gs": 450.0}
        for t in (1, 2, 3):
            target_classification.classify_feed([dict(e)], now=T0 + t)
        classification_archive.close_expired(T0 + 60)
        (row,) = rows_buffered()
        assert (row["airframe"], row["label_source"], row["type_code"]) == (None, None, None)
        assert row["baseline_class"] is not None

    def test_a_simulator_label_is_keyed_by_its_truth_hex_and_marked_sim(self):
        state.ground_truth_meta["a1b2c3"] = {"airframe": "turboprop"}
        for t in (1, 2, 3):
            e = entry("mn0123456789", node="synth-1", ground_truth_hex="a1b2c3")
            target_classification.classify_feed([e], now=T0 + t)
        classification_archive.close_expired(T0 + 60)
        (row,) = rows_buffered()
        assert (row["hex"], row["airframe"], row["label_source"], row["world"]) == (
            "a1b2c3",
            "turboprop",
            "sim_truth",
            "sim",
        )

    def test_a_target_seen_from_a_synthetic_and_a_real_node_is_real(self):
        feed(entry(node="synth-1", contributing_node_ids=["real-1"]), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        assert rows_buffered()[0]["world"] == "real"


class TestComb:
    def test_the_full_summary_fills_the_comb_and_snr_columns(self):
        full = {
            "frames": 30,
            "frac_frames_with_comb": 0.8,
            "median_n_lines": 3.0,
            "max_n_lines": 5,
            "median_spacing_hz": 102.5,
            "spacing_spread_hz": 1.5,
            "median_max_offset_hz": 210.0,
            "median_symmetry": 0.7,
            "median_rel_snr_db": -12.0,
            "median_body_snr_db": 22.0,
        }
        e = entry(
            comb={"n_lines": 3.0, "spacing_hz": 102.5, "frac_frames_with_comb": 0.8, "frames": 30}, _comb_full=full
        )
        feed(e, [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        (row,) = rows_buffered()
        assert row["comb_median_spacing_hz"] == 102.5 and row["comb_max_n_lines"] == 5
        assert row["comb_median_symmetry"] == 0.7
        assert row["median_body_snr_db"] == 22.0

    def test_the_compact_comb_alone_fills_what_it_holds(self):
        e = entry(comb={"n_lines": 2.0, "spacing_hz": 90.0, "frac_frames_with_comb": 0.5, "frames": 10})
        feed(e, [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        (row,) = rows_buffered()
        assert (row["comb_frames"], row["comb_median_n_lines"], row["comb_median_spacing_hz"]) == (10, 2.0, 90.0)
        assert row["comb_median_symmetry"] is None

    def test_no_comb_is_null_rather_than_zero(self):
        feed(entry(), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        assert rows_buffered()[0]["comb_frames"] is None

    def test_the_private_summary_never_leaves_the_feed(self):
        e = entry(comb={"frames": 1}, _comb_full={"frames": 1})
        out = target_classification.classify_feed([e], now=T0)
        assert "_comb_full" not in out[0]
        assert out[0]["comb"] == {"frames": 1}


class TestFlush:
    def _close_one(self, hex_code="mn0123456789", start=T0):
        feed(entry(hex_code), [start + 1, start + 2, start + 3])
        classification_archive.close_expired(start + 60)

    def test_flush_writes_readable_parquet_at_the_hour_path(self, archive_dir):
        self._close_one()
        files = classification_archive.flush()
        assert files == ["2026/11/17/05.parquet"]
        table = pq.read_table(archive_dir / "2026" / "11" / "17" / "05.parquet")
        assert set(table.column_names) == set(cols.ARCHIVE_COLUMNS)
        (row,) = table.to_pylist()
        assert row["hex"] == "mn0123456789" and row["world"] == "real"
        stats = classification_archive.stats()
        assert (stats["rows_buffered"], stats["rows_written"]) == (0, 1)
        assert stats["last_flush"] is not None

    def test_a_second_flush_in_the_same_hour_keeps_the_first(self, archive_dir):
        self._close_one("mnaaaaaaaaaa")
        classification_archive.flush()
        self._close_one("mnbbbbbbbbbb", start=T0 + 600)
        classification_archive.flush()
        rows = pq.read_table(archive_dir / "2026" / "11" / "17" / "05.parquet").to_pylist()
        assert sorted(r["hex"] for r in rows) == ["mnaaaaaaaaaa", "mnbbbbbbbbbb"]
        assert not list(archive_dir.rglob(".*"))

    def test_rows_land_in_the_hour_of_their_window(self, archive_dir):
        self._close_one("mnaaaaaaaaaa", start=T0 + 3540)
        self._close_one("mnbbbbbbbbbb", start=T0 + 3600)
        assert classification_archive.flush() == ["2026/11/17/05.parquet", "2026/11/17/06.parquet"]

    def test_an_open_window_is_not_flushed(self, archive_dir):
        feed(entry(), [T0 + 1, T0 + 2, T0 + 3])
        assert classification_archive.flush() == []
        assert not archive_dir.exists()

    def test_a_failed_write_keeps_the_rows_for_the_next_flush(self, archive_dir, monkeypatch):
        self._close_one()
        # A context, not monkeypatch.undo(): undo would also revert the
        # fixture's own patches and leave the archive disabled.
        with monkeypatch.context() as m:
            m.setattr(pq, "write_table", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
            with pytest.raises(OSError):
                classification_archive.flush()
            assert classification_archive.stats()["rows_buffered"] == 1
            assert "disk full" in classification_archive.stats()["last_error"]
        assert classification_archive.flush() == ["2026/11/17/05.parquet"]
        assert classification_archive.stats()["last_error"] is None

    def test_an_unreadable_hour_file_is_set_aside_not_fatal(self, archive_dir):
        path = archive_dir / "2026" / "11" / "17" / "05.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not parquet")
        self._close_one()
        assert classification_archive.flush() == ["2026/11/17/05.parquet"]
        assert pq.read_table(path).num_rows == 1
        assert len(list(path.parent.glob("05.unreadable-*"))) == 1

    def test_the_buffer_is_bounded(self, monkeypatch):
        monkeypatch.setattr(classification_archive, "MAX_BUFFERED_ROWS", 2)
        for i in range(4):
            self._close_one(f"mn{i:010d}", start=T0 + 60 * i)
        assert classification_archive.stats()["rows_buffered"] == 2
        assert classification_archive.stats()["rows_dropped"] == 2

    def test_close_all_closes_open_windows_for_shutdown(self, archive_dir):
        feed(entry(), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_all()
        assert classification_archive.flush() == ["2026/11/17/05.parquet"]


class TestDisabled:
    @pytest.mark.parametrize("value, on", [("0", False), ("1", True), ("", True), ("false", True)])
    def test_only_an_exact_zero_turns_it_off(self, monkeypatch, value, on):
        monkeypatch.setenv(classification_archive.ENABLED_ENV, value)
        assert classification_archive.enabled() is on

    def test_disabled_buffers_and_writes_nothing(self, archive_dir, monkeypatch):
        monkeypatch.setenv(classification_archive.ENABLED_ENV, "0")
        feed(entry(), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        assert classification_archive.stats()["open_windows"] == 0
        assert classification_archive.flush() == []
        assert not archive_dir.exists()
        assert classification_archive.stats()["enabled"] is False

    def test_turning_it_off_with_rows_buffered_writes_nothing(self, archive_dir, monkeypatch):
        feed(entry(), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        monkeypatch.setenv(classification_archive.ENABLED_ENV, "0")
        assert classification_archive.flush() == []
        assert not archive_dir.exists()


class TestNeverBreaksTheFeed:
    def test_an_archive_failure_is_counted_and_the_entry_still_classified(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("archive down")

        monkeypatch.setattr(classification_archive, "observe", boom)
        monkeypatch.setattr(classification_archive, "close_expired", boom)
        e = {"hex": "mn0123456789", "node_id": "real-1", "position_source": "adsb_single_node", "gs": 450.0}
        out = target_classification.classify_feed([e], now=T0)
        assert out == [e] and "motion_class" in e
        assert state.classification_archive_errors == 2
        assert state.target_classification_errors == 0

    def test_a_malformed_entry_is_counted_not_raised(self):
        e = {"hex": "mn0123456789", "node_id": "real-1", "position_source": "adsb_single_node", "gs": 450.0}
        e["contributing_node_ids"] = 5
        out = target_classification.classify_feed([e], now=T0)
        assert out == [e]
        assert state.classification_archive_errors == 1

    def test_the_scorecard_endpoint_reports_the_archive(self, client):
        feed(entry(), [T0 + 1, T0 + 2, T0 + 3])
        classification_archive.close_expired(T0 + 60)
        classification_archive.flush()
        body = client.get("/api/admin/classification/scorecard").json()
        archive = body["archive"]
        assert archive["enabled"] is True
        assert (archive["rows_buffered"], archive["rows_written"]) == (0, 1)
        assert archive["last_flush"] is not None
