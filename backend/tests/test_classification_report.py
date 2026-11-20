"""The classification report: reads the live archive and the bench's Parquet alike."""

from datetime import datetime, timezone

import pyarrow.parquet as pq
import pytest

from scripts import classification_bench, classification_report
from services import classification_archive
from services import classification_columns as cols

T0 = datetime(2026, 11, 17, 5, 0, 0, tzinfo=timezone.utc).timestamp()


def archive_entry(hex_code, cls, node="real-1"):
    return {
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
        "comb": {"n_lines": 3.0, "spacing_hz": 102.0, "frac_frames_with_comb": 0.9, "frames": 20},
    }


def write_archive(root, monkeypatch):
    """An archive tree: two hex-labelled aircraft, one unlabelled, one simulated, across two hours."""
    monkeypatch.setattr(classification_archive, "ARCHIVE_DIR", root)
    targets = [
        # (hex, motion verdict, label, node, window start offset)
        ("4ca1f2", "jet", ("4ca1f2", "jet", "hex"), "real-1", 0),
        ("c0ffee", "jet", ("c0ffee", "turboprop", "hex"), "real-1", 0),
        ("mn0000000001", "piston", None, "real-1", 60),
        ("a1b2c3", "turboprop", ("a1b2c3", "turboprop", "sim_truth"), "synth-1", 3600),
    ]
    for hex_code, verdict, label, node, offset in targets:
        for t in (1, 2, 3):
            classification_archive.observe(archive_entry(hex_code, verdict, node), label, T0 + offset + t)
    classification_archive.close_expired(T0 + 7200)
    assert classification_archive.flush() == ["2026/11/17/05.parquet", "2026/11/17/06.parquet"]
    return root


@pytest.fixture(autouse=True)
def archive_on(monkeypatch):
    monkeypatch.setenv(classification_archive.ENABLED_ENV, "1")


@pytest.fixture
def archive(tmp_path, monkeypatch):
    return write_archive(tmp_path / "archive", monkeypatch)


def report(argv, capsys):
    rc = classification_report.main([str(a) for a in argv])
    return rc, capsys.readouterr()


class TestArchiveInput:
    def test_the_default_reads_real_rows_only(self, archive, capsys):
        rc, out = report([archive], capsys)
        assert rc == 0
        assert "3 of 4 rows" in out.out
        assert "2 labelled with a class the baseline predicts, 2 targets" in out.out
        assert "motion baseline vs label, one vote per target (hex)" in out.out
        assert "label source hex" in out.out and "label source sim_truth" not in out.out
        assert "comb features per class" in out.out and "jet vs turboprop" in out.out

    def test_the_confusion_matrix_counts_the_verdicts_against_the_labels(self, archive, capsys):
        _, out = report([archive, "--label-source", "hex"], capsys)
        out_lines = out.out.splitlines()
        first_card = out_lines.index(next(line for line in out_lines if "truth \\ predicted" in line))
        matrix = {line.split()[0]: line.split()[1:] for line in out_lines[first_card + 1 : first_card + 6]}
        # Both labelled aircraft were called jet: one rightly, one a turboprop.
        assert matrix["jet"] == ["1", "0", "0", "0", "0", "0"]
        assert matrix["turboprop"] == ["1", "0", "0", "0", "0", "0"]
        assert "accuracy=0.5" in out.out

    def test_world_all_adds_the_simulated_rows(self, archive, capsys):
        _, out = report([archive, "--world", "all"], capsys)
        assert "4 of 4 rows" in out.out and "label source sim_truth" in out.out

    def test_world_sim_alone(self, archive, capsys):
        _, out = report([archive, "--world", "sim"], capsys)
        assert "1 of 4 rows" in out.out

    def test_label_source_filter(self, archive, capsys):
        _, out = report([archive, "--world", "all", "--label-source", "sim_truth"], capsys)
        assert "1 of 4 rows" in out.out

    def test_since_and_until_bound_the_window_start(self, archive, capsys):
        _, out = report([archive, "--world", "all", "--since", "2026-11-17T06:00:00"], capsys)
        assert "1 of 4 rows" in out.out
        _, out = report([archive, "--world", "all", "--until", "2026-11-17T06:00:00"], capsys)
        assert "3 of 4 rows" in out.out
        _, out = report([archive, "--world", "all", "--since", str(T0 + 30), "--until", str(T0 + 90)], capsys)
        assert "1 of 4 rows" in out.out

    def test_a_single_file_is_a_valid_input(self, archive, capsys):
        _, out = report([archive / "2026" / "11" / "17" / "05.parquet"], capsys)
        assert "3 of 3 rows" in out.out

    def test_unlabelled_rows_are_counted_but_not_scored(self, archive, capsys):
        _, out = report([archive, "--world", "real"], capsys)
        assert "3 of 4 rows" in out.out and "2 labelled" in out.out

    def test_an_archive_of_only_unlabelled_rows_says_so(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(classification_archive, "ARCHIVE_DIR", tmp_path / "a")
        for t in (1, 2, 3):
            classification_archive.observe(archive_entry("mn0000000001", "piston"), None, T0 + t)
        classification_archive.close_expired(T0 + 60)
        classification_archive.flush()
        rc, out = report([tmp_path / "a"], capsys)
        assert rc == 0 and "nothing labelled to score" in out.out


class TestExport:
    def test_the_export_is_the_filtered_labelled_rows_with_a_group_column(self, archive, tmp_path, capsys):
        dest = tmp_path / "out" / "train.parquet"
        _, out = report([archive, "--world", "all", "--export", dest], capsys)
        assert "wrote 3 labelled rows" in out.out
        table = pq.read_table(dest)
        assert set(table.column_names) == set(cols.ARCHIVE_COLUMNS) | {"group"}
        rows = table.to_pylist()
        assert sorted(r["group"] for r in rows) == ["4ca1f2", "a1b2c3", "c0ffee"]
        assert all(r["group"] == r["hex"] and r["airframe"] for r in rows)

    def test_export_respects_the_filters(self, archive, tmp_path, capsys):
        dest = tmp_path / "train.parquet"
        report([archive, "--label-source", "hex", "--export", dest], capsys)
        assert sorted(r["group"] for r in pq.read_table(dest).to_pylist()) == ["4ca1f2", "c0ffee"]

    def test_nothing_labelled_writes_no_file(self, archive, tmp_path, capsys):
        dest = tmp_path / "none.parquet"
        _, out = report([archive, "--since", str(T0 + 30), "--until", str(T0 + 90), "--export", dest], capsys)
        assert "no labelled rows to export" in out.out and not dest.exists()


class TestBenchInput:
    @pytest.fixture(scope="class")
    def bench_file(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("bench") / "cls.parquet"
        classification_bench.main(
            [
                "--duration-s", "60",
                "--nodes", "3",
                "--assoc-interval", "10",
                "--min-aircraft", "12",
                "--max-aircraft", "12",
                "--min-frames", "1",
                "--airframe-mix", "jet:0.3,turboprop:0.3,piston:0.1,helicopter:0.2,drone:0.1",
                "--out", str(out),
            ]
        )  # fmt: skip
        return out

    def test_a_bench_file_reads_as_simulated_and_sim_truth(self, bench_file, capsys):
        capsys.readouterr()
        n = pq.read_table(bench_file).num_rows
        rc, out = report([bench_file, "--world", "sim", "--label-source", "sim_truth"], capsys)
        assert rc == 0
        assert f"{n} of {n} rows" in out.out
        assert "motion baseline vs label, per row (window)" in out.out
        assert "label source sim_truth" in out.out

    def test_the_default_world_selects_none_of_a_bench_file(self, bench_file, capsys):
        capsys.readouterr()
        _, out = report([bench_file], capsys)
        assert "no rows match the filters" in out.out and "'sim': " in out.out

    def test_a_bench_file_exports_with_the_object_id_as_the_group(self, bench_file, tmp_path, capsys):
        capsys.readouterr()
        dest = tmp_path / "train.parquet"
        report([bench_file, "--world", "all", "--export", dest], capsys)
        bench_ids = {r["object_id"] for r in pq.read_table(bench_file).to_pylist()}
        rows = pq.read_table(dest).to_pylist()
        assert {r["group"] for r in rows} == bench_ids
        assert all(r["world"] == "sim" and r["label_source"] == "sim_truth" for r in rows)

    def test_a_bench_file_and_an_archive_report_together(self, bench_file, archive, capsys):
        capsys.readouterr()
        n = pq.read_table(bench_file).num_rows
        _, out = report([bench_file, archive, "--world", "all"], capsys)
        assert f"{n + 4} of {n + 4} rows" in out.out


class TestRobustness:
    def test_a_missing_path(self, tmp_path, capsys):
        rc, out = report([tmp_path / "nope"], capsys)
        assert rc == 0
        assert "no rows: the archive is empty or missing" in out.out
        assert "no Parquet files found" in out.err

    def test_an_empty_directory(self, tmp_path, capsys):
        rc, out = report([tmp_path], capsys)
        assert rc == 0 and "no rows: the archive is empty or missing" in out.out

    def test_an_unreadable_file_is_skipped_and_the_rest_reported(self, archive, capsys):
        (archive / "2026" / "11" / "17" / "07.parquet").write_bytes(b"half a file")
        rc, out = report([archive], capsys)
        assert rc == 0 and "3 of 4 rows" in out.out
        assert "07.parquet: unreadable" in out.err

    def test_a_flush_in_progress_is_not_read(self, archive, capsys):
        (archive / "2026" / "11" / "17" / ".07.parquet.tmp").write_bytes(b"half a file")
        _, out = report([archive], capsys)
        assert out.err == ""

    def test_export_with_an_empty_archive_writes_nothing(self, tmp_path, capsys):
        dest = tmp_path / "train.parquet"
        rc, _ = report([tmp_path, "--export", dest], capsys)
        assert rc == 0 and not dest.exists()
