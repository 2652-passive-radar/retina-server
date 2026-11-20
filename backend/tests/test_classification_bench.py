"""Smoke test for the offline classification bench: a short run writes a usable dataset."""

import pyarrow.parquet as pq
import pytest

from scripts import classification_bench
from services.target_identity import TARGET_CLASSES


@pytest.mark.timeout(60)
def test_short_run_writes_the_expected_dataset(tmp_path, capsys):
    out = tmp_path / "cls.parquet"
    res = classification_bench.main(
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
    table = pq.read_table(out)
    assert set(classification_bench.COLUMNS) == set(table.column_names)
    assert table.num_rows == len(res.rows) >= 1
    rows = table.to_pylist()
    present = {t.airframe for t in res.targets.values()}
    assert present <= set(TARGET_CLASSES)
    for airframe in present:
        assert any(r["airframe"] == airframe for r in rows)
    assert all(r["window_start_s"] % classification_bench.WINDOW_S == 0 for r in rows)
    printed = capsys.readouterr().out
    assert "spurious confirmed tracks per hour:" in printed
    assert classification_bench.spurious_tracks(res)["confirmed"] == len(res.tracks)
    assert "tower 49.345000,-122.972778" in printed
    assert "motion baseline vs simulator truth" in printed


def test_tower_sets_assign_each_node_its_towers_carrier():
    args = classification_bench.parse_args(["--nodes", "4", "--tower-set", "yvr3"])
    nodes = classification_bench._ring_nodes(
        args.nodes, args.center, classification_bench.TOWER_SETS["yvr3"], args.fc_hz, args.doppler_window
    )
    assert [n.fc_hz for n in nodes] == [599e6, 485e6, 473e6, 599e6]
    chan = classification_bench._ring_nodes(3, args.center, classification_bench.TOWER_SETS["chan"], None, 1000.0)
    assert {n.fc_hz for n in chan} == {521e6}
    override = classification_bench._ring_nodes(2, args.center, classification_bench.TOWER_SETS["chan"], 500e6, 1000.0)
    assert {n.fc_hz for n in override} == {500e6}


def test_colocated_layout_puts_every_node_on_one_rooftop():
    centre = (49.26, -123.25)
    nodes = classification_bench._ring_nodes(
        3, centre, classification_bench.TOWER_SETS["yvr3"], None, 1000.0, colocated=True, rx_alt_ft=330.0
    )
    assert {(n.rx_lat, n.rx_lon, n.rx_alt_ft) for n in nodes} == {(49.26, -123.25, 330.0)}
    assert len({(n.tx_lat, n.tx_lon) for n in nodes}) == 3


def test_spurious_tracks_counts_confirmed_tracks_that_never_sat_on_a_target():
    res = classification_bench.Result(duration_s=1800.0)
    audit = classification_bench.TrackAudit
    res.tracks = {
        ("n0", "a"): audit(frames=9, matched=True),
        ("n0", "b"): audit(frames=5, beside_body=True),
        ("n1", "c"): audit(frames=4),
        ("n1", "d"): audit(frames=7, matched=True, beside_body=True),
    }
    assert classification_bench.spurious_tracks(res) == {
        "confirmed": 4,
        "spurious": 2,
        "beside_body": 1,
        "per_hour": 4.0,
    }
    assert classification_bench.spurious_tracks(classification_bench.Result())["per_hour"] is None
