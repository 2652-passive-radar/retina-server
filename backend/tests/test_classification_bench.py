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
    assert "tower 49.345000,-122.972778" in printed
    assert "motion baseline vs simulator truth" in printed


def test_tower_sets_assign_each_node_its_towers_carrier():
    args = classification_bench.parse_args(["--nodes", "4", "--tower-set", "yvr3"])
    nodes = classification_bench._ring_nodes(
        args.nodes, args.center, classification_bench.TOWER_SETS["yvr3"], args.fc_hz, args.doppler_window
    )
    assert [n.fc_hz for n in nodes] == [521e6, 491e6, 533e6, 521e6]
    chan = classification_bench._ring_nodes(3, args.center, classification_bench.TOWER_SETS["chan"], None, 1000.0)
    assert {n.fc_hz for n in chan} == {521e6}
    override = classification_bench._ring_nodes(2, args.center, classification_bench.TOWER_SETS["chan"], 500e6, 1000.0)
    assert {n.fc_hz for n in override} == {500e6}
