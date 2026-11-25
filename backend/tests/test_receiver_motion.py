"""Moving receivers: per-frame receiver pose on ingest, applied to solver configs."""

import math

import pytest

from core import state
from services import receiver_motion
from services.tasks import solver_pool

RX = (49.26, -123.25, 50.0)
TXS = {"a": (49.40, -123.00, 300.0), "b": (49.10, -123.05, 250.0), "c": (49.35, -123.55, 400.0)}


@pytest.fixture(autouse=True)
def _clean():
    receiver_motion._reset_for_tests()
    yield
    receiver_motion._reset_for_tests()


def _motion(lat=RX[0], lon=RX[1], ve=12.0, vn=-4.0):
    return {"lat": lat, "lon": lon, "alt_m": RX[2], "vel_east_ms": ve, "vel_north_ms": vn, "vel_up_ms": 0.0}


def _cfg(nid):
    tx = TXS[nid]
    return {"rx_lat": RX[0], "rx_lon": RX[1], "rx_alt_ft": RX[2] / 0.3048, "tx_lat": tx[0], "tx_lon": tx[1],
            "tx_alt_ft": tx[2] / 0.3048, "fc_hz": 600e6}  # fmt: skip


@pytest.mark.parametrize(
    "bad",
    [None, "x", {"lat": None, "lon": 1}, {"lat": 91, "lon": 0}, _motion(ve=500.0), {**_motion(), "alt_m": "nan"}],
)
def test_unusable_motion_is_not_recorded(bad):
    assert not receiver_motion.record("n1", {"timestamp": 1000, "rx_motion": bad})
    assert not receiver_motion.is_moving("n1")


def test_stationary_nodes_pass_through_untouched():
    cfgs = {"a": _cfg("a")}
    assert receiver_motion.apply({"timestamp_ms": 5000}, cfgs) is cfgs


def test_pose_is_interpolated_to_the_solve_epoch():
    receiver_motion.record("a", {"timestamp": 1000, "rx_motion": _motion(lat=49.0, ve=10.0)})
    receiver_motion.record("a", {"timestamp": 2000, "rx_motion": _motion(lat=49.001, ve=20.0)})
    lat, _, _, ve, _, _ = receiver_motion.pose_at("a", 1250)
    assert lat == pytest.approx(49.00025)
    assert ve == pytest.approx(12.5)


def test_pose_is_extrapolated_by_its_velocity_but_not_too_far():
    receiver_motion.record("a", {"timestamp": 1000, "rx_motion": _motion(lat=49.0, lon=-123.0, ve=0.0, vn=10.0)})
    lat, lon, *_ = receiver_motion.pose_at("a", 3000)
    assert (lat - 49.0) * math.pi / 180 * 6_371_000 == pytest.approx(20.0, rel=1e-6)
    assert lon == -123.0
    assert receiver_motion.pose_at("a", 1000 + receiver_motion._MAX_GAP_MS + 1) is None


def test_moving_node_gets_its_pose_and_velocity_and_a_stale_one_is_dropped():
    receiver_motion.record("a", {"timestamp": 10_000, "rx_motion": _motion(lat=49.27)})
    receiver_motion.record("b", {"timestamp": 1_000, "rx_motion": _motion()})
    cfgs = {nid: _cfg(nid) for nid in TXS}
    out = receiver_motion.apply({"timestamp_ms": 10_000}, cfgs)
    assert out["a"]["rx_lat"] == pytest.approx(49.27)
    assert out["a"]["rx_vel_east_ms"] == pytest.approx(12.0)
    assert "b" not in out
    assert out["c"] is cfgs["c"]
    assert cfgs["a"]["rx_lat"] == RX[0]


def _measure(rx_vel):
    """Noise-free delay/Doppler for one aircraft from a receiver moving at rx_vel (local ENU),
    by finite difference of the bistatic range."""
    from retina_geolocator import Geometry

    def ecef(lat, lon, alt):
        return Geometry.lla2ecef(lat, lon, alt)

    def vec(v, lat, lon, alt):
        a, b = Geometry.enu2ecef(*v, lat, lon, alt), ecef(lat, lon, alt)
        return [a[i] - b[i] for i in range(3)]

    def rng(p, r, t):
        return math.dist(p, t) + math.dist(p, r) - math.dist(r, t)

    tgt, tv = (49.33, -123.12, 6000.0), (180.0, -60.0, 0.0)
    p0, r0, vp, vr = ecef(*tgt), ecef(*RX), vec(tv, *tgt), vec(rx_vel, *RX)
    meas = []
    for nid, tx in TXS.items():
        t = ecef(*tx)
        h = 1e-3
        plus = rng([p0[i] + vp[i] * h for i in range(3)], [r0[i] + vr[i] * h for i in range(3)], t)
        minus = rng([p0[i] - vp[i] * h for i in range(3)], [r0[i] - vr[i] * h for i in range(3)], t)
        meas.append({"node_id": nid, "delay_us": rng(p0, r0, t) / 299.792458,
                     "doppler_hz": -(600e6 / 299_792_458.0) * (plus - minus) / (2 * h), "snr": 15})  # fmt: skip
    s_in = {"initial_guess": {"lat": 49.35, "lon": -123.14, "alt_km": 6.0}, "measurements": meas, "n_nodes": 3,
            "timestamp_ms": 50_000}  # fmt: skip
    return s_in, tv


def test_pool_solve_models_a_moving_receiver(monkeypatch):
    pytest.importorskip("retina_geolocator.multinode_solver")
    from retina_geolocator import multinode_solver

    if not hasattr(multinode_solver, "_rx_velocity_ms"):
        pytest.skip("retina_geolocator without the receiver Doppler model")
    s_in, tv = _measure((12.0, -4.0, 0.0))
    cfgs = {nid: _cfg(nid) for nid in TXS}
    blind = solver_pool._pool_solve_multinode(s_in, cfgs)
    for nid in TXS:
        receiver_motion.record(nid, {"timestamp": 50_000, "rx_motion": _motion()})
    res = solver_pool._pool_solve_multinode(s_in, cfgs)
    err = math.hypot(res["vel_east"] - tv[0], res["vel_north"] - tv[1])
    assert err < 0.5
    assert math.hypot(blind["vel_east"] - tv[0], blind["vel_north"] - tv[1]) > 5 * max(err, 0.1)
    assert state.rx_motion_applied >= 3
