"""Comb features: spacing recovery on synthetic frames, and on frames from the real simulator."""

import math
import random

import pytest

from services.comb_features import CombAccumulator, CombObs, estimate_spacing, frame_comb

BODY_DELAY = 20.0
BODY_DOPPLER = 150.0
BODY_SNR = 20.0


def _frame(line_offsets, snr_drop=10.0, extra=()):
    delays = [BODY_DELAY] + [BODY_DELAY + 0.05 * i for i, _ in enumerate(line_offsets)]
    dopplers = [BODY_DOPPLER] + [BODY_DOPPLER + o for o in line_offsets]
    snrs = [BODY_SNR] + [BODY_SNR - snr_drop] * len(line_offsets)
    for d, f, s in extra:
        delays.append(d)
        dopplers.append(f)
        snrs.append(s)
    return frame_comb(delays, dopplers, snrs, BODY_DELAY, BODY_DOPPLER)


def _comb(spacing, ks):
    return [k * spacing for k in ks]


def test_spacing_of_a_slow_rotor_comb():
    obs = _frame(_comb(13.6, (-3, -2, -1, 1, 2, 3)))
    assert obs.n_lines == 6
    assert obs.spacing_hz == pytest.approx(13.6, rel=0.03)


def test_spacing_of_a_propeller_comb():
    obs = _frame(_comb(102.0, (-2, -1, 1, 2, 3)))
    assert obs.spacing_hz == pytest.approx(102.0, rel=0.03)


def test_a_jet_has_no_lines():
    obs = _frame([])
    assert (obs.n_lines, obs.spacing_hz, obs.symmetry, obs.mean_rel_snr_db) == (0, None, None, None)
    assert obs.offsets_hz == ()


def test_one_line_has_no_spacing():
    obs = _frame([102.0])
    assert obs.n_lines == 1
    assert obs.spacing_hz is None


def test_other_targets_at_other_delays_are_ignored():
    obs = _frame(_comb(102.0, (-1, 1, 2)), extra=[(35.0, 310.0, 15.0), (35.1, 412.0, 12.0), (3.0, -500.0, 6.0)])
    assert obs.n_lines == 3
    assert obs.spacing_hz == pytest.approx(102.0, rel=0.03)


def test_the_body_detection_is_not_a_line():
    obs = _frame([102.0, -102.0])
    assert 0.0 not in obs.offsets_hz
    assert obs.n_lines == 2


def test_a_sub_harmonic_is_not_chosen():
    offsets = [-204.0, -102.0, 102.0, 204.0, 306.0]
    spacing = estimate_spacing(offsets)
    assert spacing == pytest.approx(102.0, rel=0.03)
    assert estimate_spacing([13.6 * k for k in (1, 2, 3, 4, 5, 6)]) == pytest.approx(13.6, rel=0.03)


def test_spacing_tolerates_measurement_noise():
    rng = random.Random(4)
    offsets = [k * 102.0 * 1.02 + rng.gauss(0, 1.0) for k in (-3, -2, -1, 1, 2, 3)]
    assert estimate_spacing(offsets) == pytest.approx(104.0, rel=0.03)


def test_symmetry_and_relative_snr():
    sym = _frame([-102.0, 102.0, -204.0, 204.0], snr_drop=12.0)
    assert sym.symmetry == 1.0
    assert sym.mean_rel_snr_db == pytest.approx(-12.0)
    lopsided = _frame([102.0, 204.0, 306.0])
    assert lopsided.symmetry == 0.0
    assert lopsided.max_abs_offset_hz == pytest.approx(306.0)


def test_offsets_are_sorted_and_relative_snr_needs_a_body():
    obs = frame_comb(
        [BODY_DELAY] * 2, [BODY_DOPPLER + 204.0, BODY_DOPPLER - 102.0], [8.0, 9.0], BODY_DELAY, BODY_DOPPLER
    )
    assert obs.offsets_hz == (-102.0, 204.0)
    assert obs.mean_rel_snr_db is None


def test_accumulator_summary():
    acc = CombAccumulator()
    assert acc.summary()["frames"] == 0
    assert acc.summary()["median_spacing_hz"] is None
    assert acc.summary()["frac_frames_with_comb"] == 0.0
    acc.add(_frame([]))
    acc.add(_frame(_comb(102.0, (-1, 1, 2))))
    acc.add(_frame(_comb(102.0, (-2, -1, 1, 2))))
    s = acc.summary()
    assert s["frames"] == 3
    assert s["frac_frames_with_comb"] == pytest.approx(2 / 3)
    assert s["median_n_lines"] == 3.5
    assert s["max_n_lines"] == 4
    assert s["median_spacing_hz"] == pytest.approx(102.0, rel=0.03)
    assert s["spacing_spread_hz"] < 5.0
    assert s["median_max_offset_hz"] == pytest.approx(204.0)
    assert s["median_rel_snr_db"] == pytest.approx(-10.0)


def test_accumulator_of_no_combs_is_empty_safe():
    acc = CombAccumulator()
    acc.add(CombObs((), 0, 0.0, None, None, None))
    s = acc.summary()
    assert s["frames"] == 1 and s["max_n_lines"] == 0
    assert s["median_n_lines"] is None and s["spacing_spread_hz"] is None


def test_spacing_recovered_from_a_simulated_turboprop(monkeypatch):
    from retina_simulation.micro_doppler import AIRFRAMES, Rotor
    from retina_simulation.world import NodeConfig, SimulatedAircraft, SimulationWorld

    rx_lat, rx_lon = 49.19, -123.18
    # The shipped line strength leaves most lines under the detection floor at
    # the legacy SNR model, so boost it as the simulator's own tests do: this
    # tests the recovery of the spacing, not the interim calibration.
    monkeypatch.setitem(AIRFRAMES, "turboprop", tuple(Rotor(6, 2.05, 17.0, "forward", 15.0) for _ in range(2)))
    random.seed(3)
    world = SimulationWorld(center_lat=rx_lat, center_lon=rx_lon)
    world.add_node(
        NodeConfig(
            node_id="n",
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt_ft=0.0,
            tx_lat=rx_lat + 0.1,
            tx_lon=rx_lon + 0.3,
            tx_alt_ft=500.0,
            fc_hz=500e6,
            beam_azimuth_deg=45.0,
            beam_width_deg=120.0,
            max_range_km=60.0,
            doppler_min=-1000.0,
            doppler_max=1000.0,
        )
    )
    world.micro_doppler_enabled = True
    world.cpi_s = 0.5
    world.airframe_mix = {"turboprop": 1.0}
    world.min_aircraft = world.max_aircraft = 0
    ac = SimulatedAircraft(
        object_id="tp-1",
        lat=rx_lat + 0.04,
        lon=rx_lon + 0.06,
        alt_km=2.0,
        vel_east=0.15,
        vel_north=0.0,
        vel_up=0.0,
        heading_deg=90.0,
        speed_km_s=0.15,
        airframe="turboprop",
        rotor_phases=[0.2, 1.3],
        rpm_scale=1.02,
    )
    world.aircraft = [ac]
    acc = CombAccumulator()
    for i in range(40):
        world._time += 2.0
        ac.lon += 0.15 * 2.0 / (111.2 * math.cos(math.radians(rx_lat)))
        frame = world.generate_detections_for_node("n", i * 2000)
        if frame["delay"]:
            acc.add(frame_comb(frame["delay"], frame["doppler"], frame["snr"], frame["delay"][0], frame["doppler"][0]))
    summary = acc.summary()
    assert summary["frac_frames_with_comb"] > 0.5
    assert summary["median_spacing_hz"] == pytest.approx(102.0, rel=0.06)
