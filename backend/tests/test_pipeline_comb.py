"""Comb features on live frames: read off the detection the tracker assigned, per track."""

import pytest

from core import state
from pipeline import passive_radar
from pipeline.passive_radar import DEFAULT_NODE_CONFIG, PassiveRadarPipeline

NODE = {**DEFAULT_NODE_CONFIG, "node_id": "comb-test-node", "cpi_s": 0.5}
SNR_BODY = 25.0
# Below the tracker's 4 dB detection floor, so the tracker never follows a line.
# A line above the floor can start a track of its own; see TestTrackedLines.
SNR_LINE = 3.0
FRAMES = 14


def frame(i, bodies, lines=(), t0=1_700_000_000_000):
    """One frame: ``bodies`` are (delay_us, doppler_hz) that drift with i, ``lines`` are extra (delay, doppler, snr)."""
    delay, doppler, snr = [], [], []
    for d, f in bodies:
        delay.append(d + 0.02 * i)
        doppler.append(f + 0.5 * i)
        snr.append(SNR_BODY)
    for d, f, s in lines:
        delay.append(d + 0.02 * i)
        doppler.append(f + 0.5 * i)
        snr.append(s)
    return {"timestamp": t0 + 1000 * i, "delay": delay, "doppler": doppler, "snr": snr}


def with_lines(delay, doppler, offsets, snr=SNR_LINE):
    return [(delay, doppler + o, snr) for o in offsets]


def run(bodies, lines=(), frames=FRAMES, pipe=None):
    pipe = pipe or PassiveRadarPipeline(NODE)
    for i in range(frames):
        pipe.process_frame(frame(i, bodies, lines))
    return pipe


def confirmed(pipe):
    return [t for t in pipe.tracker.tracks if t.id]


def comb_of(pipe, track):
    return pipe._combs[track.id].summary()


class TestAccumulation:
    def test_a_track_with_a_rotor_comb_reads_its_lines(self):
        offsets = (-200.0, -100.0, 100.0, 200.0)
        pipe = run([(20.0, 100.0)], with_lines(20.0, 100.0, offsets))
        (track,) = confirmed(pipe)
        s = comb_of(pipe, track)
        assert s["frames"] >= 5
        assert s["median_n_lines"] == 4
        assert s["median_spacing_hz"] == pytest.approx(100.0, abs=3.0)
        assert s["frac_frames_with_comb"] == 1.0
        assert s["median_body_snr_db"] == SNR_BODY

    def test_a_jet_with_no_lines_has_frames_and_no_comb(self):
        pipe = run([(20.0, 100.0)])
        (track,) = confirmed(pipe)
        s = comb_of(pipe, track)
        assert s["frames"] >= 5
        assert s["frac_frames_with_comb"] == 0.0
        assert s["median_n_lines"] is None

    def test_another_aircrafts_detection_is_not_a_line(self):
        offsets = (-100.0, 100.0)
        # A second aircraft at almost the same delay, 600 Hz away: assigned to its own track.
        pipe = run([(20.0, 100.0), (20.3, 700.0)], with_lines(20.0, 100.0, offsets))
        a = next(t for t in confirmed(pipe) if abs(t.history["measurements"][-1]["doppler"] - 107) < 20)
        assert comb_of(pipe, a)["median_n_lines"] == 2
        assert comb_of(pipe, a)["max_n_lines"] == 2

    def test_without_the_exclusion_that_aircraft_would_read_as_a_line(self, monkeypatch):
        seen = []
        real = passive_radar.frame_comb

        def spy(delays, dopplers, snrs, *a, **k):
            seen.append(len(delays))
            return real(delays, dopplers, snrs, *a, **k)

        monkeypatch.setattr(passive_radar, "frame_comb", spy)
        run([(20.0, 100.0), (20.3, 700.0)], with_lines(20.0, 100.0, (-100.0, 100.0)))
        # Body plus two lines, never the neighbour's body as a fourth candidate.
        assert max(seen) == 3

    def test_a_detection_that_only_started_a_tentative_track_is_still_a_candidate(self):
        pipe = PassiveRadarPipeline(NODE)
        run([(20.0, 100.0)], with_lines(20.0, 100.0, (-100.0, 100.0)), frames=FRAMES, pipe=pipe)
        (track,) = confirmed(pipe)
        assert comb_of(pipe, track)["median_n_lines"] == 2

    def test_history_is_bounded(self, monkeypatch):
        monkeypatch.setattr(passive_radar, "COMB_HISTORY_FRAMES", 4)
        pipe = run([(20.0, 100.0)], with_lines(20.0, 100.0, (-100.0, 100.0)), frames=20)
        (track,) = confirmed(pipe)
        assert comb_of(pipe, track)["frames"] == 4


class TestTrackedLines:
    def test_a_line_the_tracker_follows_is_withheld_from_the_comb(self):
        # Lines above the tracker's floor get tracks of their own, and a detection
        # assigned to another track is not read as a line: the comb of a body whose
        # lines the tracker follows is the lines it does not.  This pins the rule as
        # specified; the bench's spurious-track count says how often it bites.
        pipe = run([(20.0, 100.0)], with_lines(20.0, 100.0, (-100.0, 100.0), snr=12.0))
        body = max(confirmed(pipe), key=lambda t: t.history["measurements"][-1]["snr"])
        assert comb_of(pipe, body)["frac_frames_with_comb"] < 0.5


class TestCost:
    def test_isolated_tracks_never_call_frame_comb(self, monkeypatch):
        calls = []
        monkeypatch.setattr(passive_radar, "frame_comb", lambda *a, **k: calls.append(1))
        run([(10.0 + 7 * k, 100.0 + 90 * k) for k in range(6)])
        assert calls == []

    def test_a_frame_with_no_detections_is_skipped(self):
        pipe = PassiveRadarPipeline(NODE)
        pipe.process_frame({"timestamp": 1000, "delay": [], "doppler": [], "snr": []})
        assert pipe._combs == {}


class TestExposure:
    def test_the_geolocated_track_carries_the_summary(self, monkeypatch):
        pipe = PassiveRadarPipeline(NODE)
        pipe._GEO_INTERVAL_S = 0.0
        run([(20.0, 100.0)], with_lines(20.0, 100.0, (-100.0, 100.0)), frames=20, pipe=pipe)
        geo = [g for g in pipe.geolocated_tracks.values() if g.comb_source is not None]
        assert geo
        assert geo[0].comb_summary()["median_n_lines"] == 2

    def test_a_track_without_a_comb_source_reports_none(self):
        pipe = PassiveRadarPipeline(NODE)
        geo = passive_radar.GeolocatedTrack("t", 0, 0, 0, 0, 0, 0, 0, 0, 3, 0)
        assert geo.comb_summary() is None
        assert pipe._combs == {}


class TestPruning:
    def test_dead_tracks_leave_the_comb_dict(self):
        pipe = run([(20.0, 100.0)], with_lines(20.0, 100.0, (-100.0, 100.0)))
        assert pipe._combs
        pipe.tracker.tracks = []
        pipe.event_writer.events.clear()
        pipe._prune_stale_tracks()
        assert pipe._combs == {}


class TestNeverRaises:
    def test_an_error_is_counted_and_the_frame_goes_on(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("comb exploded")

        monkeypatch.setattr(passive_radar, "frame_comb", boom)
        pipe = run([(20.0, 100.0)], with_lines(20.0, 100.0, (-100.0, 100.0)))
        assert state.comb_feature_errors > 0
        assert confirmed(pipe)

    def test_ragged_arrays_do_not_raise(self):
        pipe = PassiveRadarPipeline(NODE)
        pipe.process_frame({"timestamp": 1000, "delay": [1.0, 1.1, 1.2], "doppler": [1.0], "snr": [20.0, 20.0]})
        pipe.process_frame({"timestamp": 2000, "delay": [], "doppler": [], "snr": []})
