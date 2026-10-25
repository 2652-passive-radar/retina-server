"""Target classification: the hex-lookup label and the motion-only baseline.

The registry is built from a few hand-written tar1090-db rows rather than the
real export, so every expectation here is visible in the file.
"""

import math
import os

import pytest

from core import state
from services import aircraft_db, motion_classifier, target_classification
from services.target_identity import (
    BALLOON,
    DRONE,
    FAST_FIXED_WING,
    GLIDER,
    LIGHT_FIXED_WING,
    ROTORCRAFT,
    SURFACE,
    class_from_icao,
    identify,
    target_class_for,
)

AIRCRAFT_CSV = "\n".join(
    [
        "4CA1F2;EI-DCL;B738;00;BOEING 737-800;;Ryanair;",
        "A12345;N172SP;C172;00;CESSNA 172 Skyhawk;;;",
        "AE1234;;H60;10;SIKORSKY UH-60 Black Hawk;;US Army;",
        "400ABC;G-ABCD;AT76;00;;;Loganair;",
        "A99999;N208XX;C208;00;;;;",
        "A55555;N55555;C172;0001;;;Private Owner LLC;",
        "3D0001;D-1234;GLID;00;Glider;;;",
        "ADF7F0;;DRON;10;;;;",
        "000001;;;10;;;Miscode - VARIOUS;",
        "short;;;;",
    ]
)
TYPES = {
    "B738": ["BOEING 737-800", "L2J", "M"],
    "C172": ["CESSNA 172 Skyhawk", "L1P", "L"],
    "H60": ["SIKORSKY UH-60 Black Hawk", "H2T", "M"],
    "AT76": ["ATR 72-600", "L2T", "M"],
    "C208": ["CESSNA 208 Caravan", "L1T", "L"],
    "GLID": ["Glider", "L0-", "-"],
    "DRON": ["", "D0-", "-"],
}


@pytest.fixture
def registry():
    count = aircraft_db.build(AIRCRAFT_CSV, TYPES, aircraft_db.db_path())
    assert count == 9
    return aircraft_db.db_path()


class TestAircraftDb:
    def test_no_registry_answers_none(self):
        assert not aircraft_db.available()
        assert aircraft_db.lookup("4ca1f2") is None
        assert aircraft_db.age_s() is None

    def test_lookup_joins_the_type_table(self, registry):
        rec = aircraft_db.lookup("4CA1F2")
        assert rec == {
            "registration": "EI-DCL",
            "type_code": "B738",
            "flags": "00",
            "description": "BOEING 737-800",
            "operator": "Ryanair",
            "type_name": "BOEING 737-800",
            "icao_class": "L2J",
            "wtc": "M",
        }

    def test_non_icao_addresses_are_never_looked_up(self, registry):
        for h in ("~4ca1f2", "pr294e", "mn0123456789", "", None, "4ca1f"):
            assert aircraft_db.lookup(h) is None

    def test_a_registry_built_after_boot_is_picked_up(self):
        assert aircraft_db.lookup("4ca1f2") is None
        aircraft_db.build(AIRCRAFT_CSV, TYPES, aircraft_db.db_path())
        assert aircraft_db.lookup("4ca1f2")["registration"] == "EI-DCL"

    def test_a_rebuilt_registry_replaces_the_cached_answers(self, registry):
        assert aircraft_db.lookup("4ca1f2")["registration"] == "EI-DCL"
        aircraft_db.build("4CA1F2;EI-NEW;B738;00;;;;", TYPES, registry)
        st = registry.stat()
        os.utime(registry, (st.st_atime, st.st_mtime + 5))
        assert aircraft_db.lookup("4ca1f2")["registration"] == "EI-NEW"

    def test_a_failed_build_leaves_the_old_registry(self, registry):
        with pytest.raises(AttributeError):
            aircraft_db.build(AIRCRAFT_CSV, {"B738": None, 3: ["x"]}, registry)
        assert aircraft_db.lookup("4ca1f2")["registration"] == "EI-DCL"
        assert not list(registry.parent.glob(".aircraft_db.*"))


class TestIdentity:
    @pytest.mark.parametrize(
        "icao_class,wtc,expected",
        [
            ("L2J", "M", FAST_FIXED_WING),
            ("L1J", "L", FAST_FIXED_WING),
            ("L2T", "M", FAST_FIXED_WING),
            ("L1T", "L", LIGHT_FIXED_WING),
            ("L1P", "L", LIGHT_FIXED_WING),
            ("A1P", "L", LIGHT_FIXED_WING),
            ("H1P", "L", ROTORCRAFT),
            ("G1P", "L", ROTORCRAFT),
            ("D0-", "-", DRONE),
            ("B0-", "-", BALLOON),
            ("V0-", "-", SURFACE),
            ("L0-", "-", None),
            ("", None, None),
            (None, None, None),
        ],
    )
    def test_icao_class(self, icao_class, wtc, expected):
        assert class_from_icao(icao_class, wtc) == expected

    @pytest.mark.parametrize(
        "hex_code,cls",
        [
            ("4ca1f2", FAST_FIXED_WING),
            ("a12345", LIGHT_FIXED_WING),
            ("ae1234", ROTORCRAFT),
            ("400abc", FAST_FIXED_WING),
            ("a99999", LIGHT_FIXED_WING),
            ("3d0001", GLIDER),
            ("adf7f0", DRONE),
        ],
    )
    def test_registry_label(self, registry, hex_code, cls):
        assert identify(hex_code)["class"] == cls

    def test_label_names_the_airframe(self, registry):
        assert identify("ae1234") == {
            "source": "aircraft_db",
            "class": ROTORCRAFT,
            "military": True,
            "type_code": "H60",
            "type_name": "SIKORSKY UH-60 Black Hawk",
            "description": "SIKORSKY UH-60 Black Hawk",
            "icao_class": "H2T",
            "wtc": "M",
            "operator": "US Army",
        }

    def test_a_privacy_listed_aircraft_is_not_named(self, registry):
        label = identify("a55555")
        assert label["class"] == LIGHT_FIXED_WING
        assert label["privacy"] is True
        assert "registration" not in label and "operator" not in label

    def test_a_row_naming_nothing_is_no_label(self, registry):
        assert identify("000001")["military"] is True
        assert identify("000001")["class"] is None

    def test_emitter_category_labels_what_the_registry_does_not_know(self, registry):
        assert identify("abcdef") is None
        assert identify("abcdef", "A7") == {
            "source": "adsb_category",
            "class": ROTORCRAFT,
            "category": "A7",
            "military": False,
        }
        assert identify("abcdef", "B6")["class"] == DRONE
        assert identify("abcdef", "A0") is None
        assert identify("abcdef", " a7 ")["class"] == ROTORCRAFT

    def test_registry_wins_over_the_category(self, registry):
        assert identify("4ca1f2", "A1")["class"] == FAST_FIXED_WING

    def test_legacy_target_class(self):
        assert target_class_for(DRONE) == "drone"
        assert target_class_for(ROTORCRAFT) == "aircraft"
        assert target_class_for(SURFACE) is None
        assert target_class_for(None) is None


def straight_trail(speed_ms, alt_m, n=13, dt=5.0, heading_deg=90.0, lat=51.5, lon=-0.1, climb_ms=0.0):
    out = []
    for i in range(n):
        d = speed_ms * dt * i
        dn = d * math.cos(math.radians(heading_deg))
        de = d * math.sin(math.radians(heading_deg))
        out.append(
            (
                1000.0 + i * dt,
                lat + dn / 111_320.0,
                lon + de / (111_320.0 * math.cos(math.radians(lat))),
                None if alt_m is None else alt_m + climb_ms * dt * i,
            )
        )
    return out


def orbit_trail(speed_ms, radius_m, alt_m, n=25, dt=5.0, lat=51.5, lon=-0.1):
    out = []
    omega = speed_ms / radius_m
    for i in range(n):
        a = omega * dt * i
        out.append(
            (
                1000.0 + i * dt,
                lat + radius_m * math.cos(a) / 111_320.0,
                lon + radius_m * math.sin(a) / (111_320.0 * math.cos(math.radians(lat))),
                alt_m,
            )
        )
    return out


class TestMotionBaseline:
    def test_airliner_in_the_cruise(self):
        m = motion_classifier.classify(straight_trail(230, 11_000))
        assert m.cls == FAST_FIXED_WING
        assert m.confidence > 0.9
        assert m.features.speed_ms == pytest.approx(230, rel=0.02)

    def test_light_aircraft_in_the_circuit(self):
        m = motion_classifier.classify(straight_trail(50, 800))
        assert m.cls == LIGHT_FIXED_WING

    def test_hovering_then_moving_is_a_rotorcraft(self):
        hover = [(1000.0 + i * 5, 51.5, -0.1, 300.0) for i in range(7)]
        move = [(p[0] + 35, p[1], p[2], p[3]) for p in straight_trail(40, 300, n=7)]
        m = motion_classifier.classify(hover + move)
        assert m.features.hover_fraction > 0.2
        assert m.cls == ROTORCRAFT

    def test_slow_low_and_turning_hard_is_a_drone(self):
        m = motion_classifier.classify(orbit_trail(12, 60, 120))
        assert m.features.turn_dps > 10
        assert m.cls == DRONE

    def test_glider_thermalling(self):
        m = motion_classifier.classify(orbit_trail(25, 100, 1500, n=31))
        assert m.cls == GLIDER

    def test_a_balloon_drifts(self):
        m = motion_classifier.classify(straight_trail(8, 20_000, n=31))
        assert m.cls == BALLOON

    def test_noise_inside_a_chord_does_not_read_as_speed(self):
        # Alternating 300 m of jitter on a stationary target: differencing
        # consecutive points would claim 60 m/s.
        pts = [(1000.0 + i * 5, 51.5 + (0.0027 if i % 2 else 0.0), -0.1, 100.0) for i in range(25)]
        m = motion_classifier.classify(pts)
        assert m.features.speed_ms < 35

    def test_doppler_floor_rules_out_the_slow_classes(self):
        floor = motion_classifier.doppler_speed_floor_ms(-150.0, 195e6)
        assert floor == pytest.approx(299_792_458 / 195e6 * 75, rel=1e-9)
        m = motion_classifier.classify(None, speed_floor_ms=floor)
        assert m.cls == FAST_FIXED_WING
        assert m.scores[DRONE] == 0.0

    def test_a_small_doppler_decides_nothing(self):
        floor = motion_classifier.doppler_speed_floor_ms(2.0, 195e6)
        assert motion_classifier.classify(None, speed_floor_ms=floor).cls == motion_classifier.UNKNOWN

    def test_no_evidence_is_unknown(self):
        m = motion_classifier.classify([])
        assert (m.cls, m.confidence) == (motion_classifier.UNKNOWN, 0.0)
        assert motion_classifier.doppler_speed_floor_ms(None, 195e6) is None
        assert motion_classifier.doppler_speed_floor_ms(10.0, None) is None

    def test_an_impossible_target_is_unknown(self):
        m = motion_classifier.classify(straight_trail(700, 30_000))
        assert m.cls == motion_classifier.UNKNOWN
        assert sum(m.scores.values()) == 0.0

    def test_a_reported_speed_stands_in_for_a_short_trail(self):
        m = motion_classifier.classify([(1000.0, 51.5, -0.1, 10_000.0)], speed_ms=240.0)
        assert m.cls == FAST_FIXED_WING

    def test_climb_is_read_across_the_trail_not_chord_by_chord(self):
        pts = straight_trail(230, 11_000, n=25)
        noisy = [(t, la, lo, alt + (400.0 if i % 2 else -400.0)) for i, (t, la, lo, alt) in enumerate(pts)]
        assert motion_classifier.classify(noisy).features.climb_ms < 5
        climbing = motion_classifier.classify(straight_trail(120, 3_000, n=25, climb_ms=12.0))
        assert climbing.features.climb_ms == pytest.approx(12.0, rel=0.05)

    def test_negative_altitude_scores(self):
        m = motion_classifier.classify(straight_trail(50, -5))
        assert all(not math.isnan(v) for v in m.scores.values())

    def test_scores_are_a_distribution(self):
        m = motion_classifier.classify(straight_trail(60, 500))
        assert sum(m.scores.values()) == pytest.approx(1.0)
        d = m.as_dict()
        assert set(d) == {"class", "confidence", "scores", "features"}


def _feed_entry(hex_code, trail, **kw):
    for p in trail:
        state.track_histories.setdefault(hex_code, __import__("collections").deque()).append(
            [p[1], p[2], p[3] / 0.3048 if p[3] is not None else 0, p[0]]
        )
    return {"hex": hex_code, "position_source": "multinode_solve", "multinode": True, **kw}


class TestFeed:
    def test_identity_overrides_target_class_and_scores_the_baseline(self, registry):
        entry = _feed_entry("mnaaaaaaaaaa", straight_trail(230, 11_000), adsb_hex="4ca1f2", target_class=None)
        target_classification.classify_feed([entry], now=2000.0)
        assert entry["identity"]["registration"] == "EI-DCL"
        assert entry["target_class"] == "aircraft"
        assert entry["motion_class"]["class"] == FAST_FIXED_WING
        card = target_classification.scorecard(now=2000.0)
        assert (card["targets"], card["decided"], card["accuracy"], card["coverage"]) == (1, 1, 1.0, 1.0)
        assert card["confusion"][FAST_FIXED_WING][FAST_FIXED_WING] == 1

    def test_a_drone_label_marks_the_target_a_drone(self, registry):
        entry = _feed_entry("adf7f0", orbit_trail(12, 60, 120), target_class="aircraft")
        target_classification.classify_entry(entry, now=2000.0)
        assert entry["target_class"] == "drone"

    def test_a_dark_target_gets_the_baseline_only(self, registry):
        entry = _feed_entry("mnbbbbbbbbbb", straight_trail(50, 800), target_class="aircraft")
        target_classification.classify_entry(entry, now=2000.0)
        assert "identity" not in entry
        assert entry["target_class"] == "aircraft"
        assert entry["motion_class"]["class"] == LIGHT_FIXED_WING
        assert target_classification.scorecard(now=2000.0)["targets"] == 0

    def test_simulated_hexes_are_not_looked_up(self, registry):
        state.adsb_aircraft["4ca1f2"] = {"hex": "4ca1f2", "world": "sim"}
        entry = _feed_entry("4ca1f2", straight_trail(230, 11_000))
        target_classification.classify_entry(entry, now=2000.0)
        assert "identity" not in entry

    def test_the_emitter_category_labels_an_unregistered_hex(self, registry):
        state.adsb_aircraft["abcdef"] = {"hex": "abcdef", "world": "real", "category": "A7"}
        entry = _feed_entry("abcdef", [])
        target_classification.classify_entry(entry, now=2000.0)
        assert entry["identity"]["class"] == ROTORCRAFT

    def test_single_node_entries_use_the_doppler_floor(self):
        class _Pipeline:
            config = {"FC": 195e6}

        state.node_pipelines["n1"] = _Pipeline()
        entry = {
            "hex": "pr0001",
            "position_source": "single_node_ellipse_arc",
            "node_id": "n1",
            "doppler_hz": -150.0,
        }
        target_classification.classify_entry(entry, now=2000.0)
        assert entry["motion_class"]["features"]["speed_floor_ms"] == pytest.approx(115.3, abs=0.1)
        assert entry["motion_class"]["class"] == FAST_FIXED_WING

    def test_a_claimed_adsb_entry_reads_its_own_speed_not_the_hex_trail(self):
        entry = _feed_entry("abc123", straight_trail(5, 100))
        entry.update(position_source="adsb_single_node", gs=450.0, alt_baro=35_000)
        target_classification.classify_entry(entry, now=2000.0)
        features = entry["motion_class"]["features"]
        assert features["n_chords"] == 0
        assert features["speed_ms"] == pytest.approx(231.5, abs=0.1)
        assert entry["motion_class"]["class"] == FAST_FIXED_WING

    def test_an_arc_trail_is_not_read_as_motion(self):
        entry = _feed_entry("pr0002", straight_trail(230, 11_000))
        entry["position_source"] = "single_node_ellipse_arc"
        target_classification.classify_entry(entry, now=2000.0)
        assert entry["motion_class"]["class"] == motion_classifier.UNKNOWN

    def test_one_failing_entry_does_not_cost_the_feed(self, registry, monkeypatch):
        good = _feed_entry("mncccccccccc", straight_trail(230, 11_000))
        bad = {"hex": "mndddddddddd", "position_source": "multinode_solve"}

        real = target_classification.motion_for

        def _motion_for(entry):
            if entry is bad:
                raise RuntimeError("boom")
            return real(entry)

        monkeypatch.setattr(target_classification, "motion_for", _motion_for)
        target_classification.classify_feed([bad, good])
        assert "motion_class" in good
        assert "motion_class" not in bad
        assert state.target_classification_errors == 1

    def test_scorecard_is_one_vote_per_aircraft_inside_the_window(self, registry):
        for t in (1000.0, 1001.0, 1002.0):
            target_classification._record("4ca1f2", FAST_FIXED_WING, LIGHT_FIXED_WING, 0.5, t)
        target_classification._record("a12345", LIGHT_FIXED_WING, LIGHT_FIXED_WING, 0.8, 1003.0)
        target_classification._record("ae1234", ROTORCRAFT, motion_classifier.UNKNOWN, 0.2, 1004.0)
        card = target_classification.scorecard(now=1005.0)
        assert (card["targets"], card["decided"]) == (3, 2)
        assert card["accuracy"] == 0.5
        assert card["coverage"] == pytest.approx(0.667, abs=1e-3)
        assert card["per_class"][LIGHT_FIXED_WING] == {"targets": 1, "recall": 1.0, "precision": 0.5}
        assert card["per_class"][ROTORCRAFT]["recall"] is None
        later = target_classification.scorecard(now=1003.5 + target_classification.SCORE_WINDOW_S)
        assert later["targets"] == 1

    def test_empty_scorecard(self):
        card = target_classification.scorecard()
        assert (card["targets"], card["accuracy"], card["coverage"]) == (0, None, None)
