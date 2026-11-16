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
    BLADES,
    DRONE,
    HELICOPTER,
    JET,
    OTHER,
    PISTON,
    TURBOPROP,
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
        "C0FFEE;C-FDHD;DH8D;00;DHC-8-402 Dash 8;;Porter;",
        "A22222;N22222;R44;00;ROBINSON R44;;;",
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
    "DH8D": ["DHC-8-400 Dash 8Q", "L2T", "M"],
    "R44": ["ROBINSON R44", "H1P", "L"],
}


@pytest.fixture
def registry():
    count = aircraft_db.build(AIRCRAFT_CSV, TYPES, aircraft_db.db_path())
    assert count == 11
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
        "icao_class,expected",
        [
            ("L2J", JET),
            ("L1J", JET),
            ("S2J", JET),
            ("L2T", TURBOPROP),
            ("L1T", TURBOPROP),
            ("A2T", TURBOPROP),
            ("L1P", PISTON),
            ("A1P", PISTON),
            ("S1P", PISTON),
            ("S1E", PISTON),
            ("L1E", PISTON),
            ("L0-", OTHER),
            ("H1P", HELICOPTER),
            ("H2T", HELICOPTER),
            ("G1P", OTHER),
            ("T2T", OTHER),
            ("R1R", OTHER),
            ("D0-", DRONE),
            ("B0-", OTHER),
            ("V0-", OTHER),
            ("", None),
            (None, None),
        ],
    )
    def test_icao_class(self, icao_class, expected):
        assert class_from_icao(icao_class) == expected

    @pytest.mark.parametrize(
        "hex_code,cls",
        [
            ("4ca1f2", JET),
            ("a12345", PISTON),
            ("ae1234", HELICOPTER),
            ("400abc", TURBOPROP),
            ("a99999", TURBOPROP),
            ("c0ffee", TURBOPROP),
            ("a22222", HELICOPTER),
            ("3d0001", OTHER),
            ("adf7f0", DRONE),
        ],
    )
    def test_registry_label(self, registry, hex_code, cls):
        assert identify(hex_code)["class"] == cls

    def test_blades_come_from_the_type_table(self, registry):
        assert identify("c0ffee")["blades"] == 6
        assert identify("a12345")["blades"] == 2
        assert identify("a22222")["blades"] == 2
        assert "blades" not in identify("4ca1f2")
        assert BLADES["DH8D"] == 6

    def test_label_names_the_airframe(self, registry):
        assert identify("ae1234") == {
            "source": "aircraft_db",
            "class": HELICOPTER,
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
        assert label["class"] == PISTON
        assert label["privacy"] is True
        assert "registration" not in label and "operator" not in label

    def test_a_row_naming_nothing_is_no_label(self, registry):
        assert identify("000001")["military"] is True
        assert identify("000001")["class"] is None

    def test_emitter_category_labels_what_the_registry_does_not_know(self, registry):
        assert identify("abcdef") is None
        assert identify("abcdef", "A7") == {
            "source": "adsb_category",
            "class": HELICOPTER,
            "category": "A7",
            "military": False,
        }
        assert identify("abcdef", "B6")["class"] == DRONE
        assert identify("abcdef", "B4")["class"] == PISTON
        for cat in ("B1", "B2", "B3", "B7", "C1", "C2", "C3"):
            assert identify("abcdef", cat)["class"] == OTHER
        assert identify("abcdef", " a7 ")["class"] == HELICOPTER

    def test_weight_classes_say_nothing_about_the_engine(self, registry):
        for cat in ("A0", "A1", "A2", "A3", "A4", "A5", "A6"):
            assert identify("abcdef", cat) is None

    def test_registry_wins_over_the_category(self, registry):
        assert identify("4ca1f2", "A7")["class"] == JET

    def test_legacy_target_class(self):
        assert target_class_for(DRONE) == "drone"
        assert target_class_for(JET) == "aircraft"
        assert target_class_for(HELICOPTER) == "aircraft"
        assert target_class_for(OTHER) is None
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
        assert m.cls == JET
        assert m.confidence > 0.9
        assert m.features.speed_ms == pytest.approx(230, rel=0.02)

    def test_a_dash_8_cruise_is_ambiguous_between_jet_and_turboprop(self):
        m = motion_classifier.classify(straight_trail(150, 7_000))
        assert m.scores[JET] > 0
        assert m.scores[TURBOPROP] > 0
        assert m.cls in (JET, TURBOPROP, motion_classifier.UNKNOWN)

    def test_light_aircraft_in_the_circuit(self):
        m = motion_classifier.classify(straight_trail(50, 800))
        assert m.cls == PISTON

    def test_hovering_then_moving_is_a_helicopter(self):
        hover = [(1000.0 + i * 5, 51.5, -0.1, 300.0) for i in range(7)]
        move = [(p[0] + 35, p[1], p[2], p[3]) for p in straight_trail(40, 300, n=7)]
        m = motion_classifier.classify(hover + move)
        assert m.features.hover_fraction > 0.2
        assert m.cls == HELICOPTER

    def test_slow_low_and_turning_hard_is_a_drone(self):
        m = motion_classifier.classify(orbit_trail(12, 60, 120))
        assert m.features.turn_dps > 10
        assert m.cls == DRONE

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
        assert m.cls in (JET, TURBOPROP)
        assert m.scores[DRONE] == 0.0
        assert m.scores[PISTON] == 0.0

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
        assert m.cls == JET

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
        assert entry["motion_class"]["class"] == JET
        card = target_classification.scorecard(now=2000.0)
        assert (card["targets"], card["decided"], card["accuracy"], card["coverage"]) == (1, 1, 1.0, 1.0)
        assert card["confusion"][JET][JET] == 1

    def test_a_drone_label_marks_the_target_a_drone(self, registry):
        entry = _feed_entry("adf7f0", orbit_trail(12, 60, 120), target_class="aircraft")
        target_classification.classify_entry(entry, now=2000.0)
        assert entry["target_class"] == "drone"

    def test_a_dark_target_gets_the_baseline_only(self, registry):
        entry = _feed_entry("mnbbbbbbbbbb", straight_trail(50, 800), target_class="aircraft")
        target_classification.classify_entry(entry, now=2000.0)
        assert "identity" not in entry
        assert entry["target_class"] == "aircraft"
        assert entry["motion_class"]["class"] == PISTON
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
        assert entry["identity"]["class"] == HELICOPTER

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
        assert entry["motion_class"]["class"] in (JET, TURBOPROP)

    def test_a_claimed_adsb_entry_reads_its_own_speed_not_the_hex_trail(self):
        entry = _feed_entry("abc123", straight_trail(5, 100))
        entry.update(position_source="adsb_single_node", gs=450.0, alt_baro=35_000)
        target_classification.classify_entry(entry, now=2000.0)
        features = entry["motion_class"]["features"]
        assert features["n_chords"] == 0
        assert features["speed_ms"] == pytest.approx(231.5, abs=0.1)
        assert entry["motion_class"]["class"] == JET

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
            target_classification._record("4ca1f2", JET, PISTON, 0.5, t)
        target_classification._record("a12345", PISTON, PISTON, 0.8, 1003.0)
        target_classification._record("ae1234", HELICOPTER, motion_classifier.UNKNOWN, 0.2, 1004.0)
        card = target_classification.scorecard(now=1005.0)
        assert (card["targets"], card["decided"]) == (3, 2)
        assert card["accuracy"] == 0.5
        assert card["coverage"] == pytest.approx(0.667, abs=1e-3)
        assert card["per_class"][PISTON] == {"targets": 1, "recall": 1.0, "precision": 0.5}
        assert card["per_class"][HELICOPTER]["recall"] is None
        later = target_classification.scorecard(now=1003.5 + target_classification.SCORE_WINDOW_S)
        assert later["targets"] == 1

    def test_scorecard_splits_by_altitude_band(self):
        record = target_classification._record
        record("a00001", JET, JET, 0.9, 1000.0, 11_000.0)
        record("a00002", TURBOPROP, JET, 0.6, 1000.0, 7_000.0)
        record("a00003", TURBOPROP, TURBOPROP, 0.6, 1000.0, 2_000.0)
        record("a00004", PISTON, PISTON, 0.8, 1000.0, 300.0)
        record("a00005", DRONE, motion_classifier.UNKNOWN, 0.2, 1000.0, 3_000.0)
        record("a00006", JET, JET, 0.9, 1000.0, None)
        bands = target_classification.scorecard(now=1001.0)["by_altitude"]
        assert list(bands) == ["<1km", "1-3km", "3-6km", ">6km", "unknown"]
        assert bands["<1km"]["targets"] == 1
        assert bands["1-3km"]["confusion"][TURBOPROP][TURBOPROP] == 1
        assert bands["3-6km"]["decided"] == 0 and bands["3-6km"]["accuracy"] is None
        assert bands[">6km"]["confusion"][TURBOPROP][JET] == 1
        assert (bands[">6km"]["targets"], bands[">6km"]["accuracy"]) == (2, 0.5)
        assert bands["unknown"]["accuracy"] == 1.0
        assert sum(b["targets"] for b in bands.values()) == 6

    def test_a_target_labelled_other_is_not_scored(self, registry):
        entry = _feed_entry("3d0001", straight_trail(25, 1500))
        target_classification.classify_entry(entry, now=2000.0)
        assert entry["identity"]["class"] == OTHER
        assert "target_class" not in entry
        assert target_classification.scorecard(now=2000.0)["targets"] == 0

    def test_empty_scorecard(self):
        card = target_classification.scorecard()
        assert (card["targets"], card["accuracy"], card["coverage"]) == (0, None, None)


class TestSimTruth:
    def _dark_sim_entry(self, gt_hex, airframe, speed_ms=150, alt_m=7_000, **meta):
        state.ground_truth_meta[gt_hex] = {"airframe": airframe, **meta}
        entry = _feed_entry("mn" + gt_hex, straight_trail(speed_ms, alt_m), ground_truth_hex=gt_hex)
        entry["target_class"] = "aircraft"
        return entry

    def test_a_dark_sim_target_is_scored_against_the_simulator_airframe(self):
        entry = self._dark_sim_entry("a1b2c3", "turboprop")
        target_classification.classify_entry(entry, now=2000.0)
        assert "identity" not in entry
        card = target_classification.scorecard(now=2000.0)
        assert card["targets"] == 1
        assert card["confusion"][TURBOPROP][entry["motion_class"]["class"]] == 1
        assert card["by_label_source"]["sim_truth"]["targets"] == 1
        assert card["by_label_source"]["hex"]["targets"] == 0

    def test_hex_and_sim_labels_are_scored_apart_and_together(self, registry):
        hexed = _feed_entry("mnaaaaaaaaaa", straight_trail(230, 11_000), adsb_hex="4ca1f2")
        simmed = self._dark_sim_entry("a1b2c3", "jet", speed_ms=230, alt_m=11_000)
        target_classification.classify_feed([hexed, simmed], now=2000.0)
        card = target_classification.scorecard(now=2000.0)
        assert card["targets"] == 2
        by_source = card["by_label_source"]
        assert (by_source["hex"]["targets"], by_source["sim_truth"]["targets"]) == (1, 1)
        assert set(by_source["sim_truth"]) == set(card) - {"window_s", "by_label_source"}
        assert card["confusion"][JET][JET] == 2

    def test_sim_truth_never_reaches_the_identity(self):
        entry = self._dark_sim_entry("a1b2c3", "helicopter")
        target_classification.classify_entry(entry, now=2000.0)
        assert "identity" not in entry
        assert entry["target_class"] == "aircraft"

    @pytest.mark.parametrize("airframe", [None, "", "glider"])
    def test_an_unusable_airframe_records_nothing(self, airframe):
        entry = self._dark_sim_entry("a1b2c3", airframe)
        target_classification.classify_entry(entry, now=2000.0)
        assert target_classification.scorecard(now=2000.0)["targets"] == 0

    def test_an_entry_with_no_ground_truth_hex_records_nothing(self):
        entry = _feed_entry("mnbbbbbbbbbb", straight_trail(150, 7_000))
        target_classification.classify_entry(entry, now=2000.0)
        assert target_classification.scorecard(now=2000.0)["targets"] == 0

    def test_a_hex_label_wins_over_sim_truth(self, registry):
        state.ground_truth_meta["4ca1f2"] = {"airframe": "piston", "source": "live"}
        entry = _feed_entry("4ca1f2", straight_trail(230, 11_000), ground_truth_hex="4ca1f2")
        target_classification.classify_entry(entry, now=2000.0)
        by_source = target_classification.scorecard(now=2000.0)["by_label_source"]
        assert (by_source["hex"]["targets"], by_source["sim_truth"]["targets"]) == (1, 0)

    def test_a_live_mirrored_aircraft_is_looked_up_despite_the_sim_world(self, registry):
        state.adsb_aircraft["4ca1f2"] = {"hex": "4ca1f2", "world": "sim"}
        state.ground_truth_meta["4ca1f2"] = {"airframe": "jet", "source": "live"}
        entry = _feed_entry("4ca1f2", straight_trail(230, 11_000))
        target_classification.classify_entry(entry, now=2000.0)
        assert entry["identity"]["registration"] == "EI-DCL"

    def test_a_sim_spawned_hex_is_still_not_looked_up(self, registry):
        state.adsb_aircraft["4ca1f2"] = {"hex": "4ca1f2", "world": "sim"}
        state.ground_truth_meta["4ca1f2"] = {"airframe": "jet", "source": "sim"}
        entry = _feed_entry("4ca1f2", straight_trail(230, 11_000))
        target_classification.classify_entry(entry, now=2000.0)
        assert "identity" not in entry

    def test_the_empty_card_has_both_sources(self):
        by_source = target_classification.scorecard()["by_label_source"]
        assert list(by_source) == ["hex", "sim_truth"]
        assert by_source["sim_truth"]["accuracy"] is None
