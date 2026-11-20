#!/usr/bin/env python3
"""Offline classification bench and training-dataset export.

Why this exists
---------------
The motion baseline (services.motion_classifier) is scored live only against
registry labels, which are scarce for the classes that matter and say nothing
about how a jet and a turboprop differ in their Doppler.  The simulator draws
its own airframes, so it can say what every target really is.  This bench runs
the simulator's world through the same pipeline association_bench does
(per-node tracker -> track-level association -> multinode solve), then writes
one row per target per 60 s window holding

  * truth: airframe (the label), object type, whether it had a transponder;
  * the motion baseline's features and verdict, from the target's solved trail
    where the pipeline solved it, else from the Doppler speed floor alone
    (target_classification.motion_for's two paths);
  * comb features (services.comb_features) over every frame a node saw it in;
  * geometry: nodes that saw it, altitude, body SNR.

and prints the baseline's scorecard against truth and what the comb shows per
class.  Split train and test BY ``object_id``: a target's windows are not
independent.

What it is not
--------------
Comb features are read at the target's body detection, which the bench finds by
matching the frame to the target's TRUE delay and Doppler.  A receiver has no
such handle: real data will need the tracker's association of lines to a track.
The numbers say what the features carry when association is perfect, not what
a deployment gets.

Illuminators are real Vancouver DTV transmitters (--tower-set), each node
using its tower's own carrier.  The receiver area is an example point on the
Vancouver west side, not a surveyed site.  Line strength is the simulator's interim calibration (micro_doppler.AIRFRAMES), not a
measurement, and the legacy SNR model caps the body at 25 dB, which leaves a
rotor line almost never above the detection floor; --snr-model radar (the
default here) uses the bistatic radar equation with a per-airframe RCS.

Usage:  python scripts/classification_bench.py --out data/classification.parquet
"""

import argparse
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyarrow as pa  # noqa: E402
from retina_analytics.association import predict_observation  # noqa: E402
from retina_analytics.detection_association import DetectionAssociator  # noqa: E402
from retina_geolocator.multinode_solver import fit_constant_velocity, solve_multinode  # noqa: E402
from retina_simulation.micro_doppler import AIRFRAMES  # noqa: E402
from retina_simulation.orchestrator import parse_airframe_mix  # noqa: E402
from retina_simulation.world import MetroCell, NodeConfig, SimulationWorld  # noqa: E402
from retina_tracker.tracker import Tracker  # noqa: E402

from scripts.association_bench import (  # noqa: E402
    MATCH_KM,
    _beam_gate_ok,
    _frame_to_detections,
    _strip_adsb,
)
from scripts.classification_report import (  # noqa: E402
    print_card,
    print_comb_tables,
    vote_per_target,
)
from services import classification_columns as cols  # noqa: E402
from services import motion_classifier, target_classification  # noqa: E402
from services.comb_features import CombAccumulator, frame_comb  # noqa: E402
from services.frame_processor import confirmed_track_views  # noqa: E402
from services.geo import haversine_km  # noqa: E402
from services.target_identity import TARGET_CLASSES  # noqa: E402

DEFAULT_MIX = "jet:0.45,turboprop:0.15,piston:0.2,helicopter:0.1,drone:0.1"
# Public broadcast towers: (name, lat, lon, altitude_ft, fc_hz).  Altitude is
# the table's site elevation plus antenna height above ground, converted to
# feet; it assumes those two columns mean exactly that.
CHAN_DT = ("CHAN-DT", 49.345, -122.972778, 2135.0, 521e6)  # 616.9 m + 33.9 m = 650.8 m
# "yvr3" is three transmitters on three separate sites, so a multinode solve
# gets geometric spread from the towers as well as the receivers.  Every UHF
# station on Mt Seymour (CHAN included, 1.5 km from the rest) lies in one
# direction from the west side, so the other two come from across the water:
# CHEK-DT to the south and KVOS to the south-southeast, about 25 km apart.
# The low-power Fraser Valley rebroadcasters (CFEG, CHAN-DT-1, under 0.5 kW)
# would add an eastern bearing but illuminate ~30 dB less, so they are left out.
# Centre frequency of the RF channel.
CBUT_DT = ("CBUT-DT", 49.353611, -122.956389, 3453.0, 599e6)  # RF 35; 967.7 m + 84.7 m = 1052.4 m
# Mt Warburton Pike, Saturna Island.  Altitude is approximate (summit ~497 m
# plus an assumed ~30 m mast); the digital ERP was not found, only the analog
# facility's 16.4 kW.
CHEK_DT = ("CHEK-DT", 48.774444, -123.171111, 1730.0, 485e6)  # RF 16
# Mt Constitution, Orcas Island (US).  FCC CDBS: 370 kW ERP, 817.1 m AMSL.
KVOS_TV = ("KVOS-TV", 48.680556, -122.839361, 2681.0, 473e6)  # RF 14
TOWER_SETS = {"chan": (CHAN_DT,), "yvr3": (CBUT_DT, CHEK_DT, KVOS_TV)}
DEFAULT_CENTER = (49.26, -123.25)  # Vancouver west side (example receiver area)
# Nodes sit 3-15 km from the centre, cycling through these radii.
NODE_RADII_KM = (6.0, 12.0, 4.0, 15.0, 9.0, 3.0)
WINDOW_S = cols.WINDOW_S
# A solved trail shorter than this in a window is not motion: the single-node
# path (Doppler floor alone) is used instead.
MIN_TRAIL_POINTS = 3
# Where the body detection may sit from the target's noiseless observation:
# the simulator's own noise is gauss(0, 0.1-0.2 us) and gauss(0, 2-4 Hz).
BODY_DELAY_GATE_US = 1.0
BODY_DOPPLER_GATE_HZ = 10.0
METRO_RADIUS_KM = 60.0

COMB_COLUMNS = cols.COMB_COLUMNS
# The columns that mean the same in the live archive come from services.classification_columns,
# so the report reads both files by one set of names.  object_id, airframe and the has_adsb /
# object_type / n_solves / truth_* columns are the simulator's own.
COLUMNS: dict[str, pa.DataType] = {
    "object_id": pa.string(),
    "window_start_s": pa.float64(),
    "airframe": pa.string(),
    "object_type": pa.string(),
    "has_adsb": pa.bool_(),
    "motion_source": pa.string(),
    **cols.MOTION_COLUMNS,
    **cols.BASELINE_COLUMNS,
    **cols.COMB_TYPES,
    "n_nodes": pa.int64(),
    "node_ids": pa.string(),
    "n_solves": pa.int64(),
    "median_alt_m": pa.float64(),
    "truth_median_alt_m": pa.float64(),
    "median_body_snr_db": pa.float64(),
    "n_frames": pa.int64(),
}
assert set(cols.MOTION_COLUMNS) | set(cols.GEOMETRY_COLUMNS) <= set(COLUMNS)


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


@dataclass
class Target:
    """What the bench knows of one simulated aircraft: truth, sightings, solves."""

    object_id: str
    airframe: str
    object_type: str
    has_adsb: bool
    # (t_s, node_id, body_snr_db, body_doppler_hz, truth_alt_m, CombObs)
    sightings: list = field(default_factory=list)
    # (t_s, lat, lon, alt_m)
    solves: list = field(default_factory=list)


@dataclass
class TrackAudit:
    """Whether a confirmed tracker track ever sat on a real target.

    ``matched`` is set when one of its measurements fell inside the body gates
    of a target's true observation.  ``beside_body`` is set when one fell
    inside the delay gate of a target but outside its Doppler gate: that is
    where a rotor line sits, so a track with only that is the signature of the
    tracker locking onto a comb line.
    """

    frames: int = 0
    matched: bool = False
    beside_body: bool = False


@dataclass
class Result:
    targets: dict[str, Target] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    frames: int = 0
    solves: int = 0
    wall_s: float = 0.0
    duration_s: float = 0.0
    nodes: dict = field(default_factory=dict)
    # (node_id, track id) -> TrackAudit, for every track the tracker confirmed.
    tracks: dict = field(default_factory=dict)


def _ring_nodes(n_nodes, center, towers, fc_hz_override, doppler_window, colocated=False, rx_alt_ft=100.0):
    """One node per tower in turn, on a ring around ``center`` or all at it.

    ``colocated`` puts every receiver at ``center``: one rooftop with a node per
    illuminator, which is how a single-site deployment is built.  Its beams are
    left to the simulator's broadside default, since there is no metro core to
    aim at from inside it, and its multinode geometry comes from the towers
    alone.
    """
    lat0, lon0 = center
    nodes = []
    for i in range(n_nodes):
        _, tx_lat, tx_lon, tx_alt_ft, tower_fc = towers[i % len(towers)]
        if colocated:
            lat, lon, aim = lat0, lon0, None
        else:
            ring_km = NODE_RADII_KM[i % len(NODE_RADII_KM)]
            brg = 2.0 * math.pi * i / n_nodes
            lat = lat0 + math.degrees(ring_km * math.cos(brg) / 6371.0)
            lon = lon0 + math.degrees(ring_km * math.sin(brg) / (6371.0 * math.cos(math.radians(lat0))))
            # Aimed at the metro core, so every beam overlaps over the traffic.
            aim = (math.degrees(brg) + 180.0) % 360.0
        nodes.append(
            NodeConfig(
                node_id=f"cls-{i:02d}",
                rx_lat=lat,
                rx_lon=lon,
                rx_alt_ft=rx_alt_ft,
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_alt_ft=tx_alt_ft,
                fc_hz=fc_hz_override or tower_fc,
                doppler_min=-doppler_window,
                doppler_max=doppler_window,
                beam_azimuth_deg=aim,
                max_range_km=100.0,
                max_bistatic_range_km=100.0,
            )
        )
    return nodes


def build_world(args) -> SimulationWorld:
    mix = parse_airframe_mix(args.airframe_mix)
    world = SimulationWorld(center_lat=args.center[0], center_lon=args.center[1])
    for nc in _ring_nodes(
        args.nodes,
        args.center,
        TOWER_SETS[args.tower_set],
        args.fc_hz,
        args.doppler_window,
        colocated=args.layout == "colocated",
        rx_alt_ft=args.rx_alt_ft,
    ):
        world.add_node(nc)
    world.metro_cells = [MetroCell(core_lat=args.center[0], core_lon=args.center[1], radius_km=METRO_RADIUS_KM)]
    world.frac_metro_traffic = 1.0
    world.min_aircraft = args.min_aircraft
    world.max_aircraft = args.max_aircraft
    # Drones are an object type in the world, not an entry of airframe_mix,
    # which is drawn only for non-drone aircraft.
    world.frac_drone = mix.get("drone", 0.0)
    world.frac_dark = 0.15
    rest = {k: v for k, v in mix.items() if k != "drone"}
    if len(rest) == 1:
        rest["jet" if "jet" not in rest else "turboprop"] = 0.0
    world.airframe_mix = rest or {"jet": 1.0}
    world.micro_doppler_enabled = args.micro_doppler
    world.cpi_s = args.cpi_s
    world.snr_model = args.snr_model
    return world


def _body_detection(frame, d_us, f_hz):
    """Index of the frame detection that is the target's body, or None.

    The strongest detection near the target's true delay and Doppler: a comb
    line sits at the same delay but 15 dB or more below the body.  This uses
    the truth the bench holds; a receiver has only its tracker's association.
    """
    best = None
    for i, (d, f, s) in enumerate(zip(frame["delay"], frame["doppler"], frame["snr"])):
        if abs(d - d_us) <= BODY_DELAY_GATE_US and abs(f - f_hz) <= BODY_DOPPLER_GATE_HZ:
            if best is None or s > frame["snr"][best]:
                best = i
    return best


def run(args) -> Result:
    random.seed(args.seed)
    t_wall = time.perf_counter()
    world = build_world(args)
    node_cfgs = {nid: asdict(nc) for nid, nc in world.nodes.items()}
    bin_hz = 1.0 / args.cpi_s
    assoc = DetectionAssociator(
        grid_step_km=3.0,
        cv_fit=fit_constant_velocity,
        cv_chi2_max=2.0,
        cv_min_span_s=12.0,
        cv_min_epochs=4,
        cv_exclusive=True,
    )
    for nid, cfg in node_cfgs.items():
        assoc.register_node(nid, cfg)
    # The library rate-limits on wall clock; a replay compresses minutes into
    # seconds, so cadence is driven from simulated time below instead.
    assoc._ASSOC_MIN_INTERVAL_S = 0.0
    trackers = {nid: Tracker() for nid in node_cfgs}
    res = Result(nodes=world.nodes)
    node_ids = sorted(node_cfgs)
    next_send = {nid: i * (args.frame_interval / len(node_ids)) for i, nid in enumerate(node_ids)}
    last_assoc: dict[str, float] = {}
    t = 0.0
    while t < args.duration_s:
        world.step(args.dt, mode="adsb")
        t += args.dt
        ts_ms = int(t * 1000)
        due = [nid for nid in node_ids if next_send[nid] <= t]
        if not due:
            continue
        truth = [(ac.lat, ac.lon, ac.object_id, ac.speed_km_s * 1000.0) for ac in world.aircraft]
        for nid in due:
            next_send[nid] += args.frame_interval
            frame = world.generate_detections_for_node(nid, ts_ms)
            res.frames += 1
            expected = _expected_observations(world, assoc.node_geometries[nid]) if frame["delay"] else []
            _collect_sightings(res, expected, nid, frame, t, bin_hz)
            if args.blind:
                frame = _strip_adsb(frame)
            trackers[nid].process_frame(_frame_to_detections(frame), ts_ms)
            _audit_tracks(res, trackers[nid], nid, expected)
            assoc._pending_tracks[nid] = confirmed_track_views(trackers[nid], 20, ts_ms)
            if (t - last_assoc.get(nid, -1e9)) < args.assoc_interval:
                continue
            last_assoc[nid] = t
            round_ = assoc.submit_tracks_round(nid, assoc._pending_tracks.get(nid, []), ts_ms)
            for s_in in assoc.format_track_pairs_for_solver(round_.pairs):
                if s_in.get("n_nodes", 0) < 2:
                    continue
                try:
                    out = solve_multinode(s_in, node_cfgs)
                except Exception:
                    continue
                if not out or not out.get("success") or not _beam_gate_ok(out, s_in, node_cfgs, lambda _n: None):
                    continue
                d, oid = min(
                    ((haversine_km(out["lat"], out["lon"], a, b), oid) for a, b, oid, _ in truth),
                    default=(math.inf, None),
                )
                if d <= MATCH_KM and oid in res.targets:
                    res.targets[oid].solves.append((t, out["lat"], out["lon"], out.get("alt_m")))
                    res.solves += 1
    res.duration_s = t
    res.rows = build_rows(res, node_cfgs, args.min_frames)
    res.wall_s = time.perf_counter() - t_wall
    return res


def _expected_observations(world, geo):
    """Each aircraft's noiseless (delay us, Doppler Hz) at a node, as ``(aircraft, d, f)``."""
    return [
        (
            ac,
            *predict_observation(
                geo, ac.lat, ac.lon, ac.alt_km, ac.vel_east * 1000.0, ac.vel_north * 1000.0, ac.vel_up * 1000.0
            ),
        )
        for ac in world.aircraft
    ]


def _collect_sightings(res, expected, nid, frame, t, bin_hz):
    for ac, d_us, f_hz in expected:
        i = _body_detection(frame, d_us, f_hz)
        if i is None:
            continue
        tgt = res.targets.get(ac.object_id)
        if tgt is None:
            tgt = res.targets[ac.object_id] = Target(ac.object_id, ac.airframe, ac.object_type, ac.has_adsb)
        obs = frame_comb(
            frame["delay"], frame["doppler"], frame["snr"], frame["delay"][i], frame["doppler"][i], bin_hz=bin_hz
        )
        tgt.sightings.append((t, nid, frame["snr"][i], frame["doppler"][i], ac.alt_km * 1000.0, obs))


def _audit_tracks(res, tracker, nid, expected):
    """Record, for every confirmed track updated this frame, whether it sits on a target."""
    for track in tracker.tracks:
        if track.id is None or track.history["frames"][-1] != tracker.frame_count:
            continue
        m = track.history["measurements"][-1]
        if m is None:
            continue
        audit = res.tracks.setdefault((nid, track.id), TrackAudit())
        audit.frames += 1
        for _, d_us, f_hz in expected:
            if abs(m["delay"] - d_us) > BODY_DELAY_GATE_US:
                continue
            if abs(m["doppler"] - f_hz) <= BODY_DOPPLER_GATE_HZ:
                audit.matched = True
                break
            audit.beside_body = True


def spurious_tracks(res: Result) -> dict:
    """Confirmed tracks that never matched a target, and the rate per hour of simulated time."""
    spurious = [a for a in res.tracks.values() if not a.matched]
    hours = res.duration_s / 3600.0
    return {
        "confirmed": len(res.tracks),
        "spurious": len(spurious),
        "beside_body": sum(1 for a in spurious if a.beside_body),
        "per_hour": round(len(spurious) / hours, 1) if hours > 0 else None,
    }


def _motion_for_window(tgt, t0, t1, node_cfgs):
    """The baseline's verdict on one window, by the same two paths motion_for takes."""
    trail = sorted((t, lat, lon, alt) for t, lat, lon, alt in tgt.solves if t0 <= t < t1)
    if len(trail) >= MIN_TRAIL_POINTS and trail[-1][0] - trail[0][0] >= motion_classifier.MIN_CHORD_S:
        return "multinode", motion_classifier.classify(trail)
    # The least speed a Doppler implies is a lower bound, so the window's
    # largest one is the tightest; production has only the latest frame's.
    floors = [
        motion_classifier.doppler_speed_floor_ms(f, node_cfgs[nid]["fc_hz"])
        for ts, nid, _, f, _, _ in tgt.sightings
        if t0 <= ts < t1
    ]
    floors = [x for x in floors if x is not None]
    return "single_node", motion_classifier.classify(None, speed_floor_ms=max(floors) if floors else None)


def build_rows(res, node_cfgs, min_frames) -> list[dict]:
    rows = []
    for tgt in sorted(res.targets.values(), key=lambda x: x.object_id):
        windows = sorted({math.floor(s[0] / WINDOW_S) * WINDOW_S for s in tgt.sightings})
        for w0 in windows:
            seen = [s for s in tgt.sightings if w0 <= s[0] < w0 + WINDOW_S]
            if len(seen) < min_frames:
                continue
            source, motion = _motion_for_window(tgt, w0, w0 + WINDOW_S, node_cfgs)
            acc = CombAccumulator()
            for s in seen:
                acc.add(s[5])
            comb = acc.summary()
            solved = [alt for ts, _, _, alt in tgt.solves if w0 <= ts < w0 + WINDOW_S and alt is not None]
            row = {
                "object_id": tgt.object_id,
                "window_start_s": w0,
                "airframe": tgt.airframe,
                "object_type": tgt.object_type,
                "has_adsb": bool(tgt.has_adsb),
                "motion_source": source,
                **cols.motion_row(motion),
                **cols.comb_row(comb),
                "n_nodes": len({s[1] for s in seen}),
                "node_ids": ",".join(sorted({s[1] for s in seen})),
                "n_solves": len(solved),
                "median_alt_m": _median(solved),
                "truth_median_alt_m": _median([s[4] for s in seen]),
                "median_body_snr_db": _median([s[2] for s in seen]),
                "n_frames": len(seen),
            }
            rows.append(row)
    return rows


def write_parquet(rows, path) -> None:
    cols.write_parquet(rows, path, COLUMNS)


def report(res: Result, args, command: str) -> None:
    rows = res.rows
    print("classification bench")
    print(f"command: {command}")
    print(
        f"scene: {args.nodes} nodes "
        f"{'co-located at' if args.layout == 'colocated' else '3-15 km from'} {args.center[0]},{args.center[1]}, "
        f"{args.duration_s:.0f}s, seed {args.seed}, doppler +/-{args.doppler_window:.0f} Hz, "
        f"cpi={args.cpi_s}s, micro-doppler={'on' if args.micro_doppler else 'off'}, snr-model={args.snr_model}, "
        f"{'blind' if args.blind else 'ADS-B tagged'}, mix={args.airframe_mix}"
    )
    for nid, nc in sorted(res.nodes.items()):
        print(
            f"  node {nid}: {nc.rx_lat:.4f},{nc.rx_lon:.4f}  tower {nc.tx_lat:.6f},{nc.tx_lon:.6f} "
            f"@ {nc.tx_alt_ft:.0f} ft  fc={nc.fc_hz / 1e6:.0f} MHz"
        )
    print(f"towers: --tower-set {args.tower_set}")
    print(f"ran {res.frames} frames, {res.solves} solves matched to a target in {res.wall_s:.0f}s wall")
    sp = spurious_tracks(res)
    print(
        f"spurious confirmed tracks per hour: {sp['per_hour']} "
        f"({sp['spurious']} of {sp['confirmed']} confirmed tracker tracks, all nodes, never inside a target's "
        f"{BODY_DELAY_GATE_US:.0f} us / {BODY_DOPPLER_GATE_HZ:.0f} Hz body gate; "
        f"{sp['beside_body']} of them sat at a target's delay but off its Doppler, where rotor lines are)"
    )
    print("\nrows and targets per class (truth airframe)")
    print(f"  {'class':<12}{'targets':>9}{'rows':>7}{'rows from multinode solves':>30}")
    for c in TARGET_CLASSES:
        rs = [r for r in rows if r["airframe"] == c]
        print(
            f"  {c:<12}{len({r['object_id'] for r in rs}):>9}{len(rs):>7}"
            f"{sum(1 for r in rs if r['motion_source'] == 'multinode'):>30}"
        )
    print(f"  total: {len(rows)} rows, {len({r['object_id'] for r in rows})} targets")
    if not rows:
        print("no rows: nothing was seen for min-frames frames in a window")
        return
    print_card(
        f"motion baseline vs simulator truth, per row (window) [{len(rows)} rows]",
        target_classification.build_card([(r["airframe"], r["baseline_class"], r["alt_m"]) for r in rows]),
    )
    print_card(
        "motion baseline vs simulator truth, one vote per aircraft",
        target_classification.build_card(vote_per_target(rows, group="object_id")),
    )
    print_comb_tables(rows)


def _latlon(text):
    lat, lon = text.split(",")
    return float(lat), float(lon)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration-s", type=float, default=900.0, help="simulated seconds")
    p.add_argument("--airframe-mix", default=DEFAULT_MIX, help="class:weight,... (drone is the share of drones)")
    p.add_argument("--fc-hz", type=float, default=None, help="override every node's carrier (default: its tower's)")
    p.add_argument("--doppler-window", type=float, default=1000.0, help="node Doppler window, +/- Hz")
    p.add_argument("--micro-doppler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpi-s", type=float, default=0.5)
    p.add_argument("--snr-model", choices=("legacy", "radar"), default="radar")
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument(
        "--center",
        type=_latlon,
        default=DEFAULT_CENTER,
        metavar="LAT,LON",
        help="receiver area; default: Vancouver west side (example)",
    )
    p.add_argument(
        "--tower-set",
        choices=tuple(TOWER_SETS),
        default="chan",
        help="chan: every node uses CHAN-DT. yvr3: node i uses tower i mod 3 of CBUT-DT, CHEK-DT, KVOS-TV",
    )
    p.add_argument(
        "--layout",
        choices=("ring", "colocated"),
        default="ring",
        help="ring: nodes 3-15 km around --center. colocated: every node at --center (one rooftop)",
    )
    p.add_argument("--rx-alt-ft", type=float, default=100.0, help="receiver altitude, ft above sea level")
    p.add_argument("--out", help="write the per-window rows here as Parquet (e.g. data/classification.parquet)")
    p.add_argument(
        "--tagged",
        dest="blind",
        action="store_false",
        default=True,
        help="feed per-detection ADS-B tags to association (default is blind, as association_bench)",
    )
    p.add_argument("--min-aircraft", type=int, default=10)
    p.add_argument("--max-aircraft", type=int, default=16)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--frame-interval", type=float, default=2.0)
    p.add_argument("--assoc-interval", type=float, default=6.0, help="seconds between a node's association rounds")
    p.add_argument("--min-frames", type=int, default=3, help="drop windows in which a target was seen fewer times")
    args = p.parse_args(argv)
    unknown = [k for k in parse_airframe_mix(args.airframe_mix) if k not in AIRFRAMES]
    if unknown:
        p.error(f"unknown airframe(s) {unknown}")
    return args


def main(argv=None) -> Result:
    args = parse_args(argv)
    command = "python scripts/classification_bench.py " + " ".join(sys.argv[1:] if argv is None else argv)
    res = run(args)
    report(res, args, command)
    if args.out:
        write_parquet(res.rows, args.out)
        print(f"\nwrote {len(res.rows)} rows to {args.out}")
    return res


if __name__ == "__main__":
    main()
