"""Receivers that move: per-frame receiver position and velocity for the solver.

A node on a moving platform (a vehicle, a boat, a person walking) sends, with
each detection frame, where its receiver was and how fast it was moving at
the frame's timestamp, as estimated on the node from its GNSS:

    frame["rx_motion"] = {
        "lat": deg, "lon": deg, "alt_m": m,
        "vel_east_ms": m/s, "vel_north_ms": m/s, "vel_up_ms": m/s,
        "pos_sd_m": m (optional), "vel_sd_ms": m/s (optional),
    }

Absent or null means a stationary receiver: nothing here changes for it, and
it is solved from its registered config exactly as before.

Two things change for a moving receiver, and both are a matter of handing the
solver the right node config rather than changing how it solves:

* **Geometry.**  The bistatic delay is measured from where the receiver is
  now, not from its registered position.  The config's rx_lat / rx_lon /
  rx_alt_ft are replaced with the receiver's position at the solve's epoch.
* **The receiver's own Doppler.**  A moving receiver adds
  (fc/c)(u_RP + u_TR).v_R to every detection, which the solver would otherwise
  read as aircraft velocity (several Hz at walking pace, tens at road speed).
  retina_geolocator models that term when the config carries rx_vel_east_ms,
  rx_vel_north_ms and rx_vel_up_ms.

Recorded on ingest, before the per-node rate limit, so a frame dropped from
the heavy pipeline still contributes its pose; applied on the way into the
solver (solver_pool), which every solve path goes through.  The epoch a solve
describes is its input's timestamp_ms, which epoch alignment sets to the
newest measurement's time; the pose is interpolated there.
"""

import bisect
import math
import threading
from collections import deque

from core import state

# Poses kept per node: comfortably more than the solver's 45 s staleness
# window at one frame a second.
_HISTORY = 240
# A pose further than this from the solve's epoch is not used: the receiver
# could be anywhere by then, and the registered position is no better.
_MAX_GAP_MS = 3000
# Faster than this is not a ground or sea platform; a value past it is a bad
# estimate, not a fast receiver.
_MAX_SPEED_MS = 100.0
_KEYS = ("lat", "lon", "alt_m", "vel_east_ms", "vel_north_ms", "vel_up_ms")
_FT_PER_M = 1.0 / 0.3048

_lock = threading.Lock()
_poses: dict[str, deque] = {}


def _reset_for_tests() -> None:
    with _lock:
        _poses.clear()


def parse(rx_motion) -> tuple | None:
    """(lat, lon, alt_m, ve, vn, vu) from a frame's rx_motion, or None when it
    is absent, null or not a usable estimate."""
    if not isinstance(rx_motion, dict):
        return None
    try:
        vals = tuple(float(rx_motion.get(k) if rx_motion.get(k) is not None else 0.0) for k in _KEYS)
    except (TypeError, ValueError):
        return None
    if rx_motion.get("lat") is None or rx_motion.get("lon") is None:
        return None
    lat, lon, alt_m, ve, vn, vu = vals
    if not all(math.isfinite(v) for v in vals):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and -500.0 <= alt_m <= 10_000.0):
        return None
    if math.sqrt(ve * ve + vn * vn + vu * vu) > _MAX_SPEED_MS:
        return None
    return vals


def record(node_id: str, frame: dict) -> bool:
    """File the frame's receiver pose under the node.  True if one was filed."""
    pose = parse(frame.get("rx_motion"))
    if pose is None:
        if frame.get("rx_motion") is not None:
            state.bump_counter("rx_motion_rejected")
        return False
    t_ms = frame.get("timestamp")
    if t_ms is None:
        return False
    t_ms = int(t_ms)
    with _lock:
        hist = _poses.setdefault(node_id, deque(maxlen=_HISTORY))
        if hist and t_ms <= hist[-1][0]:
            # Out of order or repeated: keep the history sorted for bisect.
            if t_ms == hist[-1][0]:
                hist[-1] = (t_ms, *pose)
            return True
        hist.append((t_ms, *pose))
    return True


def is_moving(node_id: str) -> bool:
    """True if the node has sent a receiver pose (it is on a moving platform)."""
    with _lock:
        return bool(_poses.get(node_id))


def pose_at(node_id: str, t_ms: float) -> tuple | None:
    """The receiver's (lat, lon, alt_m, ve, vn, vu) at t_ms: linear between the
    poses either side, or the nearest one moved on by its velocity when t_ms is
    outside them; None if the nearest is more than _MAX_GAP_MS away."""
    with _lock:
        hist = list(_poses.get(node_id) or ())
    if not hist:
        return None
    times = [h[0] for h in hist]
    i = bisect.bisect_left(times, t_ms)
    if 0 < i < len(hist) and times[i] - times[i - 1] <= 2 * _MAX_GAP_MS:
        a, b = hist[i - 1], hist[i]
        f = (t_ms - a[0]) / (b[0] - a[0])
        return tuple(a[k] + f * (b[k] - a[k]) for k in range(1, 7))
    near = min((h for h in (hist[i - 1] if i > 0 else None, hist[i] if i < len(hist) else None) if h),
               key=lambda h: abs(h[0] - t_ms))  # fmt: skip
    dt_s = (t_ms - near[0]) / 1000.0
    if abs(dt_s) * 1000.0 > _MAX_GAP_MS:
        return None
    lat, lon, alt_m, ve, vn, vu = near[1:]
    lat2 = lat + math.degrees(vn * dt_s / 6_371_000.0)
    lon2 = lon + math.degrees(ve * dt_s / (6_371_000.0 * max(math.cos(math.radians(lat)), 1e-6)))
    return lat2, lon2, alt_m + vu * dt_s, ve, vn, vu


def apply(s_in: dict, node_cfgs: dict) -> dict:
    """``node_cfgs`` with every moving node's receiver placed at the solve's
    epoch and carrying its velocity.  Stationary nodes are passed through as
    the same objects; the input dict is never mutated."""
    if not node_cfgs or not isinstance(s_in, dict):
        return node_cfgs
    with _lock:
        moving = [nid for nid in node_cfgs if _poses.get(nid)]
    if not moving:
        return node_cfgs
    t_ms = s_in.get("timestamp_ms")
    if t_ms is None:
        ts = [m.get("t_s") for m in (s_in.get("measurements") or ()) if m.get("t_s") is not None]
        t_ms = max(ts) * 1000.0 if ts else None
    out = dict(node_cfgs)
    for nid in moving:
        pose = pose_at(nid, t_ms) if t_ms is not None else None
        if pose is None:
            # A moving node with no pose near this epoch: solving it at its
            # registered position would put the error the compensation
            # exists to remove straight back in, so it is left out.
            out.pop(nid)
            state.bump_counter("rx_motion_stale")
            continue
        lat, lon, alt_m, ve, vn, vu = pose
        out[nid] = {
            **node_cfgs[nid],
            "rx_lat": lat,
            "rx_lon": lon,
            "rx_alt_ft": alt_m * _FT_PER_M,
            "rx_vel_east_ms": ve,
            "rx_vel_north_ms": vn,
            "rx_vel_up_ms": vu,
        }
        state.bump_counter("rx_motion_applied")
    return out
