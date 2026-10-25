"""Classify a target from its motion alone: the baseline classifier.

No identity, no signature, no learned weights.  A target is described by what
the radar measured it doing (how fast it moves, how high, how hard it turns,
how fast it climbs, whether it ever stops) and each class in
target_identity.TARGET_CLASSES is given a flight envelope over those same
quantities.  A class scores the product of how well each measured feature sits
inside its envelope, weighted by how common the class is, and the scores are
normalised into a distribution.  It is
deliberately the simplest thing that could work, so anything cleverer has a
number to beat: services.target_classification scores it against the
hex-lookup label wherever a target has both.

Envelopes are trapezoids (zero, full, full, zero).  A feature the target did
not yield (no altitude on a single-node track, no turn rate on a straight line
too short to measure) is left out of the product rather than guessed, so it
neither helps nor hurts any class and the confidence falls with the evidence.

Position noise is the trap.  A multinode solve is good to a few hundred metres,
so speed differenced between consecutive 1 Hz points is mostly noise; every
rate here is taken over chords at least MIN_CHORD_S long instead.

Single-node tracks have no usable position at all (the drawn point is an arc
midpoint), but their Doppler still bounds the speed from below: the bistatic
range rate is at most twice the target's speed, so |v| >= lambda * |f_d| / 2.
That floor alone rules a drone out of a target the radar sees closing at
100 m/s.
"""

import math
from dataclasses import dataclass, field

from services.geo import haversine_km
from services.target_identity import (
    BALLOON,
    DRONE,
    FAST_FIXED_WING,
    GLIDER,
    LIGHT_FIXED_WING,
    ROTORCRAFT,
    TARGET_CLASSES,
)

C_M_S = 299_792_458.0
MIN_CHORD_S = 10.0
# A chord this slow is a hover.  Below the jitter a solve's position carries
# over MIN_CHORD_S, so a hovering target reads as hovering rather than moving.
HOVER_SPEED_MS = 5.0
# A heading is only meaningful while the target is going somewhere.
MIN_HEADING_SPEED_MS = 10.0
# Below this a "best" class is no better than a guess between two or three.
MIN_CONFIDENCE = 0.35
# Nor is a verdict the motion did not move: the best class must have gained at
# least this much over its prior, or the answer is the prior talking.
MIN_LIFT = 1.25
UNKNOWN = "unknown"

_INF = math.inf


@dataclass(frozen=True)
class Envelope:
    speed_ms: tuple[float, float, float, float]
    alt_m: tuple[float, float, float, float]
    turn_dps: tuple[float, float, float, float]
    climb_ms: tuple[float, float, float, float]
    can_hover: bool
    # How common the class is among the targets a node sees, before any motion
    # is measured.  Envelopes overlap (a light aircraft and a helicopter both
    # cruise at 50 m/s), and where the motion cannot split them the commoner
    # one is the better guess.  Drones are weighted above their share of the
    # sky because missing one is the costlier mistake.
    prior: float


# The full (flat) span of each trapezoid is how the class typically flies, the
# tapers how far it can go.  Where a class is unbounded below, an open end
# means "anything under the typical span is fine", not "impossible".
ENVELOPES: dict[str, Envelope] = {
    FAST_FIXED_WING: Envelope(
        speed_ms=(50, 110, 300, 400),
        alt_m=(-300, 1_500, 13_000, 18_000),
        turn_dps=(-_INF, 0, 3.5, 8),
        climb_ms=(-_INF, 0, 20, 50),
        can_hover=False,
        prior=0.45,
    ),
    LIGHT_FIXED_WING: Envelope(
        speed_ms=(15, 30, 65, 110),
        alt_m=(-_INF, 0, 3_000, 7_000),
        turn_dps=(-_INF, 0, 6, 15),
        climb_ms=(-_INF, 0, 5, 10),
        can_hover=False,
        prior=0.25,
    ),
    ROTORCRAFT: Envelope(
        speed_ms=(-_INF, 0, 70, 90),
        alt_m=(-_INF, 0, 1_000, 4_000),
        turn_dps=(-_INF, 0, 5, 15),
        climb_ms=(-_INF, 0, 8, 15),
        can_hover=True,
        prior=0.10,
    ),
    GLIDER: Envelope(
        speed_ms=(12, 18, 45, 70),
        alt_m=(0, 300, 3_500, 8_000),
        turn_dps=(-_INF, 0, 15, 25),
        climb_ms=(-_INF, 0, 5, 10),
        can_hover=False,
        prior=0.05,
    ),
    BALLOON: Envelope(
        speed_ms=(-_INF, 0, 12, 25),
        alt_m=(-_INF, 0, 35_000, 40_000),
        turn_dps=(-_INF, 0, 2, 5),
        climb_ms=(-_INF, 0, 5, 8),
        can_hover=True,
        prior=0.05,
    ),
    DRONE: Envelope(
        speed_ms=(-_INF, 0, 20, 40),
        alt_m=(-_INF, 0, 150, 600),
        turn_dps=(-_INF, 0, 45, 90),
        climb_ms=(-_INF, 0, 8, 15),
        can_hover=True,
        prior=0.10,
    ),
}
assert set(ENVELOPES) == set(TARGET_CLASSES)

# A fixed-wing target that hovers for a fifth of its chords is not fixed wing,
# but one noisy chord is not a hover, so the penalty scales with the fraction.
_HOVER_PENALTY = 0.02


def trapezoid(x: float, env: tuple[float, float, float, float]) -> float:
    a, b, c, d = env
    if b <= x <= c:
        return 1.0
    if x < b:
        return 1.0 if a == -_INF else max(0.0, (x - a) / (b - a))
    if x >= d:
        return 0.0
    return (d - x) / (d - c)


def at_least(floor: float, env: tuple[float, float, float, float]) -> float:
    """How plausible "faster than ``floor``" is under an envelope: its upper side only."""
    _, _, c, d = env
    if floor <= c:
        return 1.0
    if floor >= d:
        return 0.0
    return (d - floor) / (d - c)


@dataclass
class MotionFeatures:
    span_s: float = 0.0
    n_chords: int = 0
    speed_ms: float | None = None
    max_speed_ms: float | None = None
    speed_floor_ms: float | None = None
    alt_m: float | None = None
    turn_dps: float | None = None
    climb_ms: float | None = None
    hover_fraction: float | None = None

    def as_dict(self) -> dict:
        out = {"span_s": round(self.span_s, 1), "n_chords": self.n_chords}
        for k in ("speed_ms", "max_speed_ms", "speed_floor_ms", "alt_m", "turn_dps", "climb_ms", "hover_fraction"):
            v = getattr(self, k)
            if v is not None:
                out[k] = round(v, 2)
        return out


@dataclass
class MotionClass:
    cls: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    features: MotionFeatures = field(default_factory=MotionFeatures)

    def as_dict(self) -> dict:
        return {
            "class": self.cls,
            "confidence": round(self.confidence, 3),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "features": self.features.as_dict(),
        }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def doppler_speed_floor_ms(doppler_hz: float | None, fc_hz: float | None) -> float | None:
    """The least speed a bistatic Doppler shift implies: lambda * |f_d| / 2."""
    if doppler_hz is None or not fc_hz or fc_hz <= 0:
        return None
    return (C_M_S / fc_hz) * abs(doppler_hz) / 2.0


def _climb_ms(pts) -> float | None:
    """Mean vertical rate between the trail's first and last thirds.

    A solved altitude is the noisiest coordinate a dark solve gives, so it is
    never differenced chord by chord: the median of each third stands in for
    its altitude, and the rate is taken over the time between them.
    """
    alts = [(p[0], p[3]) for p in pts if p[3] is not None]
    if len(alts) < 6:
        return None
    third = len(alts) // 3
    head, tail = alts[:third], alts[-third:]
    dt = _median([t for t, _ in tail]) - _median([t for t, _ in head])
    if dt < 3 * MIN_CHORD_S:
        return None
    return abs(_median([a for _, a in tail]) - _median([a for _, a in head])) / dt


def extract_features(
    trail: list[tuple[float, float, float, float | None]] | None,
    *,
    speed_floor_ms: float | None = None,
    alt_m: float | None = None,
    speed_ms: float | None = None,
) -> MotionFeatures:
    """Motion features from a trail of ``(t_s, lat, lon, alt_m | None)``, oldest first.

    ``alt_m`` and ``speed_ms`` stand in for the trail's own when it yields none:
    an altitude an ADS-B fix vouches for, or the speed an entry reports before
    its trail is a chord long.
    """
    f = MotionFeatures(speed_floor_ms=speed_floor_ms)
    pts = sorted((p for p in (trail or ()) if p[0] is not None), key=lambda p: p[0])
    if len(pts) >= 2:
        f.span_s = pts[-1][0] - pts[0][0]
    chords = []
    i = 0
    for j in range(1, len(pts)):
        dt = pts[j][0] - pts[i][0]
        if dt >= MIN_CHORD_S:
            chords.append((pts[i], pts[j], dt))
            i = j
    f.n_chords = len(chords)
    speeds, headings = [], []
    for a, b, dt in chords:
        v = haversine_km(a[1], a[2], b[1], b[2]) * 1000.0 / dt
        speeds.append(v)
        headings.append((_bearing_deg(a[1], a[2], b[1], b[2]), v, a[0], b[0]))
    f.speed_ms = _median(speeds) if speeds else speed_ms
    if len(speeds) >= 2:
        f.max_speed_ms = sorted(speeds)[min(len(speeds) - 1, int(0.9 * len(speeds)))]
    f.climb_ms = _climb_ms(pts)
    if speeds:
        f.hover_fraction = sum(1 for v in speeds if v < HOVER_SPEED_MS) / len(speeds)
    turns = []
    for (h1, v1, t1a, t1b), (h2, v2, t2a, t2b) in zip(headings, headings[1:]):
        if v1 < MIN_HEADING_SPEED_MS or v2 < MIN_HEADING_SPEED_MS:
            continue
        dh = abs((h2 - h1 + 180) % 360 - 180)
        dt = (t2a + t2b) / 2 - (t1a + t1b) / 2
        if dt > 0:
            turns.append(dh / dt)
    f.turn_dps = _median(turns)
    alts = [p[3] for p in pts if p[3] is not None]
    f.alt_m = _median(alts) if alts else alt_m
    return f


def classify_features(f: MotionFeatures) -> MotionClass:
    """Score every class's envelope against the features and pick the best."""
    informative = (
        f.speed_ms is not None or f.speed_floor_ms is not None or f.alt_m is not None or f.hover_fraction is not None
    )
    if not informative:
        return MotionClass(UNKNOWN, 0.0, {}, f)
    raw: dict[str, float] = {}
    for cls, env in ENVELOPES.items():
        s = env.prior
        if f.speed_ms is not None:
            s *= trapezoid(f.speed_ms, env.speed_ms)
        if f.max_speed_ms is not None:
            s *= at_least(f.max_speed_ms, env.speed_ms)
        if f.speed_floor_ms is not None:
            s *= at_least(f.speed_floor_ms, env.speed_ms)
        if f.alt_m is not None:
            s *= trapezoid(f.alt_m, env.alt_m)
        if f.turn_dps is not None:
            s *= trapezoid(f.turn_dps, env.turn_dps)
        if f.climb_ms is not None:
            s *= trapezoid(f.climb_ms, env.climb_ms)
        if f.hover_fraction and not env.can_hover:
            s *= max(0.0, 1.0 - f.hover_fraction) * (_HOVER_PENALTY ** min(1.0, f.hover_fraction * 5))
        raw[cls] = s
    total = sum(raw.values())
    if total <= 0:
        return MotionClass(UNKNOWN, 0.0, {k: 0.0 for k in raw}, f)
    scores = {k: v / total for k, v in raw.items()}
    best = max(scores, key=scores.get)
    conf = scores[best]
    decided = conf >= MIN_CONFIDENCE and conf >= MIN_LIFT * ENVELOPES[best].prior
    return MotionClass(best if decided else UNKNOWN, conf, scores, f)


def classify(
    trail: list[tuple[float, float, float, float | None]] | None,
    *,
    speed_floor_ms: float | None = None,
    alt_m: float | None = None,
    speed_ms: float | None = None,
) -> MotionClass:
    return classify_features(extract_features(trail, speed_floor_ms=speed_floor_ms, alt_m=alt_m, speed_ms=speed_ms))
